# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which sweep cells a change can move: those whose kernel it builds differently, and
those whose row in the table it changes. PR Gate (llk-sfpu-ulp-pr-check.yaml) builds the
sweep at the change and at its first parent, and runs only these cells on a card; a
cell neither rebuilds nor re-judges measures what main already measured.

A build is compared by what the card loads -- every allocated section of every ELF the
variant runs, its own and the shared ones -- not by the file: debug info names the
build directory, and two builds of the same tree differ there and nowhere else.

``--ulp-variants=PATH`` (with ``--compile-producer``) has the sweep write one
:class:`Built` per cell; ``python -m helpers.ulp_affected select`` reads it with both
builds and both tables. Run from ``python_tests``."""

import argparse
import hashlib
import json
import shutil
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .llk_params import MathOperation
from .ulp_budget_diff import resolve
from .ulp_table import BudgetKey, Table, from_plain, load_table, to_plain

# Set by --ulp-variants: where the sweep records each cell it built.
VARIANTS_PATH: Path | None = None

# Every variant runs these beside its own ELFs, relative to the artefacts directory.
SHARED_ELF_DIR = Path("shared") / "elf"

# A build's ELFs, relative to its directory (TestConfig.build_elfs).
_ELF_DIR = "elf"

_SHF_ALLOC = 0x2
_SHT_NOBITS = 8

# A collapsed list shows this many cells and says how many it withheld.
_MAX_ROWS = 40


@dataclass(frozen=True)
class Built:
    """One sweep cell as the producer built it: its pytest node, the op and key it is
    judged under, and its build directory under the artefacts root
    (``<test source>/<variant hash>``). Cells of one variant share a build: the formats
    are runtime arguments."""

    node: str
    op: MathOperation
    key: BudgetKey
    build: str

    def describe(self) -> str:
        """As the sweep names the cell."""
        k = self.key
        return (
            f"{self.op.name} {k.input_format.name}->{k.output_format.name} "
            f"approx={k.approx_mode.name} dest_acc={k.dest_acc.name}"
        )

    def to_json(self) -> str:
        return json.dumps(to_plain(self))

    @classmethod
    def from_json(cls, line: str) -> "Built":
        return from_plain(cls, json.loads(line))


def prepare_variants_file() -> None:
    """Create ``--ulp-variants``' file empty, once per session on the controller."""
    if VARIANTS_PATH is not None:
        VARIANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        VARIANTS_PATH.write_text("", encoding="utf-8")


def record_built(node: str, op: MathOperation, key: BudgetKey, build: str) -> None:
    """Append one :class:`Built` under ``--ulp-variants``; nothing without it. One line
    per write, so the xdist workers' appends do not interleave."""
    if VARIANTS_PATH is None:
        return
    with VARIANTS_PATH.open("a", encoding="utf-8") as out:
        out.write(Built(node, op, key, build).to_json() + "\n")


def read_built(path: Path) -> list[Built]:
    return [
        Built.from_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def loaded_digest(elf: Path) -> str:
    """SHA-256 over what the card loads from *elf*: each allocated section's type,
    address and size, and the bytes of those that have any. A 32-bit little-endian
    ELF, which is what the RISC-V toolchain writes."""
    data = Path(elf).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 1:
        raise ValueError(f"{elf}: not a 32-bit little-endian ELF")
    (shoff,) = struct.unpack_from("<I", data, 0x20)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x2E)
    digest = hashlib.sha256()
    for index in range(shnum):
        _, kind, flags, address, offset, size = struct.unpack_from(
            "<IIIIII", data, shoff + index * shentsize
        )
        if not flags & _SHF_ALLOC:
            continue
        digest.update(struct.pack("<III", kind, address, size))
        if kind != _SHT_NOBITS:
            digest.update(data[offset : offset + size])
    return digest.hexdigest()


def build_digest(root: Path, build: str) -> str | None:
    """One digest over every ELF the card loads for *build* under the artefacts
    directory *root*, the shared ones included; ``None`` when *root* has no such
    build."""
    own = sorted((root / build / _ELF_DIR).glob("*.elf"))
    if not own:
        return None
    digest = hashlib.sha256()
    for elf in [*own, *sorted((root / SHARED_ELF_DIR).glob("*.elf"))]:
        digest.update(elf.relative_to(root).name.encode())
        digest.update(loaded_digest(elf).encode())
    return digest.hexdigest()


class Reason(Enum):
    """Why a cell is run."""

    REBUILT = "its kernel builds differently"
    NEW_BUILD = "the first parent builds no such kernel"
    ROW = "its row in the table changed"


@dataclass(frozen=True)
class Selected:
    built: Built
    reasons: tuple[Reason, ...]


def select(
    cells: list[Built],
    head_root: Path,
    base_root: Path,
    base_table: Table,
    head_table: Table,
) -> list[Selected]:
    """Each cell the change can move, in the order recorded. A cell whose head build is
    missing is an error: the producer recorded it, so it built it."""
    head: dict[str, str] = {}
    base: dict[str, str | None] = {}
    selected = []
    for cell in cells:
        if cell.build not in head:
            found = build_digest(head_root, cell.build)
            if found is None:
                raise ValueError(
                    f"{cell.node}: no ELF under {head_root / cell.build / _ELF_DIR}, "
                    "though the sweep recorded building it"
                )
            head[cell.build] = found
            base[cell.build] = build_digest(base_root, cell.build)
        reasons = []
        if base[cell.build] is None:
            reasons.append(Reason.NEW_BUILD)
        elif base[cell.build] != head[cell.build]:
            reasons.append(Reason.REBUILT)
        if resolve(base_table, cell.op, cell.key) != resolve(
            head_table, cell.op, cell.key
        ):
            reasons.append(Reason.ROW)
        if reasons:
            selected.append(Selected(cell, tuple(reasons)))
    return selected


def ship(selected: list[Selected], head_root: Path, out: Path) -> None:
    """Copy the head builds *selected* runs, and the shared ELFs, into *out* in the
    artefacts layout, without object files: a ``--compile-consumer`` run with *out* as
    its artefacts directory finds every ELF it needs there."""
    for build in sorted({s.built.build for s in selected}):
        shutil.copytree(
            head_root / build,
            out / build,
            ignore=shutil.ignore_patterns("obj"),
            dirs_exist_ok=True,
        )
    shutil.copytree(
        head_root / SHARED_ELF_DIR, out / SHARED_ELF_DIR, dirs_exist_ok=True
    )


def render_selection(selected: list[Selected], total: int) -> str:
    lines = ["### SFPU ULP sweep cells this change can move", ""]
    if not selected:
        lines.append(
            f"None of the {total} sweep cells: no kernel builds differently and no "
            "row changed, so the sweep has nothing to run."
        )
        return "\n".join(lines) + "\n"
    ops = sorted({s.built.op for s in selected}, key=lambda op: op.name)
    lines.append(
        f"{len(selected)} of {total} cells, over {len(ops)} op(s): "
        + ", ".join(op.name for op in ops)
        + "."
    )
    lines.append("")
    for reason in Reason:
        count = sum(reason in s.reasons for s in selected)
        if count:
            lines.append(f"- {count} because {reason.value}")
    lines += ["", "<details><summary>Cells</summary>", ""]
    ordered = sorted(selected, key=lambda s: (s.built.op.name, s.built.key.sort_key))
    for s in ordered[:_MAX_ROWS]:
        why = "; ".join(reason.value for reason in s.reasons)
        lines.append(f"- {s.built.describe()}: {why}")
    if len(selected) > _MAX_ROWS:
        lines.append(f"- … and {len(selected) - _MAX_ROWS} more")
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m helpers.ulp_affected")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser(
        "select", help="the sweep cells a change rebuilds or re-judges"
    )
    cmd.add_argument(
        "--variants",
        type=Path,
        required=True,
        help="--ulp-variants' records from the change's build",
    )
    cmd.add_argument(
        "--head-build",
        type=Path,
        required=True,
        help="the change's artefacts directory (tt-llk-build)",
    )
    cmd.add_argument(
        "--base-build",
        type=Path,
        required=True,
        help="the first parent's artefacts directory; a build missing there is new",
    )
    cmd.add_argument("--base-table", type=Path, required=True)
    cmd.add_argument("--head-table", type=Path, required=True)
    cmd.add_argument(
        "--nodes",
        type=Path,
        required=True,
        help="write the selected pytest node ids here, one per line (pytest @FILE)",
    )
    cmd.add_argument(
        "--ship",
        type=Path,
        help="copy the selected head builds here, in the artefacts layout",
    )
    cmd.add_argument("--out", type=Path, help="also write the report here")
    args = parser.parse_args(argv)
    try:
        cells = read_built(args.variants)
        if not cells:
            raise ValueError(f"{args.variants} records no cell: the build ran nothing")
        selected = select(
            cells,
            args.head_build,
            args.base_build,
            load_table(args.base_table),
            load_table(args.head_table),
        )
        args.nodes.write_text(
            "".join(f"{s.built.node}\n" for s in selected), encoding="utf-8"
        )
        if args.ship is not None and selected:
            ship(selected, args.head_build, args.ship)
    except (ValueError, OSError) as refused:
        print(f"cannot select: {refused}")
        return 2
    report = render_selection(selected, len(cells))
    print(report, end="")
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
