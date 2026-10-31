# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The sweep cells a change can move, for PR Gate (llk-sfpu-ulp-pr-check.yaml): those
whose kernel builds differently from the change's first parent, and those whose row in
the table changed. Builds are compared by what the card loads, the ELFs' allocated
sections: debug info names the build directory.

``--ulp-variants=PATH`` makes the sweep record each cell it builds. Then, from
``python_tests``, with BASE and HEAD the two artefacts directories (``tt-llk-build``)::

    python -m helpers.ulp_affected --records PATH --base BASE --head HEAD \\
        --base-table T --nodes OUT"""

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from .llk_params import MathOperation
from .ulp_budget_diff import resolve
from .ulp_table import BudgetKey, from_plain, load_table, to_plain

# Set by --ulp-variants.
VARIANTS_PATH: Path | None = None


@dataclass(frozen=True)
class Built:
    """One sweep cell: its pytest node, the op and key it is judged under, and its build
    directory under the artefacts root. Cells of one variant share a build."""

    node: str
    op: MathOperation
    key: BudgetKey
    build: str

    def describe(self) -> str:
        k = self.key
        return (
            f"{self.op.name} {k.input_format.name}->{k.output_format.name} "
            f"approx={k.approx_mode.name} dest_acc={k.dest_acc.name}"
        )


def record_built(node: str, op: MathOperation, key: BudgetKey, build: str) -> None:
    """Append one record under ``--ulp-variants``: one write per line, so xdist workers
    do not interleave."""
    if VARIANTS_PATH is not None:
        with VARIANTS_PATH.open("a", encoding="utf-8") as out:
            out.write(json.dumps(to_plain(Built(node, op, key, build))) + "\n")


def read_built(path: Path) -> list[Built]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [from_plain(Built, json.loads(line)) for line in lines if line.strip()]


def loaded_digest(elf: Path) -> str:
    """SHA-256 over each allocated section of a 32-bit little-endian ELF: its type,
    address and size, and its bytes unless it is NOBITS."""
    data = Path(elf).read_bytes()
    if data[:6] != b"\x7fELF\x01\x01":
        raise ValueError(f"{elf}: not a 32-bit little-endian ELF")
    (shoff,) = struct.unpack_from("<I", data, 0x20)
    entry_size, count = struct.unpack_from("<HH", data, 0x2E)
    digest = hashlib.sha256()
    for i in range(count):
        _, kind, flags, addr, offset, size = struct.unpack_from(
            "<6I", data, shoff + i * entry_size
        )
        if flags & 0x2:  # SHF_ALLOC
            digest.update(struct.pack("<3I", kind, addr, size))
            if kind != 8:  # SHT_NOBITS has no bytes in the file
                digest.update(data[offset : offset + size])
    return digest.hexdigest()


def build_digest(root: Path, build: str) -> str | None:
    """One digest over *build*'s ELFs and the shared ones; ``None`` when *root* has no
    such build."""
    own = sorted((root / build / "elf").glob("*.elf"))
    if not own:
        return None
    shared = sorted((root / "shared" / "elf").glob("*.elf"))
    parts = "".join(f"{elf.name}:{loaded_digest(elf)}\n" for elf in own + shared)
    return hashlib.sha256(parts.encode()).hexdigest()


def select(cells, head: Path, base: Path, base_table, head_table):
    """``(cell, why)`` for each cell the change can move, in record order."""
    picked = []
    for cell in cells:
        now = build_digest(head, cell.build)
        if now is None:
            raise ValueError(
                f"{cell.node}: recorded, but {head / cell.build} has no ELF"
            )
        was = build_digest(base, cell.build)
        why = []
        if now != was:
            why.append("new build" if was is None else "rebuilt")
        if resolve(base_table, cell.op, cell.key) != resolve(
            head_table, cell.op, cell.key
        ):
            why.append("row changed")
        if why:
            picked.append((cell, " and ".join(why)))
    return picked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m helpers.ulp_affected")
    parser.add_argument("--records", type=Path, required=True, help="--ulp-variants'")
    parser.add_argument("--base", type=Path, required=True, help="the parent's build")
    parser.add_argument("--head", type=Path, required=True, help="the change's build")
    parser.add_argument("--base-table", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True, help="picked node ids")
    args = parser.parse_args(argv)
    try:
        cells = read_built(args.records)
        if not cells:
            raise ValueError(f"{args.records} records no cell")
        # The head table is this session's (CHIP_ARCH).
        tables = load_table(args.base_table), load_table()
        picked = select(cells, args.head, args.base, *tables)
    except (ValueError, OSError) as refused:
        print(f"cannot select: {refused}")
        return 2
    args.nodes.write_text("".join(f"{cell.node}\n" for cell, _ in picked))
    print(f"### SFPU ULP: {len(picked)} of {len(cells)} sweep cells to run\n")
    for cell, why in sorted(picked, key=lambda p: p[0].describe()):
        print(f"- {cell.describe()}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
