#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Work out what a port consists of by looking at the generator, not at a description of it.

Every field here used to be read from `agentic_port/manifests/<op>.yaml` in tt-dm-codegen. That file
is not ours: its schema says it is "written by the classify stage" of the porting orchestrator that
lives in the same repository, for that orchestrator's own translate, review and gate phases. We were
reading a sibling pipeline's working notes, which is why they drift out from under us -- nobody
maintains them on our behalf and nothing tells us when they change.

The drift is not hypothetical. `tilize.yaml` lists eight `kernel_paths`, but `ops/tilize/spec.py`
selects `compute_tilize_typecast.cpp` on the numeric-fuse path and `writer_blockfloat_rne.cpp` on the
blockfloat path, and neither appears in the list. The same omission on `untilize` left
`rm_shard_split.h` unvendored and 112 cases writing uncorrelated data, which we diagnosed by hand and
fixed by copying the header in rather than by fixing the reason it was missing.

So each field is resolved one of three ways, in descending order of preference:

  measured    the prototype leg decides which cases codegen can serve, replacing every declaration
              of scope. `gate.py` already grades this way -- a case the prototype fails is excused --
              so the declarations were only ever an approximation of something we measure anyway.
  derived     read off the generator tree, the tt-metal tree or the ttnn module, then verified. A
              derived name that turns out not to exist stops the run here, at second zero, instead of
              yielding a plausible wrong answer forty minutes later on hardware.
  declared    what genuinely cannot be derived, kept in `axes/<op>.yaml` beside this file, in our
              own repository, on our own branch. Six lines an op.

The only irreducible declaration is the sweep axis map. A sweep names its parameters whatever it
likes and sometimes buries them inside a bundled one (`ri_specs.shape`, `pad_specs.padding`), so no
amount of looking at the tree recovers which parameter carries the shape.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AXES = HERE / "axes"

# Where a generator keeps templates shared between ops. `f895be71b` moved them from the first to the
# second, so both are tried rather than one being hardcoded: a repin across that commit would
# otherwise resolve every shared kernel to nothing.
SHARED_TEMPLATE_DIRS = ("common/templates", "common/kernels/codegen")

KERNEL_SUFFIXES = (".cpp", ".h", ".hpp")

# A kernel filename as it appears quoted in builder source, e.g. `"writer_blockfloat_rne.cpp"`. The
# builder picks between kernels at build time, so the set it can reach is the set it mentions.
QUOTED_KERNEL = re.compile(r"""['"]([A-Za-z0-9_./-]+\.(?:cpp|h|hpp))['"]""")

# `#include "rm_shard_split.h"`. Angle-bracket includes are the toolchain's, not the generator's.
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


class Missing(SystemExit):
    """A name that had to exist did not. Raised rather than returned so no caller can ignore it."""

    def __init__(self, what: str) -> None:
        super().__init__(f"discover: {what}")


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def resolve_category(repo: Path, op: str, explicit: str | None = None) -> str:
    """Which directory under `ttnn/cpp/ttnn/operations` this op lives in.

    The manifest's `native_entry` used to be the tiebreaker, because `untilize` exists twice --
    `data_movement/untilize` and `experimental/quasar/untilize`, identical down to the filenames. But
    the entry point only ever encoded whether the op is mainline or experimental, and the tree already
    says that: a mainline op is the one that is not under `experimental/`.

    Ambiguity that survives that rule is a genuine question rather than something to guess at, so it
    stops the run and asks for `--category`. That is a better failure than picking one, because
    picking wrong scaffolds a port into a directory nobody is looking at.
    """
    if explicit:
        chosen = repo / "ttnn/cpp/ttnn/operations" / explicit / op
        if not chosen.is_dir():
            raise Missing(f"--category {explicit} has no {op} directory at {chosen}")
        return explicit

    root = repo / "ttnn/cpp/ttnn/operations"
    if not root.is_dir():
        raise Missing(f"no ttnn operations tree at {root}")

    found = []
    for path in sorted(root.glob(f"*/{op}")) + sorted(root.glob(f"*/*/{op}")):
        # A port's own `codegen` subdirectory matches when an op is named after its parent.
        if path.is_dir() and "codegen" not in path.relative_to(root).parts[:-1]:
            found.append(str(path.parent.relative_to(root)))

    mainline = [c for c in found if not c.startswith("experimental")]
    for candidates, note in ((mainline, "mainline"), (found, "experimental")):
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise Missing(
                f"{op} names more than one {note} directory, so pass --category to choose: "
                + ", ".join(candidates)
            )
    raise Missing(f"no directory named {op} under {root}")


def resolve_builder(codegen_root: Path, op: str) -> str:
    """The generator source the agent transliterates.

    `ops/<op>/spec.py` in six of the seven manifests. The seventh is `move`, which shares
    `ops/identity/spec.py` -- an op does not have to have a builder to itself. So convention is tried
    first and a search second, rather than convention being assumed.
    """
    conventional = Path("ops") / op / "spec.py"
    if (codegen_root / conventional).is_file():
        return str(conventional)

    # Some other op's builder serves this one. The one that names it is the one that builds it.
    hits = []
    for path in sorted((codegen_root / "ops").glob("*/*.py")):
        if path.name not in ("spec.py", "builder.py"):
            continue
        if re.search(rf"\b{re.escape(op)}\b", _read(path)):
            hits.append(str(path.relative_to(codegen_root)))
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise Missing(f"{op} is named by more than one builder, so it is ambiguous: " + ", ".join(hits))
    raise Missing(f"no builder for {op} under {codegen_root}/ops")


def shared_template_dir(codegen_root: Path) -> Path:
    for candidate in SHARED_TEMPLATE_DIRS:
        if (codegen_root / candidate).is_dir():
            return codegen_root / candidate
    raise Missing(
        f"no shared template directory under {codegen_root}; tried " + ", ".join(SHARED_TEMPLATE_DIRS)
    )


def kernel_seeds(codegen_root: Path, op: str, builder: str) -> set[str]:
    """Every kernel filename the builder could reach, before following includes.

    Two sources, because neither alone is complete. The op's own template directory covers the
    kernels it selects between at build time -- `compute_tilize.cpp` versus
    `compute_tilize_typecast.cpp` -- which a curated list gets wrong as soon as a variant is added.
    Builder source covers shared kernels the op reaches into, like `writer_blockfloat_rne.cpp`, which
    live outside the op's directory and so would otherwise be invisible.

    Over-inclusion is the safe direction and is chosen deliberately: an extra vendored kernel costs a
    file in the diff, a missing one costs a silent wrong answer on hardware.
    """
    seeds = set()

    op_templates = codegen_root / "ops" / op / "templates"
    if op_templates.is_dir():
        seeds |= {p.name for p in op_templates.iterdir() if p.suffix in KERNEL_SUFFIXES}

    # The builder's whole directory, not just the one file: builders split helpers across modules and
    # a kernel named in a sibling is as reachable as one named in `spec.py`.
    for path in sorted((codegen_root / builder).parent.glob("*.py")):
        for name in QUOTED_KERNEL.findall(_read(path)):
            # Templates are referenced by basename; a dotted path in the source is a python import.
            if "/" not in name:
                seeds.add(name)
    return seeds


def kernel_closure(codegen_root: Path, op: str, builder: str) -> tuple[list[str], list[str]]:
    """Resolve the seeds against the template directories and follow their includes.

    The include walk is what a hand-written list structurally cannot get right. `rm_shard_split.h` is
    not a kernel anyone selects -- it is a header another kernel includes -- so it appears in no list
    of kernels and its absence surfaces only as wrong data at runtime. Following includes finds it
    for the same reason a compiler would.

    Returns the repo-relative paths to vendor, and the seed names that resolved to nothing.
    """
    op_templates = codegen_root / "ops" / op / "templates"
    shared = shared_template_dir(codegen_root)

    resolved: dict[str, Path] = {}
    unresolved: set[str] = set()
    pending = sorted(kernel_seeds(codegen_root, op, builder))

    while pending:
        name = pending.pop()
        base = name.rsplit("/", 1)[-1]
        if base in resolved or base in unresolved:
            continue
        for directory in (op_templates, shared):
            candidate = directory / base
            if candidate.is_file():
                resolved[base] = candidate
                pending.extend(INCLUDE.findall(_read(candidate)))
                break
        else:
            unresolved.add(base)

    paths = sorted(str(p.relative_to(codegen_root)) for p in resolved.values())
    # Includes reach headers the toolchain provides too, so an unresolved name is only worth
    # reporting when it looks like it came from this generator.
    return paths, sorted(n for n in unresolved if n.endswith(KERNEL_SUFFIXES))


def force_entries(op: str, category: str) -> dict[str, str]:
    """The two private entry points a verify run calls to pin one implementation.

    Conventional since the public op stopped taking an `implementation` argument, and the convention
    is checkable: `measure.py` resolves these against the built ttnn and fails loudly when a name is
    not bound. A declaration cannot be checked the same way -- one naming a symbol that does not
    exist yields forced-native measurements on both legs, which reads as a port that agrees with
    itself perfectly.
    """
    family = category.replace("/", "_") or "data_movement"
    module = f"ttnn._ttnn.operations.{family}"
    return {
        "force_native": f"{module}.{op}_force_native",
        "force_codegen": f"{module}.{op}_force_codegen",
    }


def load_axes(op: str) -> dict:
    """The sweep axis map, and which suites to enumerate. The one thing still written down.

    `yaml` is imported here rather than at module scope so that `--category-only` runs on a bare
    runner. Resolving the category is the first thing a run needs and happens before any dependency is
    installed, and a top-level import would make that step fail on a missing PyYAML while asking a
    question that needs nothing but the directory listing.
    """
    import yaml

    path = AXES / f"{op}.yaml"
    if not path.is_file():
        raise Missing(
            f"no axis map for {op} at {path}. It names which sweep parameter carries the shape, the "
            "dtype and the layout, which cannot be derived because a sweep names its parameters "
            "freely. Copy a neighbouring file and adjust it."
        )
    axes = yaml.safe_load(path.read_text()) or {}
    for required in ("shape", "dtype", "layout"):
        if not axes.get(required):
            raise Missing(f"{path} names no {required} parameter")
    return axes


def resolve_sweep(op: str, axes: dict) -> dict:
    """Which sweep defines the graded surface.

    `common.sweeps.codegen_<op>` in seven of seven manifests, so it is convention rather than
    information. Verification is the import itself, which happens in `ledger.py` where the module is
    actually needed; naming it here without importing keeps this module usable off the runner.
    """
    return {
        "module": axes.get("sweep_module") or f"common.sweeps.codegen_{op}",
        "suites": axes.get("suites") or None,
    }


def describe(repo: Path, codegen_root: Path, op: str, category: str | None = None) -> dict:
    axes = load_axes(op)
    resolved_category = resolve_category(repo, op, category)
    builder = resolve_builder(codegen_root, op)
    kernels, unresolved = kernel_closure(codegen_root, op, builder)
    if not kernels:
        raise Missing(f"{op} resolved no kernels, which no port can be true of")

    descriptor = {
        "op": op,
        "category": resolved_category,
        "native_entry": f"ttnn.{op}" if not resolved_category.startswith("experimental") else None,
        "builder": builder,
        "kernels": kernels,
        "unresolved_kernels": unresolved,
        "sweep": resolve_sweep(op, axes),
        "axes": {k: v for k, v in axes.items() if k not in ("sweep_module", "suites")},
    }
    descriptor.update(force_entries(op, resolved_category))
    return descriptor


def compare_manifest(descriptor: dict, manifest_path: Path) -> list[str]:
    """Report where the manifest we no longer read disagrees with what is actually there.

    Kept as reporting only, and deliberately not as a gate. The point of this module is that the
    manifest is not authoritative, so a disagreement is evidence about the manifest, not about the
    port. It is worth surfacing because it is exactly the signal that would have saved the
    `rm_shard_split.h` afternoon, and because whoever maintains that file deserves to be told.
    """
    if not manifest_path.is_file():
        return []
    manifest = yaml.safe_load(manifest_path.read_text()) or {}

    notes = []
    declared = {p.rsplit("/", 1)[-1] for p in (manifest.get("kernel_paths") or [])}
    found = {p.rsplit("/", 1)[-1] for p in descriptor["kernels"]}
    for name in sorted(found - declared):
        notes.append(f"kernel reachable but unlisted in the manifest: {name}")
    for name in sorted(declared - found):
        notes.append(f"kernel listed in the manifest but not found in the generator: {name}")

    for field, ours in (("codegen_builder", descriptor["builder"]), ("native_entry", descriptor["native_entry"])):
        theirs = manifest.get(field)
        if theirs and ours and theirs != ours:
            notes.append(f"{field}: manifest says {theirs}, the tree says {ours}")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True)
    ap.add_argument("--repo", default=".", help="tt-metal checkout")
    ap.add_argument("--codegen-root", default=".codegen", help="pinned tt-dm-codegen checkout")
    ap.add_argument("--category", default=None, help="only needed when the op name is ambiguous")
    ap.add_argument(
        "--category-only",
        action="store_true",
        help="print just the category and stop; needs the tt-metal tree and nothing else",
    )
    ap.add_argument("--out", default=None, help="write JSON here instead of stdout")
    ap.add_argument(
        "--compare-manifest",
        default=None,
        help="report where a manifest disagrees with the tree; never changes the result",
    )
    args = ap.parse_args()

    if args.category_only:
        print(resolve_category(Path(args.repo), args.op, args.category))
        return 0

    descriptor = describe(Path(args.repo), Path(args.codegen_root), args.op, args.category)

    if args.compare_manifest:
        for note in compare_manifest(descriptor, Path(args.compare_manifest)):
            print(f"discover: {note}", file=sys.stderr)

    for name in descriptor["unresolved_kernels"]:
        print(f"discover: warning: {name} is referenced but was not found in any template directory", file=sys.stderr)

    text = json.dumps(descriptor, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
