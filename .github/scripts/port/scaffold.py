#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic scaffold for a codegen op port.

Creates everything the translate agent should not have to invent: the kernel copies, the empty
host files, and the build registration. Running this before the initial build means the agent only
ever edits the *contents* of files that already exist and are already registered, so every rebuild
it triggers is a plain incremental compile with no CMake re-configure.

The kernel set is read from the port manifest's `kernel_paths`, which is authoritative -- it names
exactly the tt-dm-codegen templates the generator actually uses, so nothing here has to guess.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# The three DeviceOperation components every codegen op scaffolds under `codegen/`.
COMPONENTS = ("device_operation", "program_factory", "supported")

# tt-metal's `check-spdx-licenses` static check requires this on every in-tree source. Kernel
# templates ship header-less from tt-dm-codegen, so they are stamped on copy.
SPDX = "// SPDX-FileCopyrightText: © {year} Tenstorrent USA, Inc.\n" "//\n" "// SPDX-License-Identifier: Apache-2.0\n"


def spdx_header() -> str:
    return SPDX.format(year=datetime.now(timezone.utc).year)


def find_op_dir(ttmetal: Path, op: str, category: str | None = None) -> Path:
    """Locate the native op directory without hardcoding its category.

    Several op names appear both as a top-level category op and under `experimental/`
    (`data_movement/pad` and `experimental/quasar/pad`, for instance), so the shallower match wins
    and `--category` is available when that heuristic is wrong.
    """
    if category:
        candidate = ttmetal / "ttnn/cpp/ttnn/operations" / category / op
        if not (candidate / "device").is_dir():
            sys.exit(f"scaffold: {candidate} has no device/ subdir")
        return candidate

    for pattern in (f"ttnn/cpp/ttnn/operations/*/{op}", f"ttnn/cpp/ttnn/operations/*/*/{op}"):
        dirs = [p for p in sorted(ttmetal.glob(pattern)) if (p / "device").is_dir()]
        if len(dirs) == 1:
            return dirs[0]
        if len(dirs) > 1:
            sys.exit(
                f"scaffold: ambiguous op directory for {op!r}: {[str(d) for d in dirs]} "
                "-- pass --category to disambiguate"
            )
    sys.exit(f"scaffold: no op directory with a device/ subdir found for {op!r}")


def stamp_spdx(path: Path) -> None:
    text = path.read_text()
    if "SPDX-License-Identifier" in text:
        return
    path.write_text(f"{spdx_header()}\n{text}")


def copy_kernels(manifest: dict, codegen_root: Path, op_dir: Path) -> list[Path]:
    """Copy the manifest's kernel templates into `<op>/codegen/kernels/`.

    A native kernel sharing a basename would leave two indistinguishable files in the op's working
    set, so colliding copies are prefixed and quoted includes among the copied set are repointed.
    """
    entries = manifest.get("kernel_paths") or []
    if not entries:
        sys.exit("scaffold: manifest has no kernel_paths")

    # Manifest entries may carry a trailing `# comment`; yaml keeps those out, but be defensive.
    sources = [codegen_root / str(e).split("#")[0].strip() for e in entries]
    missing = [s for s in sources if not s.is_file()]
    if missing:
        sys.exit("scaffold: manifest kernel source missing: " + ", ".join(map(str, missing)))

    names = [s.name for s in sources]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        sys.exit("scaffold: duplicate manifest basenames: " + ", ".join(dupes))

    device_kernels = op_dir / "device" / "kernels"
    native_names = set()
    if device_kernels.is_dir():
        native_names = {
            p.name
            for p in device_kernels.rglob("*")
            if p.is_file() and "codegen_templates" not in p.relative_to(device_kernels).parts
        }
    renames = {n: f"codegen_{n}" for n in names if n in native_names}

    kernels_dir = op_dir / "codegen" / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sources:
        dest = kernels_dir / renames.get(source.name, source.name)
        shutil.copy2(source, dest)
        stamp_spdx(dest)
        copied.append(dest)

    for path in copied:
        text = path.read_text()
        rewritten = text
        for old, new in renames.items():
            rewritten = rewritten.replace(f'#include "{old}"', f'#include "{new}"')
        if rewritten != text:
            path.write_text(rewritten)
    return copied


def write_stubs(op_dir: Path, op: str) -> list[Path]:
    """Create the six host files empty-but-valid so they compile and register before translate."""
    codegen_dir = op_dir / "codegen"
    codegen_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for component in COMPONENTS:
        hpp = codegen_dir / f"{op}_codegen_{component}.hpp"
        cpp = codegen_dir / f"{op}_codegen_{component}.cpp"
        if not hpp.is_file():
            hpp.write_text(f"{spdx_header()}\n#pragma once\n")
            written.append(hpp)
        if not cpp.is_file():
            cpp.write_text(f'{spdx_header()}\n#include "{hpp.name}"\n')
            written.append(cpp)
    return written


def register_sources(category_dir: Path, op: str) -> list[str]:
    """Add the three codegen `.cpp` files to the category `sources.cmake`.

    Anchored on the op's own free-function entry (`<op>/<op>.cpp`), which every ported op already
    has listed, so the codegen entries land in the op's own block rather than at the end of a list.
    """
    path = category_dir / "sources.cmake"
    text = path.read_text()
    entries = [f"{op}/codegen/{op}_codegen_{c}.cpp" for c in COMPONENTS]
    added = [e for e in entries if e not in text]
    if not added:
        return []

    anchor = f"    {op}/{op}.cpp\n"
    if anchor not in text:
        sys.exit(f"scaffold: cannot find anchor {anchor.strip()!r} in {path}")
    block = "".join(f"    {e}\n" for e in added)
    path.write_text(text.replace(anchor, block + anchor, 1))
    return added


def register_kernel_globs(category_dir: Path, op: str, kernels_dir: Path) -> list[str]:
    """Add one glob per extension present, so copied headers ship alongside the `.cpp` kernels.

    `file(GLOB_RECURSE kernels ...)` drives the installed ttnn-runtime package and the JIT resolves
    a kernel's own `#include "<local>.h"` from that install tree, so a `*.cpp`-only glob would ship
    a kernel whose header is missing -- broken packaged build, working source checkout.
    """
    path = category_dir / "CMakeLists.txt"
    text = path.read_text()
    suffixes = sorted({f.suffix for f in kernels_dir.iterdir() if f.is_file() and f.suffix})
    globs = [f"{op}/codegen/kernels/*{s}" for s in suffixes] or [f"{op}/codegen/kernels/*.cpp"]
    added = [g for g in globs if g not in text]
    if not added:
        return []

    # Land directly after the op's own native kernel globs to keep the file grouped by op.
    native = [ln for ln in text.splitlines(keepends=True) if ln.strip().startswith(f"{op}/device/kernels/")]
    if not native:
        sys.exit(f"scaffold: cannot find native kernel glob for {op!r} in {path}")
    anchor = native[-1]
    block = "".join(f"    {g}\n" for g in added)
    path.write_text(text.replace(anchor, anchor + block, 1))
    return added


def verify(op_dir: Path, op: str, kernels: list[Path]) -> list[str]:
    errors = [
        f"missing or empty: {p}"
        for c in COMPONENTS
        for s in ("hpp", "cpp")
        for p in [op_dir / "codegen" / f"{op}_codegen_{c}.{s}"]
        if not p.is_file() or p.stat().st_size == 0
    ]
    errors += [f"missing copied kernel: {p}" for p in kernels if not p.is_file()]

    category_dir = op_dir.parent
    sources_text = (category_dir / "sources.cmake").read_text()
    cmake_text = (category_dir / "CMakeLists.txt").read_text()
    for c in COMPONENTS:
        entry = f"{op}/codegen/{op}_codegen_{c}.cpp"
        if entry not in sources_text:
            errors.append(f"sources.cmake missing: {entry}")
    if f"{op}/codegen/kernels/" not in cmake_text:
        errors.append(f"CMakeLists.txt missing any glob for {op}/codegen/kernels/")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True)
    ap.add_argument("--ttmetal-home", default=".")
    ap.add_argument("--codegen-root", required=True, help="tt-dm-codegen checkout root")
    ap.add_argument("--manifest", default=None, help="defaults to <codegen-root>/agentic_port/manifests/<op>.yaml")
    ap.add_argument("--category", default=None, help="operations/<category>/<op>, when the name is ambiguous")
    args = ap.parse_args()

    ttmetal = Path(args.ttmetal_home).resolve()
    codegen_root = Path(args.codegen_root).resolve()
    manifest_path = (
        Path(args.manifest) if args.manifest else codegen_root / "agentic_port/manifests" / f"{args.op}.yaml"
    )
    if not manifest_path.is_file():
        sys.exit(f"scaffold: manifest not found: {manifest_path}")

    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    op_dir = find_op_dir(ttmetal, args.op, args.category)

    kernels = copy_kernels(manifest, codegen_root, op_dir)
    stubs = write_stubs(op_dir, args.op)
    sources_added = register_sources(op_dir.parent, args.op)
    globs_added = register_kernel_globs(op_dir.parent, args.op, op_dir / "codegen" / "kernels")

    errors = verify(op_dir, args.op, kernels)
    result = {
        "op": args.op,
        "op_dir": str(op_dir.relative_to(ttmetal)),
        "manifest": str(manifest_path),
        "kernels_copied": [str(p.relative_to(ttmetal)) for p in kernels],
        "stubs_written": [str(p.relative_to(ttmetal)) for p in stubs],
        "sources_cmake_added": sources_added,
        "kernel_globs_added": globs_added,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
