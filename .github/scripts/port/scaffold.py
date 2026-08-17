#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic scaffold for a codegen op port.

Creates everything the translate agent should not have to invent: the kernel copies, the empty
host files, the build registration, and the routing test. Running this before the initial build
means the agent only ever edits the *contents* of files that already exist and are already
registered, so every rebuild it triggers is a plain incremental compile with no CMake re-configure.

The kernel set comes from `discover.py`, which walks the builder's own template directories and
follows their includes rather than trusting a written list. A list cannot name a header that is only
reachable because another kernel includes it, and that omission is not theoretical: it is how
`rm_shard_split.h` went unvendored and left 112 `untilize` cases writing uncorrelated data.

The routing test is generated rather than written, for the same reason the gates are not the agent's
to edit. Every out-of-scope case in the coverage ledger has to fall back to native under `auto`, that
is the entire assertion, and it is identical for every op -- so an emitter covers the whole set by
construction while an agent writing tests by hand covers whatever it thought of. `gate.py` re-renders
it on every `verify` call and refuses to measure a tree where it has drifted, which makes the test a
deliverable the agent cannot weaken but also cannot forget. The pipeline this replaces emitted the
same file from its phase 8; `test_repeat_codegen_routing.py` in tests/ttnn/nightly is its output.

Emitting it needs the sweep module, and therefore a working `ttnn`, so it does not happen during the
pre-build scaffold pass -- `--emit-test-only` runs it once the tree is built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Safe at module scope even during the pre-build pass: `ttnn_names` imports ttnn only when asked for
# a name, and answers "no name" if it cannot.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttnn_names  # noqa: E402

# The three DeviceOperation components every codegen op scaffolds under `codegen/`.
COMPONENTS = ("device_operation", "program_factory", "supported")

TEST_DIR = "tests/ttnn/nightly/unit_tests/operations"

# tt-metal's `check-spdx-licenses` static check requires this on every in-tree source. Kernel
# templates ship header-less from tt-dm-codegen, so they are stamped on copy.
SPDX = "// SPDX-FileCopyrightText: © {year} Tenstorrent USA, Inc.\n" "//\n" "// SPDX-License-Identifier: Apache-2.0\n"
# The same notice for Python, where `//` is a syntax error rather than a comment.
SPDX_PY = "# SPDX-FileCopyrightText: © {year} Tenstorrent USA, Inc.\n" "#\n" "# SPDX-License-Identifier: Apache-2.0\n"


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


def copy_kernels(descriptor: dict, codegen_root: Path, op_dir: Path, resume: bool = False) -> list[Path]:
    """Copy the op's kernel templates into `<op>/codegen/kernels/`.

    A native kernel sharing a basename would leave two indistinguishable files in the op's working
    set, so colliding copies are prefixed and quoted includes among the copied set are repointed.

    The set comes from `discover.py`, which walks the builder's template directories and follows
    includes. It used to come from a manifest's `kernel_paths`, and the difference is not cosmetic: a
    hand-maintained list cannot express a header that is reachable only because another kernel
    includes it, which is how `rm_shard_split.h` went unvendored and left 112 `untilize` cases writing
    uncorrelated data against a green build.
    """
    entries = descriptor.get("kernels") or []
    if not entries:
        sys.exit("scaffold: the descriptor names no kernels")

    sources = [codegen_root / str(e).strip() for e in entries]
    missing = [s for s in sources if not s.is_file()]
    if missing:
        sys.exit("scaffold: kernel source missing: " + ", ".join(map(str, missing)))

    names = [s.name for s in sources]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        # Two template directories holding the same basename cannot both be vendored, and picking one
        # silently would compile the wrong kernel.
        sys.exit("scaffold: two templates share a basename: " + ", ".join(dupes))

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
    # Two lists, because they answer different questions. `expected` is every kernel that must be
    # here, and it is what `verify()` checks -- a resume that kept ten files and copied one still has
    # to prove all eleven exist. `copied` is only the files this pass wrote, and it is what the include
    # rewrite below may touch, because rewriting a kernel the agent edited would be the overwrite this
    # is avoiding.
    expected = []
    copied = []
    kept = []
    for source in sources:
        dest = kernels_dir / renames.get(source.name, source.name)
        expected.append(dest)
        # Under `--resume` the kernels on the branch are the previous attempt's, and a kernel is one of
        # the few files a port is *allowed* to have edited -- the SPDX stamp and the include rewrite
        # below both mean a faithful copy is not byte-identical to its template either. Overwriting
        # would silently discard the fix that a run just spent forty minutes verifying.
        #
        # Absent files are still copied, which is the point of running this pass at all on a resume: a
        # generator that gained a header between attempts is how the previous port lost every in-scope
        # case at runtime, with a green build.
        if resume and dest.exists():
            if dest.read_bytes() != source.read_bytes():
                kept.append(dest)
            continue
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

    # Reported rather than resolved, and deliberately so. A vendored kernel that differs from its
    # template is either a fix the port needs or a change the generator made since, and telling those
    # apart needs the template as it stood when the port was written -- which only a repin has. Until
    # then, silence would hide generator changes and overwriting would discard verified fixes, so the
    # divergence is named and left alone.
    for path in kept:
        print(f"scaffold: kept {path.relative_to(op_dir)}, which differs from its template")
    if kept and copied:
        print(f"scaffold: vendored {len(copied)} new kernel(s) alongside {len(kept)} kept")
    return expected


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


def already_listed(text: str, entry: str) -> bool:
    """Whether a CMake list already carries `entry` -- as its own entry, not inside another one.

    `entry in text` was the obvious test, and it is wrong for exactly one shape of op name. The day
    `untilize` merged, scaffolding `tilize` found `tilize/codegen/kernels/*.cpp` already present,
    because it is a substring of `untilize/codegen/kernels/*.cpp`. It added nothing, then failed its
    own verification, which looks at the start of a line and was right to. Every op whose name is a
    suffix of an already-ported op has this problem: `tilize`/`untilize`, and `pad` against anything
    ending in `pad`.
    """
    return any(line.strip() == entry for line in text.splitlines())


def register_sources(category_dir: Path, op: str) -> list[str]:
    """Add the three codegen `.cpp` files to the category `sources.cmake`.

    Anchored on the op's own free-function entry (`<op>/<op>.cpp`), which every ported op already
    has listed, so the codegen entries land in the op's own block rather than at the end of a list.
    """
    path = category_dir / "sources.cmake"
    text = path.read_text()
    entries = [f"{op}/codegen/{op}_codegen_{c}.cpp" for c in COMPONENTS]
    added = [e for e in entries if not already_listed(text, e)]
    if not added:
        return []

    anchor = f"    {op}/{op}.cpp\n"
    if anchor not in text:
        sys.exit(f"scaffold: cannot find anchor {anchor.strip()!r} in {path}")
    block = "".join(f"    {e}\n" for e in added)
    path.write_text(text.replace(anchor, block + anchor, 1))
    return added


def glob_block(text: str) -> tuple[int, int]:
    """Line span of the `file(GLOB_RECURSE kernels ...)` call: first line, and the closing line.

    The op's name appears in this CMakeLists in two unrelated roles, and only one of them takes a
    glob. `untilize` has `untilize/device/kernels/*.cpp` in this call *and* two explicit
    `untilize/device/kernels/dataflow/*.cpp` entries in the `target_sources` FILE_SET further down.
    Anchoring on the last match anywhere in the file put the codegen globs in that FILES list, where
    CMake resolves every entry as a real path and expands nothing -- `Cannot find source file:
    untilize/codegen/kernels/*.cpp`, at configure time, before a single object was compiled. `pad`
    has no explicit entries, so its two glob lines were the only matches and the wrong search
    happened to be right.

    Located by paren balance rather than by finding the next `)`, so a nested call inside the glob
    list would not end the block early.
    """
    lines = text.splitlines(keepends=True)
    marker = next((i for i, line in enumerate(lines) if "GLOB_RECURSE kernels" in line), None)
    if marker is None:
        return -1, -1
    start = next((i for i in range(marker, -1, -1) if "file(" in lines[i]), marker)
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("(") - lines[i].count(")")
        if depth <= 0:
            return start, i
    return -1, -1


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
    added = [g for g in globs if not already_listed(text, g)]
    if not added:
        return []

    start, close = glob_block(text)
    if start < 0:
        sys.exit(f"scaffold: cannot find the file(GLOB_RECURSE kernels ...) call in {path}")

    lines = text.splitlines(keepends=True)
    # Beside the op's own globs to keep the call grouped by op, but confined to the call either way:
    # an op with no native glob of its own goes in last rather than somewhere it would be a path.
    own = [i for i in range(start, close) if lines[i].strip().startswith(f"{op}/")]
    at = own[-1] + 1 if own else close
    indent = re.match(r"\s*", lines[at - 1]).group(0) or "    "
    lines[at:at] = [f"{indent}{glob}\n" for glob in added]
    path.write_text("".join(lines))
    return added


# --------------------------------------------------------------------------------------------
# Routing test emitter
# --------------------------------------------------------------------------------------------

# Standalone on purpose. This file ships to nightly CI and runs there with no part of the porting
# harness present, so it repeats a few lines of input construction rather than importing them.
_TEMPLATE = """{spdx}
# Generated in full from this op's coverage data and replaced whenever that data changes, so edits
# made here do not survive.
#
# Every one of these {count} cases is outside what the codegen path supports. `ttnn.{op}` decides
# internally which implementation serves a call, so each case must be served natively and must
# produce exactly what the native path produces. They are grouped by the condition that rejects
# them.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

_DTYPES = {{
    "bfloat16": ttnn.bfloat16,
    "float32": ttnn.float32,
    "int32": ttnn.int32,
    "uint32": ttnn.uint32,
    "uint16": ttnn.uint16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
}}
_LAYOUTS = {{"row_major": ttnn.ROW_MAJOR_LAYOUT, "tile": ttnn.TILE_LAYOUT}}


def _make_input(shape, dtype):
    if dtype == "uint16":
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == "int32":
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == "uint32":
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


# (dtype, layout, shape, kwargs)
_ROUTING = [
{rows}
]

_ROUTING_IDS = [
{ids}
]


@pytest.mark.parametrize("dtype,layout,shape,kwargs", _ROUTING, ids=_ROUTING_IDS)
def test_{op}_codegen_routing(device, dtype, layout, shape, kwargs):
    torch_input = _make_input(shape, dtype)
    tt_input = ttnn.from_torch(torch_input, dtype=_DTYPES[dtype], layout=_LAYOUTS[layout], device=device)

    # The forced-native entry rather than `ttnn.{op}` itself: the public entry is the thing under
    # test here, so using it to produce the reference would compare it against itself and pass no
    # matter where it routed.
    golden = ttnn.to_torch({force_native}(tt_input, **kwargs))
    # The native call above compiled and cached its program, so a correct fallback reuses it and
    # leaves the cache flat. Only a mis-route to codegen compiles something new. Taking the
    # snapshot after the native call is what makes the two distinguishable -- before it, both a
    # fallback and a mis-route add exactly one entry.
    entries_before = device.num_program_cache_entries()
    routed = ttnn.to_torch(ttnn.{op}(tt_input, **kwargs))

    assert_equal(golden, routed)
    assert device.num_program_cache_entries() == entries_before, (
        "an unsupported case routed to the codegen path (the program cache grew); expected native"
    )
"""


def test_path(op: str, category: str) -> str:
    """Repo-relative path of the emitted routing test. Also `gate.py`'s allowed write path."""
    return f"{TEST_DIR}/{category}/test_{op}_codegen_routing.py"


def _literal(value) -> str:
    """Render a ledger value as Python source that reconstructs it.

    Refuses rather than guesses. A value that fell through to `str()` would land in the file as
    `<ttnn.Shape object at 0x...>`, and the emitted test would fail to import -- which is a worse
    outcome than no test, because it looks like a broken port rather than a broken emitter.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in sorted(value.items())) + "}"

    kind = type(value)
    module = kind.__module__ or ""
    if "ttnn" in module or "tt_lib" in module:
        if kind.__name__ == "Shape":
            return f"ttnn.Shape({_literal(list(value))})"
        # Before the enum guess below, because a name is evidence and a lowercased repr is an
        # inference. `MemoryConfig` reaches here: its repr carries C++ scope markers, so the pattern
        # below correctly refuses it, and only the name gets it into the file.
        named = ttnn_names.constant_name(value)
        if named:
            return f"ttnn.{named}"
        # `DataType.BFLOAT16`, `Layout.TILE` and friends round-trip through their lowercase ttnn
        # alias. The pattern is strict because the obvious loose version -- accept anything with a
        # dot in it -- turns the default `<module.Class object at 0x...>` repr into a confident
        # `ttnn.something object at 0x...>`, which is both wrong and unparseable.
        text = str(value)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_]+", text):
            return f"ttnn.{text.rsplit('.', 1)[-1].lower()}"
    raise TypeError(f"cannot render {kind.__name__} from {module!r} as a literal: {value!r}")


# Names that exist only in the private repository this port was generated from. tt-metal is public,
# so each of these in a landed diff points a maintainer at something they cannot open -- and the
# emitted test is the one file here written wholly by machine, which makes it the one most likely to
# carry them without anyone reading it first. The header this replaced said "AUTO-GENERATED", named a
# script no tt-metal reader can run, and explained itself in terms of a coverage ledger and an
# `invalidate_vector` that are not in this repository either.
#
# `codegen` alone is deliberately absent: `prim::<op>_codegen` and `codegen/` are real tt-metal
# names. What cannot ship is prose about the generator, not the word itself.
PRIVATE_NAMES = (
    r"tt-dm-codegen",
    r"AUTO-GENERATED",
    r"invalidate_vector",
    r"coverage ledger",
    r"port_scope",
    r"builder_utils",
    r"spec\.py",
    r"porting guide",
    r"scope:(?:in|out)",
    r"phase \d",
    r"\bmanifests?\b",
)


def private_names(text: str) -> list[str]:
    """Which private-repo names a would-be public file carries, in the order they appear."""
    found = []
    for pattern in PRIVATE_NAMES:
        hit = re.search(pattern, text, re.IGNORECASE)
        if hit and hit.group(0) not in found:
            found.append(hit.group(0))
    return found


def forced_entry(op: str, descriptor: dict | None, which: str) -> str:
    """The dotted path of a forced entry.

    `discover.py` builds this from the op and its category, since the public op stopped taking an
    `implementation` argument and the two private names became conventional. The convention is
    checkable and is checked: `measure.py` resolves the same path against the built ttnn and fails
    loudly when it is not bound, which is the failure a wrong path should produce. What it must never
    do is resolve to something plausible -- a forced-codegen name that falls back to the public op
    measures native against native and reads as a port in perfect agreement with itself.
    """
    resolved = (descriptor or {}).get(which)
    if resolved:
        return resolved
    sys.exit(
        f"scaffold: no {which} entry was resolved for {op}, so the routing test has no way to reach "
        f"the native path except through the public entry it is meant to be testing"
    )


def render_routing_test(
    op: str, category: str, cases: list[dict], descriptor: dict | None = None, year: int | None = None
) -> str:
    """Build the routing test source for every out-of-scope case in the ledger.

    Pure function of its arguments, so `gate.py` can re-render it and compare. Anything time- or
    environment-dependent in here would turn that comparison into a flake.
    """
    out_of_scope = [c for c in cases if c.get("scope") == "out"]
    # Grouped by the condition that rejects them, so the file reads as a list of reasons rather than
    # a wall of tuples, and a reviewer can see which general condition each case represents.
    by_reason: dict[str, list[dict]] = {}
    for case in out_of_scope:
        by_reason.setdefault(str(case.get("reason") or "no reason recorded"), []).append(case)

    rows, ids = [], []
    for reason in sorted(by_reason):
        comment = f"    # {reason}"
        rows.append(comment)
        ids.append(comment)
        for case in sorted(by_reason[reason], key=lambda c: c["case_id"]):
            shape, kwargs = case.get("shape"), case.get("kwargs") or {}
            rows.append(
                f"    ({_literal(case['dtype'])}, {_literal(case['layout'])}, "
                f"{_literal(shape)}, {_literal(kwargs)}),"
            )
            # The pytest id carries the case id so a nightly failure points straight back at the
            # ledger entry, plus the parameters so it is legible without cross-referencing. Rendered
            # through `_literal`, because `str()` on a ttnn object embeds its address and the pin
            # would then see a different file on every run.
            args = "|".join(f"{key}={_literal(value)}" for key, value in sorted(kwargs.items()))
            label = f"{case['case_id']}|{case['dtype']}|{case['layout']}|{shape}"
            ids.append(f"    {_literal(f'{label}|{args}' if args else label)},")
    text = _TEMPLATE.format(
        spdx=SPDX_PY.format(year=year if year is not None else datetime.now(timezone.utc).year),
        op=op,
        category=category,
        count=len(out_of_scope),
        rows="\n".join(rows),
        ids="\n".join(ids),
        force_native=forced_entry(op, descriptor, "force_native"),
    )
    leaked = private_names(text)
    if leaked:
        sys.exit(
            f"scaffold: the emitted routing test names {', '.join(leaked)}, which cannot ship. "
            f"tt-metal is public and the repository this port was generated from is not, so a "
            f"reader following that reference has nowhere to go. State the constraint, not where "
            f"it came from."
        )
    return text


def emit_routing_test(
    ttmetal: Path, op: str, category: str, cases: list[dict], descriptor: dict | None = None
) -> dict:
    path = ttmetal / test_path(op, category)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_routing_test(op, category, cases, descriptor)
    path.write_text(text)
    return {
        "path": test_path(op, category),
        "cases": len([c for c in cases if c.get("scope") == "out"]),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


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
        if not already_listed(sources_text, entry):
            errors.append(f"sources.cmake missing: {entry}")
    # Where the globs landed, not just whether the string is present anywhere. The weaker check
    # passed with `errors: []` on a tree whose CMake configure could not succeed, because a glob
    # written into a `target_sources` FILES list is still a substring of the file.
    start, close = glob_block(cmake_text)
    cmake_lines = cmake_text.splitlines()
    placed = [i for i, line in enumerate(cmake_lines) if line.strip().startswith(f"{op}/codegen/kernels/")]
    if not placed:
        errors.append(f"CMakeLists.txt missing any glob for {op}/codegen/kernels/")
    elif start < 0:
        errors.append("CMakeLists.txt has no file(GLOB_RECURSE kernels ...) call to hold the globs")
    else:
        stray = [i + 1 for i in placed if not start < i < close]
        if stray:
            errors.append(
                f"CMakeLists.txt has {op}/codegen/kernels/ globs outside file(GLOB_RECURSE kernels ...) "
                f"at line(s) {stray}; CMake will read them as literal paths"
            )
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True)
    ap.add_argument("--ttmetal-home", default=".")
    ap.add_argument("--codegen-root", required=True, help="tt-dm-codegen checkout root")
    ap.add_argument("--descriptor", default=None, help="JSON from discover.py; resolved here if absent")
    ap.add_argument("--category", default=None, help="operations/<category>/<op>, when the name is ambiguous")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="an existing port is on this branch: add what the generator gained, keep what is already "
        "there. Without it, kernels are copied over unconditionally, which discards hand-written fixes.",
    )
    ap.add_argument(
        "--emit-test-only",
        action="store_true",
        help="emit just the routing test; needs a built tree, so it runs after the build",
    )
    args = ap.parse_args()

    ttmetal = Path(args.ttmetal_home).resolve()
    codegen_root = Path(args.codegen_root).resolve()

    # Resolved here when the caller has not already done it, so this stays runnable by hand. The
    # workflow passes `--descriptor` so that one resolution is shared by every step of a run and no
    # two steps can disagree about what the port consists of.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import discover

    if args.descriptor:
        descriptor = json.loads(Path(args.descriptor).read_text())
    else:
        descriptor = discover.describe(ttmetal, codegen_root, args.op, args.category)

    op_dir = find_op_dir(ttmetal, args.op, args.category or descriptor.get("category"))

    if args.emit_test_only:
        # Imported here rather than at module scope: building the ledger imports the generator's
        # sweep module and therefore ttnn, which does not exist yet during the pre-build pass.
        import ledger

        category = op_dir.parent.name
        emitted = emit_routing_test(
            ttmetal, args.op, category, ledger.build_ledger(descriptor), descriptor
        )
        print(json.dumps({"op": args.op, "category": category, **emitted}, indent=2))
        return 0

    kernels = copy_kernels(descriptor, codegen_root, op_dir, resume=args.resume)
    stubs = write_stubs(op_dir, args.op)
    sources_added = register_sources(op_dir.parent, args.op)
    globs_added = register_kernel_globs(op_dir.parent, args.op, op_dir / "codegen" / "kernels")

    errors = verify(op_dir, args.op, kernels)
    result = {
        "op": args.op,
        "op_dir": str(op_dir.relative_to(ttmetal)),
        "builder": descriptor.get("builder"),
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
