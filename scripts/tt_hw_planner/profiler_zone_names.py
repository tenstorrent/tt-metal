# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A run folder whose path makes two device-profiler zone labels collide cannot be profiled.

tt-metal names every device profiler zone by the string `name,__FILE__,__LINE__,KERNEL_PROFILER`
(kernel_profiler.hpp: PROFILER_MSG_NAME) and keys it by a 16-bit fold of its FNV-1a hash
(profiler.cpp: hash16CT). Two different labels on the same 16 bits make populateZoneSrcLocations
TT_THROW "Source location hashes are colliding", and every profiler read from then on fails. __FILE__
is an absolute path, so the folder a checkout lives in is part of every label -- WH Galaxy,
2026-09-28: in /tmp/tt_hw_planner_qwen_image_edit_1790623639 the two "QKT@V MM+Pack" zones of the SDPA
streaming kernel (lines 1680 and 1789) folded to the same 42956, the profiler stopped reading
mid-forward, and half the baseline's ops had no device time. The same labels under the source
checkout, or under the previous run's folder, do not collide.

So the tool checks a candidate folder BEFORE creating a checkout there, and moves on to the next
name when it would collide. Everything is read from tt-metal's own source -- the label format, the
hash constants, and which macros open a zone -- so an upstream change is followed, not contradicted;
when those definitions cannot be read the check says so and does not guess.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

# Where tt-metal defines what this module mirrors, relative to a checkout's root.
_PROFILER_HEADER = Path("tt_metal") / "tools" / "profiler" / "kernel_profiler.hpp"
_PROFILER_IMPL = Path("tt_metal") / "impl" / "profiler" / "profiler.cpp"
# The trees device kernels (and the headers they include) are compiled from.
_KERNEL_TREES = ("tt_metal", "ttnn")
_SOURCE_SUFFIXES = (".cpp", ".cc", ".h", ".hpp")
# The label macro every device zone expands to; the zone macros are found as whatever reaches it.
_LABEL_MACRO = "PROFILER_MSG_NAME"
# How many names are tried before giving up and warning.
_MAX_NAME_TRIES = 16

_DEFINE_RE = re.compile(r"^\s*#\s*define\s+(\w+)\s*\(([^)]*)\)(.*)$")
_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="ignore")
    except OSError:
        return ""


@lru_cache(maxsize=None)
def _label_suffix(repo: Path):
    """What PROFILER_MSG appends after __FILE__ and the line: ',KERNEL_PROFILER' today, or None."""
    m = re.search(r'#define\s+PROFILER_MSG\s+__FILE__\s+","\s+\$Line\s+"([^"]*)"', _read(repo / _PROFILER_HEADER))
    return m.group(1) if m else None


@lru_cache(maxsize=None)
def _hash_constants(repo: Path):
    """(FNV offset basis, FNV prime) exactly as hash32CT / hash16CT use them, or None."""
    src = _read(repo / _PROFILER_IMPL)
    basis = re.search(r"hash32CT\(\s*str\.c_str\(\),\s*str\.length\(\),\s*UINT32_C\((\d+)\)\s*\)", src)
    prime = re.search(r"\(basis\s*\^\s*str\[0\]\)\s*\*\s*UINT32_C\((\d+)\)", src)
    return (int(basis.group(1)), int(prime.group(1))) if basis and prime else None


def hash16(label: str, constants) -> int:
    """tt-metal's hash16CT: 32-bit FNV-1a, its halves XOR-folded to 16 bits."""
    basis, prime = constants
    r = basis
    for c in label.encode():
        r = ((r ^ c) * prime) & 0xFFFFFFFF
    return ((r & 0xFFFF) ^ (r >> 16)) & 0xFFFF


def _sources(repo: Path):
    for tree in _KERNEL_TREES:
        base = repo / tree
        if base.is_dir():
            for p in base.rglob("*"):
                if p.suffix in _SOURCE_SUFFIXES and p.is_file():
                    yield p


def _defines(text: str):
    """(name, params, body) for every function-like #define, continuation lines joined."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _DEFINE_RE.match(lines[i])
        if m:
            body = m.group(3)
            while body.rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                body = body.rstrip()[:-1] + " " + lines[i]
            yield m.group(1), m.group(2), body
        i += 1


@lru_cache(maxsize=None)
def _zone_sites(repo: Path) -> tuple:
    """((name, repo-relative file, line), ...) for every device zone opened in a kernel tree."""
    files = [(p, _read(p)) for p in _sources(repo)]
    files = [(p, t) for p, t in files if "Zone" in t]
    macros = {}
    for _p, text in files:
        for name, params, body in _defines(text):
            macros.setdefault(name, set()).add(body)
    zone = {n for n, bodies in macros.items() if any(_LABEL_MACRO in b for b in bodies)}
    grew = True
    while grew:  # wrappers of wrappers: a macro that expands to a zone macro opens a zone too
        grew = False
        for n, bodies in macros.items():
            if n not in zone and any(re.search(r"\b(%s)\s*\(" % "|".join(map(re.escape, zone)), b) for b in bodies):
                zone.add(n)
                grew = True
    if not zone:
        return ()
    call = re.compile(r"\b(%s)\s*\(" % "|".join(map(re.escape, sorted(zone))))
    sites = []
    for p, text in files:
        rel = p.relative_to(repo).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for m in call.finditer(line):
                names = _STRING_RE.findall(line[m.end() :])
                if names:
                    sites.append((names[-1], rel, lineno))
    return tuple(sites)


def colliding_labels(roots, repo: Path):
    """Pairs of different zone labels that share a 16-bit hash when kernels are compiled from `roots`
    (a checkout's own folder and any other it compiles from), or None when tt-metal's definitions
    cannot be read from `repo` -- an unknown, not a clean bill."""
    repo = Path(repo).resolve()
    suffix, constants = _label_suffix(repo), _hash_constants(repo)
    if suffix is None or constants is None:
        return None
    sites = _zone_sites(repo)
    seen: dict = {}
    pairs = []
    for root in dict.fromkeys(str(Path(r)) for r in roots):
        for name, rel, line in sites:
            label = "%s,%s/%s,%d%s" % (name, root, rel, line, suffix)
            h = hash16(label, constants)
            other = seen.setdefault(h, label)
            if other != label:
                pairs.append((other, label))
    return pairs


def collision_free(path: Path, repo: Path) -> Path:
    """`path`, or the first `<path>_<k>` that does not exist and makes no two zone labels collide.

    Checked together with `repo`, the checkout the new one is made from: a run compiles some
    kernels from each, so a label under one can collide with a label under the other."""
    path = Path(path)
    tried = []
    for k in range(_MAX_NAME_TRIES):
        cand = path if k == 0 else path.with_name("%s_%d" % (path.name, k))
        if cand.exists():
            continue
        bad = colliding_labels([cand, repo], repo)
        if bad is None:
            print(
                "[worktree] WARN could not read tt-metal's profiler zone definitions under %s; the run "
                "folder was not checked for zone-label hash collisions" % repo,
                file=sys.stderr,
            )
            return cand
        if not bad:
            return cand
        tried.append((cand, bad[0]))
        print(
            "[worktree] %s would make two profiler zone labels share a 16-bit hash (%s | %s); "
            "trying another name" % (cand, bad[0][0][:90], bad[0][1][:90]),
            file=sys.stderr,
        )
    print(
        "[worktree] WARN every one of %d candidate folders collides; using %s anyway -- device "
        "profiling there will fail at the first colliding zone" % (_MAX_NAME_TRIES, path),
        file=sys.stderr,
    )
    return path
