#!/usr/bin/env python3
"""inc_scan.py REV|WORKTREE [OVERLAY]: static recursive #include closure of the fabric router kernel, resolved against the
router compile's -I list, reading files from a git revision (git show REV:path) or the working tree (WORKTREE), with an
optional overlay dir (offline_build.resolve mapping) taking precedence. Prints tree-relative paths. Conditional includes
(#if) are all followed, so it over-approximates; used to find files only one side's router includes."""
import os, re, subprocess, sys

TREE = "/home/aagarwal/tt-metal"
INC = [".", "ttnn", "ttnn/cpp", "tt_metal/hw/inc", "tt_metal/tt-llk/common", "tt_metal/tt-llk/tools/include",
       "tt_metal/hostdevcommon/api", "tt_metal/api", "tt_metal/hw/ckernels/blackhole/metal/common",
       "tt_metal/hw/ckernels/blackhole/metal/llk_io", "tt_metal/hw/inc/internal/tt-1xx",
       "tt_metal/hw/inc/internal/tt-1xx/blackhole", "tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines",
       "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc", "tt_metal/tt-llk/tt_llk_blackhole/common/inc",
       "tt_metal/tt-llk/tt_llk_blackhole/llk_lib", "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/hal",
       "tt_metal/hw/firmware/src/tt-1xx"]
rev = sys.argv[1]
over = {}
if len(sys.argv) > 2:
    sys.path.insert(0, TREE + "/fabric_exp/consolidation")
    from offline_build import resolve
    over = {dst: src for src, dst in resolve(sys.argv[2])}
cache = {}


def read(path):
    if path in cache:
        return cache[path]
    txt = None
    if path in over:
        txt = open(over[path], errors="replace").read()
    elif rev == "WORKTREE":
        p = os.path.join(TREE, path)
        txt = open(p, errors="replace").read() if os.path.isfile(p) else None
    else:
        r = subprocess.run(["git", "-C", TREE, "show", f"{rev}:{path}"], capture_output=True, text=True, errors="replace")
        txt = r.stdout if r.returncode == 0 else None
    cache[path] = txt
    return txt


INCL = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)


def find(name, cur_dir):
    for d in [cur_dir] + INC:
        p = os.path.normpath(os.path.join(d, name))
        if read(p) is not None:
            return p
    return None


start = sys.argv[3:] or []
kernel = ("tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp" if over else
          "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp")
todo = [kernel, "tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc",
        "tt_metal/hw/firmware/src/tt-1xx/subordinate_active_erisck.cc"]
seen = set()
while todo:
    f = todo.pop()
    if f in seen:
        continue
    seen.add(f)
    t = read(f)
    if t is None:
        continue
    for name in INCL.findall(t):
        p = find(name, os.path.dirname(f))
        if p and p not in seen:
            todo.append(p)
print("\n".join(sorted(p for p in seen if read(p) is not None)))
