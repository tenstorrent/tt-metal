#!/usr/bin/env python3
"""Private relink build of selected ttnn op unity TUs (MiMo moe track).

Reads (read-only) the shared build's ninja commands (`ninja -t commands ttnn/_ttnncpp.so` run in SHARED_BUILD), recompiles ONLY the given ops' unity TU
from THIS worktree's sources with the shared build's flags, and relinks a PRIVATE _ttnncpp.so into <out>/lib. Nothing under the shared tree is written.
Use at run time with LD_LIBRARY_PATH=<out>/lib (the _ttnn.so RUNPATH is searched after LD_LIBRARY_PATH).
Usage: priv_relink.py --worktree W --out OUT unified_routed_expert_ffn [moe_fused_swiglu moe_compute ...]
"""
import argparse
import os
import re
import subprocess
import sys
import time

SHARED = "/mnt/tt-data/ssinghal/mimo/tt-metal"
SB = SHARED + "/build_Release"

ap = argparse.ArgumentParser()
ap.add_argument("--worktree", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("ops", nargs="+")
a = ap.parse_args()
W, OUT = os.path.abspath(a.worktree), os.path.abspath(a.out)
assert OUT != SB and not OUT.startswith(SHARED + "/"), "refusing to write inside the shared tree"
os.makedirs(OUT + "/obj", exist_ok=True)
os.makedirs(OUT + "/lib", exist_ok=True)
cmds = subprocess.run(
    ["ninja", "-C", SB, "-t", "commands", "ttnn/_ttnncpp.so"], capture_output=True, text=True, check=True
).stdout.split("\n")
link = [c for c in cmds if " -o ttnn/_ttnncpp.so " in c][0]
procs = []
objmap = {}
for op in a.ops:
    cc = [c for c in cmds if f"/{op}/CMakeFiles/" in c and "unity_0_cxx.cxx.o -c" in c]
    assert len(cc) == 1, (op, len(cc))
    cc = cc[0]
    m = re.search(r"-o (\S+unity_0_cxx\.cxx\.o)", cc)
    shared_obj_rel = m.group(1)
    src = re.search(r" -c (\S+unity_0_cxx\.cxx)", cc).group(1)
    unity = open(src).read().replace(SHARED + "/ttnn/cpp/", W + "/ttnn/cpp/")
    uf = f"{OUT}/obj/{op}_unity_0_cxx.cxx"
    open(uf, "w").write(unity)
    of = f"{OUT}/obj/{op}_unity_0_cxx.cxx.o"
    cc = cc.replace(" -c " + src, " -c " + uf).replace("-o " + shared_obj_rel, "-o " + of)
    cc = re.sub(r" -MD| -MT \S+| -MF \S+", "", cc)
    cc = cc.replace(f"-I{SHARED}/ttnn/cpp ", f"-I{W}/ttnn/cpp -I{SHARED}/ttnn/cpp ", 1)
    cc = cc.replace(
        f"-I{SHARED}/ttnn/cpp/ttnn/operations/experimental/", f"-I{W}/ttnn/cpp/ttnn/operations/experimental/"
    )
    objmap[shared_obj_rel] = of
    procs.append((op, cc))
t0 = time.time()
running = []
for op, cc in procs:
    print(f"[priv_relink] compiling {op}", flush=True)
    running.append((op, subprocess.Popen(cc, shell=True, cwd=SB)))
for op, p in running:
    rc = p.wait()
    print(f"[priv_relink] {op} rc={rc} ({time.time() - t0:.0f}s)", flush=True)
    if rc:
        sys.exit(rc)
for so, of in objmap.items():
    assert so in link, so
    link = link.replace(so, of)
link = link.replace(" -o ttnn/_ttnncpp.so ", f" -o {OUT}/lib/_ttnncpp.so ").replace(
    " -Xlinker --dependency-file=ttnn/CMakeFiles/ttnncpp.dir/link.d", ""
)
link = re.sub(r"^: && ", "", link)
link = re.sub(r" && :$", "", link)
print("[priv_relink] linking", flush=True)
rc = subprocess.run(link, shell=True, cwd=SB).returncode
print(f"[priv_relink] link rc={rc} ({time.time() - t0:.0f}s)", flush=True)
sys.exit(rc)
