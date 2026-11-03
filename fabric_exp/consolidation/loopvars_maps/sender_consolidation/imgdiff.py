#!/usr/bin/env python3
"""imgdiff.py BASE NEW [OUT]: the ELFs (case__hash/erisc/erisc.elf, relative) whose loadable bytes differ between two
cov_m builds (bytecmp.py's comparison), one per line, sorted; written to OUT or stdout. Shared ELFs only."""
import concurrent.futures, glob, os, subprocess, sys
COV = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
OC = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy"


def image(elf):
    return subprocess.run([OC, "-O", "binary", elf, "/dev/stdout"], capture_output=True, check=True).stdout


def rels(b):
    return {os.path.relpath(e, f"{COV}/{b}") for e in glob.glob(f"{COV}/{b}/*/*/*.elf")}


base, new = sys.argv[1:3]
shared = sorted(rels(base) & rels(new))
with concurrent.futures.ThreadPoolExecutor(32) as ex:
    same = list(ex.map(lambda r: image(f"{COV}/{base}/{r}") == image(f"{COV}/{new}/{r}"), shared))
out = "".join(r + "\n" for r, s in zip(shared, same) if not s)
if len(sys.argv) > 3:
    open(sys.argv[3], "w").write(out)
else:
    sys.stdout.write(out)
