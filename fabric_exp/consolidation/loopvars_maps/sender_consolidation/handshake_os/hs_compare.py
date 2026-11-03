"""hs_compare.py A B: run_handshakes' size (nm -S) and each image's loadable bytes in two cov_m builds, per case and
ERISC: the function's size in A and B, and the image delta B - A (min / max over the case's configs)."""
import collections, glob, os, subprocess, sys
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
BIN = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"


def fsize(elf):
    for l in subprocess.run([BIN + "nm", "-S", "-C", elf], capture_output=True, text=True).stdout.splitlines():
        p = l.split(" ", 3)
        if len(p) == 4 and "run_handshakes" in p[3]:
            return int(p[1], 16)
    return 0


def loadable(elf):
    out = subprocess.run([BIN + "readelf", "-lW", elf], capture_output=True, text=True).stdout
    return sum(int(l.split()[4], 16) for l in out.splitlines() if l.strip().startswith("LOAD"))


a, b = sys.argv[1:3]
rows = collections.defaultdict(list)
for e in sorted(glob.glob(f"{CB}/{b}/*/*/*.elf")):
    rel = os.path.relpath(e, f"{CB}/{b}")
    ea = f"{CB}/{a}/{rel}"
    if not os.path.exists(ea):
        continue
    case, erisc = rel.split("__")[0], rel.split("/")[1]
    rows[(case, "E0" if erisc == "active_erisc" else "E1")].append(
        (fsize(ea), fsize(e), loadable(e) - loadable(ea), loadable(ea), loadable(e)))
print(f"{'case':22} ERISC  run_handshakes A -> B (bytes)   image delta B-A (min..max)   largest image A -> B")
for (case, er), v in sorted(rows.items()):
    fa = sorted({x[0] for x in v}); fb = sorted({x[1] for x in v})
    d = [x[2] for x in v]
    print(f"{case:22} {er}    {'/'.join(map(str, fa)):>8} -> {'/'.join(map(str, fb)):<10}         {min(d):+5d}..{max(d):+5d}"
          f"            {max(x[3] for x in v):6d} -> {max(x[4] for x in v):6d}")
