#!/usr/bin/env python3
"""Compare the transposed block bit dumps: every <case>_f1.bin and <case>_f4.bin against <case>_f0.bin, byte for byte.
usage: bits_cmp.py <dump_dir> [<other_dump_dir>]   (with a second dir: every file of the first against the same name there)"""
import glob, os, sys
a = sys.argv[1]
b = sys.argv[2] if len(sys.argv) > 2 else None
bad = 0
for f0 in sorted(glob.glob(os.path.join(a, "*_f0.bin"))):
    case = os.path.basename(f0)[:-7]
    ref = open(f0, "rb").read()
    pairs = [(f"{case}_f{k}.bin", os.path.join(a, f"{case}_f{k}.bin"), ref) for k in (1, 4)]
    inp = open(os.path.join(a, f"{case}_in0.bin"), "rb").read()
    pairs += [(f"{case}_in{k}.bin", os.path.join(a, f"{case}_in{k}.bin"), inp) for k in (1, 4)]
    if b:
        pairs = [(n, os.path.join(b, n), open(os.path.join(a, n), "rb").read()) for n in (f"{case}_f{k}.bin" for k in (0, 1, 4))]
    for name, path, want in pairs:
        if not os.path.exists(path):
            print(f"{name}: missing"); bad += 1; continue
        got = open(path, "rb").read()
        nd = sum(x != y for x, y in zip(got, want)) + abs(len(got) - len(want))
        print(f"{name}: {len(got)} bytes, {nd} differ"); bad += nd > 0
print("all equal" if bad == 0 else f"{bad} files differ")
sys.exit(1 if bad else 0)
