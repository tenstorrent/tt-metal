#!/usr/bin/env python3
# bits_pair.py <dir>: compares <case>_vec.npz with <case>_blk.npz byte for byte in one bits folder.
import glob, os, sys
import numpy as np
d = sys.argv[1]; same = diff = 0
for f in sorted(glob.glob(f"{d}/*_vec.npz")):
    g = f[:-8] + "_blk.npz"
    if not os.path.exists(g): continue
    a, b = np.load(f)["res"], np.load(g)["res"]
    if a.tobytes() == b.tobytes(): same += 1
    else:
        diff += 1; print("   differs", os.path.basename(f)[:-8], int((a != b).sum()), "bytes")
print(f"== vec vs blk in {os.path.basename(d)}: {same} identical, {diff} differ")
