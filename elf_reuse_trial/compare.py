#!/usr/bin/env python3
"""Compare two sessions' ELF manifests; strip and recompare any that differ (needs both trees).

Usage: compare.py <dir X> <dir Y> [<tree X> <tree Y> <objcopy>]   (dirs hold elfs.sha256)
"""
import json
import os
import subprocess
import sys
import tempfile


def load(d):
    out = {}
    for line in open(f"{d}/elfs.sha256"):
        digest, path = line.rstrip("\n").split(None, 1)
        out[path] = digest
    return out


x, y = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(x) & set(y))
differ = [p for p in common if x[p] != y[p]]
result = {"x": sys.argv[1], "y": sys.argv[2], "x_elfs": len(x), "y_elfs": len(y), "common": len(common),
          "identical": len(common) - len(differ), "different": len(differ),
          "only_x": len(set(x) - set(y)), "only_y": len(set(y) - set(x)), "different_examples": differ[:20]}
if differ and len(sys.argv) > 5:
    tree_x, tree_y, objcopy = sys.argv[3:6]
    same_stripped = 0
    with tempfile.TemporaryDirectory() as tmp:
        for p in differ:
            a, b = f"{tmp}/a", f"{tmp}/b"
            subprocess.run([objcopy, "-S", os.path.join(tree_x, p), a], check=True)
            subprocess.run([objcopy, "-S", os.path.join(tree_y, p), b], check=True)
            same_stripped += open(a, "rb").read() == open(b, "rb").read()
    result["different_but_identical_stripped"] = same_stripped
print(json.dumps(result))
