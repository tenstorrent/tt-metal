#!/usr/bin/env python3
"""mk_overlay.py SRC_TREE DST REL_FILE [--tests DIR]: a tree of symlinks into SRC_TREE whose path down to REL_FILE is
real directories (every sibling a symlink) and REL_FILE a real copy, to edit in place; its own empty generated/fabric.
--tests: tests/ links to DIR instead of SRC_TREE/tests. REL_FILE '-' makes a plain root (no copied file)."""
import argparse, os, shutil
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("rel"); ap.add_argument("--tests", default="")
a = ap.parse_args()
src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
assert not os.path.exists(dst), dst
os.makedirs(dst)
parts = [] if a.rel == "-" else a.rel.split("/")
skip_top = {"generated", ".git", ".scratch"} | ({"tests"} if a.tests else set())
for e in os.listdir(src):
    if e in skip_top or (parts and e == parts[0]):
        continue
    os.symlink(os.path.join(src, e), os.path.join(dst, e))
if a.tests:
    os.symlink(os.path.abspath(a.tests), os.path.join(dst, "tests"))
os.makedirs(os.path.join(dst, "generated", "fabric"))
cs, cd = src, dst
for i, p in enumerate(parts):
    cs, cd = os.path.join(cs, p), os.path.join(cd, p)
    if i == len(parts) - 1:
        shutil.copy2(cs, cd)
        break
    os.makedirs(cd)
    for e in os.listdir(cs):
        if e != parts[i + 1]:
            os.symlink(os.path.join(cs, e), os.path.join(cd, e))
