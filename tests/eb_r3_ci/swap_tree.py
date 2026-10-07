#!/usr/bin/env python3
"""Round 3 eltwise binary: put one tree's device files into both /work and the installed ttnn package (the JIT reads kernel
sources from /work and headers from the package), and log where each landed. usage: swap_tree.py <tarball> <RT>"""
import os, shutil, subprocess, sys, tarfile, tempfile
tar, rt = sys.argv[1], sys.argv[2]
tmp = tempfile.mkdtemp()
with tarfile.open(tar) as t:
    t.extractall(tmp)
n = in_work = in_rt = 0
for root, _, files in os.walk(tmp):
    for f in files:
        src = os.path.join(root, f); rel = os.path.relpath(src, tmp); n += 1
        shutil.copy2(src, os.path.join("/work", rel)); in_work += 1
        for cand in {os.path.join(rt, rel), os.path.join(rt, rel.replace("ttnn/cpp/", "cpp/", 1))}:
            if os.path.exists(cand):
                shutil.copy2(src, cand); in_rt += 1
print(f"SWAP {n} files: {in_work} into /work, {in_rt} also under the runtime root {rt}")
