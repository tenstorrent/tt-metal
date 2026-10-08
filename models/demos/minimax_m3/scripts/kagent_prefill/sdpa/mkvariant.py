#!/usr/bin/env python3
"""Make a JIT variant root: a symlink farm of the worktree in which only the sdpa kernels dir is a real copy, so
TT_METAL_HOME/TT_METAL_RUNTIME_ROOT=<root> compiles edited copies of the kernels (probes) without touching W.
usage: mkvariant.py <name>  -> prints the root; edit <root>/<KREL>/... afterwards"""
import os
import shutil
import sys

W = "/mnt/data/kernel-agent/dev/prefill-sdpa/tt-metal"
KREL = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels"
root = f"/mnt/data/kernel-agent/dev/prefill-sdpa/variants/{sys.argv[1]}"
if os.path.exists(root):
    shutil.rmtree(root)
parts = KREL.split("/")
cur_src, cur_dst = W, root
os.makedirs(root)
for i, p in enumerate(parts):
    for e in os.listdir(cur_src):
        if e == p:
            continue
        os.symlink(os.path.join(cur_src, e), os.path.join(cur_dst, e))
    cur_src = os.path.join(cur_src, p)
    cur_dst = os.path.join(cur_dst, p)
    if i < len(parts) - 1:
        os.makedirs(cur_dst)
shutil.copytree(cur_src, cur_dst)
print(root)
