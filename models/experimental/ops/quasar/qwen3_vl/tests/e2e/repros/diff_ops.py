# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Compare the captured op outputs of two runs op by op and print where they first diverge.

usage: diff_ops.py RUN_A RUN_B   (run folders saved with --qwen-capture-ops; older runs saved decode_ops.pt)
"""
import sys
from pathlib import Path

import torch


def _ops_file(run):
    return run / "ops.pt" if (run / "ops.pt").exists() else run / "decode_ops.pt"


def diff_ops(a, b):
    """Print each op whose outputs differ, up to the first significant divergence."""
    print(f"ops: {len(a)} vs {len(b)}")
    for (i, na, _, ta), (_, nb, _, tb) in zip(a, b):
        if na != nb:
            print(f"#{i}: op sequence differs: {na} vs {nb}")
            return
        if ta.shape != tb.shape:
            print(f"#{i} {na}: shape {tuple(ta.shape)} vs {tuple(tb.shape)}")
            return
        fa, fb = torch.isfinite(ta.float()), torch.isfinite(tb.float())
        if not (fa.all() and fb.all()):  # NaN/Inf makes diff.max() NaN, which compares false below
            print(
                f"#{i} {na} {tuple(ta.shape)}: non-finite outputs {int((~fa).sum())} vs {int((~fb).sum())}"
                f"/{ta.numel()}"
            )
            print("first significant divergence ^")
            return
        diff = (ta.float() - tb.float()).abs()
        if diff.max() > 0:
            pcc = torch.corrcoef(torch.stack([ta.flatten().double(), tb.flatten().double()]))[0, 1].item()
            print(
                f"#{i} {na} {tuple(ta.shape)}: max_abs_diff={diff.max().item():.4g} pcc={pcc:.5f} "
                f"differing={int((diff > 0).sum())}/{diff.numel()}"
            )
            if diff.max() > 1e-2:
                print("first significant divergence ^")
                return
    print("no significant divergence")


def main():
    diff_ops(*(torch.load(_ops_file(Path(p)), weights_only=False) for p in sys.argv[1:3]))


if __name__ == "__main__":
    main()
