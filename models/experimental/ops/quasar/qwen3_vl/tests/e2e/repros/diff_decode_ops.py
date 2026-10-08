# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Compare the decode_ops.pt of two runs op by op and print where they first diverge.

usage: diff_decode_ops.py RUN_A RUN_B   (run folders saved with --qwen-dump-decode-ops)
"""
import sys
from pathlib import Path

import torch


def main():
    a, b = (torch.load(Path(p) / "decode_ops.pt", weights_only=False) for p in sys.argv[1:3])
    print(f"ops: {len(a)} vs {len(b)}")
    for (i, na, _, ta), (_, nb, _, tb) in zip(a, b):
        if na != nb:
            print(f"#{i}: op sequence differs: {na} vs {nb}")
            return
        if ta.shape != tb.shape:
            print(f"#{i} {na}: shape {tuple(ta.shape)} vs {tuple(tb.shape)}")
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


if __name__ == "__main__":
    main()
