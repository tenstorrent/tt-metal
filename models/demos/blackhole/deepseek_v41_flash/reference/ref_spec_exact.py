# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline exactness report: spec-decode streams (greedy_dev_g_k{k}.pt) vs the device plain greedy stream (greedy_dev_g.pt, k=0), both written by
tests/test_spec_loop.py with the top1-top2 logit gap of every emitted token. At the first divergence of each user the gap of the plain and the spec
argmax shows whether it is a near-tie (small gap, numerics of the 1-row vs 1+k-row verify path) or a real difference (large gap).
"""

import glob
import re
import sys

import torch

D = "/mnt/tt-data/ssinghal/dsv4-spec-accept"
PFX = sys.argv[1] if len(sys.argv) > 1 else "greedy_dev_g"  # e.g. greedy_paged_g for the paged runs
base = torch.load(f"{D}/{PFX}.pt")
bs, bg = base["stream"], base.get("gap")
for f in sorted(glob.glob(f"{D}/{PFX}_k*.pt")):
    k = int(re.search(r"_k(\d+)", f).group(1))
    d = torch.load(f)
    s, g = d["stream"], d.get("gap")
    B = s.shape[0]
    same, rows = 0, []
    for b in range(B):
        L = min(int((s[b] >= 0).sum()), int((bs[b] >= 0).sum()))
        ne = (s[b, :L] != bs[b, :L]).nonzero()
        if len(ne) == 0:
            same += 1
            continue
        i = int(ne[0])
        gp = bg[b][i - 1] if bg is not None and i > 0 and len(bg[b]) >= i else float("nan")
        gs = g[b][i - 1] if g is not None and i > 0 and len(g[b]) >= i else float("nan")
        rows.append((b, i, gp, gs))
    print(f"k={k}: identical {same}/{B}")
    for b, i, gp, gs in rows:
        print(
            f"   user {b} first divergence at generated token {i}: top1-top2 logit gap plain {gp:.3f} / spec {gs:.3f}"
        )
