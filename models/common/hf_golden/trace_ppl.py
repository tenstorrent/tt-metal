#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wikitext perplexity of a golden trace, from its own stored final hidden state.

    python trace_ppl.py <trace_dir> <checkpoint_snapshot_dir> <lm_head_key> [--skip N] [--count M]

logits = last_hidden_state @ lm_head^T at every position (the trace stores only the last position's logits);
scores wikitext token t+1 from position t, over wikitext tokens [skip, skip + count) so traces with different
prefixes are scored on the same text."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("trace", type=Path)
ap.add_argument("ckpt", type=Path)
ap.add_argument("lm_head_key")
ap.add_argument("--skip", type=int, default=1, help="wikitext tokens not scored at the start (no context)")
ap.add_argument("--count", type=int, default=None)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
DEV = args.device

meta = json.loads((args.trace / "metadata.json").read_text())
n, prefix_ids = meta["n_tokens"], meta.get("prefix_ids") or []
p = len(prefix_ids)
cache = load_file(str((args.trace / meta["token_cache"]).resolve()))["token_ids"].flatten()
text = cache[: n - p].long()  # wikitext ids in this trace, after the prefix

index = json.loads((args.ckpt / "model.safetensors.index.json").read_text())["weight_map"]
with safe_open(str(args.ckpt / index[args.lm_head_key]), "pt", device=DEV) as f:
    w = f.get_tensor(args.lm_head_key).to(torch.bfloat16)

h = load_file(str(args.trace / "hidden/final.safetensors"), device=DEV)["model.last_hidden_state"][0]
w = w.to(h.dtype)  # fp32 final states (DeepSeek-V4.1's head upcasts its bf16 weight) score in fp32; bf16 unchanged
count = args.count or (len(text) - args.skip)
# wikitext token j sits at position p + j; it is predicted from position p + j - 1
targets = text[args.skip : args.skip + count].to(DEV)
pos = torch.arange(args.skip, args.skip + count, device=DEV) + p - 1
nll, top1, top5 = [], [], []
for s in range(0, count, 1024):
    logits = (h[pos[s : s + 1024]] @ w.T).float()
    t = targets[s : s + 1024]
    nll.append(torch.nn.functional.cross_entropy(logits, t, reduction="none"))
    top = logits.topk(5, dim=-1).indices
    top1.append(top[:, 0] == t)
    top5.append((top == t[:, None]).any(-1))
nll, top1, top5 = torch.cat(nll), torch.cat(top1).float(), torch.cat(top5).float()
q = count // 4
quarters = " ".join(f"{nll[i * q : (i + 1) * q].mean().exp().item():6.2f}" for i in range(4))
print(
    f"{args.trace.parent.parent.parent.name:14s} prefix={p:3d} scored={count:6d}  ppl={nll.mean().exp().item():6.3f}  "
    f"top1={top1.mean().item() * 100:5.2f}%  top5={top5.mean().item() * 100:5.2f}%  ppl by quarter: {quarters}"
)
