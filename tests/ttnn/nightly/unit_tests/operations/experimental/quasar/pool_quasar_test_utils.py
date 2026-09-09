# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Quasar pool tests (mirrors binary_ng_quasar_test_utils.py)."""

import torch


def _build_input(pattern, batch, in_h, in_w, channels, seed, mod):
    """Returns an NHWC float32 tensor; caller quantizes to bf16."""
    shape = (batch, in_h, in_w, channels)
    if pattern == "random":
        torch.manual_seed(seed)
        return torch.rand(shape)
    if pattern == "ones":
        return torch.ones(shape)
    if pattern == "zeros":
        return torch.zeros(shape)
    if pattern.startswith("const:"):
        return torch.full(shape, float(pattern.split(":", 1)[1]))
    if pattern == "sticks":
        s = torch.arange(batch * in_h * in_w, dtype=torch.float32).reshape(batch, in_h, in_w, 1)
        if mod:
            s = s % mod
        return s.expand(shape).contiguous()
    if pattern == "channels":
        c = torch.arange(channels, dtype=torch.float32).reshape(1, 1, 1, channels)
        if mod:
            c = c % mod
        return c.expand(shape).contiguous()
    raise ValueError(f"unknown PATTERN={pattern!r}")


def _dump_mismatches(got, golden, out_h, out_w, channels, n_dump):
    """got/golden: (sticks, C) float tensors. Prints the n_dump worst sticks."""
    diff = (got - golden).abs()
    per_stick = diff.max(dim=1).values
    bad = int((per_stick > 0).sum().item())
    n = min(n_dump, bad)
    if n == 0:
        return
    worst = torch.topk(per_stick, n).indices
    print(f"\nQPOOL: {bad}/{got.shape[0]} sticks mismatch; worst {n}:")
    for s in worst.tolist():
        b, r = divmod(s, out_h * out_w)
        oh, ow = divmod(r, out_w)
        ch = int(diff[s].argmax().item())
        k = min(8, channels)
        print(
            f"  stick {s} (b={b}, oh={oh}, ow={ow}) worst ch={ch} "
            f"exp={golden[s, ch].item():.4f} got={got[s, ch].item():.4f} | "
            f"ch0..{k - 1} exp={[round(v, 3) for v in golden[s, :k].tolist()]} "
            f"got={[round(v, 3) for v in got[s, :k].tolist()]}"
        )
