# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 6 — fp32 long-context precision (two-pass softmax).

The online-softmax recurrence accumulates SFPU-exp rounding across the KV
blocks (error ~ sqrt(num_kv_blocks)). At fp32 the tight rms target (0.02) is
breached only at the longest context (S_kv = 8192 → 256 blocks: online device
rms 0.0284). The descriptor switches the fp32 / non-causal / no-mask /
S_kv > 4096 regime to a NON-online two-pass softmax (global max in pass 1, exp
once per element in pass 2, no per-block correction), which clears the target
with margin.

These tests pin the exact failing golden shape (Q1x1x8192x64 fp32) plus a few
guards that the two-pass gate does not regress the online path.
"""

import math

import pytest
import torch

import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _pcc(out, ref):
    out = out.to(torch.float32).flatten()
    ref = ref.to(torch.float32).flatten()
    return torch.corrcoef(torch.stack([out, ref]))[0, 1].item()


def _rms(out, ref):
    out = out.to(torch.float32)
    ref = ref.to(torch.float32)
    return (torch.sqrt(torch.mean((out - ref) ** 2)) / ref.std()).item()


def _reference(Q, K, V, scale):
    Qf, Kf, Vf = (t.to(torch.float32) for t in (Q, K, V))
    s = scale if scale is not None else 1.0 / math.sqrt(Qf.shape[-1])
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    return torch.matmul(torch.softmax(scores, dim=-1), Vf)


def _run(device, shape, *, scale=None, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(shape, dtype=torch.float32)
    K = torch.randn(shape, dtype=torch.float32)
    V = torch.randn(shape, dtype=torch.float32)
    ref = _reference(Q, K, V, scale)

    tQ = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tK = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tV = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(tQ, tK, tV, scale=scale)
    out_t = ttnn.to_torch(out)

    assert not torch.isnan(out_t).any(), "NaN in two-pass output"
    return out_t, ref


# The exact failing golden cells: Q1x1x8192x64 fp32, mask_mode=none, both
# scale modes (auto == explicit since 1/sqrt(64) == 0.125, but exercise both).
@pytest.mark.parametrize("scale", [None, 0.125], ids=["auto", "explicit"])
def test_fp32_s8192_two_pass_clears_target(device, scale):
    out, ref = _run(device, (1, 1, 8192, 64), scale=scale)
    pcc, rms = _pcc(out, ref), _rms(out, ref)
    # Golden fp32 tolerance is (PCC 0.999, rms 0.02). Two-pass should clear both
    # with margin (host sim ~0.004; online path missed at 0.0284).
    assert pcc >= 0.999, f"PCC {pcc} < 0.999"
    assert rms <= 0.02, f"rms {rms} > 0.02 (two-pass did not clear the target)"


# Multi-head long-context fp32 (also triggers two-pass).
def test_fp32_s8192_multihead(device):
    out, ref = _run(device, (1, 2, 8192, 64))
    assert _pcc(out, ref) >= 0.999
    assert _rms(out, ref) <= 0.02


# Non-regression: S=4096 fp32 stays on the ONLINE path (S_kv_t == 128, gate is
# S_kv_t > 128) and still meets the target.
def test_fp32_s4096_still_online(device):
    out, ref = _run(device, (1, 1, 4096, 64))
    assert _pcc(out, ref) >= 0.999
    assert _rms(out, ref) <= 0.02


# Non-regression: a short fp32 case is untouched by the gate.
def test_fp32_short_unaffected(device):
    out, ref = _run(device, (1, 1, 512, 64))
    assert _pcc(out, ref) >= 0.999
    assert _rms(out, ref) <= 0.02
