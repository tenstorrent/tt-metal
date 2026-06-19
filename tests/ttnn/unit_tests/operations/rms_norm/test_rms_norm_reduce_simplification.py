# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Refinement 5 — Remove the redundant DEST-level reduce-accumulate chunking.
#
# PASS-1 is now a single `square` + single `reduce<SUM,REDUCE_ROW>` over the whole
# resident shard (no num_chunks / reduce_block / Accumulate loop). These tests
# target the cases that, under the OLD kernel, exercised num_chunks > 1 — i.e. a
# resident shard WIDER than DEST_AUTO_LIMIT (4 for fp32_dest_acc, 8 for bf16 acc).
# If the single-reduce path were wrong (e.g. only reducing the first DEST-worth of
# tiles), the Sum(x^2) would be too small and the output too large by a constant
# factor — exactly the failure class Refinement 1 chased. We assert exactness on
# all-ones and high PCC vs torch on random input, across both regimes and layouts.
#
# DO NOT DELETE — documents the Refinement 5 correctness contract.

import math

import pytest
import torch

import ttnn
from ttnn.operations.rms_norm import rms_norm


def _torch_rms(x, gamma, eps=1e-6):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(var + eps)
    if gamma is not None:
        out = out * gamma.reshape(-1)
    return out


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0
    return float((a @ b) / denom)


# Shapes whose resident shard is WIDE enough that the OLD kernel used num_chunks > 1.
#   - Regime A-fallback wide rows: (2,512,1024) Wt=32, (1024,1024) Wt=32 — shard=32 tiles
#     >> DEST limit, so the old PASS-1 looped 4-8 chunks. Single reduce must match.
#   - Regime A tall: (4,1,512,512) Wt=16 — shard=16 tiles, old looped 2-4 chunks.
#   - Regime B wide: (1,1,32,32768) Wt_s=16, (128,8192) Wt_s=16 — per-core shard
#     16 tiles, old looped multiple chunks within the shard.
TILE_SHAPES = [
    (4, 1, 512, 512),  # Regime A, shard 16 tiles
    (2, 512, 1024),  # Regime A-fallback, shard 32 tiles (8 fp32 chunks old)
    (1024, 1024),  # Regime A-fallback, shard 32 tiles
    (1, 1, 32, 32768),  # Regime B, Wt_s=16
    (128, 8192),  # Regime B, Wt_s=16
    (1, 1, 64, 12288),  # Regime B, Wt_s=12
]


@pytest.mark.parametrize("shape", TILE_SHAPES, ids=[str(s) for s in TILE_SHAPES])
@pytest.mark.parametrize("with_gamma", [False, True], ids=["no_gamma", "gamma"])
def test_reduce_simplification_tile_allones(device, shape, with_gamma):
    """All-ones: Sum(x^2)=W, rms=1, output==1.0 exactly (mod bf16 rounding).

    A truncated reduce (only first chunk) would yield output = sqrt(W / Wt_s_first)
    >> 1 — the bug this catches."""
    x = torch.ones(shape, dtype=torch.bfloat16)
    gamma = torch.ones((1, 1, 1, shape[-1]), dtype=torch.bfloat16) if with_gamma else None

    ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) if with_gamma else None
    out = ttnn.to_torch(rms_norm(ti, gamma=tg)).to(torch.float32).reshape(shape)
    err = (out - 1.0).abs().max().item()
    assert err < 0.05, f"shape={shape} gamma={with_gamma} max|out-1|={err}"


@pytest.mark.parametrize("shape", TILE_SHAPES, ids=[str(s) for s in TILE_SHAPES])
@pytest.mark.parametrize("with_gamma", [False, True], ids=["no_gamma", "gamma"])
def test_reduce_simplification_tile_random(device, shape, with_gamma):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32) if with_gamma else None
    ref = _torch_rms(x, gamma)

    ti = ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tg = (
        ttnn.from_torch(gamma.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if with_gamma
        else None
    )
    out = ttnn.to_torch(rms_norm(ti, gamma=tg)).to(torch.float32).reshape(shape)
    pcc = _pcc(out, ref)
    assert pcc > 0.999, f"shape={shape} gamma={with_gamma} pcc={pcc}"


# ROW_MAJOR multi-chunk: a single (b,c,h) stick of W elements; W wide enough that
# the tilized resident shard spans many reduce_block chunks under the old kernel.
RM_SHAPES = [
    (1, 1, 32, 1024),  # RM Regime A, Wt=32 -> old 4-8 chunks
    (1, 32, 1024),  # RM Regime A
    (1, 1, 32, 8192),  # RM, wide (Regime A or B by fit)
    (1, 32, 8192),  # RM wide
]


@pytest.mark.parametrize("shape", RM_SHAPES, ids=[str(s) for s in RM_SHAPES])
@pytest.mark.parametrize("with_gamma", [False, True], ids=["no_gamma", "gamma"])
def test_reduce_simplification_rm_random(device, shape, with_gamma):
    torch.manual_seed(1)
    x = torch.randn(shape, dtype=torch.float32)
    gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32) if with_gamma else None
    ref = _torch_rms(x, gamma)

    ti = ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tg = (
        ttnn.from_torch(gamma.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        if with_gamma
        else None
    )
    out = ttnn.to_torch(rms_norm(ti, gamma=tg)).to(torch.float32).reshape(shape)
    pcc = _pcc(out, ref)
    assert pcc > 0.999, f"shape={shape} gamma={with_gamma} pcc={pcc}"
