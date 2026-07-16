# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unit test for the GDN create-heads op `ttnn.experimental.nlp_create_qkv_heads_gdn`.

The op is a pure address-arithmetic scatter (fused token-major [B,1,S,(Nq+Nk+Nv)*D] ->
head-major Q/K/V [B,H,S,D], no compute), so its output must be BIT-IDENTICAL to a torch
slice+reshape+permute reference. This is also the primary gate for the batched-barrier
reader/writer change: batching the NoC transfers only relocates barriers, so `max_abs_diff`
must stay exactly 0.

The B>1 shapes exercise the writer's per-block batch-rollover bookkeeping.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc


def _reference(A, nq, nk, nv, d):
    """[B, 1, S, (nq+nk+nv)*d] -> head-major q/k/v, each [B, H, S, d]."""
    B, _, S, _ = A.shape
    q, k, v = torch.split(A, [nq * d, nk * d, nv * d], dim=-1)

    def to_heads(t, h):
        return t.reshape(B, S, h, d).permute(0, 2, 1, 3).contiguous()

    return to_heads(q, nq), to_heads(k, nk), to_heads(v, nv)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "batch, seq_len, nq, nk, nv, head_dim",
    [
        (1, 128, 4, 4, 12, 128),  # GDN Qwen3.6-27B SIM_TP=4 shape (a few seq tile-rows)
        (1, 2048, 4, 4, 12, 128),  # GDN full prefill window
        (2, 256, 4, 4, 12, 128),  # B>1 -> exercises the writer batch-rollover
        (1, 64, 2, 2, 3, 64),  # smaller / odd head counts + head_dim
    ],
)
def test_nlp_create_qkv_heads_gdn(batch, seq_len, nq, nk, nv, head_dim, dtype, device):
    torch.manual_seed(1234)
    width = (nq + nk + nv) * head_dim
    A = torch.randn(batch, 1, seq_len, width)
    if dtype == ttnn.bfloat16:
        # Snap to the bf16 grid so the reference and the (bf16-tile-moving) op agree exactly.
        A = A.to(torch.bfloat16).to(torch.float32)

    in0 = ttnn.from_torch(A, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_gdn(in0, num_q_heads=nq, num_k_heads=nk, num_v_heads=nv)

    assert list(q.padded_shape) == [batch, nq, seq_len, head_dim]
    assert list(k.padded_shape) == [batch, nk, seq_len, head_dim]
    assert list(v.padded_shape) == [batch, nv, seq_len, head_dim]

    ref_q, ref_k, ref_v = _reference(A, nq, nk, nv, head_dim)
    for name, got_t, ref in [("Q", q, ref_q), ("K", k, ref_k), ("V", v, ref_v)]:
        got = ttnn.to_torch(got_t).to(torch.float32)
        max_abs = (got - ref).abs().max().item()
        passing, pcc = comp_pcc(ref, got, 1.0)
        logger.info(f"{name}: max_abs_diff={max_abs} pcc={pcc}")
        # Pure relocation of tiles -> must be exact, not just high-PCC.
        assert max_abs == 0.0, f"{name} not bit-identical (max_abs_diff={max_abs})"
        assert passing, f"{name} PCC {pcc} below 1.0"
