# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for ``ttnn.experimental.deepseek_fused_qkv``.

The op fuses the deepseek_v4_flash decode ``_qkv`` calc (attention.py lines 680-696)
into one Blackhole device op with DRAM-sharded weights:

    Q  : q_a = rmsnorm_w(hidden @ Wqa); q = q_a @ Wqb; per-head RMSNorm(Dh) + RoPE(Rd)
    KV : kv = rmsnorm_w(hidden @ Wkv); RoPE(Rd)   (parallel, disjoint cores)

This test is self-contained: it builds random weights + gains + cos/sin + rotate matrix,
runs the op on device, and compares ``(q, kv)`` against a torch fp32 reference of exactly
that math.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tt.layers import dram_width_sharded_weight

PCC_THRESHOLD = 0.99
TILE = 32

# The op is built up in stages (see the plan todos): the KV compute partition lands first, then
# the Q path. Flip to True once the Q path is implemented so the harness also asserts q PCC.
CHECK_Q = True


def _interleaved_rotate_matrix(rope_dim: int) -> torch.Tensor:
    """[rope_dim, rope_dim] interleaved ``rotate_half`` matrix (matches attention.py)."""
    r = torch.zeros(rope_dim, rope_dim, dtype=torch.float32)
    for p in range(rope_dim // 2):
        r[2 * p, 2 * p + 1] = 1.0
        r[2 * p + 1, 2 * p] = -1.0
    return r


def _rms_norm(x: torch.Tensor, eps: float, weight: torch.Tensor | None = None) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x if weight is None else x * weight


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    d = x.shape[-1]
    nope, rope = x[..., : d - rope_dim], x[..., d - rope_dim :]
    rotated = rope * cos + (rope @ rot) * sin
    return torch.cat([nope, rotated], dim=-1)


def _torch_reference(hidden, wqa, wqb, wkv, qa_g, kv_g, cos, sin, rot, rope_dim, num_heads, eps):
    """Mirror of ``_qkv`` in torch fp32. Weights are torch nn.Linear [out, in]."""
    q_a = _rms_norm(hidden @ wqa.t(), eps, qa_g)  # [1, q_lora]
    q = q_a @ wqb.t()  # [1, H*Dh]
    h, dh = num_heads, q.shape[-1] // num_heads
    q = q.reshape(1, 1, h, dh)
    q = _rms_norm(q, eps)  # per-head, unweighted
    q = _apply_rope(q, cos, sin, rot, rope_dim)  # [1, 1, H, Dh]

    kv = _rms_norm(hidden @ wkv.t(), eps, kv_g)  # [1, Dh]
    kv = kv.reshape(1, 1, 1, -1)
    kv = _apply_rope(kv, cos, sin, rot, rope_dim)  # [1, 1, 1, Dh]
    return q, kv


def _dram_interleaved(t: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


# (name, D, q_lora, H, Dh, Rd)
@pytest.mark.parametrize(
    "name, D, q_lora, H, Dh, Rd",
    (
        ("small", 512, 256, 8, 256, 64),
        ("real", 4096, 1024, 64, 512, 64),
    ),
)
def test_deepseek_fused_qkv_op(device, reset_seeds, name, D, q_lora, H, Dh, Rd):
    eps = 1e-6
    torch.manual_seed(0)

    hidden = torch.randn(1, 1, 1, D, dtype=torch.float32) * 0.1
    wqa = torch.randn(q_lora, D, dtype=torch.float32) * (D**-0.5)
    wqb = torch.randn(H * Dh, q_lora, dtype=torch.float32) * (q_lora**-0.5)
    wkv = torch.randn(Dh, D, dtype=torch.float32) * (D**-0.5)
    qa_g = torch.randn(q_lora, dtype=torch.float32) * 0.1 + 1.0
    kv_g = torch.randn(Dh, dtype=torch.float32) * 0.1 + 1.0
    cos = torch.randn(1, 1, 1, Rd, dtype=torch.float32)
    sin = torch.randn(1, 1, 1, Rd, dtype=torch.float32)
    rot = _interleaved_rotate_matrix(Rd)

    q_ref, kv_ref = _torch_reference(hidden.reshape(1, D), wqa, wqb, wkv, qa_g, kv_g, cos, sin, rot, Rd, H, eps)

    # Device inputs.
    hidden_tt = _dram_interleaved(hidden, device)
    wqa_tt = dram_width_sharded_weight(wqa, device)
    wqb_tt = dram_width_sharded_weight(wqb, device)
    wkv_tt = dram_width_sharded_weight(wkv, device)
    qa_g_tt = _dram_interleaved(qa_g.reshape(1, 1, 1, q_lora), device)
    kv_g_tt = _dram_interleaved(kv_g.reshape(1, 1, 1, Dh), device)
    cos_tt = _dram_interleaved(cos, device)
    sin_tt = _dram_interleaved(sin, device)
    trans_mat = _interleaved_rotate_matrix(TILE).reshape(1, 1, TILE, TILE)
    trans_mat_tt = _dram_interleaved(trans_mat, device)

    q_tt, kv_tt = ttnn.experimental.deepseek_fused_qkv(
        hidden_tt,
        wqa_tt,
        wqb_tt,
        wkv_tt,
        qa_g_tt,
        kv_g_tt,
        cos_tt,
        sin_tt,
        trans_mat_tt,
        eps,
        Rd,
        H,
    )

    kv_got = ttnn.to_torch(kv_tt).float().reshape(1, 1, 1, Dh)
    kv_pass, kv_msg = comp_pcc(kv_ref, kv_got, pcc=PCC_THRESHOLD)
    logger.info(f"[fused_qkv {name}] kv {comp_allclose(kv_ref, kv_got)}")
    logger.info(f"[fused_qkv {name}] kv PCC: {kv_msg}")

    if CHECK_Q:
        q_got = ttnn.to_torch(q_tt).float().reshape(1, 1, H, Dh)
        q_pass, q_msg = comp_pcc(q_ref, q_got, pcc=PCC_THRESHOLD)
        logger.info(f"[fused_qkv {name}] q  {comp_allclose(q_ref, q_got)}")
        logger.info(f"[fused_qkv {name}] q  PCC: {q_msg}")

    assert kv_pass, f"kv PCC < {PCC_THRESHOLD} ({name}): {kv_msg}"
    if CHECK_Q:
        assert q_pass, f"q PCC < {PCC_THRESHOLD} ({name}): {q_msg}"
