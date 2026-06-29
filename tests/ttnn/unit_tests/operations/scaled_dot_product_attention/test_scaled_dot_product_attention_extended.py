# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Extended tests for scaled_dot_product_attention — focused shape/parameter coverage.

Small matrix: small/medium/large + GQA + MQA + cross-attn + mask + explicit scale.
Run with:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_extended.py
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _run_sdpa(device, q_shape, k_shape, v_shape, *, attn_mask=None, is_causal=False, scale=None, pcc_threshold=0.995):
    torch.manual_seed(42)

    torch_Q = torch.randn(q_shape, dtype=torch.bfloat16)
    torch_K = torch.randn(k_shape, dtype=torch.bfloat16)
    torch_V = torch.randn(v_shape, dtype=torch.bfloat16)

    # GQA/MQA: replicate K/V heads to match Q for torch reference
    H_q = q_shape[1]
    H_kv = k_shape[1]
    Kf = torch_K.float()
    Vf = torch_V.float()
    if H_q != H_kv:
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)

    expected = torch.nn.functional.scaled_dot_product_attention(
        torch_Q.float(),
        Kf,
        Vf,
        attn_mask=attn_mask.float() if attn_mask is not None else None,
        is_causal=is_causal,
        scale=scale,
    )

    ttnn_Q = ttnn.from_torch(
        torch_Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        torch_K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        torch_V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_mask = None
    if attn_mask is not None:
        ttnn_mask = ttnn.from_torch(
            attn_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, attn_mask=ttnn_mask, is_causal=is_causal, scale=scale)
    torch_output = ttnn.to_torch(output)

    assert list(output.shape) == list(q_shape)

    out_f = torch_output.float().flatten()
    exp_f = expected.flatten()
    out_c = out_f - out_f.mean()
    exp_c = exp_f - exp_f.mean()
    denom = torch.sqrt((out_c**2).sum()) * torch.sqrt((exp_c**2).sum())
    pcc = (out_c * exp_c).sum() / denom if denom > 0 else 1.0

    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} < {pcc_threshold}"


def _make_causal_mask(B, S_q, S_kv):
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch.bfloat16)
    mask.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    return mask


# --- Shape variations ---


def test_sdpa_single_tile(device):
    _run_sdpa(device, (1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32))


def test_sdpa_multi_tile(device):
    _run_sdpa(device, (1, 1, 128, 128), (1, 1, 128, 128), (1, 1, 128, 128))


def test_sdpa_multi_q_blocks(device):
    """S_q not divisible by B_q_t: exercises the block-size divisor logic."""
    _run_sdpa(device, (1, 1, 192, 64), (1, 1, 192, 64), (1, 1, 192, 64))


def test_sdpa_long_seq(device):
    _run_sdpa(device, (1, 1, 512, 64), (1, 1, 512, 64), (1, 1, 512, 64))


# --- Head configurations ---


def test_sdpa_gqa(device):
    _run_sdpa(device, (1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64))


def test_sdpa_mqa(device):
    _run_sdpa(device, (1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64))


# --- Cross-attention ---


def test_sdpa_cross_attention(device):
    _run_sdpa(device, (1, 4, 64, 64), (1, 4, 128, 64), (1, 4, 128, 64))


# --- Mask variations ---


def test_sdpa_custom_mask(device):
    mask = _make_causal_mask(1, 128, 128)
    _run_sdpa(device, (1, 1, 128, 128), (1, 1, 128, 128), (1, 1, 128, 128), attn_mask=mask)


def test_sdpa_custom_mask_multi_block(device):
    """Mask with multiple Q-blocks and KV-blocks — the key correctness case."""
    mask = _make_causal_mask(1, 256, 256)
    _run_sdpa(device, (1, 1, 256, 256), (1, 1, 256, 256), (1, 1, 256, 256), attn_mask=mask)


def test_sdpa_explicit_scale(device):
    _run_sdpa(device, (1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64), scale=0.125)
