# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4 tests — L1 budget fit for large head dims.

Tests that large head dims (D=256, 512, 1024) no longer OOM and produce
correct results across all supported dtypes and mask modes. These shapes
previously failed with "Statically allocated circular buffers ... grow to
N B which is beyond max L1 size of 1499136 B".
"""

import pytest
import torch
import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _run_sdpa(device, B, H, S, D, dtype, *, is_causal=False, scale=None, mask=False):
    """Helper: run SDPA and compare against torch reference."""
    torch_dtype = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.bfloat8_b: torch.bfloat16,  # bf8b doesn't have a torch equivalent
    }[dtype]

    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, dtype=torch_dtype)
    k = torch.randn(B, H, S, D, dtype=torch_dtype)
    v = torch.randn(B, H, S, D, dtype=torch_dtype)

    q_t = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    k_t = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    v_t = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    kwargs = {}
    ref_kwargs = {}
    if is_causal:
        kwargs["is_causal"] = True
        ref_kwargs["is_causal"] = True
    if scale is not None:
        kwargs["scale"] = scale
        ref_kwargs["scale"] = scale
    else:
        ref_kwargs["scale"] = 1.0 / (D**0.5)
    if mask and not is_causal:
        attn_mask = torch.zeros(B, 1, S, S, dtype=torch_dtype)
        attn_mask[:, :, :, S // 2 :] = float("-inf")
        kwargs["attn_mask"] = ttnn.from_torch(attn_mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        ref_kwargs["attn_mask"] = attn_mask

    out_t = scaled_dot_product_attention(q_t, k_t, v_t, **kwargs)
    out = ttnn.to_torch(out_t)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, **ref_kwargs)

    max_diff = (out.float() - ref.float()).abs().max().item()
    return max_diff


@pytest.mark.parametrize("D", [256, 512, 1024])
def test_large_d_bf16(device, D):
    """Large head dims with bf16 — previously OOM at D >= 512."""
    max_diff = _run_sdpa(device, 1, 1, 128, D, ttnn.bfloat16)
    assert max_diff < 0.05, f"D={D} bf16: max_diff={max_diff}"
    print(f"D={D} bf16: max_diff={max_diff}")


@pytest.mark.parametrize("D", [256, 512, 1024])
def test_large_d_fp32(device, D):
    """Large head dims with fp32 — previously OOM at D >= 256."""
    max_diff = _run_sdpa(device, 1, 1, 128, D, ttnn.float32)
    assert max_diff < 0.05, f"D={D} fp32: max_diff={max_diff}"
    print(f"D={D} fp32: max_diff={max_diff}")


@pytest.mark.parametrize("D", [256, 512, 1024])
def test_large_d_bf8b(device, D):
    """Large head dims with bf8b — previously OOM at D >= 512."""
    max_diff = _run_sdpa(device, 1, 1, 128, D, ttnn.bfloat8_b)
    assert max_diff < 0.1, f"D={D} bf8b: max_diff={max_diff}"
    print(f"D={D} bf8b: max_diff={max_diff}")


@pytest.mark.parametrize("D", [512, 1024])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
def test_large_d_causal(device, D, dtype):
    """Large head dims with causal masking — previously OOM."""
    max_diff = _run_sdpa(device, 1, 1, 128, D, dtype, is_causal=True)
    threshold = 0.1 if dtype == ttnn.bfloat8_b else 0.05
    assert max_diff < threshold, f"D={D} {dtype} causal: max_diff={max_diff}"
    print(f"D={D} {dtype} causal: max_diff={max_diff}")


@pytest.mark.parametrize("D", [512, 1024])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
def test_large_d_custom_mask(device, D, dtype):
    """Large head dims with custom additive mask — previously OOM."""
    max_diff = _run_sdpa(device, 1, 1, 128, D, dtype, mask=True)
    threshold = 0.1 if dtype == ttnn.bfloat8_b else 0.05
    assert max_diff < threshold, f"D={D} {dtype} mask: max_diff={max_diff}"
    print(f"D={D} {dtype} mask: max_diff={max_diff}")


@pytest.mark.parametrize("D", [512, 1024])
def test_large_d_explicit_scale(device, D):
    """Large head dims with explicit scale — previously OOM."""
    max_diff = _run_sdpa(device, 1, 1, 128, D, ttnn.bfloat16, scale=0.125)
    assert max_diff < 0.05, f"D={D} bf16 explicit scale: max_diff={max_diff}"
    print(f"D={D} bf16 explicit scale: max_diff={max_diff}")


def test_large_d_multi_head(device):
    """Large head dims with multiple heads — previously OOM."""
    max_diff = _run_sdpa(device, 1, 8, 128, 512, ttnn.bfloat16)
    assert max_diff < 0.05, f"multi-head D=512 bf16: max_diff={max_diff}"
    print(f"multi-head D=512 bf16: max_diff={max_diff}")


def test_large_d_gqa(device):
    """Large head dims with GQA — previously OOM."""
    torch.manual_seed(42)
    B, H_q, H_kv, S, D = 1, 8, 2, 128, 512
    q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)

    q_t = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_t = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_t = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_t = scaled_dot_product_attention(q_t, k_t, v_t)
    out = ttnn.to_torch(out_t)

    # GQA reference: expand K/V heads
    k_exp = k.repeat_interleave(H_q // H_kv, dim=1)
    v_exp = v.repeat_interleave(H_q // H_kv, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp)

    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < 0.05, f"GQA D=512 bf16: max_diff={max_diff}"
    print(f"GQA D=512 bf16: max_diff={max_diff}")


def test_large_d_cross_attention(device):
    """Large head dims with cross-attention (S_q != S_kv) — previously OOM."""
    torch.manual_seed(42)
    B, H, S_q, S_kv, D = 1, 4, 64, 128, 512
    q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)

    q_t = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_t = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_t = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_t = scaled_dot_product_attention(q_t, k_t, v_t)
    out = ttnn.to_torch(out_t)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < 0.05, f"cross-attn D=512 bf16: max_diff={max_diff}"
    print(f"cross-attn D=512 bf16: max_diff={max_diff}")


@pytest.mark.parametrize("D", [512, 1024])
def test_large_d_fp32_dest_acc_false(device, D):
    """Large head dims with fp32_dest_acc_en=False — previously OOM."""
    from ttnn.operations.scaled_dot_product_attention import default_compute_kernel_config
    import struct

    torch.manual_seed(42)
    B, H, S = 1, 1, 128
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    q_t = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_t = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_t = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )
    out_t = scaled_dot_product_attention(q_t, k_t, v_t, compute_kernel_config=cfg)
    out = ttnn.to_torch(out_t)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < 0.05, f"D={D} bf16 fp32_acc=False: max_diff={max_diff}"
    print(f"D={D} bf16 fp32_acc=False: max_diff={max_diff}")


def test_d64_regression(device):
    """Regression: D=64 (smallest multi-tile D) still works."""
    max_diff = _run_sdpa(device, 1, 1, 128, 64, ttnn.bfloat16)
    assert max_diff < 0.05, f"D=64 bf16: max_diff={max_diff}"


def test_d32_regression(device):
    """Regression: D=32 (single-tile D, D_BLOCK=1, no chunking) still works."""
    max_diff = _run_sdpa(device, 1, 1, 128, 32, ttnn.bfloat16)
    assert max_diff < 0.05, f"D=32 bf16: max_diff={max_diff}"
