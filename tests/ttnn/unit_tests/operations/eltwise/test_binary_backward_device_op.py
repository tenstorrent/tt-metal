# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared binary_backward device operation (issue #56601).

Exercises MUL_BW as the first op routed through the shared device op. Covers
dtypes x shapes vs torch, mixed operand dtypes (input vs other vs grad_output),
memory configs, preallocated outputs, program-cache keying, and validation
rejections that flip the caller back onto the composite path.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_TORCH_OF = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}


def _pt_and_tt(shape, low, high, device, dtype, memory_config, seed=213919):
    torch.manual_seed(seed)
    torch_dtype = _TORCH_OF.get(dtype, torch.bfloat16)
    pt = torch.rand(shape, dtype=torch_dtype) * (high - low) + low
    tt = ttnn.from_torch(
        pt.float() if torch_dtype != torch.float32 else pt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=memory_config,
    )
    # bfloat8_b round-trips through the tile packer, so read back the exact
    # value the device sees for the golden comparison.
    return ttnn.to_torch(tt).float(), tt


def _torch_mul_bw(grad, a, b):
    # d(a*b)/da = grad*b, d(a*b)/db = grad*a
    return grad * b, grad * a


# ---------------------------------------------------------------------------
# forward correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 320, 384),
        (1, 3, 320, 384),
        (4, 8, 512, 512),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram", "l1"],
)
def test_mul_bw_correctness(shape, dtype, memory_config, device):
    # (4,8,512,512) at float32 in L1 needs ~32MiB per operand across 5 buffers; skip that combo only.
    if shape == (4, 8, 512, 512) and dtype == ttnn.float32 and memory_config == ttnn.L1_MEMORY_CONFIG:
        pytest.skip("fp32 (4,8,512,512) x 5 buffers exceeds L1 budget")
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, dtype, memory_config, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, dtype, memory_config, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, dtype, memory_config, seed=213921)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=memory_config)
    grad_a_tt = ttnn.to_torch(out[0]).float()
    grad_b_tt = ttnn.to_torch(out[1]).float()

    pcc = 0.999 if dtype == ttnn.bfloat16 else 0.9999
    assert_with_pcc(grad_a_pt, grad_a_tt, pcc)
    assert_with_pcc(grad_b_pt, grad_b_tt, pcc)


# ---------------------------------------------------------------------------
# mixed operand dtypes — the exact class the tanh_bw factory bug bit (#56061)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grad_dtype,a_dtype,b_dtype",
    [
        (ttnn.float32, ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.bfloat16, ttnn.float32),
    ],
)
def test_mul_bw_mixed_operand_dtypes(grad_dtype, a_dtype, b_dtype, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, a_dtype, mc, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, b_dtype, mc, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, grad_dtype, mc, seed=213921)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)
    grad_a_tt = ttnn.to_torch(out[0]).float()
    grad_b_tt = ttnn.to_torch(out[1]).float()

    assert_with_pcc(grad_a_pt, grad_a_tt, 0.999)
    assert_with_pcc(grad_b_pt, grad_b_tt, 0.999)


# ---------------------------------------------------------------------------
# preallocated outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preallocate",
    ["both", "input_only", "other_only"],
)
def test_mul_bw_preallocated(preallocate, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=213921)

    input_grad = None
    other_grad = None
    if preallocate in ("both", "input_only"):
        input_grad = ttnn.empty_like(a_tt)
    if preallocate in ("both", "other_only"):
        other_grad = ttnn.empty_like(b_tt)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(
        g_tt,
        a_tt,
        b_tt,
        are_required_outputs=[True, True],
        memory_config=mc,
        input_grad=input_grad,
        other_grad=other_grad,
    )
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)


# ---------------------------------------------------------------------------
# partial mask stays on the composite path (device op contract requires both)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask", [[True, False], [False, True]])
def test_mul_bw_partial_mask_routes_to_composite(mask, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=213921)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, are_required_outputs=mask, memory_config=mc)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)
    if mask[0]:
        assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    if mask[1]:
        assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)


# ---------------------------------------------------------------------------
# program-cache keying: one entry per dtype-set for the device-op path
# ---------------------------------------------------------------------------


def test_mul_bw_program_cache_keying(device):
    shape = (1, 1, 320, 384)
    mc = ttnn.DRAM_MEMORY_CONFIG

    def _run(dtype):
        _, a = _pt_and_tt(shape, -1.0, 1.0, device, dtype, mc, seed=1)
        _, b = _pt_and_tt(shape, -5.0, 5.0, device, dtype, mc, seed=2)
        _, g = _pt_and_tt(shape, -3.0, 3.0, device, dtype, mc, seed=3)
        return ttnn.mul_bw(g, a, b, memory_config=mc)

    start = device.num_program_cache_entries()
    _run(ttnn.bfloat16)
    _run(ttnn.bfloat16)  # second call must not add an entry
    after_bf16 = device.num_program_cache_entries()
    assert after_bf16 - start == 1, f"expected 1 cache entry for bfloat16 mul_bw, got {after_bf16 - start}"
    _run(ttnn.float32)
    after_f32 = device.num_program_cache_entries()
    assert after_f32 - after_bf16 == 1, f"expected 1 additional entry for float32 mul_bw, got {after_f32 - after_bf16}"
