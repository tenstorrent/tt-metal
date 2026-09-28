# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared BinaryBackwardDeviceOperation.

Covers mul_bw as the first op routed through it. Tests focus on the shared layer's contract:
output specs, validation, program-cache keying, and broadcast reduce.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp


_TORCH_OF = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}


def _pt_and_tt(shape, low, high, device, dtype, memory_config, seed=213919, layout=ttnn.TILE_LAYOUT):
    torch.manual_seed(seed)
    torch_dtype = _TORCH_OF.get(dtype, torch.bfloat16)
    pt = torch.rand(shape, dtype=torch_dtype) * (high - low) + low
    tt = ttnn.from_torch(
        pt.float() if torch_dtype != torch.float32 else pt,
        device=device,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
    )
    return ttnn.to_torch(tt).float(), tt


def _torch_mul_bw(grad, a, b):
    return grad * b, grad * a


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 320, 384),
        (1, 3, 320, 384),
    ],
)
@pytest.mark.parametrize(
    "dtype, ulp, expected_pcc",
    [
        (ttnn.bfloat16, 4, None),
        (ttnn.float32, 4, None),
        (ttnn.bfloat8_b, None, 0.99),
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram", "l1"],
)
def test_mul_bw_correctness(shape, dtype, ulp, expected_pcc, memory_config, device):
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, dtype, memory_config, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, dtype, memory_config, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, dtype, memory_config, seed=213921)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=memory_config)
    grad_a_tt = ttnn.to_torch(out[0]).float()
    grad_b_tt = ttnn.to_torch(out[1]).float()

    assert out[0].dtype == a_tt.dtype
    assert out[1].dtype == b_tt.dtype

    if expected_pcc is not None:
        assert_with_pcc(grad_a_pt, grad_a_tt, expected_pcc)
        assert_with_pcc(grad_b_pt, grad_b_tt, expected_pcc)
        return

    torch_out_dtype = _TORCH_OF[dtype]
    assert_with_ulp(
        expected_result=grad_a_pt.to(torch_out_dtype),
        actual_result=grad_a_tt.to(torch_out_dtype),
        ulp_threshold=ulp,
    )
    assert_with_ulp(
        expected_result=grad_b_pt.to(torch_out_dtype),
        actual_result=grad_b_tt.to(torch_out_dtype),
        ulp_threshold=ulp,
    )


@pytest.mark.parametrize(
    "grad_dtype,a_dtype,b_dtype,pcc",
    [
        (ttnn.float32, ttnn.bfloat16, ttnn.bfloat16, 0.999),
        (ttnn.bfloat16, ttnn.float32, ttnn.bfloat16, 0.999),
        (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat16, 0.99),
    ],
)
def test_mul_bw_mixed_operand_dtypes(grad_dtype, a_dtype, b_dtype, pcc, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, a_dtype, mc, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, b_dtype, mc, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, grad_dtype, mc, seed=213921)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)
    grad_a_tt = ttnn.to_torch(out[0]).float()
    grad_b_tt = ttnn.to_torch(out[1]).float()

    assert_with_pcc(grad_a_pt, grad_a_tt, pcc)
    assert_with_pcc(grad_b_pt, grad_b_tt, pcc)


@pytest.mark.parametrize("preallocate", ["both", "input_only", "other_only"])
def test_mul_bw_preallocated(preallocate, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=213921)

    input_grad = ttnn.empty_like(a_tt) if preallocate in ("both", "input_only") else None
    other_grad = ttnn.empty_like(b_tt) if preallocate in ("both", "other_only") else None

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

    if input_grad is not None:
        assert out[0].buffer_address() == input_grad.buffer_address()
    if other_grad is not None:
        assert out[1].buffer_address() == other_grad.buffer_address()

    assert_with_ulp(
        expected_result=grad_a_pt.to(torch.bfloat16),
        actual_result=ttnn.to_torch(out[0]).to(torch.bfloat16),
        ulp_threshold=1,
    )
    assert_with_ulp(
        expected_result=grad_b_pt.to(torch.bfloat16),
        actual_result=ttnn.to_torch(out[1]).to(torch.bfloat16),
        ulp_threshold=1,
    )


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
        assert_with_ulp(
            expected_result=grad_a_pt.to(torch.bfloat16),
            actual_result=ttnn.to_torch(out[0]).to(torch.bfloat16),
            ulp_threshold=1,
        )
    if mask[1]:
        assert_with_ulp(
            expected_result=grad_b_pt.to(torch.bfloat16),
            actual_result=ttnn.to_torch(out[1]).to(torch.bfloat16),
            ulp_threshold=1,
        )


def _torch_ref_mul_bw(g_pt, a_pt, b_pt):
    a = a_pt.detach().clone().float().requires_grad_(True)
    b = b_pt.detach().clone().float().requires_grad_(True)
    (a * b).backward(g_pt.float())
    return a.grad, b.grad


@pytest.mark.parametrize(
    "grad_shape,input_shape,other_shape",
    [
        ((1, 1, 32, 128), (1, 1, 1, 128), (1, 1, 32, 128)),
        ((1, 1, 32, 128), (1, 1, 32, 128), (1, 1, 1, 128)),
        ((2, 4, 32, 32), (2, 1, 32, 32), (1, 4, 32, 32)),
        ((2, 3, 32, 128), (128,), (2, 3, 32, 128)),
    ],
)
def test_mul_bw_broadcast_returns_operand_shape_grads(grad_shape, input_shape, other_shape, device):
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(input_shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    b_pt, b_tt = _pt_and_tt(other_shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    g_pt, g_tt = _pt_and_tt(grad_shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    grad_a_pt, grad_b_pt = _torch_ref_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)

    assert list(out[0].shape) == list(a_pt.shape)
    assert list(out[1].shape) == list(b_pt.shape)
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)


def test_mul_bw_broadcast_with_preallocated_operand_shape(device):
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt((1, 1, 1, 128), -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    b_pt, b_tt = _pt_and_tt((1, 1, 32, 128), -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    g_pt, g_tt = _pt_and_tt((1, 1, 32, 128), -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    input_grad = ttnn.empty_like(a_tt)
    other_grad = ttnn.empty_like(b_tt)

    grad_a_pt, grad_b_pt = _torch_ref_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(
        g_tt,
        a_tt,
        b_tt,
        are_required_outputs=[True, True],
        memory_config=mc,
        input_grad=input_grad,
        other_grad=other_grad,
    )

    assert out[0].buffer_address() == input_grad.buffer_address()
    assert out[1].buffer_address() == other_grad.buffer_address()
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)


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
    _run(ttnn.bfloat16)
    after_bf16 = device.num_program_cache_entries()
    added_bf16 = after_bf16 - start
    assert added_bf16 == 1

    _run(ttnn.float32)
    added_f32 = device.num_program_cache_entries() - after_bf16
    assert added_f32 == 1


def test_mul_bw_alternating_prealloc_slot_isolates_program_cache(device):
    shape = (1, 1, 32, 32)
    output_mc = ttnn.DRAM_MEMORY_CONFIG
    prealloc_mc = ttnn.L1_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, output_mc, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, output_mc, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, output_mc, seed=3)

    def _make_l1(shape):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=prealloc_mc,
        )

    grad_a_pt = g_pt.float() * b_pt.float()
    grad_b_pt = g_pt.float() * a_pt.float()

    start = device.num_program_cache_entries()

    input_grad_l1 = _make_l1(shape)
    out_a = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=output_mc, input_grad=input_grad_l1)
    assert out_a[0].buffer_address() == input_grad_l1.buffer_address()
    assert out_a[0].memory_config().buffer_type == ttnn.BufferType.L1
    assert out_a[1].memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out_a[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out_a[1]).float(), 0.999)

    other_grad_l1 = _make_l1(shape)
    out_b = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=output_mc, other_grad=other_grad_l1)
    assert out_b[1].buffer_address() == other_grad_l1.buffer_address()
    assert out_b[0].memory_config().buffer_type == ttnn.BufferType.DRAM
    assert out_b[1].memory_config().buffer_type == ttnn.BufferType.L1
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out_b[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out_b[1]).float(), 0.999)

    added = device.num_program_cache_entries() - start
    assert added == 2


@pytest.mark.parametrize("bad_role", ["grad", "input", "other"])
def test_mul_bw_int_operand_raises(bad_role, device, expect_error):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    int_tensor = ttnn.zeros(shape, dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    grad, input_, other = (
        (int_tensor, a_tt, b_tt)
        if bad_role == "grad"
        else (g_tt, int_tensor, b_tt)
        if bad_role == "input"
        else (g_tt, a_tt, int_tensor)
    )

    with expect_error(RuntimeError, "floating-point"):
        ttnn.mul_bw(grad, input_, other, memory_config=mc)


@pytest.mark.parametrize("bad_role", ["grad", "input"])
def test_mul_bw_scalar_overload_int_operand_raises(bad_role, device, expect_error):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=2)
    int_tensor = ttnn.zeros(shape, dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    grad, input_ = (int_tensor, a_tt) if bad_role == "grad" else (g_tt, int_tensor)

    with expect_error(RuntimeError, "floating-point"):
        ttnn.mul_bw(grad, input_, 2.0, memory_config=mc)


def test_mul_bw_row_major_operand_routes_to_composite(device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)
    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)


def test_mul_bw_sharded_preallocated_grad_routes_to_composite(device):
    shape = (1, 1, 32, 32)
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, dram, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, dram, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, dram, seed=3)
    sharded_mc = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_grad = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=sharded_mc,
    )

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, input_grad=input_grad)
    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)


def test_mul_bw_fp32_grad_bf16_operands_uses_fp32_dest_acc(device):
    shape = (1, 1, 320, 384)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.float32, mc, seed=3)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)
    grad_a_tt = ttnn.to_torch(out[0]).float()
    grad_b_tt = ttnn.to_torch(out[1]).float()

    assert_with_ulp(
        expected_result=grad_a_pt.to(torch.bfloat16),
        actual_result=grad_a_tt.to(torch.bfloat16),
        ulp_threshold=4,
    )
    assert_with_ulp(
        expected_result=grad_b_pt.to(torch.bfloat16),
        actual_result=grad_b_tt.to(torch.bfloat16),
        ulp_threshold=4,
    )


def test_mul_bw_rejects_preallocated_output_dtype_mismatch(device, expect_error):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    wrong_dtype = ttnn.zeros(shape, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    correct = ttnn.empty_like(b_tt)

    with expect_error(RuntimeError, "dtype"):
        ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc, input_grad=wrong_dtype, other_grad=correct)


def test_mul_bw_output_preserves_input_padded_shape(device):
    logical_shape = (1, 1, 40, 40)
    padded_shape = (1, 1, 96, 96)
    torch.manual_seed(41)
    a_pt = torch.randn(logical_shape, dtype=torch.bfloat16)
    b_pt = torch.randn(logical_shape, dtype=torch.bfloat16)
    g_pt = torch.randn(logical_shape, dtype=torch.bfloat16)

    def _tilize_pad(pt):
        rm = ttnn.from_torch(pt, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        return ttnn.tilize_with_val_padding(rm, padded_shape, 0.0)

    a_tt, b_tt, g_tt = _tilize_pad(a_pt), _tilize_pad(b_pt), _tilize_pad(g_pt)
    assert tuple(a_tt.padded_shape) == padded_shape

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for grad_tt in (out[0], out[1]):
        assert tuple(grad_tt.padded_shape) == padded_shape

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)
    r = tuple(slice(0, s) for s in logical_shape)
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0])[r].float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1])[r].float(), 0.999)


def test_mul_bw_sharded_stays_on_composite(device):
    shape = (1, 1, 32, 32)
    sharded_mc = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, sharded_mc, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, sharded_mc, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, sharded_mc, seed=3)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=sharded_mc)
    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)
