# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared BinaryBackwardDeviceOperation (mul_bw first consumer)."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp


_TORCH_OF = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}


def _pt_and_tt(
    shape, low, high, device, dtype, memory_config, seed=213919, layout=ttnn.TILE_LAYOUT, required_grad=False
):
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
    # Golden operates on the device's post-quantization view so bf16 truncation matches; requires_grad wraps that view.
    pt_ref = ttnn.to_torch(tt).float()
    if required_grad:
        pt_ref = pt_ref.detach().requires_grad_(True)
    return pt_ref, tt


def _assert_mul_bw_grads(out, golden, *, ulp=None, pcc=None, out_dtype=torch.bfloat16, slots=(0, 1)):
    for i in slots:
        if golden[i] is None:
            continue
        actual = ttnn.to_torch(out[i])
        expected = golden[i]
        if ulp is not None:
            assert_with_ulp(
                expected_result=expected.to(out_dtype),
                actual_result=actual.to(out_dtype),
                ulp_threshold=ulp,
            )
        else:
            assert_with_pcc(expected.float(), actual.float(), pcc)


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
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, dtype, memory_config, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, dtype, memory_config, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, dtype, memory_config, seed=3)

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt)
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=memory_config)

    assert out[0].dtype == a_tt.dtype
    assert out[1].dtype == b_tt.dtype

    if expected_pcc is not None:
        _assert_mul_bw_grads(out, golden, pcc=expected_pcc)
        return

    torch_out_dtype = _TORCH_OF[dtype]
    _assert_mul_bw_grads(out, golden, ulp=ulp, out_dtype=torch_out_dtype)


@pytest.mark.parametrize(
    "grad_dtype,a_dtype,b_dtype,use_ulp,pcc",
    [
        (ttnn.float32, ttnn.bfloat16, ttnn.bfloat16, True, None),
        (ttnn.bfloat16, ttnn.float32, ttnn.bfloat16, True, None),
        (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat16, False, 0.99),
    ],
)
def test_mul_bw_mixed_operand_dtypes(grad_dtype, a_dtype, b_dtype, use_ulp, pcc, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, a_dtype, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, b_dtype, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, grad_dtype, mc, seed=3)

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt)
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)

    if use_ulp:
        _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16)
    else:
        _assert_mul_bw_grads(out, golden, pcc=pcc)


def test_mul_bw_broadcast_auto_alloc_preserves_operand_dtype(device):
    # Guards silent grad-dtype leak from ttnn::multiply on the composite broadcast auto-alloc branch.
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt((1, 1, 1, 128), -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt((1, 1, 32, 128), -5.0, 5.0, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt((1, 1, 32, 128), -3.0, 3.0, device, ttnn.float32, mc, seed=3)

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt)
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)

    assert out[0].dtype == a_tt.dtype
    assert out[1].dtype == b_tt.dtype
    # PCC not ULP: bf16 sum-reduce over 32 rows in the composite broadcast path accumulates ~tens of ULP.
    _assert_mul_bw_grads(out, golden, pcc=0.99)


@pytest.mark.parametrize("preallocate", ["both", "input_only", "other_only"])
def test_mul_bw_preallocated(preallocate, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    input_grad = ttnn.empty_like(a_tt) if preallocate in ("both", "input_only") else None
    other_grad = ttnn.empty_like(b_tt) if preallocate in ("both", "other_only") else None

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt)
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

    _assert_mul_bw_grads(out, golden, ulp=1, out_dtype=torch.bfloat16)


@pytest.mark.parametrize("mask", [[True, False], [False, True]])
def test_mul_bw_partial_mask_routes_to_composite(mask, device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt, are_required_outputs=mask)
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, are_required_outputs=mask, memory_config=mc)

    slots = [i for i, req in enumerate(mask) if req]
    _assert_mul_bw_grads(out, golden, ulp=1, out_dtype=torch.bfloat16, slots=slots)


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
    a_pt, a_tt = _pt_and_tt(input_shape, -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(other_shape, -5.0, 5.0, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(grad_shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt)
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)

    assert list(out[0].shape) == list(a_pt.shape)
    assert list(out[1].shape) == list(b_pt.shape)
    # Composite broadcast path does bf16 sum-reduce over up to 128 terms; ULP=4 is unreachable, PCC matches nightly.
    _assert_mul_bw_grads(out, golden, pcc=0.99)


def test_mul_bw_broadcast_with_preallocated_operand_shape(device):
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt((1, 1, 1, 128), -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt((1, 1, 32, 128), -5.0, 5.0, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt((1, 1, 32, 128), -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    input_grad = ttnn.empty_like(a_tt)
    other_grad = ttnn.empty_like(b_tt)

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt)
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
    # Same composite broadcast reduce as above; ULP is unreachable in bf16, PCC matches nightly.
    _assert_mul_bw_grads(out, golden, pcc=0.99)


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
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)
    int_tensor = ttnn.zeros(shape, dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    grad, input_ = (int_tensor, a_tt) if bad_role == "grad" else (g_tt, int_tensor)

    with expect_error(RuntimeError, "floating-point"):
        ttnn.mul_bw(grad, input_, 2.0, memory_config=mc)


def test_mul_bw_row_major_operand_routes_to_composite(device):
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(
        shape, -1.0, 1.0, device, ttnn.bfloat16, mc, layout=ttnn.ROW_MAJOR_LAYOUT, required_grad=True, seed=1
    )
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt)
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)
    # ROW_MAJOR routes through composite ttnn::multiply(tilize(a_row_major)); precision matches nightly PCC bar.
    _assert_mul_bw_grads(out, golden, pcc=0.99)


def test_mul_bw_sharded_preallocated_grad_routes_to_composite(device):
    shape = (1, 1, 32, 32)
    dram = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, dram, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, dram, required_grad=True, seed=2)
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

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt)
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, input_grad=input_grad)
    _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16)


def test_mul_bw_fp32_grad_bf16_operands_uses_fp32_dest_acc(device):
    shape = (1, 1, 320, 384)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.float32, mc, seed=3)

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt)
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)
    _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16)


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

    golden = ttnn.get_golden_function(ttnn.mul_bw)(
        g_pt.float(), a_pt.float().requires_grad_(True), b_pt.float().requires_grad_(True)
    )
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for grad_tt in (out[0], out[1]):
        assert tuple(grad_tt.padded_shape) == padded_shape

    r = tuple(slice(0, s) for s in logical_shape)
    for i in (0, 1):
        assert_with_ulp(
            expected_result=golden[i].to(torch.bfloat16),
            actual_result=ttnn.to_torch(out[i])[r].to(torch.bfloat16),
            ulp_threshold=4,
        )


def test_mul_bw_sharded_stays_on_composite(device):
    shape = (1, 1, 32, 32)
    sharded_mc = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, sharded_mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, sharded_mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, sharded_mc, seed=3)

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt)
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=sharded_mc)
    _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16)
