# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared BinaryBackwardDeviceOperation (mul/add/sub/rsub/div_bw)."""

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
def test_mul_bw_partial_mask_served_by_fused_path(mask, device):
    # Fused device op serves partial masks directly; unused slot is returned as None.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    golden = ttnn.get_golden_function(ttnn.mul_bw)(g_pt, a_pt, b_pt, are_required_outputs=mask)
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, are_required_outputs=mask, memory_config=mc)

    # Composite path must nullopt the unrequested slot (no silent full-output on partial mask).
    for i, req in enumerate(mask):
        assert (out[i] is not None) == req, f"slot {i} present={out[i] is not None} but mask said {req}"
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


# ---------------------------------------------------------------------------
# add_bw / sub_bw / rsub_bw / div_bw — shared device-op routing + edge cases.
# ---------------------------------------------------------------------------

# (op_fn, other_low, other_high) — div_bw needs non-zero other.
_OP_TABLE = [
    ("add", ttnn.add_bw, -5.0, 5.0),
    ("sub", ttnn.sub_bw, -5.0, 5.0),
    ("rsub", ttnn.rsub_bw, -5.0, 5.0),
    ("div", ttnn.div_bw, 0.1, 5.0),
]


@pytest.mark.parametrize("op_name,op_fn,other_low,other_high", _OP_TABLE)
def test_bw_fused_multi_batch_no_cb_backpressure(op_name, op_fn, other_low, other_high, device):
    # Reader sustains >2 tiles/core without CB starvation at this batch×channel×H×W.
    shape = (1, 3, 320, 384)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, other_low, other_high, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)
    golden = ttnn.get_golden_function(op_fn)(g_pt, a_pt, b_pt)
    out = op_fn(g_tt, a_tt, b_tt, memory_config=mc)
    _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "input_val,other_val,grad_val",
    [
        (1e-20, 1e-20, 1.0),  # tiny: other² = 1e-40 denormal → 1/other² = inf
        (1e30, 1e20, 1.0),  # huge: other² = 1e40 overflows fp32; (input/other)/other = -1e-10
        (-2.0, 3e-10, 1.0),  # mid: other² = 9e-20 risks precision loss
    ],
)
def test_div_bw_fused_handles_extreme_other(input_val, other_val, grad_val, device):
    # other_grad stays finite and matches (input/other)/other across fp32 range edges where other² fails.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG

    def _const(val):
        pt = torch.full(shape, val, dtype=torch.float32)
        tt = ttnn.from_torch(pt, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, memory_config=mc)
        return pt, tt

    a_pt, a_tt = _const(input_val)
    b_pt, b_tt = _const(other_val)
    g_pt, g_tt = _const(grad_val)
    out = ttnn.div_bw(g_tt, a_tt, b_tt, memory_config=mc)
    other_grad = ttnn.to_torch(out[1]).float()
    assert torch.isfinite(other_grad).all(), f"other_grad has inf/nan at input={input_val}, other={other_val}"
    # Expected via composite formulation.
    expected = -grad_val * (a_pt / b_pt) / b_pt
    if torch.isfinite(expected).all():
        rel = ((other_grad - expected).abs() / expected.abs().clamp(min=1e-30)).max().item()
        assert rel < 1e-3, f"max rel err {rel} at input={input_val}, other={other_val}"


@pytest.mark.parametrize("op_name,op_fn,other_low,other_high", _OP_TABLE)
@pytest.mark.parametrize(
    "grad_shape,input_shape,other_shape",
    [
        ((1, 1, 32, 128), (1, 1, 1, 128), (1, 1, 32, 128)),
        ((1, 1, 32, 128), (1, 1, 32, 128), (1, 1, 1, 128)),
        ((2, 4, 32, 32), (2, 1, 32, 32), (1, 4, 32, 32)),
    ],
)
def test_bw_broadcast_returns_operand_shape_grads(
    op_name, op_fn, other_low, other_high, grad_shape, input_shape, other_shape, device
):
    # Broadcast grads sum-reduce back to each operand's shape per d(a op b)/da and /db semantics.
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(input_shape, -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(other_shape, other_low, other_high, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(grad_shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)
    golden = ttnn.get_golden_function(op_fn)(g_pt, a_pt, b_pt)
    out = op_fn(g_tt, a_tt, b_tt, memory_config=mc)
    assert list(out[0].shape) == list(
        a_pt.shape
    ), f"{op_name}: input_grad shape {list(out[0].shape)} != {list(a_pt.shape)}"
    assert list(out[1].shape) == list(
        b_pt.shape
    ), f"{op_name}: other_grad shape {list(out[1].shape)} != {list(b_pt.shape)}"
    # add/sub/rsub are exact apart from the sum-reduce; div's recip adds real precision loss.
    if op_name == "div":
        _assert_mul_bw_grads(out, golden, pcc=0.99)
    else:
        _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16)


@pytest.mark.parametrize("op_name,op_fn,other_low,other_high", _OP_TABLE)
@pytest.mark.parametrize("mask", [[True, False], [False, True]])
def test_bw_partial_mask_served_by_fused_path(op_name, op_fn, other_low, other_high, mask, device):
    # Fused device op accepts any mask; the wrapper returns None for the unrequested slot.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, other_low, other_high, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)
    golden = ttnn.get_golden_function(op_fn)(g_pt, a_pt, b_pt, are_required_outputs=mask)
    out = op_fn(g_tt, a_tt, b_tt, are_required_outputs=mask, memory_config=mc)
    # Composite path must nullopt the unrequested slot (no silent full-output on partial mask).
    for i, req in enumerate(mask):
        assert (out[i] is not None) == req, f"{op_name}: slot {i} present={out[i] is not None} but mask said {req}"
    slots = [i for i, req in enumerate(mask) if req]
    _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16, slots=slots)


@pytest.mark.parametrize("op_name,op_fn,other_low,other_high", _OP_TABLE)
@pytest.mark.parametrize("bad_role", ["grad", "input", "other"])
def test_bw_int_operand_raises(op_name, op_fn, other_low, other_high, bad_role, device, expect_error):
    # Hard check runs before composite fallback — int/uint operands now TT_FATAL for all 5 ops.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, b_tt = _pt_and_tt(shape, other_low, other_high, device, ttnn.bfloat16, mc, seed=2)
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
        op_fn(grad, input_, other, memory_config=mc)


def test_rsub_bw_output_dtype_follows_input_other(device):
    # input_grad dtype follows input, other_grad dtype follows other (not grad_tensor's dtype).
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.float32, mc, seed=3)  # grad is fp32, operands bf16
    out = ttnn.rsub_bw(g_tt, a_tt, b_tt, memory_config=mc)
    assert out[0].dtype == a_tt.dtype, f"input_grad dtype {out[0].dtype} != input dtype {a_tt.dtype}"
    assert out[1].dtype == b_tt.dtype, f"other_grad dtype {out[1].dtype} != other dtype {b_tt.dtype}"


@pytest.mark.parametrize("op_name,op_fn", [("add", ttnn.add_bw), ("sub", ttnn.sub_bw), ("rsub", ttnn.rsub_bw)])
def test_bw_fp32_dest_acc_still_enabled_on_grad_only_kernel(op_name, op_fn, device):
    # fp32_dest_acc_en still derives from input_df/other_df even when 1-input kernel skips those CBs.
    shape = (1, 1, 320, 384)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.float32, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.float32, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)
    golden = ttnn.get_golden_function(op_fn)(g_pt, a_pt, b_pt)
    out = op_fn(g_tt, a_tt, b_tt, memory_config=mc)
    _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16)


@pytest.mark.parametrize("op_name,op_fn,other_low,other_high", _OP_TABLE + [("mul", ttnn.mul_bw, -5.0, 5.0)])
def test_bw_divergent_input_other_mc_resolves_per_slot(op_name, op_fn, other_low, other_high, device):
    # Without explicit output mc, input_grad lands on input.mc and other_grad on other.mc;
    # divergent per-operand memory_config is now served directly by the fused device op.
    shape = (1, 1, 32, 32)
    a_pt, a_tt = _pt_and_tt(
        shape, -1.0, 1.0, device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, required_grad=True, seed=1
    )
    b_pt, b_tt = _pt_and_tt(
        shape, other_low, other_high, device, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, required_grad=True, seed=2
    )
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, seed=3)
    golden = ttnn.get_golden_function(op_fn)(g_pt, a_pt, b_pt)
    out = op_fn(g_tt, a_tt, b_tt)  # no memory_config kwarg — defaults per slot
    assert (
        out[0].memory_config() == a_tt.memory_config()
    ), f"{op_name}: input_grad mc {out[0].memory_config()} != {a_tt.memory_config()}"
    assert (
        out[1].memory_config() == b_tt.memory_config()
    ), f"{op_name}: other_grad mc {out[1].memory_config()} != {b_tt.memory_config()}"
    _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16)


@pytest.mark.parametrize("op_name,op_fn,other_low,other_high", _OP_TABLE + [("mul", ttnn.mul_bw, -5.0, 5.0)])
def test_bw_divergent_mc_swapped_resolves_per_slot(op_name, op_fn, other_low, other_high, device):
    # Swap of test_bw_divergent_input_other_mc_resolves_per_slot: input L1 / other DRAM.
    shape = (1, 1, 32, 32)
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(
        shape, other_low, other_high, device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, required_grad=True, seed=2
    )
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, seed=3)
    golden = ttnn.get_golden_function(op_fn)(g_pt, a_pt, b_pt)
    out = op_fn(g_tt, a_tt, b_tt)
    assert out[0].memory_config() == a_tt.memory_config()
    assert out[1].memory_config() == b_tt.memory_config()
    _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16)


@pytest.mark.parametrize("mask", [[True, False], [False, True]])
@pytest.mark.parametrize("op_name,op_fn,other_low,other_high", _OP_TABLE + [("mul", ttnn.mul_bw, -5.0, 5.0)])
def test_bw_partial_mask_with_divergent_mc_per_slot(op_name, op_fn, other_low, other_high, mask, device):
    # Partial mask + divergent-mc: the requested slot must land on its own operand's mc,
    # proving WRITE_{INPUT,OTHER}_GRAD gates key off the per-slot mc, not a single global one.
    shape = (1, 1, 32, 32)
    a_pt, a_tt = _pt_and_tt(
        shape, -1.0, 1.0, device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, required_grad=True, seed=1
    )
    b_pt, b_tt = _pt_and_tt(
        shape, other_low, other_high, device, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, required_grad=True, seed=2
    )
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, seed=3)
    golden = ttnn.get_golden_function(op_fn)(g_pt, a_pt, b_pt, are_required_outputs=mask)
    out = op_fn(g_tt, a_tt, b_tt, are_required_outputs=mask)
    for i, (req, operand_tt) in enumerate(zip(mask, (a_tt, b_tt))):
        if req:
            assert out[i] is not None
            assert out[i].memory_config() == operand_tt.memory_config()
        else:
            assert out[i] is None
    slots = [i for i, req in enumerate(mask) if req]
    _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16, slots=slots)


def test_add_bw_partial_mask_fused_skips_unrequested_slot(device):
    # Caller drops preallocated for the masked-off slot and the NoC writer gate is off, so the
    # preallocated buffer must stay at its sentinel value. Compute still runs both chains (#63473).
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)
    sentinel = 7.0
    preallocated_other = ttnn.full(
        shape, sentinel, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc
    )
    out = ttnn.add_bw(
        g_tt, a_tt, b_tt, memory_config=mc, other_grad=preallocated_other, are_required_outputs=[True, False]
    )
    assert out[0] is not None and out[1] is None
    assert torch.all(ttnn.to_torch(preallocated_other) == sentinel)


@pytest.mark.parametrize(
    "bad_preallocated_kind",
    ["sharded", "row_major", "mixed_dtype"],
)
def test_add_bw_bad_preallocated_in_masked_slot_still_fused(device, bad_preallocated_kind):
    # A sharded / ROW_MAJOR / mixed-dtype preallocated in the masked-off slot would otherwise
    # soft-route the call to composite or smuggle a bad spec into the fused program. Caller
    # drops it first, so the fused path serves the requested slot and leaves the buffer alone.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)
    sentinel = 7.0
    if bad_preallocated_kind == "sharded":
        bad_mc = ttnn.create_sharded_memory_config(
            shape=shape,
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        preallocated_other = ttnn.full(
            shape, sentinel, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=bad_mc
        )
    elif bad_preallocated_kind == "row_major":
        preallocated_other = ttnn.full(
            shape, sentinel, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mc
        )
    else:
        preallocated_other = ttnn.full(
            shape, sentinel, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc
        )
    out = ttnn.add_bw(
        g_tt, a_tt, b_tt, memory_config=mc, other_grad=preallocated_other, are_required_outputs=[True, False]
    )
    assert out[0] is not None and out[1] is None
    assert torch.all(ttnn.to_torch(preallocated_other) == sentinel)


@pytest.mark.parametrize("op_name,op_fn,other_low,other_high", _OP_TABLE)
def test_bw_both_required_false_raises(op_name, op_fn, other_low, other_high, device, expect_error):
    # Composite raised "at least one gradient is expected"; fused must not silently succeed.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, b_tt = _pt_and_tt(shape, other_low, other_high, device, ttnn.bfloat16, mc, seed=2)
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)
    with expect_error(RuntimeError, "at least one"):
        op_fn(g_tt, a_tt, b_tt, are_required_outputs=[False, False], memory_config=mc)


@pytest.mark.parametrize("op_name,op_fn,other_low,other_high", _OP_TABLE)
def test_bw_mixed_dtype_preallocated_grad_routes_to_composite(op_name, op_fn, other_low, other_high, device):
    # fp32 preallocated grad with bf16 operands (mixed-precision training); composite's assign coerces.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, required_grad=True, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, other_low, other_high, device, ttnn.bfloat16, mc, required_grad=True, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)
    fp32_input_grad = ttnn.zeros(shape, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    golden = ttnn.get_golden_function(op_fn)(g_pt, a_pt, b_pt)
    out = op_fn(g_tt, a_tt, b_tt, memory_config=mc, input_grad=fp32_input_grad)
    assert out[0].dtype == ttnn.float32, f"{op_name}: preallocated fp32 input_grad lost its dtype on composite"
    # ULP on the fp32 slot catches silent coercion PCC would mask; div composite uses bf16
    # intermediates (multiply + reciprocal at input.dtype) so ULP is unreachable — PCC there.
    if op_name == "div":
        _assert_mul_bw_grads(out, golden, pcc=0.99, slots=[0])
    else:
        assert_with_ulp(
            expected_result=golden[0].to(torch.float32), actual_result=ttnn.to_torch(out[0]), ulp_threshold=4
        )
    _assert_mul_bw_grads(out, golden, ulp=4, out_dtype=torch.bfloat16, slots=[1])


def test_add_bw_non_32x32_tile_routes_to_composite(device, expect_error):
    # Pins the soft route: fused device op defers to composite, which then fails downstream
    # in copy_device_operation (empty_like drops the custom tile). End-to-end unsupported.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    tile = ttnn.Tile((16, 32))
    mk = lambda seed: ttnn.from_torch(
        torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        device=device,
        memory_config=mc,
    )
    a_tt, b_tt, g_tt = mk(1), mk(2), mk(3)
    with expect_error(RuntimeError, "same tile shape"):
        ttnn.add_bw(g_tt, a_tt, b_tt, memory_config=mc)
