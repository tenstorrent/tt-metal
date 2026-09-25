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

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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
# bit-addressable floats use ULP because mul_bw is a pure multiply (kernel result is
# round_to_dtype(grad*operand)); block-float types keep PCC (shared-exponent quantisation).
@pytest.mark.parametrize(
    "dtype, ulp, expected_pcc",
    [
        (ttnn.bfloat16, 4, None),
        (ttnn.float32, 4, None),
        (ttnn.bfloat8_b, None, 0.99),
        (ttnn.bfloat4_b, None, 0.93),
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["dram", "l1"],
)
def test_mul_bw_correctness(shape, dtype, ulp, expected_pcc, memory_config, device):
    if shape == (4, 8, 512, 512) and dtype == ttnn.float32 and memory_config == ttnn.L1_MEMORY_CONFIG:
        pytest.skip("fp32 (4,8,512,512) x 5 buffers exceeds L1 budget")
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, dtype, memory_config, seed=213919)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, dtype, memory_config, seed=213920)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, dtype, memory_config, seed=213921)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=memory_config)
    grad_a_tt = ttnn.to_torch(out[0]).float()
    grad_b_tt = ttnn.to_torch(out[1]).float()

    assert out[0].dtype == a_tt.dtype, f"input_grad dtype {out[0].dtype} != input dtype {a_tt.dtype}"
    assert out[1].dtype == b_tt.dtype, f"other_grad dtype {out[1].dtype} != other dtype {b_tt.dtype}"

    if expected_pcc is not None:
        assert_with_pcc(grad_a_pt, grad_a_tt, expected_pcc)
        assert_with_pcc(grad_b_pt, grad_b_tt, expected_pcc)
        return

    # Compare in operand dtype so ULP is measured in the space the device sees; comparing
    # at fp32 for a bf16 result would score every pack-back rounding as ~800000 fp32 ULPs.
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


# ---------------------------------------------------------------------------
# mixed operand dtypes — the exact class the tanh_bw factory bug bit (#56061)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grad_dtype,a_dtype,b_dtype,pcc",
    [
        (ttnn.float32, ttnn.bfloat16, ttnn.bfloat16, 0.999),
        (ttnn.bfloat16, ttnn.float32, ttnn.bfloat16, 0.999),
        (ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, 0.999),
        # bfloat8_b is a shared-exponent block format, so per-tile quantisation is looser;
        # the nightly mul_bw sweep parametrises it for every operand, so cover it here too.
        (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat16, 0.99),
        (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, 0.99),
        (ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat8_b, 0.99),
        (ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.float32, 0.99),
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
    # Address parity: op must write into the caller's buffers, not silently allocate fresh.
    if input_grad is not None:
        assert out[0].buffer_address() == input_grad.buffer_address(), (
            f"input_grad buffer_address diverged (out={out[0].buffer_address()} "
            f"preallocated={input_grad.buffer_address()}); op ignored the preallocated tensor"
        )
    if other_grad is not None:
        assert out[1].buffer_address() == other_grad.buffer_address(), (
            f"other_grad buffer_address diverged (out={out[1].buffer_address()} "
            f"preallocated={other_grad.buffer_address()}); op ignored the preallocated tensor"
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
# broadcasting operands — PyTorch autograd parity via composite reduce_to_shape
# ---------------------------------------------------------------------------


def _torch_autograd_mul_bw(g_pt, a_pt, b_pt):
    # Ground truth: exactly what AccumulateGrad does on the CPU side of PyTorch autograd.
    a = a_pt.detach().clone().float().requires_grad_(True)
    b = b_pt.detach().clone().float().requires_grad_(True)
    (a * b).backward(g_pt.float())
    return a.grad, b.grad


@pytest.mark.parametrize(
    "grad_shape,input_shape,other_shape,label",
    [
        # forward: input broadcasts up on dim 2 -> grad_output at (1,1,32,128); reduce input_grad only.
        ((1, 1, 32, 128), (1, 1, 1, 128), (1, 1, 32, 128), "leading_bcast_input"),
        # forward: other broadcasts up on dim 2 -> reduce other_grad only.
        ((1, 1, 32, 128), (1, 1, 32, 128), (1, 1, 1, 128), "leading_bcast_other"),
        # both operands broadcast on different axes -> reduce both grads.
        ((2, 4, 32, 32), (2, 1, 32, 32), (1, 4, 32, 32), "channel_bcast_both"),
        # scalar-broadcast: one operand is a single-element tile.
        ((2, 4, 32, 32), (1, 1, 32, 32), (2, 4, 32, 32), "scalar_like_input"),
        # cross-rank: operand.rank < grad.rank; needs sum + rank-drop reshape.
        ((2, 3, 32, 128), (128,), (2, 3, 32, 128), "cross_rank_input"),
        ((2, 3, 32, 128), (2, 3, 32, 128), (32, 128), "cross_rank_other"),
        # regression: same-shape stays on the fast (single-launch) path.
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32), "no_bcast_fast_path"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_mul_bw_broadcast_matches_pytorch_autograd(grad_shape, input_shape, other_shape, label, device):
    # PyTorch parity: composite mul_bw's reduce_to_shape must return grads at operand
    # shapes matching torch.autograd's AccumulateGrad, not the broadcast shape.
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(input_shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    b_pt, b_tt = _pt_and_tt(other_shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    g_pt, g_tt = _pt_and_tt(grad_shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    grad_a_pt, grad_b_pt = _torch_autograd_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)

    # Shape parity first — silently-broadcast-shape grads were the pre-existing bug.
    assert list(out[0].shape) == list(a_pt.shape), (
        f"[{label}] input_grad shape={list(out[0].shape)} != input shape={list(a_pt.shape)}; "
        f"composite reduce_to_shape must match PyTorch AccumulateGrad's shape contract"
    )
    assert list(out[1].shape) == list(
        b_pt.shape
    ), f"[{label}] other_grad shape={list(out[1].shape)} != other shape={list(b_pt.shape)}"
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)


def test_mul_bw_broadcast_with_preallocated_operand_shape(device):
    # Preallocated grads sized to operand shape (not grad shape) still round-trip; the
    # reduce path writes into them via ttnn::assign, not through the multiply's output slot.
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt((1, 1, 1, 128), -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    b_pt, b_tt = _pt_and_tt((1, 1, 32, 128), -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    g_pt, g_tt = _pt_and_tt((1, 1, 32, 128), -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    input_grad = ttnn.empty_like(a_tt)
    other_grad = ttnn.empty_like(b_tt)

    grad_a_pt, grad_b_pt = _torch_autograd_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(
        g_tt,
        a_tt,
        b_tt,
        are_required_outputs=[True, True],
        memory_config=mc,
        input_grad=input_grad,
        other_grad=other_grad,
    )

    assert (
        out[0].buffer_address() == input_grad.buffer_address()
    ), "preallocated input_grad address changed; reduce path must ttnn::assign into caller's buffer"
    assert out[1].buffer_address() == other_grad.buffer_address()
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
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
    added_bf16 = after_bf16 - start
    assert added_bf16 == 1, (
        f"expected 1 program cache entry for two identical bfloat16 mul_bw calls (second a hit), got {added_bf16}; "
        "more means the hash separates runs it should share, fewer means it collides distinct programs"
    )
    _run(ttnn.float32)
    added_f32 = device.num_program_cache_entries() - after_bf16
    assert added_f32 == 1, (
        f"expected 1 additional entry when switching bfloat16 -> float32 mul_bw, got {added_f32}; "
        "0 means the hash collides dtypes, >1 means the second float32 program was not cached"
    )


# ---------------------------------------------------------------------------
# rejection paths (routing gate + device op TT_FATALs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_role", ["grad", "input", "other"])
def test_mul_bw_int_operand_raises(bad_role, device, expect_error):
    # PyTorch autograd also throws on int/uint gradient requests; mul_bw matches that
    # instead of routing to a silent composite via mul_int_tile.
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


def test_mul_bw_all_int32_operands_raise(device, expect_error):
    # Pure int32 for all three: PyTorch would raise on any of these having requires_grad;
    # ttnn matches so no silent-composite mul_int_tile masquerading as a valid backward.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    torch.manual_seed(51)
    a_pt = torch.randint(-8, 8, shape, dtype=torch.int32)
    b_pt = torch.randint(-8, 8, shape, dtype=torch.int32)
    g_pt = torch.randint(-8, 8, shape, dtype=torch.int32)
    to_tt = lambda pt: ttnn.from_torch(pt, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32, memory_config=mc)
    a_tt, b_tt, g_tt = to_tt(a_pt), to_tt(b_pt), to_tt(g_pt)

    with expect_error(RuntimeError, "floating-point"):
        ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)


def test_mul_bw_row_major_operand_routes_to_composite(device):
    # ROW_MAJOR operand: device op requires TILE, ttnn.multiply accepts ROW_MAJOR.
    # Gate must fall back — no MUL_BW device-op fatal.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1, layout=ttnn.ROW_MAJOR_LAYOUT)
    _, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    try:
        ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)
    except RuntimeError as exc:
        assert "MUL_BW device op" not in str(exc) and "MUL_BW operation" not in str(
            exc
        ), f"routing gate leaked a ROW_MAJOR operand into MUL_BW device op:\n{exc}"


def test_mul_bw_sharded_preallocated_grad_routes_to_composite(device):
    # Preallocated sharded input_grad + interleaved operands + default mem_config:
    # compute_output_specs returns preallocated->tensor_spec() verbatim, so a sharded
    # buffer would slip past an input-only gate. Must fall back to composite.
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
    # Guards issue #43196: fp32 grad + bf16 operands. Without fp32_dest_acc_en, an fp32
    # tile is unpacked into a bf16-configured DEST and tile-aligned corruption follows.
    # PCC 0.999 does not catch it; ULP does.
    shape = (1, 1, 320, 384)
    mc = ttnn.DRAM_MEMORY_CONFIG
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.float32, mc, seed=3)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc)
    grad_a_tt = ttnn.to_torch(out[0]).float()
    grad_b_tt = ttnn.to_torch(out[1]).float()

    # Assert in bf16 (the DEST-facing output dtype); ULP measured in fp32 would score
    # every pack-back rounding as ~800000 ULPs.
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
    # Dtype-mismatched preallocated grad: gate rejects and the composite's ttnn.multiply
    # also raises; either path must fail loudly rather than silently coerce.
    shape = (1, 1, 32, 32)
    mc = ttnn.DRAM_MEMORY_CONFIG
    _, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, mc, seed=1)
    _, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, mc, seed=2)
    _, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, mc, seed=3)

    wrong_dtype = ttnn.zeros(shape, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    correct = ttnn.empty_like(b_tt)

    with expect_error(RuntimeError, "dtype"):
        ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=mc, input_grad=wrong_dtype, other_grad=correct)


def test_mul_bw_default_mem_config_divergent_operands_stays_on_composite(device):
    # A1: routing gate rejects when input and other differ mem_configs without an explicit
    # output_memory_config; this test pins the correctness half of that fallback.
    shape = (1, 1, 32, 32)
    a_pt, a_tt = _pt_and_tt(shape, -1.0, 1.0, device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, seed=1)
    b_pt, b_tt = _pt_and_tt(shape, -5.0, 5.0, device, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, seed=2)
    g_pt, g_tt = _pt_and_tt(shape, -3.0, 3.0, device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, seed=3)

    out = ttnn.mul_bw(g_tt, a_tt, b_tt)

    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)


def test_mul_bw_output_preserves_input_padded_shape(device):
    # Input padded beyond tile alignment via tilize_with_val_padding: the factory writes
    # input.physical_volume()/TILE_HW pages, so the output must carry the same padded shape
    # or the writer runs past the allocation (same class as #56061 5daac64).
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
    assert tuple(a_tt.padded_shape) == padded_shape, f"tilize_with_val_padding didn't produce {padded_shape}"

    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for grad_tt, name in ((out[0], "input_grad"), (out[1], "other_grad")):
        assert tuple(grad_tt.padded_shape) == padded_shape, (
            f"{name} padded_shape {tuple(grad_tt.padded_shape)} != input's {padded_shape}; "
            f"writer would overrun on the {padded_shape[-2] * padded_shape[-1] // 32 // 32}-tile input"
        )
    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)
    r = tuple(slice(0, s) for s in logical_shape)
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0])[r].float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1])[r].float(), 0.999)


def test_mul_bw_sharded_stays_on_composite(device):
    # Sharded operands must fall back to composite (fused device op is interleaved-only);
    # same regression class as sigmoid_bw in #56061 (b59f52d).
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

    # If the gate regresses, the per-operand sharded rejection would fire with the
    # MUL_BW-keyed message; composite accepts and returns a matching-precision result.
    out = ttnn.mul_bw(g_tt, a_tt, b_tt, memory_config=sharded_mc)
    grad_a_pt, grad_b_pt = _torch_mul_bw(g_pt, a_pt, b_pt)
    assert_with_pcc(grad_a_pt, ttnn.to_torch(out[0]).float(), 0.999)
    assert_with_pcc(grad_b_pt, ttnn.to_torch(out[1]).float(), 0.999)
