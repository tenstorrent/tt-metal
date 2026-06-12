# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Cache-correctness sweep for moreh ops (batch A) migrated to the descriptor framework (#46506).

Companion to test_descriptor_cache.py — same module-scoped `cache_device` fixture (program
cache enabled, l1_small_size=32768), same two-pronged contract:

  * not stale       : every call's result matches the op's torch golden, even on a
                      program-cache HIT (a frozen per-call rt-arg would make a later call wrong).
  * not over-caching: on the cache-HIT loop (attributes held FIXED, only input DATA varied so a
                      fresh allocation gives a fresh address), the op must NOT create a new
                      program-cache entry per call.

Per-op invocations + torch references are copied verbatim from each op's own nightly test
(tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_*.py); shapes/cases are not invented.

Verdict after reading the device factories under ttnn/cpp/ttnn/operations/moreh/<op>/device/:
all six ops are predicted OK. Every per-call scalar they carry (abs_pow `p`; clip_grad_norm
`norm_type`/`max_norm`) lives in `operation_attributes_t` and is therefore covered by the DEFAULT
program hash, so varying it legitimately adds a cache entry (correct, NOT a bug). All tensor
addresses are bound as Buffer* in the descriptor rt-args (patched on the cache-hit path). There
is no scalar that is baked into rt-args yet EXCLUDED from the hash (frozen) nor one that is
hashed-but-should-be-reapplied. The tests below therefore:
  - probe the HIT path by holding attributes FIXED and varying input DATA, bounding entries <= 1;
  - additionally vary the scalar (p / norm_type / max_norm) asserting NOT-stale, but do NOT bound
    the entry count on that loop, because a new entry there is the correct hash-covered behavior.
"""

import random

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import TILE_HEIGHT, TILE_WIDTH


@pytest.fixture(scope="module")
def cache_device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# moreh_abs_pow  (ttnn.operations.moreh.abs(input, p) -> |x|^p, element-wise).
# Scalar `p` -> (floored_p, decimal, p_is_negative) baked into reader/compute rt-args, but `p`
# is in operation_attributes_t and there is NO custom compute_program_hash, so the default hash
# covers it. Verdict: OK.
#   * fresh-data loop with FIXED p: must not go stale and must add <= 1 entry (HIT path).
#   * varying-p loop: must not go stale (entry count not bounded — new entry is correct).
# Reference torch formula matches the device op (element-wise |x|^p), tolerances mirror the
# moreh comp_allclose_and_pcc defaults used by sibling moreh tests (rtol=atol=0.1, pcc=0.999).
# ---------------------------------------------------------------------------
def _abs_pow_check(cache_device, x, p):
    npu_dtype = ttnn.bfloat16
    tt_input = ttnn.from_torch(x.bfloat16(), dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=cache_device)
    tt_out = ttnn.operations.moreh.abs(tt_input, p)
    actual = ttnn.to_torch(tt_out)
    expected = torch.abs(x).pow(p)
    passing, out = comp_allclose_and_pcc(expected, actual, pcc=0.999, rtol=0.1, atol=0.1)
    return passing, out


def test_moreh_abs_pow_cache(cache_device):
    torch.manual_seed(2024)
    shape = [1, 1, TILE_HEIGHT, TILE_WIDTH]
    p_fixed = 2.0

    # HIT path: fixed p, fresh data each iter (fresh address).
    # WARM-UP then assert ZERO growth (robust to ops that build multiple sub-programs).
    def _hit():
        x = torch.rand(shape, dtype=torch.float32) + 0.5  # keep away from 0 (bf16 |x|^p precision)
        passing, out = _abs_pow_check(cache_device, x, p_fixed)
        assert passing, f"abs_pow stale on cache hit (fixed p={p_fixed}): {out}"

    _hit()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _hit()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"abs_pow: cache grew past {base} across fresh allocations (over-caching)"

    # Not-stale across varying p (p is hash-covered -> a new entry per p is correct, not bounded).
    x = torch.rand(shape, dtype=torch.float32) + 0.5
    for p in [2.0, 3.0, 0.8, 1.0]:
        passing, out = _abs_pow_check(cache_device, x, p)
        assert passing, f"abs_pow stale at p={p}: {out}"


# ---------------------------------------------------------------------------
# moreh_clip_grad_norm  (ttnn.operations.moreh.clip_grad_norm(grads, max_norm, norm_type)).
# This single public op drives step1 (|x|^norm_type partial sums), step2 (total_norm =
# sum^(1/norm_type)), and step3 (in-place grad *= clip_coef). Per-call scalars:
#   * norm_type -> `decimal` baked into step1 & step2 reader rt-args; norm_type is in each step's
#     operation_attributes_t, no custom hash -> default hash covers it.  OK.
#   * max_norm  -> affects clip_coef, which step3 consumes as a TENSOR (clip_coef_clamped, bound
#     as Buffer*, patched on hit), not a baked scalar.  OK.
# Invocation + torch golden + tolerances copied verbatim from test_moreh_clip_grad_norm.py
# (clip_grad_norm_ reference, comp_allclose_and_pcc rtol=atol=0.1). Single-parameter, fixed shape
# for a clean cache-entry count.
# ---------------------------------------------------------------------------
def _clip_grad_norm_check(cache_device, grad, max_norm, norm_type):
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.float32

    param = torch.nn.Parameter(torch.empty(grad.shape, dtype=cpu_dtype))
    param.grad = grad.clone()
    cpu_total_norm = torch.nn.utils.clip_grad_norm_([param], max_norm, norm_type)

    npu_inputs = [
        ttnn.from_torch(grad.clone().bfloat16(), dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=cache_device)
    ]
    npu_total_norm = ttnn.operations.moreh.clip_grad_norm(npu_inputs, max_norm, norm_type)
    actual_total_norm = ttnn.to_torch(npu_total_norm).reshape(1)

    pass_total_norm, out_total_norm = comp_allclose_and_pcc(actual_total_norm, cpu_total_norm, rtol=0.1, atol=0.1)
    # Check the in-place-clipped grad as well.
    actual_grad = ttnn.to_torch(npu_inputs[0])
    pass_grad, out_grad = comp_allclose_and_pcc(param.grad, actual_grad, rtol=0.1, atol=0.1)
    return (pass_total_norm and pass_grad), f"total_norm:{out_total_norm} grad:{out_grad}"


def test_moreh_clip_grad_norm_cache(cache_device):
    torch.manual_seed(2023)
    random.seed(2023)
    shape = [2, 2, 2 * TILE_HEIGHT, 2 * TILE_WIDTH]
    max_norm_fixed = 1.0
    norm_type_fixed = 2.0

    # HIT path: fixed attributes, fresh grad data each iter.
    # WARM-UP then assert ZERO growth: clip_grad_norm is COMPOSITE (step1 + step2 + step3 each
    # cache one program), so the first call settles the cache at N entries; subsequent passes with
    # the same config must add none, regardless of N.
    def _hit():
        grad = torch.empty(shape, dtype=torch.float32).uniform_(0, 2.5)
        passing, out = _clip_grad_norm_check(cache_device, grad, max_norm_fixed, norm_type_fixed)
        assert passing, f"clip_grad_norm stale on cache hit (fixed max_norm/norm_type): {out}"

    _hit()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _hit()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"clip_grad_norm: cache grew past {base} across fresh allocations (over-caching)"

    # Not-stale across varying norm_type (hash-covered -> new entries are correct, not bounded).
    grad = torch.empty(shape, dtype=torch.float32).uniform_(0, 2.5)
    for norm_type in [2.0, 2.2, 0.8]:
        passing, out = _clip_grad_norm_check(cache_device, grad, max_norm_fixed, norm_type)
        assert passing, f"clip_grad_norm stale at norm_type={norm_type}: {out}"

    # Not-stale across varying max_norm (changes clip_coef tensor; norm_type fixed = same programs).
    for max_norm in [2.0, 1.0, -1.0]:
        passing, out = _clip_grad_norm_check(cache_device, grad, max_norm, norm_type_fixed)
        assert passing, f"clip_grad_norm stale at max_norm={max_norm}: {out}"


# ---------------------------------------------------------------------------
# moreh_dot  (ttnn.operations.moreh.dot(a, b) -> 1x1x1x1 dot product).
# No per-call scalar: rt-args are buffer addresses (bound as Buffer*, patched) plus num_tiles
# (shape-derived, hashed). Verdict: OK. Pure HIT-path probe: vary input DATA only.
# Invocation + torch golden + tolerances copied from test_moreh_dot.py run_moreh_dot_test
# (bfloat16, reshape to 1-D, torch.matmul, comp_allclose_and_pcc pcc=0.999 rtol=atol=0.1).
# ---------------------------------------------------------------------------
def test_moreh_dot_cache(cache_device):
    torch.manual_seed(3072)
    input_shape = [1, 1, 1, 32]  # single tile, copied from test_moreh_dot.py cases
    ttnn_dtype = ttnn.bfloat16

    def _run():
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_other = torch.rand(input_shape, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, device=cache_device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)
        tt_other = ttnn.from_torch(torch_other, device=cache_device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)

        tt_out = ttnn.operations.moreh.dot(tt_input, tt_other, dtype=ttnn_dtype)
        tt_out = ttnn.to_torch(tt_out).to(torch.bfloat16)

        torch_out = torch.matmul(
            torch.reshape(torch_input, (torch_input.shape[-1],)),
            torch.reshape(torch_other, (torch_other.shape[-1],)),
        )
        passing, out = comp_allclose_and_pcc(torch_out, tt_out[0][0][0][0], pcc=0.999, rtol=0.1, atol=0.1)
        assert passing, f"moreh_dot stale on cache hit: {out}"

    # WARM-UP then assert ZERO growth across fresh allocations.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_dot: cache grew past {base} across fresh allocations (addr not patched)"


# ---------------------------------------------------------------------------
# moreh_dot_backward  (ttnn.operations.moreh.dot_backward(output_grad, a, b, input_grad, other_grad)).
# No per-call scalar: rt-args are buffer addresses (Buffer*, patched) + num_tiles + has_*_grad
# flags (structurally fixed for a given requires_grad case). Verdict: OK. HIT-path probe: vary
# input DATA only. Invocation + torch golden + tolerances copied from test_moreh_dot_backward.py
# run_moreh_dot_backward (1-D matmul backward, comp_allclose_and_pcc pcc=0.999 rtol=atol=0.1).
# ---------------------------------------------------------------------------
def test_moreh_dot_backward_cache(cache_device):
    torch.manual_seed(3072)
    input_shape = [1, 1, 1, 32]
    output_shape = [1, 1, 1, 1]
    npu_dtype = ttnn.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    def _run():
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_other = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_output_grad = torch.rand(output_shape, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, npu_dtype, layout=npu_layout, device=cache_device)
        tt_other = ttnn.from_torch(torch_other, npu_dtype, layout=npu_layout, device=cache_device)
        tt_output_grad = ttnn.from_torch(torch_output_grad, npu_dtype, layout=npu_layout, device=cache_device)

        torch_input_grad_holder = torch.full(input_shape, float("nan"), dtype=torch.bfloat16)
        torch_other_grad_holder = torch.full(input_shape, float("nan"), dtype=torch.bfloat16)
        tt_input_grad = ttnn.from_torch(torch_input_grad_holder, npu_dtype, layout=npu_layout, device=cache_device)
        tt_other_grad = ttnn.from_torch(torch_other_grad_holder, npu_dtype, layout=npu_layout, device=cache_device)

        # torch reference: 1-D matmul backward
        ti = torch.reshape(torch_input, (input_shape[-1],)).requires_grad_(True)
        to = torch.reshape(torch_other, (input_shape[-1],)).requires_grad_(True)
        torch_out = torch.matmul(ti, to)
        torch_out.backward(torch_output_grad[0][0][0][0])

        ttnn.operations.moreh.dot_backward(
            tt_output_grad, tt_input, tt_other, input_grad=tt_input_grad, other_grad=tt_other_grad
        )

        ttcpu_input_grad = ttnn.to_torch(tt_input_grad)
        ttcpu_other_grad = ttnn.to_torch(tt_other_grad)
        pass_i, out_i = comp_allclose_and_pcc(ti.grad, ttcpu_input_grad.reshape(-1), pcc=0.999, rtol=0.1, atol=0.1)
        pass_o, out_o = comp_allclose_and_pcc(to.grad, ttcpu_other_grad.reshape(-1), pcc=0.999, rtol=0.1, atol=0.1)
        assert pass_i and pass_o, f"moreh_dot_backward stale on cache hit: input_grad:{out_i} other_grad:{out_o}"

    # WARM-UP then assert ZERO growth across fresh allocations.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_dot_backward: cache grew past {base} across fresh allocations (addr not patched)"


# ---------------------------------------------------------------------------
# moreh_fold  (ttnn.operations.moreh.fold(input, None, output_size, kernel_size, dilation,
# padding, stride)). All shape params live in operation_attributes_t (hashed); rt-args are buffer
# addresses + shape-derived counts. No per-call scalar. Verdict: OK. HIT-path probe: vary input
# DATA only (params fixed). Invocation + torch golden (torch.nn.Fold) + tolerances copied from
# test_moreh_fold.py run_fold_test (bfloat16, comp_allclose_and_pcc rtol=atol=0.05).
# ---------------------------------------------------------------------------
def test_moreh_fold_cache(cache_device):
    torch.manual_seed(2024)
    # single-tile bf16 case copied verbatim from test_moreh_fold.py
    input_shape = (1, 32, 32)
    output_size = (7, 11)
    kernel_size = (4, 4)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)

    torch_fold = torch.nn.Fold(
        output_size=output_size, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
    )

    def _run():
        # +1 keeps bf16 output away from 0 (rounding precision), per the source test's comment.
        torch_input = torch.randn(input_shape, dtype=torch.bfloat16) + 1
        expected = torch_fold(torch_input)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=cache_device, dtype=ttnn.bfloat16)
        tt_out = ttnn.operations.moreh.fold(tt_input, None, output_size, kernel_size, dilation, padding, stride)
        actual = ttnn.to_torch(tt_out)
        passing, out = comp_allclose_and_pcc(expected, actual, rtol=0.05, atol=0.05)
        assert passing, f"moreh_fold stale on cache hit: {out}"

    # WARM-UP then assert ZERO growth across fresh allocations.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_fold: cache grew past {base} across fresh allocations (over-caching)"
