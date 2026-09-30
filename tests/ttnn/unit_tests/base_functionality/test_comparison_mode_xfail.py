# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.base_functionality.test_comparison_mode_context import (
    SINGLE_TILE,
    _bit_patterns,
    _registered_golden_output,
    _to_device,
    comparison_mode,
)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="pow_bw returns inf for zero inputs even where the derivative is finite")
def test_pow_bw_with_zero_input_in_comparison_mode(device):
    grad = _to_device(torch.full(SINGLE_TILE, 2.0, dtype=torch.bfloat16), device)
    input_tensor = _to_device(torch.zeros(SINGLE_TILE, dtype=torch.bfloat16), device)

    # The derivative of x**1 is 1 everywhere, so the gradient at x = 0 is grad (2.0), which the torch golden returns.
    # After computing exponent * x**(exponent - 1) * grad, pow_bw in unary_backward.cpp replaces every input <= 0
    # with inf. The fix is to drop that blanket where(lez(input), inf, ...) and only emit a non-finite value where
    # x**(exponent - 1) is itself undefined, i.e. a zero input with an exponent below 1.
    with comparison_mode():
        ttnn.pow_bw(grad, input_tensor, 1.0)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="pow_bw returns inf for negative inputs with an integer exponent")
def test_pow_bw_with_negative_input_and_integer_exponent(device):
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_tensor = _to_device(torch.full(SINGLE_TILE, -2.0, dtype=torch.bfloat16), device)

    # For an integer exponent x**n is defined for negative x and n * x**(n - 1) is finite: -4 at x = -2 for n = 2.
    # pow_bw in unary_backward.cpp masks every input <= 0 to inf, and the golden in unary_backward.py masks negative
    # inputs the same way to agree with it, so comparison passes while both are wrong. The fix is to restrict the
    # device mask to non-integer exponents and drop the matching masked_fill_ from the golden.
    with comparison_mode():
        (input_grad,) = ttnn.pow_bw(grad, input_tensor, 2.0)

    torch.testing.assert_close(ttnn.to_torch(input_grad).float(), torch.full(SINGLE_TILE, -4.0))


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="rsqrt_bw returns +inf at zero regardless of the gradient sign")
def test_rsqrt_bw_at_zero_keeps_gradient_sign(device):
    torch_grad = torch.ones(SINGLE_TILE, dtype=torch.bfloat16)
    torch_grad[..., 1::2] = -1
    grad = _to_device(torch_grad, device)
    input_tensor = _to_device(torch.zeros(SINGLE_TILE, dtype=torch.bfloat16), device)

    # The derivative of x**-0.5 is -0.5 * x**-1.5, which tends to -inf as x -> 0+, so the gradient at zero is
    # -inf * grad: -inf for positive and +inf for negative gradients. rsqrt_bw in unary_backward.cpp overwrites
    # every zero input with +inf. The fix is to write -sign(grad) * inf there instead.
    with comparison_mode():
        (input_grad,) = ttnn.rsqrt_bw(grad, input_tensor)

    expected = -torch.sign(torch_grad.float()) * float("inf")
    assert torch.equal(ttnn.to_torch(input_grad).float(), expected)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="hardswish_bw approximates 1/3 with 0.3333")
def test_hardswish_bw_uses_exact_one_third(device):
    torch_input = torch.linspace(-2.9, 2.9, 1024).reshape(SINGLE_TILE)
    grad = _to_device(torch.ones(SINGLE_TILE), device, dtype=ttnn.float32)
    input_tensor = _to_device(torch_input, device, dtype=ttnn.float32)

    # Inside (-3, 3) the derivative of x * relu6(x + 3) / 6 is x / 3 + 0.5. hardswish_bw in unary_backward.cpp
    # multiplies by the literal 0.3333, so float32 results are off by up to ~1e-4 (x / 30000), which PCC cannot see
    # but a float32 tolerance can. The fix is to use 1.0f / 3.0f.
    with comparison_mode():
        (input_grad,) = ttnn.hardswish_bw(grad, input_tensor)

    torch.testing.assert_close(ttnn.to_torch(input_grad), torch_input / 3 + 0.5, rtol=0, atol=2e-5)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="repeat_bw returns no gradient when every repeat factor is 1")
def test_repeat_bw_with_all_one_repeats_in_comparison_mode(device):
    torch_grad = torch.rand(SINGLE_TILE, dtype=torch.bfloat16)
    grad = _to_device(torch_grad, device)
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # Repeating by [1, 1, 1, 1] is the identity, so its gradient is the incoming gradient unchanged. repeat_bw in
    # unary_backward.cpp handles zero factors and a repeat along dim 0 or dim 1, then falls through and returns an
    # empty list for the identity case. The fix is to return grad when every factor is 1.
    with comparison_mode():
        gradients = ttnn.repeat_bw(grad, input_tensor, [1, 1, 1, 1])

    assert len(gradients) == 1
    assert torch.equal(ttnn.to_torch(gradients[0]), torch_grad)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="prod_bw divides the product by each input, which yields NaN at zeros")
def test_prod_bw_with_zero_input(device):
    torch_input = torch.ones(SINGLE_TILE, dtype=torch.bfloat16)
    torch_input[0, 0, 0, 0] = 0
    torch_grad = torch.zeros(SINGLE_TILE, dtype=torch.bfloat16)
    torch_grad[0, 0, 0, 0] = 1
    input_tensor = _to_device(torch_input, device)
    grad = _to_device(torch_grad, device)

    # The gradient of prod(x) with respect to x_i is the product of the other elements: 1 at the single zero and 0
    # elsewhere. For dim=None, prod_bw in unary_backward.cpp computes prod(x) * reciprocal(x), i.e. 0 * inf = NaN at
    # the zero. The fix is to count zeros: with one zero use the product of the non-zero elements at its position,
    # with more return zeros. The all-dims gradient is a full tile holding the scalar first, so this runs outside
    # comparison mode.
    (input_grad,) = ttnn.prod_bw(grad, input_tensor)

    expected = torch.zeros(SINGLE_TILE)
    expected[0, 0, 0, 0] = 1
    torch.testing.assert_close(ttnn.to_torch(input_grad).float(), expected)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="ema scales the first sample by 1 - alpha instead of copying it")
def test_ema_first_sample_in_comparison_mode(device):
    input_tensor = _to_device(torch.full(SINGLE_TILE, 4.0, dtype=torch.bfloat16), device)

    # out[0] = in[0] and out[t] = alpha * out[t - 1] + (1 - alpha) * in[t], so a constant input gives a constant
    # output, which the golden computes. The ema compute kernel (ema_compute.cpp) starts from a zeroed running state
    # and applies the recurrence to the first sample too, giving (1 - alpha) * in[0] (3.0 here) and a ramp toward 4.
    # The fix is to seed the running state with the first sample instead of zero.
    with comparison_mode():
        ttnn.ema(input_tensor, 0.25)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="logaddexp exponentiates before adding and overflows for large finite inputs")
@pytest.mark.parametrize(
    "operation",
    [ttnn.logaddexp, ttnn.logaddexp_, ttnn.logaddexp2, ttnn.logaddexp2_],
    ids=["logaddexp", "logaddexp_", "logaddexp2", "logaddexp2_"],
)
def test_logaddexp_family_with_large_finite_inputs_in_comparison_mode(device, operation):
    input_a = _to_device(torch.full(SINGLE_TILE, 1000.0), device, dtype=ttnn.float32)
    input_b = _to_device(torch.full(SINGLE_TILE, 1000.0), device, dtype=ttnn.float32)

    # log(exp(a) + exp(b)) is finite for finite inputs (1000 + log(2) here, 1001 for the base-2 form), and the torch
    # golden computes it stably. The LOGADDEXP and LOGADDEXP2 kernels (binary_op_utils.cpp) exponentiate both operands
    # before the add, so exp(1000) overflows to inf in float32. The fix is the stable form
    # max(a, b) + log1p(exp(-|a - b|)), with the base-2 equivalent for logaddexp2.
    with comparison_mode():
        operation(input_a, input_b)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="moreh_adam passes the integer step to power_tile, which expects float bits")
def test_moreh_adam_param_out_matches_reference(device):
    torch_param = torch.randn(SINGLE_TILE, dtype=torch.bfloat16)
    torch_grad = torch.randn(SINGLE_TILE, dtype=torch.bfloat16)
    zeros = torch.zeros(SINGLE_TILE, dtype=torch.bfloat16)
    param, grad, exp_avg, exp_avg_sq = (_to_device(t, device) for t in (torch_param, torch_grad, zeros, zeros))
    param_out, exp_avg_out, exp_avg_sq_out = (_to_device(t, device) for t in (torch_param, zeros, zeros))
    adam_kwargs = {"lr": 0.1, "step": 1}

    # With step = 1 the bias corrections are 1 - beta1 and 1 - beta2, so one Adam step moves every parameter by about
    # lr * sign(grad), which the golden computes. moreh_adam.cpp hands the integer step to power_tile, whose exponent
    # operand is read as float bits, so beta ** step uses a denormal exponent and the bias correction collapses to ~0.
    # The golden skips param_out for that reason; the fix is to pass the float bits of step in moreh_adam.cpp and then
    # drop the skip from the golden in moreh.py.
    with comparison_mode():
        outputs = ttnn.moreh_adam(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            param_out=param_out,
            exp_avg_out=exp_avg_out,
            exp_avg_sq_out=exp_avg_sq_out,
            **adam_kwargs,
        )

    golden_param = _registered_golden_output(ttnn.moreh_adam, param, grad, exp_avg, exp_avg_sq, **adam_kwargs)[0]
    torch.testing.assert_close(ttnn.to_torch(outputs[0]).float(), golden_param.float(), rtol=0, atol=0.03)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="reglu's split helper assumes a rank-4 input")
def test_reglu_on_rank_3_input_in_comparison_mode(device):
    input_tensor = _to_device(torch.randn((1, 32, 64), dtype=torch.bfloat16), device)

    # The default dim=-1 is documented as the last dimension, so a rank-3 input must split its last axis like the
    # torch golden does. reglu rewrites -1 to 3 and split_tensor_for_glu indexes padded_shape()[3] and builds
    # 4-element slice bounds, so rank-3 inputs fail. Normalizing dim against the input rank and building
    # rank-sized slice bounds (or unsqueezing to rank 4 and squeezing back) would fix it.
    with comparison_mode():
        ttnn.reglu(input_tensor)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="uint32 to float32 bitcast truncates the float32 mantissa on device")
def test_bitcast_uint32_to_float32_keeps_all_bits(device):
    torch_input = _bit_patterns(torch.rand(SINGLE_TILE, dtype=torch.float32) * 1000 + 1, torch.int32)
    input_tensor = _to_device(torch_input, device, dtype=ttnn.uint32)

    # A bitcast reinterprets every bit, so random float32 mantissas must survive unchanged; the golden views the
    # input bits exactly and PCC hides the loss. The device copies through the identity copy_tile/pack_tile kernel
    # without fp32 destination accumulation, so values come back rounded to a 10-bit mantissa (429.2942 -> 429.25).
    # Enabling fp32_dest_acc_en with unpack-to-dest for 32-bit bitcasts would keep the bit pattern intact.
    with comparison_mode():
        output = ttnn.bitcast(input_tensor, ttnn.float32)

    assert torch.equal(ttnn.to_torch(output), ttnn.to_torch(input_tensor).view(torch.float32))


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.xfail(strict=True, reason="uint16 sort returns a padding index for a value equal to the pad sentinel")
@pytest.mark.parametrize("descending, sentinel", [(False, 2**16 - 1), (True, 0)], ids=["ascending", "descending"])
def test_sort_uint16_with_padding_sentinel_value(device, descending, sentinel):
    width = 40
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(ttnn.uint16)
    torch_input = torch.stack([torch.randperm(width) + 1 for _ in range(32)]).reshape(1, 1, 32, width)
    torch_input[..., 7] = sentinel
    input_tensor = _to_device(torch_input.to(torch.int64).to(torch_dtype), device, dtype=ttnn.uint16)

    # The row is padded to 64 with the extreme value for the sort direction, and one real value equals that pad.
    # Every value is distinct, so the indices are unique and must equal torch's; instead the final position, where
    # the real sentinel-valued element belongs, receives an index from the padded tail. The kernel should break the
    # tie in favor of logical indices (or mask padded lanes by index) rather than relying on a value sentinel.
    _, indices = ttnn.sort(input_tensor, dim=-1, descending=descending)

    expected_indices = torch.sort(torch_input.to(torch.int64), dim=-1, descending=descending).indices
    assert torch.equal(ttnn.to_torch(indices).to(torch.int64), expected_indices)
