# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

# These backward passes all evaluate a term of the form k / x^2 by forming x^2 and
# inverting it. That intermediate has a much narrower usable range than the result,
# so the gradient is lost at both ends of the divisor's range:
#
#   |x| < 1.0842e-19   x^2 falls below the smallest normal and flushes to zero,
#                      so the reciprocal returns infinity
#   |x| > 2**63        1/x^2 falls below the smallest normal -- the square is still
#                      perfectly representable here, the reciprocal is what flushes
#   |x| > 2**64        x^2 exceeds the largest normal, so the reciprocal gets inf
#
# All three come back as inf or 0 where the exact gradient is an ordinary number.
# Each case below is (divisor, numerator), the numerator picked so the exact answer
# is a normal float rather than something the device would flush anyway. The first
# entry of each end sits just inside the working band, as a control that the change
# leaves it alone.
#
# The registered goldens are torch's own backward passes, which evaluate the same
# derivative as -grad * ((a / b) / b) -- dividing twice rather than squaring -- so
# they return the finite value and the comparison is meaningful.
#
# float32 and bfloat16 share an exponent field and therefore share all three
# thresholds; both are covered so a later change cannot regress one of them quietly.

SMALL_END = ((1e-18, 1e-30), (1e-19, 1e-30), (1e-20, 1e-30), (1e-22, 1e-30))
LARGE_END = ((1e18, 1e20), (1e19, 1e20), (1e20, 1e20), (1e25, 1e20))
CASES = SMALL_END + LARGE_END
CASE_IDS = [
    "1e-18_control", "1e-19", "1e-20", "1e-22",
    "1e18_control", "1e19", "1e20", "1e25",
]

# reciprocal_bw gets a shorter list on purpose. Torch evaluates that particular
# derivative as -grad * (1/x) * (1/x), so its own reference stops being usable once
# (1/x)^2 leaves the range -- below 5.4e-20 it overflows, above roughly 2.7e22 it
# underflows to zero. What is left is still on both sides of the working band, which
# is what the test needs.
RECIP_CASES = ((1e-18, 1e-20), (1e-19, 1e-20), (8e-20, 1e-20), (1e18, 1e20), (1e19, 1e20), (1e20, 1e20))
RECIP_CASE_IDS = ["1e-18_control", "1e-19", "8e-20", "1e18_control", "1e19", "1e20"]

DTYPES = ((torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16))
DTYPE_IDS = ("float32", "bfloat16")

SHAPE = torch.Size([1, 1, 32, 32])


def _to_device(pt, ttnn_dtype, device):
    return ttnn.from_torch(pt.detach(), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _check(name, tt_out, golden, divisor):
    got = ttnn.to_torch(tt_out).float()
    want = golden.float()
    # Where the reference itself is not representable there is nothing to compare
    # against, and where it is subnormal the device flushes it identically before
    # and after this change; neither is being judged here.
    comparable = torch.isfinite(want) & ((want == 0) | (want.abs() >= 1.17549435e-38))

    lost = int((comparable & ~torch.isfinite(got)).sum())
    assert lost == 0, (
        f"{name} with divisor {divisor:g}: {lost} of {got.numel()} gradients came back "
        f"non-finite where the reference is finite (reference {float(want.flatten()[0]):g}, "
        f"device {float(got.flatten()[0]):g})"
    )
    zeroed = int((comparable & (want != 0) & (got == 0)).sum())
    assert zeroed == 0, (
        f"{name} with divisor {divisor:g}: {zeroed} of {got.numel()} gradients came back "
        f"zero where the reference is {float(want.flatten()[0]):g}"
    )
    torch.testing.assert_close(got[comparable], want[comparable], rtol=2e-2, atol=0.0)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("divisor, numerator", CASES, ids=CASE_IDS)
def test_bw_rdiv_extreme_divisor(divisor, numerator, torch_dtype, ttnn_dtype, device):
    in_data = torch.full(SHAPE, divisor, dtype=torch_dtype).requires_grad_(True)
    grad_data = torch.ones(SHAPE, dtype=torch_dtype)

    tt_out = ttnn.rdiv_bw(
        _to_device(grad_data, ttnn_dtype, device), _to_device(in_data, ttnn_dtype, device), numerator
    )
    golden = ttnn.get_golden_function(ttnn.rdiv_bw)(grad_data, in_data, numerator)
    _check("rdiv_bw", tt_out[0], golden[0], divisor)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("divisor, grad_scale", RECIP_CASES, ids=RECIP_CASE_IDS)
def test_bw_reciprocal_extreme_input(divisor, grad_scale, torch_dtype, ttnn_dtype, device):
    # The gradient is -grad / x^2, so grad is scaled to keep the exact answer a normal
    # float at both ends rather than only in the middle.
    in_data = torch.full(SHAPE, divisor, dtype=torch_dtype).requires_grad_(True)
    grad_data = torch.full(SHAPE, grad_scale, dtype=torch_dtype)

    tt_out = ttnn.reciprocal_bw(
        _to_device(grad_data, ttnn_dtype, device), _to_device(in_data, ttnn_dtype, device)
    )
    golden = ttnn.get_golden_function(ttnn.reciprocal_bw)(grad_data, in_data)
    _check("reciprocal_bw", tt_out[0], golden[0], divisor)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("divisor, numerator", CASES, ids=CASE_IDS)
def test_bw_div_extreme_divisor(divisor, numerator, torch_dtype, ttnn_dtype, device):
    # Only the second output -- the gradient with respect to the divisor -- forms the
    # square. The first is checked too, since it shares the reciprocal.
    in_data = torch.full(SHAPE, numerator, dtype=torch_dtype).requires_grad_(True)
    other_data = torch.full(SHAPE, divisor, dtype=torch_dtype).requires_grad_(True)
    grad_data = torch.ones(SHAPE, dtype=torch_dtype)

    tt_out = ttnn.div_bw(
        _to_device(grad_data, ttnn_dtype, device),
        _to_device(in_data, ttnn_dtype, device),
        _to_device(other_data, ttnn_dtype, device),
    )
    golden = ttnn.get_golden_function(ttnn.div_bw)(grad_data, in_data, other_data)
    _check("div_bw[input_grad]", tt_out[0], golden[0], divisor)
    _check("div_bw[other_grad]", tt_out[1], golden[1], divisor)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("divisor, numerator", CASES, ids=CASE_IDS)
def test_bw_addcdiv_extreme_divisor(divisor, numerator, torch_dtype, ttnn_dtype, device):
    value = 1.0
    in_data = torch.ones(SHAPE, dtype=torch_dtype).requires_grad_(True)
    tensor1_data = torch.full(SHAPE, numerator, dtype=torch_dtype).requires_grad_(True)
    tensor2_data = torch.full(SHAPE, divisor, dtype=torch_dtype).requires_grad_(True)
    grad_data = torch.ones(SHAPE, dtype=torch_dtype)

    tt_out = ttnn.addcdiv_bw(
        _to_device(grad_data, ttnn_dtype, device),
        _to_device(in_data, ttnn_dtype, device),
        _to_device(tensor1_data, ttnn_dtype, device),
        _to_device(tensor2_data, ttnn_dtype, device),
        value,
    )
    golden = ttnn.get_golden_function(ttnn.addcdiv_bw)(grad_data, in_data, tensor1_data, tensor2_data, value)
    # tensor2 is the divisor; its gradient is the one that formed the square.
    _check("addcdiv_bw[tensor2_grad]", tt_out[2], golden[2], divisor)
