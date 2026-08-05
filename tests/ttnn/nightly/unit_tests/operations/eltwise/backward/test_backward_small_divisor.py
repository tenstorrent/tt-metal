# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

# These backward passes all evaluate a term of the form k / x^2.
#
# Forming x^2 first throws the answer away whenever the square underflows: below
# |x| = 1.0842e-19 -- the square root of the smallest normal float -- the square
# flushes to zero and its reciprocal is infinity, while the exact gradient is an
# ordinary number well inside the range. 1e-18 sits just above that window and is
# here as a control that the working band is unchanged.
#
# The registered goldens are torch's own backward passes, which evaluate the same
# derivative as -grad * ((a / b) / b) -- dividing twice rather than squaring -- so
# they return the finite value and the comparison is meaningful. The two dtypes
# fail identically today, because float32 and bfloat16 share an exponent field and
# therefore share the underflow threshold; both are covered so a later change
# cannot regress one of them quietly.
DIVISORS = (1e-18, 1e-19, 1e-20, 1e-22)
DIVISOR_IDS = ["1e-18_control", "1e-19", "1e-20", "1e-22"]

# reciprocal_bw gets a shorter list on purpose. Torch evaluates that particular
# derivative as -grad * (1/x) * (1/x), so its own reference overflows once
# |x| < 5.4e-20 and stops being usable as a golden there -- not because the device
# is wrong below that point, but because there is nothing left to compare against.
# The window that is still checkable, (5.4e-20, 1.0842e-19), is exactly where the
# squared form fails, so the cases below do exercise the defect.
RECIP_DIVISORS = (1e-18, 1e-19, 8e-20)
RECIP_DIVISOR_IDS = ["1e-18_control", "1e-19", "8e-20"]

DTYPES = ((torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16))
DTYPE_IDS = ("float32", "bfloat16")

SHAPE = torch.Size([1, 1, 32, 32])


def _to_device(pt, ttnn_dtype, device):
    return ttnn.from_torch(pt.detach(), dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _check(name, tt_out, golden, divisor):
    got = ttnn.to_torch(tt_out).float()
    want = golden.float()
    representable = torch.isfinite(want)

    lost = int((representable & ~torch.isfinite(got)).sum())
    assert lost == 0, (
        f"{name} with divisor {divisor:g}: {lost} of {got.numel()} gradients came back "
        f"non-finite where the reference is finite (reference {float(want.flatten()[0]):g}, "
        f"device {float(got.flatten()[0]):g})"
    )
    torch.testing.assert_close(got[representable], want[representable], rtol=2e-2, atol=0.0)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("divisor", DIVISORS, ids=DIVISOR_IDS)
def test_bw_rdiv_small_divisor(divisor, torch_dtype, ttnn_dtype, device):
    scalar = 1e-30
    in_data = torch.full(SHAPE, divisor, dtype=torch_dtype).requires_grad_(True)
    grad_data = torch.ones(SHAPE, dtype=torch_dtype)

    tt_out = ttnn.rdiv_bw(
        _to_device(grad_data, ttnn_dtype, device), _to_device(in_data, ttnn_dtype, device), scalar
    )
    golden = ttnn.get_golden_function(ttnn.rdiv_bw)(grad_data, in_data, scalar)
    _check("rdiv_bw", tt_out[0], golden[0], divisor)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("divisor", RECIP_DIVISORS, ids=RECIP_DIVISOR_IDS)
def test_bw_reciprocal_small_input(divisor, torch_dtype, ttnn_dtype, device):
    # The gradient is -grad / x^2, so grad is scaled down to keep the exact answer
    # representable for every divisor in the list rather than only the larger ones.
    in_data = torch.full(SHAPE, divisor, dtype=torch_dtype).requires_grad_(True)
    grad_data = torch.full(SHAPE, 1e-20, dtype=torch_dtype)

    tt_out = ttnn.reciprocal_bw(
        _to_device(grad_data, ttnn_dtype, device), _to_device(in_data, ttnn_dtype, device)
    )
    golden = ttnn.get_golden_function(ttnn.reciprocal_bw)(grad_data, in_data)
    _check("reciprocal_bw", tt_out[0], golden[0], divisor)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("divisor", DIVISORS, ids=DIVISOR_IDS)
def test_bw_div_small_divisor(divisor, torch_dtype, ttnn_dtype, device):
    # Only the second output -- the gradient with respect to the divisor -- forms
    # the square. The first is checked too, since it shares the reciprocal.
    in_data = torch.full(SHAPE, 1e-20, dtype=torch_dtype).requires_grad_(True)
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
@pytest.mark.parametrize("divisor", DIVISORS, ids=DIVISOR_IDS)
def test_bw_addcdiv_small_divisor(divisor, torch_dtype, ttnn_dtype, device):
    value = 1.0
    in_data = torch.ones(SHAPE, dtype=torch_dtype).requires_grad_(True)
    tensor1_data = torch.full(SHAPE, 1e-20, dtype=torch_dtype).requires_grad_(True)
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
