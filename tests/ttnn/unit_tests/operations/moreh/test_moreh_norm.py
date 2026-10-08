# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_norm import (
    make_no_tie_inf_input,
    make_torch_tensors,
    run_moreh_norm_backward,
    run_moreh_norm_output_mode,
    torch_norm,
    ttnn_norm,
)
from tests.ttnn.unit_tests.operations.test_utils import (
    compute_output_shape,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
)

pytestmark = pytest.mark.use_module_device

# Only p in {0, inf, -inf} runs the moreh_norm device op; other p go through moreh_abs_pow + moreh_sum.


@pytest.mark.merge_gate
@pytest.mark.parametrize("p", [0.0, float("inf"), float("-inf")], ids=["p0", "inf", "minus_inf"])
@pytest.mark.parametrize("dim", [3, 2, 1], ids=["w", "h", "nc"])
def test_moreh_norm(dim, p, device):
    torch.manual_seed(0)
    # dim picks the factory (W, H or NC) and p the compute variant. 63 x 63 spans two tiles in H and W without
    # filling them, so the W and H readers apply their padding masks.
    run_moreh_norm_output_mode(
        [2, 3, 63, 63],
        p,
        dim,
        0.06,
        0.06,
        device,
        keepdim=True,
        compute_kernel_options=False,
        use_provided_output=False,
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "dim, keepdim, fp32_dest_acc_en",
    [
        # inf over several dims chains one device call per dim.
        (None, True, False),
        ([0, 1], False, False),
        (3, True, True),
    ],
    ids=["all_dims", "nc_keepdim_false", "fp32_dest_acc"],
)
def test_moreh_norm_corner_cases(dim, keepdim, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_norm_output_mode(
        [2, 3, 32, 32],
        float("inf"),
        dim,
        0.06,
        0.06,
        device,
        keepdim=keepdim,
        compute_kernel_options=fp32_dest_acc_en,
        use_provided_output=False,
    )


@pytest.mark.merge_gate
def test_moreh_norm_provided_output(device):
    torch.manual_seed(0)
    # Not run_moreh_norm_output_mode: it pre-fills the output with torch.empty, whose leftover memory can already hold
    # the expected values (the test before this one computes the same values). NaN makes an unwritten value fail.
    torch_input, torch_output_grad = make_torch_tensors([2, 3, 32, 32], 3, keepdim=True)
    expected, _ = torch_norm(torch_input, torch_output_grad, p=float("inf"), dim=3, keepdim=True)
    tt_output = create_ttnn_tilized_tensor(torch.full(expected.shape, float("nan")), device, ttnn.bfloat16)
    ttnn.moreh_norm(
        create_ttnn_tilized_tensor(torch_input.detach(), device, ttnn.bfloat16),
        float("inf"),
        dim=3,
        keepdim=True,
        output=tt_output,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    actual = ttnn.to_torch(tt_output).reshape(expected.shape)
    passing, out = comp_allclose(expected.detach(), actual, rtol=0.06, atol=0.06)
    assert passing, out


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "p, dim, keepdim, fp32_dest_acc_en",
    [
        # The reduced dim decides which tile broadcasts output_grad needs (compile-time args): W, H, both, none.
        # A fractional p takes the decimal-exponent path; a negative p sets the power helpers' sign flags.
        (2.5, 3, True, False),
        (-2.5, 2, True, False),
        (2.0, [2, 3], True, False),
        (2.0, 1, True, False),
        (2.0, [0, 1], False, False),
        (2.5, 3, True, True),
    ],
    ids=["w_p2_5", "h_p_minus_2_5", "hw", "nc", "nc_keepdim_false", "fp32_dest_acc"],
)
def test_moreh_norm_backward(p, dim, keepdim, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_norm_backward(
        [2, 3, 63, 63], p, dim, 0.06, 0.06, device, keepdim=keepdim, compute_kernel_options=fp32_dest_acc_en
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize("p, dim", [(float("inf"), 3), (float("-inf"), 2)], ids=["inf_w", "minus_inf_h"])
def test_moreh_norm_backward_p_inf(p, dim, device):
    torch.manual_seed(0)
    # Not run_moreh_norm_backward: random values can tie on max/min |x| once rounded to bfloat16, and torch and
    # the kernel then split the gradient differently. The nightly no-tie input keeps the extreme unique.
    input_shape = [2, 3, 63, 63]
    torch_input = make_no_tie_inf_input(input_shape, dim)
    output_grad_shape, _ = compute_output_shape(input_shape, dim, keepdim=True)
    torch_output_grad = torch.empty(output_grad_shape).uniform_(-1, 1)
    _, expected_input_grad = torch_norm(torch_input, torch_output_grad, p=p, dim=dim, keepdim=True, do_backward=True)
    _, actual_input_grad = ttnn_norm(
        torch_input,
        torch_output_grad,
        p=p,
        dim=dim,
        keepdim=True,
        compute_kernel_options=False,
        do_backward=True,
        device=device,
    )

    passing, out = comp_allclose(expected_input_grad, actual_input_grad, rtol=0.06, atol=0.06)
    assert passing, out


@pytest.mark.merge_gate
def test_moreh_norm_backward_allocated_input_grad(device):
    torch.manual_seed(0)
    # Not the nightly helpers: they always pass input_grad, so the op never allocates its own.
    torch_input, torch_output_grad = make_torch_tensors([2, 3, 32, 32], 3, keepdim=True)
    torch_output, expected_input_grad = torch_norm(
        torch_input, torch_output_grad, p=2.0, dim=3, keepdim=True, do_backward=True
    )
    tt_input_grad = ttnn.moreh_norm_backward(
        create_ttnn_tilized_tensor(torch_input.detach(), device, ttnn.bfloat16),
        create_ttnn_tilized_tensor(torch_output.detach(), device, ttnn.bfloat16),
        create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16),
        2.0,
        dim=3,
        keepdim=True,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    passing, out = comp_allclose(expected_input_grad, ttnn.to_torch(tt_input_grad), rtol=0.06, atol=0.06)
    assert passing, out
