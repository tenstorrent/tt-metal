# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_linear import moreh_linear_backward
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options, to_ttnn

pytestmark = pytest.mark.use_module_device

# input_grad and weight_grad come from moreh_matmul calls (plus moreh_sum when the batch dims differ); bias_grad is the
# op's own device op, with a SingleCore factory for a scalar [1, 1] bias and a MultiCore factory for a [1, N] bias.


def run_moreh_linear_backward_bias_only_test(output_shape, bias_shape, device):
    # Not the nightly helper: it pads tiles with 0, which the bias sum absorbs, so a broken mask would pass; and it
    # can't request bias_grad alone. With only bias_grad requested the op runs just its own bias kernels.
    torch_input = torch.randint(-2, 3, output_shape[:-1] + [64], dtype=torch.float32)
    torch_weight = torch.randint(-2, 3, [output_shape[-1], 64], dtype=torch.float32)
    torch_bias = torch.randint(-10, 10, bias_shape, dtype=torch.float32).requires_grad_()
    torch_output_grad = torch.randint(-2, 3, output_shape, dtype=torch.float32)
    torch.nn.functional.linear(torch_input, torch_weight, torch_bias).backward(torch_output_grad)

    tt_bias_grad = to_ttnn(torch.full(bias_shape, float("nan")), device=device)
    # The check reads this buffer, not the return value: the op must write into it.
    ttnn.moreh_linear_backward(
        create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16),
        to_ttnn(torch_input, device=device),
        to_ttnn(torch_weight, device=device),
        are_required_outputs=[False, False, True],
        bias=to_ttnn(torch_bias.detach(), device=device),
        bias_grad=tt_bias_grad,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    actual_bias_grad = ttnn.to_torch(tt_bias_grad).reshape(bias_shape)
    passing, output_pcc = comp_allclose_and_pcc(torch_bias.grad, actual_bias_grad, pcc=0.999, rtol=0.1, atol=0.1)
    assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shapes, requires_input_grad, requires_weight_grad, requires_bias_grad, fp32_dest_acc_en",
    [
        # (input, weight, bias, output)
        (([32, 64], [96, 64], [1, 96], [32, 96]), True, True, True, False),
        (([32, 64], [96, 64], [1, 1], [32, 96]), True, True, True, False),
        # The batch dims of output_grad and weight_grad differ, so weight_grad goes through moreh_sum.
        (([2, 3, 32, 64], [96, 64], [1, 96], [2, 3, 32, 96]), True, True, True, False),
        (([32, 64], [96, 64], [1, 96], [32, 96]), True, False, False, False),
        (([32, 64], [96, 64], [1, 96], [32, 96]), True, True, True, True),
    ],
    ids=["bias_1d", "bias_scalar", "batched_weight_grad", "input_grad_only", "fp32_dest_acc"],
)
def test_moreh_linear_backward(
    shapes, requires_input_grad, requires_weight_grad, requires_bias_grad, fp32_dest_acc_en, device
):
    torch.manual_seed(0)
    assert moreh_linear_backward(
        shapes,
        requires_input_grad,
        requires_weight_grad,
        requires_bias_grad,
        get_compute_kernel_options(fp32_dest_acc_en),
        device,
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "output_shape, bias_shape",
    [
        # 33 x 50 leaves the last tile partly filled in H and W, so the bias kernels mask it.
        ([33, 50], [1, 1]),
        ([2, 33, 50], [1, 50]),
        # 149 tiles wide: a prime above any device's core count, so the MultiCore split leaves a second core group and
        # the factory builds its second compute kernel.
        ([32, 149 * 32], [1, 149 * 32]),
    ],
    ids=["single_core_unaligned", "multi_core_unaligned", "multi_core_group_2"],
)
def test_moreh_linear_backward_bias_only(output_shape, bias_shape, device):
    torch.manual_seed(0)
    run_moreh_linear_backward_bias_only_test(output_shape, bias_shape, device)
