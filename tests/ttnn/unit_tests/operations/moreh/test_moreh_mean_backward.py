# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device

# input_grad is output_grad broadcast back over the reduced dims, divided by how many values were averaged. Reducing
# W or H sets the wt_need_bcast / ht_need_bcast compile-time args; the other dims broadcast through reader runtime
# args. 63 x 63 leaves the last tile in H and W partly filled.


def run_moreh_mean_backward_test(
    input_shape, dim, device, keepdim=True, fp32_dest_acc_en=False, create_input_grad=False
):
    # Not nightly run_moreh_mean_backward: its output_grad is in [0, 1), so after dividing by the number of averaged
    # values (63 for one W dim, 23814 for all dims) every expected value is within its 0.1 tolerance of zero, and an op
    # that writes zeros, or nothing, still passes. Here output_grad is scaled by that number, so the expected values
    # are in [0, 1); input_grad starts as NaN, so a value the op never writes fails.
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.mean(torch_input, dim=dim, keepdim=keepdim)
    num_averaged = torch_input.numel() // torch_output.numel()
    torch_output_grad = torch.rand(torch_output.shape) * num_averaged
    torch_output.backward(torch_output_grad)

    tt_output_grad = create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16)
    compute_kernel_config = get_compute_kernel_options(fp32_dest_acc_en)
    if create_input_grad:
        tt_input_grad = ttnn.moreh_mean_backward(
            tt_output_grad,
            dim=dim,
            keepdim=keepdim,
            input_grad_shape=tuple(input_shape),
            compute_kernel_config=compute_kernel_config,
        )
    else:
        tt_input_grad = create_ttnn_tilized_tensor(torch.full(input_shape, float("nan")), device, ttnn.bfloat16)
        ttnn.moreh_mean_backward(
            tt_output_grad,
            dim=dim,
            keepdim=keepdim,
            input_grad=tt_input_grad,
            compute_kernel_config=compute_kernel_config,
        )

    passing, out = comp_allclose(torch_input.grad, ttnn.to_torch(tt_input_grad), rtol=0.1, atol=0.1)
    assert passing, out


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "dim",
    [[3], [2], [2, 3], [1], None],
    ids=["w", "h", "hw", "c", "all_dims"],
)
def test_moreh_mean_backward(dim, device):
    torch.manual_seed(0)
    run_moreh_mean_backward_test([2, 3, 63, 63], dim, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, dim, keepdim, fp32_dest_acc_en, create_input_grad",
    [
        # keepdim=False drops N and C here.
        ([2, 3, 63, 63], [0, 1], False, False, False),
        ([3, 4, 5, 17, 22], [2], True, False, False),
        ([2, 3, 63, 63], [3], True, True, False),
        # No input_grad passed: the op allocates it from input_grad_shape.
        ([2, 3, 63, 63], [1, 3], True, False, True),
        # 149 tiles: a prime above any device's core count, so the work split leaves a second core group and the
        # factory builds its second compute kernel.
        ([1, 1, 32, 149 * 32], [2], True, False, False),
    ],
    ids=["keepdim_false", "rank_5", "fp32_dest_acc", "allocated_input_grad", "core_group_2"],
)
def test_moreh_mean_backward_corner_cases(input_shape, dim, keepdim, fp32_dest_acc_en, create_input_grad, device):
    torch.manual_seed(0)
    run_moreh_mean_backward_test(
        input_shape,
        dim,
        device,
        keepdim=keepdim,
        fp32_dest_acc_en=fp32_dest_acc_en,
        create_input_grad=create_input_grad,
    )
