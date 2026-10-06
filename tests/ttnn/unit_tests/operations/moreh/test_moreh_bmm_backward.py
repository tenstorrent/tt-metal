# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
)

pytestmark = pytest.mark.use_module_device


def run_moreh_bmm_backward_test(input_shape, mat2_shape, device, are_required_outputs=(True, True)):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_mat2 = torch.rand(mat2_shape, requires_grad=True)
    torch_output = torch.bmm(torch_input, torch_mat2)
    torch_output_grad = torch.rand(torch_output.shape)
    torch_output.backward(torch_output_grad)

    input_requires_grad, mat2_requires_grad = are_required_outputs
    tt_output_grad = create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16)
    tt_input = create_ttnn_tilized_tensor(torch_input.detach(), device, ttnn.bfloat16)
    tt_mat2 = create_ttnn_tilized_tensor(torch_mat2.detach(), device, ttnn.bfloat16)
    tt_input_grad = (
        create_ttnn_tilized_tensor(torch.full(input_shape, float("nan")), device, ttnn.bfloat16)
        if input_requires_grad
        else None
    )
    tt_mat2_grad = (
        create_ttnn_tilized_tensor(torch.full(mat2_shape, float("nan")), device, ttnn.bfloat16)
        if mat2_requires_grad
        else None
    )
    tt_grads = ttnn.moreh_bmm_backward(
        tt_output_grad,
        tt_input,
        tt_mat2,
        are_required_outputs=are_required_outputs,
        input_grad=tt_input_grad,
        mat2_grad=tt_mat2_grad,
        compute_kernel_config=get_compute_kernel_options(True),
    )

    for required, torch_grad, tt_grad in zip(are_required_outputs, (torch_input.grad, torch_mat2.grad), tt_grads):
        if not required:
            assert tt_grad is None
            continue
        passing, output_pcc = comp_allclose_and_pcc(torch_grad, ttnn.to_torch(tt_grad), pcc=0.998, rtol=0.1, atol=0.1)
        assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_bmm_backward(device):
    torch.manual_seed(0)
    run_moreh_bmm_backward_test([1, TILE_HEIGHT, TILE_WIDTH], [1, TILE_WIDTH, TILE_HEIGHT], device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, mat2_shape, are_required_outputs",
    [
        ([1, TILE_HEIGHT, TILE_WIDTH], [1, TILE_WIDTH, TILE_HEIGHT], (True, False)),
        ([1, TILE_HEIGHT, TILE_WIDTH], [1, TILE_WIDTH, TILE_HEIGHT], (False, True)),
        ([3, TILE_HEIGHT - 1, TILE_WIDTH - 1], [3, TILE_HEIGHT - 1, TILE_WIDTH - 1], (True, True)),
    ],
    ids=["input_grad_only", "mat2_grad_only", "unaligned"],
)
def test_moreh_bmm_backward_corner_cases(input_shape, mat2_shape, are_required_outputs, device):
    torch.manual_seed(0)
    run_moreh_bmm_backward_test(input_shape, mat2_shape, device, are_required_outputs=are_required_outputs)
