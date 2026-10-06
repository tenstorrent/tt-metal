# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
)

pytestmark = pytest.mark.use_module_device

INPUT_SHAPE = [2, 3, TILE_HEIGHT, TILE_WIDTH]
# Spans two tiles in H and W without filling them: exercises the tile loops and the padding masks.
UNALIGNED_SHAPE = [2, 3, TILE_HEIGHT * 2 - 1, TILE_WIDTH * 2 - 1]

# On a rank-4 input: dim 3 -> W factory, dim 2 -> H factory, dim 1 -> NC factory.
FACTORY_DIMS = [3, 2, 1]
FACTORY_IDS = ["w", "h", "nc"]


def run_moreh_mean_test(input_shape, dim, device, keepdim=True, provide_output=False):
    torch_input = torch.rand(input_shape)
    torch_output = torch.mean(torch_input, dim=dim, keepdim=keepdim)

    tt_input = create_ttnn_tilized_tensor(torch_input, device, ttnn.bfloat16)
    tt_output = (
        create_ttnn_tilized_tensor(torch.zeros_like(torch_output), device, ttnn.bfloat16) if provide_output else None
    )
    tt_output = ttnn.moreh_mean(
        tt_input,
        dim=dim,
        keepdim=keepdim,
        output=tt_output,
        compute_kernel_config=get_compute_kernel_options(False),
    )
    actual = ttnn.to_torch(tt_output).reshape(torch_output.shape)

    passing, out = comp_allclose(torch_output, actual, rtol=0.1, atol=0.1)
    assert passing, out


def run_moreh_mean_backward_test(input_shape, dim, device, keepdim=True, provide_input_grad=False):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.mean(torch_input, dim=dim, keepdim=keepdim)
    torch_output_grad = torch.rand(torch_output.shape)
    torch_output.backward(torch_output_grad)

    tt_output_grad = create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16)
    tt_input_grad = (
        create_ttnn_tilized_tensor(torch.zeros(input_shape), device, ttnn.bfloat16) if provide_input_grad else None
    )
    tt_input_grad = ttnn.to_torch(
        ttnn.moreh_mean_backward(
            tt_output_grad,
            dim=dim,
            keepdim=keepdim,
            input_grad_shape=None if provide_input_grad else input_shape,
            input_grad=tt_input_grad,
            compute_kernel_config=get_compute_kernel_options(False),
        )
    )

    passing, out = comp_allclose(torch_input.grad, tt_input_grad, rtol=0.1, atol=0.1)
    assert passing, out


@pytest.mark.merge_gate
@pytest.mark.parametrize("dim", FACTORY_DIMS, ids=FACTORY_IDS)
def test_moreh_mean(dim, device):
    torch.manual_seed(0)
    run_moreh_mean_test(INPUT_SHAPE, dim, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        (UNALIGNED_SHAPE, 3, True),
        (UNALIGNED_SHAPE, 2, True),
        (UNALIGNED_SHAPE, 1, True),
        (UNALIGNED_SHAPE, [2, 3], True),
        (INPUT_SHAPE, None, True),
        (INPUT_SHAPE, [0, 1], False),
        ([3, 4, 5, TILE_HEIGHT - 15, TILE_WIDTH - 10], [0, 2, 4], True),
    ],
    ids=["w_unaligned", "h_unaligned", "nc_unaligned", "hw", "all_dims", "nc_keepdim_false", "rank_5"],
)
def test_moreh_mean_corner_cases(input_shape, dim, keepdim, device):
    torch.manual_seed(0)
    run_moreh_mean_test(input_shape, dim, device, keepdim=keepdim)


@pytest.mark.merge_gate
def test_moreh_mean_provided_output(device):
    torch.manual_seed(0)
    run_moreh_mean_test(INPUT_SHAPE, 3, device, provide_output=True)


@pytest.mark.merge_gate
def test_moreh_mean_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_mean_test(INPUT_SHAPE, 3, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros(INPUT_SHAPE), device, ttnn.bfloat16)
    run_moreh_mean_test(INPUT_SHAPE, 3, device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
def test_moreh_mean_backward(device):
    torch.manual_seed(0)
    run_moreh_mean_backward_test(INPUT_SHAPE, 1, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        (UNALIGNED_SHAPE, [2, 3], True),
        (INPUT_SHAPE, None, True),
        (INPUT_SHAPE, [0, 1], False),
    ],
    ids=["hw_unaligned", "all_dims", "nc_keepdim_false"],
)
def test_moreh_mean_backward_corner_cases(input_shape, dim, keepdim, device):
    torch.manual_seed(0)
    run_moreh_mean_backward_test(input_shape, dim, device, keepdim=keepdim)


@pytest.mark.merge_gate
def test_moreh_mean_backward_provided_input_grad(device):
    torch.manual_seed(0)
    run_moreh_mean_backward_test(INPUT_SHAPE, [0, 1], device, keepdim=False, provide_input_grad=True)
