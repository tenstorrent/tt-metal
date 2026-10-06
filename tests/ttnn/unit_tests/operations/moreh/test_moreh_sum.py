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

INPUT_SHAPE = [2, 3, TILE_HEIGHT, TILE_WIDTH]
# Spans two tiles in H and W without filling them: exercises the tile loops and the padding masks.
UNALIGNED_SHAPE = [2, 3, TILE_HEIGHT * 2 - 1, TILE_WIDTH * 2 - 1]

# On a rank-4 input: dim 3 -> W factory, dim 2 -> H factory, dim 1 -> NC factory.
FACTORY_DIMS = [3, 2, 1]
FACTORY_IDS = ["w", "h", "nc"]


def run_moreh_sum_test(input_shape, dim, keepdim, dtype, device, provide_output=False, value_range=(-2, 3)):
    torch_dtype = torch.int32 if dtype == ttnn.int32 else torch.bfloat16
    torch_input = torch.randint(*value_range, input_shape, dtype=torch_dtype)
    torch_output = torch.sum(torch_input, dim, keepdim=keepdim, dtype=torch_dtype)

    tt_input = create_ttnn_tilized_tensor(torch_input, device, dtype)
    tt_output = create_ttnn_tilized_tensor(torch.zeros_like(torch_output), device, dtype) if provide_output else None
    tt_output = ttnn.moreh_sum(
        tt_input,
        dim,
        keepdim=keepdim,
        output=tt_output,
        compute_kernel_config=get_compute_kernel_options(dtype == ttnn.int32),
    )
    actual = ttnn.to_torch(tt_output).reshape(torch_output.shape)

    if dtype == ttnn.int32:
        assert torch.equal(torch_output, actual)
    else:
        passing, output_pcc = comp_allclose_and_pcc(torch_output, actual, pcc=0.999, rtol=0.12, atol=0.12)
        assert passing, output_pcc


def run_moreh_sum_backward_test(input_shape, dim, keepdim, device):
    torch_input = torch.randint(-2, 3, input_shape, dtype=torch.bfloat16, requires_grad=True)
    torch_output = torch.sum(torch_input, dim, keepdim=keepdim)
    torch_output_grad = torch.randint(-2, 3, torch_output.shape, dtype=torch.bfloat16)
    torch_output.backward(torch_output_grad)

    tt_input = create_ttnn_tilized_tensor(torch_input.detach(), device, ttnn.bfloat16)
    tt_output_grad = create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16)
    tt_input_grad = ttnn.to_torch(
        ttnn.moreh_sum_backward(
            tt_output_grad,
            input=tt_input,
            dim=dim,
            keepdim=keepdim,
            compute_kernel_config=get_compute_kernel_options(False),
        )
    )

    passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, tt_input_grad, pcc=0.999, rtol=0.1, atol=0.1)
    assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize("dim", FACTORY_DIMS, ids=FACTORY_IDS)
def test_moreh_sum(dim, device):
    torch.manual_seed(0)
    run_moreh_sum_test(INPUT_SHAPE, dim, True, ttnn.bfloat16, device)


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
        ([2, 3, 2, 4, TILE_HEIGHT, TILE_WIDTH], 2, True),
        ([5], None, False),
    ],
    ids=["w_unaligned", "h_unaligned", "nc_unaligned", "hw", "all_dims", "nc_keepdim_false", "rank_6", "rank_1"],
)
def test_moreh_sum_corner_cases(input_shape, dim, keepdim, device):
    torch.manual_seed(0)
    run_moreh_sum_test(input_shape, dim, keepdim, ttnn.bfloat16, device)


@pytest.mark.merge_gate
def test_moreh_sum_provided_output(device):
    torch.manual_seed(0)
    run_moreh_sum_test(INPUT_SHAPE, 3, True, ttnn.bfloat16, device, provide_output=True)


@pytest.mark.merge_gate
def test_moreh_sum_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_sum_test(INPUT_SHAPE, 3, True, ttnn.bfloat16, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros(INPUT_SHAPE), device, ttnn.bfloat16)
    run_moreh_sum_test(INPUT_SHAPE, 3, True, ttnn.bfloat16, device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
@pytest.mark.parametrize("dim", FACTORY_DIMS, ids=FACTORY_IDS)
def test_moreh_sum_int(dim, device):
    torch.manual_seed(0)
    run_moreh_sum_test(INPUT_SHAPE, dim, True, ttnn.int32, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, dim, value_range",
    [
        (UNALIGNED_SHAPE, 3, (-2, 3)),
        # Magnitudes up to 2^24 need exact 32-bit integer accumulation; float math would round them.
        ([3, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1], 3, (-(2**24), 2**24)),
    ],
    ids=["w_unaligned", "large_magnitude"],
)
def test_moreh_sum_int_corner_cases(input_shape, dim, value_range, device):
    torch.manual_seed(0)
    run_moreh_sum_test(input_shape, dim, True, ttnn.int32, device, value_range=value_range)


@pytest.mark.merge_gate
def test_moreh_sum_backward(device):
    torch.manual_seed(0)
    run_moreh_sum_backward_test(INPUT_SHAPE, 1, True, device)


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
def test_moreh_sum_backward_corner_cases(input_shape, dim, keepdim, device):
    torch.manual_seed(0)
    run_moreh_sum_backward_test(input_shape, dim, keepdim, device)
