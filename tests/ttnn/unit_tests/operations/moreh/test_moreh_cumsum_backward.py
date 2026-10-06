# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import TILE_HEIGHT, TILE_WIDTH, create_ttnn_tilized_tensor

pytestmark = pytest.mark.use_module_device

INPUT_SHAPE = [2, 3, TILE_HEIGHT, TILE_WIDTH]


def run_moreh_cumsum_backward_test(input_shape, dim, device, provide_input_grad=False):
    torch_input = torch.randint(-2, 3, input_shape, dtype=torch.bfloat16, requires_grad=True)
    torch_output = torch.cumsum(torch_input, dim)
    torch_output_grad = torch.randint(-2, 3, torch_output.shape, dtype=torch.bfloat16)
    torch_output.backward(torch_output_grad)

    tt_output_grad = create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16)
    tt_input_grad = (
        create_ttnn_tilized_tensor(torch.full(input_shape, float("nan")), device, ttnn.bfloat16)
        if provide_input_grad
        else None
    )
    tt_input_grad = ttnn.to_torch(ttnn.moreh_cumsum_backward(tt_output_grad, dim, input_grad=tt_input_grad))

    passing, output_pcc = comp_allclose_and_pcc(torch_input.grad, tt_input_grad, pcc=0.999, rtol=0.1, atol=0.1)
    assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_cumsum_backward(device):
    torch.manual_seed(0)
    run_moreh_cumsum_backward_test([2, 3, TILE_HEIGHT, TILE_WIDTH], 1, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, dim",
    [
        # On a rank-4 input only dim 0 is accumulated directly; other dims are permuted to dim 0 first.
        # Keep it tile-aligned: dim 0 on tile-misaligned shapes is disabled in nightly (#44858).
        (INPUT_SHAPE, 0),
        ([2, 3, TILE_HEIGHT - 1, TILE_WIDTH - 1], 1),
        # Rank < 4 is reshaped to rank 4 before the permute.
        ([10, TILE_HEIGHT, TILE_WIDTH + 1], 1),
    ],
    ids=["dim_0", "unaligned", "rank_3"],
)
def test_moreh_cumsum_backward_corner_cases(input_shape, dim, device):
    torch.manual_seed(0)
    run_moreh_cumsum_backward_test(input_shape, dim, device)


@pytest.mark.merge_gate
def test_moreh_cumsum_backward_provided_input_grad(device):
    torch.manual_seed(0)
    run_moreh_cumsum_backward_test(INPUT_SHAPE, 1, device, provide_input_grad=True)


@pytest.mark.merge_gate
def test_moreh_cumsum_backward_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_cumsum_backward_test(INPUT_SHAPE, 1, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros(INPUT_SHAPE), device, ttnn.bfloat16)
    run_moreh_cumsum_backward_test(INPUT_SHAPE, 1, device)
    assert device.num_program_cache_entries() == num_program_cache_entries
