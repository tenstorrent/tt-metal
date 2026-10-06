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

INPUT_SHAPE = [TILE_HEIGHT, TILE_WIDTH]


def run_moreh_abs_pow_test(input_shape, p, device):
    # Magnitudes in [0.5, 2) with random signs: exercises abs() and keeps log(|x|) away from 0.
    magnitude = torch.rand(input_shape) * 1.5 + 0.5
    sign = torch.randint(0, 2, input_shape) * 2 - 1
    torch_input = (magnitude * sign).to(torch.bfloat16)
    torch_output = torch.abs(torch_input.float()) ** p

    tt_input = create_ttnn_tilized_tensor(torch_input, device, ttnn.bfloat16)
    tt_output = ttnn.to_torch(ttnn.moreh_abs_pow(tt_input, p, compute_kernel_config=get_compute_kernel_options(False)))

    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output, pcc=0.99, rtol=0.1, atol=0.1)
    assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_abs_pow(device):
    torch.manual_seed(0)
    # p = 2.5 runs both the integer power (|x|^2) and the fractional power (|x|^0.5 via exp/log).
    run_moreh_abs_pow_test([TILE_HEIGHT, TILE_WIDTH], 2.5, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, p",
    [
        # Decimal part 0: the exp(log(|x|) * 0) factor must come out as exactly 1.
        (INPUT_SHAPE, 3.0),
        # Floored part 0: the integer power runs with exponent 0.
        (INPUT_SHAPE, 0.5),
        # floor(-1.5) = -2 sets p_is_negative: |x|^-1.5 = |x|^0.5 / |x|^2.
        (INPUT_SHAPE, -1.5),
        # Spans two tiles in H and W without filling them: exercises the tile loop and the W padding mask.
        ([2, 3, TILE_HEIGHT * 2 - 1, TILE_WIDTH * 2 - 1], 2.5),
    ],
    ids=["p_integer", "p_below_one", "p_negative", "multi_tile_unaligned"],
)
def test_moreh_abs_pow_corner_cases(input_shape, p, device):
    torch.manual_seed(0)
    run_moreh_abs_pow_test(input_shape, p, device)


@pytest.mark.merge_gate
def test_moreh_abs_pow_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_abs_pow_test(INPUT_SHAPE, 2.5, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros(INPUT_SHAPE), device, ttnn.bfloat16)
    run_moreh_abs_pow_test(INPUT_SHAPE, 2.5, device)
    assert device.num_program_cache_entries() == num_program_cache_entries
