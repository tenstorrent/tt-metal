# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device

# Nightly has no moreh_abs_pow test, so this file has its own helper. The host splits p into floor(p), applied as an
# integer power, and the fraction, applied as exp(log(|x|) * fraction); a negative floor(p) takes the reciprocal.


def run_moreh_abs_pow_test(input_shape, p, device, fp32_dest_acc_en=False, provide_output=False):
    # Magnitudes in [0.5, 2) with random signs: exercises abs() and keeps log(|x|) away from 0.
    magnitude = torch.rand(input_shape) * 1.5 + 0.5
    sign = torch.randint(0, 2, input_shape) * 2 - 1
    torch_input = (magnitude * sign).to(torch.bfloat16)
    torch_output = torch.abs(torch_input.float()) ** p

    # Zeros, so an output the op never writes fails.
    tt_output = create_ttnn_tilized_tensor(torch.zeros(input_shape), device, ttnn.bfloat16) if provide_output else None
    tt_output = ttnn.moreh_abs_pow(
        create_ttnn_tilized_tensor(torch_input, device, ttnn.bfloat16),
        p,
        output=tt_output,
        compute_kernel_config=get_compute_kernel_options(fp32_dest_acc_en),
    )

    passing, output_pcc = comp_allclose_and_pcc(torch_output, ttnn.to_torch(tt_output), pcc=0.99, rtol=0.1, atol=0.1)
    assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "p",
    [
        # Integer and fractional power both active.
        2.5,
        # Fraction 0: exp(log(|x|) * 0) must come out as exactly 1.
        3.0,
        # floor(p) = 0: the integer power runs with exponent 0.
        0.5,
        # floor(-1.5) = -2 sets p_is_negative: |x|^-1.5 = |x|^0.5 / |x|^2.
        -1.5,
    ],
    ids=["p_2_5", "p_integer", "p_below_one", "p_negative"],
)
def test_moreh_abs_pow(p, device):
    torch.manual_seed(0)
    run_moreh_abs_pow_test([32, 32], p, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, fp32_dest_acc_en, provide_output",
    [
        # Spans two tiles in H and W without filling them: the tile loop and the W padding mask.
        ([2, 3, 63, 63], False, False),
        ([32, 32], True, False),
        # moreh_norm passes `output` when it builds a norm from abs_pow.
        ([32, 32], False, True),
    ],
    ids=["multi_tile_unaligned", "fp32_dest_acc", "provided_output"],
)
def test_moreh_abs_pow_corner_cases(input_shape, fp32_dest_acc_en, provide_output, device):
    torch.manual_seed(0)
    run_moreh_abs_pow_test(input_shape, 2.5, device, fp32_dest_acc_en=fp32_dest_acc_en, provide_output=provide_output)


@pytest.mark.merge_gate
def test_moreh_abs_pow_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_abs_pow_test([32, 32], 2.5, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Without this, the equality below would also pass for an op that never caches a program.
    assert num_program_cache_entries > 0
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses. Row-major,
    # so creating it runs no device program of its own.
    tt_placeholder = ttnn.from_torch(torch.zeros([32, 32]), dtype=ttnn.bfloat16, device=device)
    run_moreh_abs_pow_test([32, 32], 2.5, device)
    assert device.num_program_cache_entries() == num_program_cache_entries
