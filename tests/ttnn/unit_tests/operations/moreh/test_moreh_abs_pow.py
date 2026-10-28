# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device

# Nightly has no moreh_abs_pow test, so this file has its own helper.


def run_moreh_abs_pow_test(input_shape, p, device, fp32_dest_acc_en=False, provide_output=False):
    # |x| in [0.5, 2) with random signs, so log(|x|) stays finite.
    magnitude = torch.rand(input_shape) * 1.5 + 0.5
    sign = torch.randint(0, 2, input_shape) * 2 - 1
    torch_input = (magnitude * sign).to(torch.bfloat16)
    torch_output = torch.abs(torch_input.float()) ** p

    # NaN, so an output the op never writes fails.
    tt_output = (
        create_ttnn_tilized_tensor(torch.full(input_shape, float("nan")), device, ttnn.bfloat16)
        if provide_output
        else None
    )
    result = ttnn.moreh_abs_pow(
        create_ttnn_tilized_tensor(torch_input, device, ttnn.bfloat16),
        p,
        output=tt_output,
        compute_kernel_config=get_compute_kernel_options(fp32_dest_acc_en),
    )
    # Check the provided buffer itself, not the return value.
    actual = ttnn.to_torch(tt_output if provide_output else result)

    passing, output_pcc = comp_allclose_and_pcc(torch_output, actual, pcc=0.99, rtol=0.1, atol=0.1)
    assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "p",
    [
        2.5,
        3.0,
        0.5,
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
        ([2, 3, 63, 63], False, False),
        ([32, 32], True, False),
        ([32, 32], False, True),
    ],
    ids=["multi_tile_unaligned", "fp32_dest_acc", "provided_output"],
)
def test_moreh_abs_pow_corner_cases(input_shape, fp32_dest_acc_en, provide_output, device):
    torch.manual_seed(0)
    run_moreh_abs_pow_test(input_shape, 2.5, device, fp32_dest_acc_en=fp32_dest_acc_en, provide_output=provide_output)
