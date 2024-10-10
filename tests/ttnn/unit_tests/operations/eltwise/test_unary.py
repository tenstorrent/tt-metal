# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from models.utility_functions import torch_random, skip_for_grayskull, is_wormhole_b0, is_blackhole


def run_unary_test(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    # torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * (2*3.1415927) - 3.1415927
    torch_input_tensor = torch.rand((h, w), dtype=torch.float32) * (2 * 3.1415927) - 3.1415927
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_pcc = []
    output_mae = []
    for i in range(100):
        output_tensor = ttnn_function(input_tensor)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)

        output_pcc.append(assert_with_pcc(torch_output_tensor, output_tensor, pcc)[1])

        atol_delta = torch.mean(torch.abs(torch_output_tensor - output_tensor)).item()
        output_mae.append(atol_delta)

    print(sum(output_pcc) / 100, sum(output_mae) / 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cos(device, h, w):
    run_unary_test(device, h, w, ttnn.sin, 0.9)


"""
format: (pcc, mae)

APPROX=true
maclaurin:
    fp16: 0.9999972932212358 | 0.0017714691162109375

    fp32: 0.9999999798622439 | 0.00019229984696721657

remez:
    fp16: 0.9996687018469175 | 0.00463958740234375

    fp32: 0.9996886910032607 | 0.0033697818079963327

APPROX=false
maclaurin:
    fp16: 0.9999973136387109 | 0.0017852783203125

    fp32: 0.9999999845037294 | 0.00019388526270631702

remez:
    fp16: 0.9999973069930642 | 0.0017734527587890624

    fp32: 0.9999999678024156 | 0.00020105742689338513

==================================================================

SINE BUG TESTING [confirmed need to remode duplicate for WHB0]

Original:                0.9999940764190391 0.0005072979559190571
Without duplicate  x^11: 0.9999999844634588 0.00019079122837865726

"""
