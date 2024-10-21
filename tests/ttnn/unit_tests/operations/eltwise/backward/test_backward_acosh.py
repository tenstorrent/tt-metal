# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_val,
    compare_pcc,
    compare_results,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_acosh(input_shapes, device):
    in_data, input_tensor = data_gen_with_val(input_shapes, device, val=1, required_grad=True)
    # in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)

    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device, True)

    print("input_tensor : ", input_tensor)
    # print("grad_tensor : ",grad_tensor)

    tt_output_tensor_on_device = ttnn.acosh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.acosh_bw)
    golden_tensor = golden_function(grad_data, in_data, device=device)

    # torch.set_printoptions(linewidth=200, threshold = 10000 , precision=5, sci_mode = False, edgeitems=17)
    # print("golden_tensor",golden_tensor)

    # ttnn.set_printoptions(profile="full")
    print("tt_output_tensor_on_device", tt_output_tensor_on_device)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


def data_gen_pt_tt(input_shapes, device, required_grad=False, val=1):
    pt_tensor = (torch.ones(input_shapes, requires_grad=required_grad) * val).bfloat16()
    tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return pt_tensor, tt_tensor


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_bw_acosh_nan_inf(input_shapes, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True, val=0.5)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device, False, val=1)

    print("input_tensor", input_tensor)
    print("grad_tensor", grad_tensor)

    tt_output_tensor_on_device = ttnn.acosh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.acosh_bw)
    golden_tensor = golden_function(grad_data, in_data, device=device)

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)

    print("tt_output_tensor_on_device", tt_output_tensor_on_device)
    print("golden_tensor", golden_tensor)
    assert comp_pass
