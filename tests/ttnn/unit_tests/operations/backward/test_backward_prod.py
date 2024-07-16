# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import (
    data_gen_pt_tt,
    data_gen_pt_tt_prod,
    compare_results,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),  # 0
        (torch.Size([1, 1, 320, 384])),  # 1
        (torch.Size([4, 2, 32, 32])),  # 2
        (torch.Size([1, 3, 320, 384])),  # 3
        (torch.Size([4, 3, 32, 32])),  # 4
        (torch.Size([4, 3, 64, 64])),  # 5
        (torch.Size([4, 3, 320, 320])),  # 6
        (torch.Size([4, 3, 32, 32])),  # 7
        (torch.Size([1, 3, 320, 320])),  # 8
        (torch.Size([1, 4, 320, 384])),  # 9
        (torch.Size([4, 4, 32, 32])),  # 10
        (torch.Size([5, 4, 32, 32])),  # 11
        (torch.Size([6, 4, 32, 32])),  # 12
        (torch.Size([4, 5, 32, 32])),  # 13
        (torch.Size([4, 6, 32, 32])),  # 14
        (torch.Size([4, 10, 32, 32])),  # 15
        (torch.Size([4, 20, 32, 32])),  # 16
        (torch.Size([4, 30, 32, 32])),  # 17
        (torch.Size([4, 31, 32, 32])),  # 18
        (torch.Size([4, 32, 32, 32])),  # 19
        (torch.Size([4, 33, 32, 32])),  # 20
        (torch.Size([4, 63, 32, 32])),  # 21
        (torch.Size([4, 64, 32, 32])),  # 22
        (torch.Size([32, 64, 64, 64])),  # 23
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-4, -3, -2, -1, 0, 1, 2, 3],
)
@pytest.mark.parametrize("all_dimensions", [True, False])
def test_bw_prod(input_shapes, all_dimensions, dim, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt_prod(input_shapes, device, all_dimensions, dim)
    if all_dimensions == False:
        pyt_y = torch.prod(in_data, dim=dim, keepdim=True)
    else:
        pyt_y = torch.prod(in_data).view(1, 1, 1, 1)
    tt_output_tensor_on_device = ttnn.prod_bw(grad_tensor, input_tensor, all_dimensions=all_dimensions, dim=dim)
    in_data.retain_grad()
    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)

    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([32, 64, 64, 64])),
    ),
)
def test_bw_prod_default_both(input_shapes, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt_prod(input_shapes, device)
    pyt_y = torch.prod(in_data).view(1, 1, 1, 1)
    tt_output_tensor_on_device = ttnn.prod_bw(grad_tensor, input_tensor)
    in_data.retain_grad()
    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)

    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([32, 64, 64, 64])),
    ),
)
@pytest.mark.parametrize("all_dimensions", [True, False])
def test_bw_prod_default_dim(input_shapes, all_dimensions, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt_prod(input_shapes, device, all_dimensions)
    if all_dimensions == False:
        pyt_y = torch.prod(in_data, dim=0, keepdim=True)
    else:
        pyt_y = torch.prod(in_data).view(1, 1, 1, 1)
    tt_output_tensor_on_device = ttnn.prod_bw(grad_tensor, input_tensor, all_dimensions=all_dimensions)
    in_data.retain_grad()
    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)

    assert comp_pass
