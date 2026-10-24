# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
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
@pytest.mark.parametrize("value", [0.05, 1.0, 0.5, 5.0])
def test_bw_addcdiv(input_shapes, value, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor1_data, tensor1_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor2_data, tensor2_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)

    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, False)

    tt_output_tensor_on_device = ttnn.addcdiv_bw(grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value)

    golden_function = ttnn.get_golden_function(ttnn.addcdiv_bw)
    golden_tensor = golden_function(grad_data, in_data, tensor1_data, tensor2_data, value)

    comp_pcc = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pcc


# Test added for issue #60822: large numerator product and tensor2 == 0 edge cases
@pytest.mark.parametrize(
    "grad_val, t1_val, t2_val, val",
    [
        (1e20, 1e20, 1e20, 1.0),
        (-1.0, 1.0, 0.0, 1.0),
        (1.0, -1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, -2.0),
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, 0.0),
    ],
)
def test_bw_addcdiv_edge_cases(grad_val, t1_val, t2_val, val, device):
    in_data = torch.zeros(torch.Size([1, 1, 32, 32]), requires_grad=True, dtype=torch.bfloat16)
    t1_data = torch.ones(torch.Size([1, 1, 32, 32]), requires_grad=True, dtype=torch.bfloat16) * t1_val
    t2_data = torch.ones(torch.Size([1, 1, 32, 32]), requires_grad=True, dtype=torch.bfloat16) * t2_val
    grad_data = torch.ones(torch.Size([1, 1, 32, 32]), requires_grad=False, dtype=torch.bfloat16) * grad_val

    in_tensor = ttnn.Tensor(in_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    t1_tensor = ttnn.Tensor(t1_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    t2_tensor = ttnn.Tensor(t2_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    grad_tensor = ttnn.Tensor(grad_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    tt_output_tensor = ttnn.addcdiv_bw(grad_tensor, in_tensor, t1_tensor, t2_tensor, val)
    golden_function = ttnn.get_golden_function(ttnn.addcdiv_bw)
    golden_tensor = golden_function(grad_data, in_data, t1_data, t2_data, val)

    assert compare_results(tt_output_tensor, golden_tensor)

