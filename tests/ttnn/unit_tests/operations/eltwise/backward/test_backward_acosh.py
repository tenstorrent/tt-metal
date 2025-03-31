# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    compare_results,
)


# Test added for issue #6583
@pytest.mark.parametrize(
    "in_val, grad_val",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 1.0),
        (0.5, 0.5),
        (1.0, 0.5),
        (0.0, 0.0),
    ],
)
def test_bw_acosh_edge_cases(in_val, grad_val, device):
    in_data = (torch.ones(torch.Size([1, 1, 32, 32]), requires_grad=True) * in_val).bfloat16()
    input_tensor = ttnn.Tensor(in_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    grad_data = (torch.ones(torch.Size([1, 1, 32, 32]), requires_grad=False) * grad_val).bfloat16()
    grad_tensor = ttnn.Tensor(grad_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    tt_output_tensor_on_device = ttnn.acosh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.acosh_bw)
    golden_tensor = golden_function(grad_data, in_data, device=device)
    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_acosh(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device, True)

    tt_output_tensor_on_device = ttnn.acosh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.acosh_bw)
    golden_tensor = golden_function(grad_data, in_data, device=device)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
