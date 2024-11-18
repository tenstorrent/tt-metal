# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_logaddexp2(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -20, 20, device, True)

    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.logaddexp2_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.logaddexp2_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)
    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


def test_bw_logaddexp2_example(device):
    grad_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    x1_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True)
    x2_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True)
    golden_function = ttnn.get_golden_function(ttnn.logaddexp2_bw)
    golden_tensor = golden_function(grad_tensor, x1_torch, x2_torch)
    grad_tt = ttnn.from_torch(grad_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    x1_tt = ttnn.from_torch(x1_torch, layout=ttnn.TILE_LAYOUT, device=device)
    x2_tt = ttnn.from_torch(x2_torch, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.logaddexp2_bw(grad_tt, x1_tt, x2_tt)
    tt_out_1 = ttnn.to_torch(y_tt[1])
    tt_out_0 = ttnn.to_torch(y_tt[0])
    comp_pass_1 = torch.allclose(tt_out_1, golden_tensor[1])
    comp_pass_0 = torch.allclose(tt_out_0, golden_tensor[0])
    assert comp_pass_1 and comp_pass_0
