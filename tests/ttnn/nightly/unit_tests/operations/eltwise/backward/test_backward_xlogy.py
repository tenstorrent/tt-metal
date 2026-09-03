# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_xlogy(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True, seed=0)
    other_data, other_tensor = data_gen_with_range(input_shapes, 1, 5, device, True, seed=1)

    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device, seed=3)

    tt_output_tensor_on_device = ttnn.xlogy_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.xlogy_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)
    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


def test_bw_xlogy_zero_divisor(device):
    """Regression test for #55316: grad wrt y at y == 0 must match torch autograd sign and NaN rules."""
    input_shapes = torch.Size([1, 1, 1, 4])
    in_data = torch.tensor([[[[2.0, -2.0, 0.0, -2.0]]]], dtype=torch.bfloat16, requires_grad=True)
    other_data = torch.zeros((1, 1, 1, 4), dtype=torch.bfloat16, requires_grad=True)
    grad_data = torch.tensor([[[[1.0, 1.0, 1.0, -1.0]]]], dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(in_data, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    other_tensor = ttnn.from_torch(other_data, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    grad_tensor = ttnn.from_torch(grad_data, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_output_tensor_on_device = ttnn.xlogy_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.xlogy_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)
    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
