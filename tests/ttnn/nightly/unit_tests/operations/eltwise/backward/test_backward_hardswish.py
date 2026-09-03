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
def test_bw_hardswish(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.hardswish_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.hardswish_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass

@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.float32,
        ttnn.bfloat16,
    ),
)
def test_bw_hardswish_boundary(dtype, device):
    """Regression test for boundary values x = +/-3 and accuracy inside (-3, 3)."""
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    input_shapes = torch.Size([1, 1, 32, 32])

    in_data = torch.zeros(input_shapes, dtype=torch_dtype, requires_grad=True)
    in_data[..., 0, 0] = -3.0
    in_data[..., 0, 1] = 3.0
    in_data[..., 0, 2] = 2.5
    in_data[..., 0, 3] = -2.5
    in_data[..., 0, 4] = 0.0

    grad_data = torch.ones(input_shapes, dtype=torch_dtype)

    input_tensor = ttnn.Tensor(in_data, dtype).to(ttnn.TILE_LAYOUT).to(device)
    grad_tensor = ttnn.Tensor(grad_data, dtype).to(ttnn.TILE_LAYOUT).to(device)

    tt_output_tensor_on_device = ttnn.hardswish_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.hardswish_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
