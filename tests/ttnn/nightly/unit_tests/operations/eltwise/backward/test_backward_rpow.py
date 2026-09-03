# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
    data_gen_with_range,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent",
    (
        1.5,
        5.7,
        0.0,
        15.2,
    ),
)
def test_bw_rpow(input_shapes, exponent, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -201, 199, device, True)

    tt_output_tensor_on_device = ttnn.rpow_bw(grad_tensor, input_tensor, exponent)

    golden_function = ttnn.get_golden_function(ttnn.rpow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass

@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.float32,
        ttnn.bfloat16,
    ),
)
@pytest.mark.parametrize(
    "exponent",
    (
        1.5,
        2.0,
        5.7,
    ),
)
def test_bw_rpow_negative_input(dtype, exponent, device):
    """Regression test for negative inputs: rpow_bw should match autograd of base ** x."""
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    input_shapes = torch.Size([1, 1, 32, 32])

    in_data = torch.zeros(input_shapes, dtype=torch_dtype, requires_grad=True)
    in_data[..., 0, 0] = -2.0
    in_data[..., 0, 1] = -1.0
    in_data[..., 0, 2] = 0.0
    in_data[..., 0, 3] = 0.5
    in_data[..., 0, 4] = 1.0
    in_data[..., 0, 5] = 2.0

    grad_data = torch.ones(input_shapes, dtype=torch_dtype)

    input_tensor = ttnn.Tensor(in_data, dtype).to(ttnn.TILE_LAYOUT).to(device)
    grad_tensor = ttnn.Tensor(grad_data, dtype).to(ttnn.TILE_LAYOUT).to(device)

    tt_output_tensor_on_device = ttnn.rpow_bw(grad_tensor, input_tensor, exponent)

    golden_function = ttnn.get_golden_function(ttnn.rpow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
