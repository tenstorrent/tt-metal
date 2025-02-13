# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range
from tests.ttnn.utils_for_testing import assert_with_pcc

@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)

def test_device_hypot_no_bcast(input_shapes, device):
    torch.manual_seed(0)
    in_data1 = torch.rand(input_shapes, dtype=torch.bfloat16) * (200 - 100)
    in_data2 = torch.rand(input_shapes, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.hypot)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert comp_pass >= 0.9998

@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 3, 32, 32]), torch.Size([1, 1, 1, 1])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
    ),
)

def test_device_hypot_scalar(a_shape, b_shape, device):
    torch.manual_seed(0)
    in_data1 = torch.rand(a_shape, dtype=torch.bfloat16) * (200 - 100)
    in_data2 = torch.rand(b_shape, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.hypot)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert comp_pass >= 0.9998

@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([2, 3, 1, 4]), torch.Size([2, 3, 5, 4])), #ROW_A
        (torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 1, 4])), #ROW_B
    ),
)

def test_device_hypot_bcast_row(a_shape, b_shape, device):
    torch.manual_seed(0)
    in_data1 = torch.rand(a_shape, dtype=torch.bfloat16) * (200 - 100)
    in_data2 = torch.rand(b_shape, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.hypot)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert comp_pass >= 0.9998

@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([2, 3, 5, 1]), torch.Size([2, 3, 5, 4])), #COL_A
        (torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 5, 1])), #COL_B
    ),
)

def test_device_hypot_bcast_col(a_shape, b_shape, device):
    torch.manual_seed(0)
    in_data1 = torch.rand(a_shape, dtype=torch.bfloat16) * (200 - 100)
    in_data2 = torch.rand(b_shape, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.hypot)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert comp_pass >= 0.9998

@pytest.mark.parametrize(
    "a_shape, b_shape",
    (
        (torch.Size([1, 1, 31, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 2, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([2, 3, 128, 1])),
    ),
)

def test_device_hypot_invalid_bcast(a_shape, b_shape, device):
    torch.manual_seed(0)
    in_data1 = torch.rand(a_shape, dtype=torch.bfloat16) * (200 - 100)
    in_data2 = torch.rand(b_shape, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as e:
        output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)
        assert "Broadcasting rule violation" in str(e.value)
