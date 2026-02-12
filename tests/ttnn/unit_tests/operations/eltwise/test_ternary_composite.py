# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_val,
    compare_pcc,
)
from tests.ttnn.utils_for_testing import assert_with_ulp


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [1.0, 5.0, 10.0])
def test_ternary_addcmul_ttnn(input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -80, 80, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -90, 90, device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [1.0, 5.0, 10.0])
def test_ternary_addcdiv_ttnn(input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.addcdiv(input_tensor1, input_tensor2, input_tensor3, value=value)
    golden_fn = ttnn.get_golden_function(ttnn.addcdiv)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


def create_full_range_tensor(input_shape, dtype, value_ranges):
    """Create a tensor with values spanning multiple ranges."""
    num_elements = torch.prod(torch.tensor(input_shape)).item()

    num_ranges = len(value_ranges)
    elements_per_range = num_elements // num_ranges
    remainder = num_elements % num_ranges

    segments = []
    for i, (low, high) in enumerate(value_ranges):
        range_elements = elements_per_range + (1 if i < remainder else 0)

        segment = torch.linspace(low, high, steps=range_elements, dtype=dtype)
        segments.append(segment)

    in_data = torch.cat(segments)
    in_data = in_data.reshape(input_shape)
    return in_data


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 2, 32, 128])),),
)
@pytest.mark.parametrize("value", [2.75])
def test_ternary_addcdiv_float32_full_range(input_shapes, value, device):
    """Test addcdiv with float32 inputs spanning the range of [-1e15, 1e15]."""
    value_ranges_a = [
        (-100, 100),
        (-300, 300),
        (-750, 500),
        (-1000, 1000),
        (-1e4, 1e4),
        (-1e5, 1e5),
        (-1e7, 1e7),
        (-1e10, 1e10),
        (-1e15, 1e15),
        (1e8, 1e10),
        (1e12, 1e15),
        (-1e10, -1e8),
        (-1e15, -1e12),
        (-1e-5, 1e-5),
        (-1e-10, 1e-10),
    ]

    value_ranges_b = [
        (-50, 200),
        (-400, 600),
        (-2000, 3000),
        (-2e4, 2e4),
        (-3e5, 3e5),
        (-5e6, 5e6),
        (-2e8, 2e8),
        (-8e9, 8e9),
        (-1e14, 1e14),
        (2e8, 5e9),
        (8e11, 8e14),
        (-5e8, -2e7),
        (-8e14, -8e11),
        (-2e-4, 2e-4),
        (-2e-8, 2e-8),
        (-3e7, 3e7),
    ]

    value_ranges_c = [
        (-200, 50),
        (-800, 200),
        (-5000, 2000),
        (-1e3, 1e4),
        (-1e4, 1e5),
        (-2e5, 2e6),
        (-1e7, 1e8),
        (-1e9, 1e10),
        (-5e13, 5e14),
        (5e7, 2e9),
        (1e11, 2e14),
        (-3e7, -5e6),
        (-2e13, -2e11),
        (-5e-6, 5e-6),
        (-1e-7, 1e-7),
        (-1e6, 1e7),
    ]

    torch_input_tensor_a = create_full_range_tensor(
        input_shape=input_shapes, dtype=torch.float32, value_ranges=value_ranges_a
    )
    torch_input_tensor_b = create_full_range_tensor(
        input_shape=input_shapes, dtype=torch.float32, value_ranges=value_ranges_b
    )
    torch_input_tensor_c = create_full_range_tensor(
        input_shape=input_shapes, dtype=torch.float32, value_ranges=value_ranges_c
    )

    golden_function = ttnn.get_golden_function(ttnn.addcdiv)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, torch_input_tensor_c, value=value)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_c = ttnn.from_torch(
        torch_input_tensor_c,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.addcdiv(input_tensor_a, input_tensor_b, input_tensor_c, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(output_tensor, torch_output_tensor, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("predicate", [0, 1])
def test_ternary_where_opt_output(input_shapes, predicate, device):
    in_data1, input_tensor1 = data_gen_with_val(input_shapes, device, val=predicate)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -90, 90, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    ttnn.where(input_tensor1, input_tensor2, input_tensor3, output_tensor=output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.where)
    golden_tensor = golden_fn(in_data1.bool(), in_data2, in_data3)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [1.0, 5.0, 10.0])
def test_lerp_overload_ttnn(input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.lerp(input_tensor1, input_tensor2, value)
    golden_fn = ttnn.get_golden_function(ttnn.lerp)
    golden_tensor = golden_fn(in_data1, in_data2, value)

    output_torch = output_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    assert_with_ulp(golden_tensor, output_torch, ulp_threshold=2)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_lerp_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.lerp(input_tensor1, input_tensor2, input_tensor3)
    golden_fn = ttnn.get_golden_function(ttnn.lerp)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3)

    output_torch = output_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    assert_with_ulp(golden_tensor, output_torch, ulp_threshold=2)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value1", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("value2", [1.0, 5.0, 10.0])
def test_mac_overload_ttnn(input_shapes, value1, value2, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.mac(input_tensor1, value1, value2)
    golden_fn = ttnn.get_golden_function(ttnn.mac)
    golden_tensor = golden_fn(in_data1, value1, value2)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_mac_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.mac(input_tensor1, input_tensor2, input_tensor3)
    golden_fn = ttnn.get_golden_function(ttnn.mac)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass
