# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_equal
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([5, 8, 1024, 1024])),
    ),
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-5, 5, -10, 10),
        (-100, 100, -300, 300),
        (-2147483647, 2147483647, -21474, 21474),
    ],
)
def test_binary_max_int32(input_shapes, low_a, high_a, low_b, high_b, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_a = torch_input_a[:num_elements].reshape(input_shapes)
    torch_input_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_b = torch_input_b[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_result = ttnn.maximum(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)
    assert torch.equal(golden, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "input_a_val, input_b_val",
    [
        (-1, 1),
        (1, 0),
        (0, 0),
        (2147483647, -2147483647),
        (11, 53),
    ],
)
def test_binary_max_fill_val_int32(input_shapes, input_a_val, input_b_val, device):
    torch_input_a = torch.ones(input_shapes, dtype=torch.int32) * input_a_val
    torch_input_b = torch.ones(input_shapes, dtype=torch.int32) * input_b_val

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.maximum(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)
    assert torch.equal(golden, output_tensor)


@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    [
        (torch.Size([1, 2, 32]), torch.Size([1, 2, 32])),
        (torch.Size([1]), torch.Size([1, 5, 12])),
        (torch.Size([1, 2, 32, 64, 125]), torch.Size([1, 2, 32, 1, 1])),
        (torch.Size([]), torch.Size([])),
        (torch.Size([5]), torch.Size([1])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -300, 300),
        (-2147483647, 2147483647, -21474, 21474),
    ],
)
def test_binary_max_int32_bcast(input_shape_a, input_shape_b, low_a, high_a, low_b, high_b, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shape_a)).item()), 1)
    torch_input_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_a = torch_input_a[:num_elements].reshape(input_shape_a)

    num_elements = max(int(torch.prod(torch.tensor(input_shape_b)).item()), 1)
    torch_input_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_b = torch_input_b[:num_elements].reshape(input_shape_b)

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_result = ttnn.maximum(tt_in_a, tt_in_b, use_legacy=None)
    output_tensor = ttnn.to_torch(tt_result)
    assert torch.equal(golden, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([5, 10, 1024, 1024])),
    ),
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -300, 300),
        (-2147483647, 2147483647, -21474, 21474),
    ],
)
def test_binary_max_int32_opt(input_shapes, low_a, high_a, low_b, high_b, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_a = torch_input_a[:num_elements].reshape(input_shapes)
    torch_input_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_b = torch_input_b[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    output_tensor = torch.zeros(input_shapes, dtype=torch.int32)

    cq_id = 0

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = ttnn.from_torch(
        output_tensor,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn.maximum(tt_in_a, tt_in_b, output_tensor=tt_out, queue_id=cq_id)
    output_tensor = ttnn.to_torch(tt_out)
    assert torch.equal(golden, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([5, 8, 1024, 1024])),
    ),
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -300, 300),
        (-1.0 * 10**38, 1.0 * 10**38, -1.7 * 10**38, 1.7 * 10**38),
    ],
)
def test_binary_max_fp32(input_shapes, low_a, high_a, low_b, high_b, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.float32)
    torch_input_a = torch_input_a[:num_elements].reshape(input_shapes)
    torch_input_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.float32)
    torch_input_b = torch_input_b[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_result = ttnn.maximum(tt_in_a, tt_in_b)
    comp_pass = compare_equal([tt_result], [golden])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "input_a_val, input_b_val",
    [
        (0.36719, 0.5),
        (0, 0.06719),
        (0, 0.002),
        (3.4 * 10**38, 1),
        (-1, -3.4 * 10**38),
        (3.4 * 10**38, -3.4 * 10**38),
        (float("inf"), 1),
        (1, -float("inf")),
        (3.4 * 10**38, float("inf")),
        (-3.4 * 10**38, -float("inf")),
        (11, 1),
    ],
)
def test_binary_max_fill_val_fp32(input_shapes, input_a_val, input_b_val, device):
    torch_input_a = torch.ones(input_shapes, dtype=torch.float32) * input_a_val
    torch_input_b = torch.ones(input_shapes, dtype=torch.float32) * input_b_val

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.maximum(tt_in_a, tt_in_b)

    comp_pass = compare_equal([tt_result], [golden])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "input_a_val, input_b_val",
    [
        (0.36719, 0.5),
        (0.0034, 0.0023),
        (0, 0.06719),
        (0, 0.002),
        (3.4 * 10**38, 1),
        (-1, -3.4 * 10**38),
        (3.4 * 10**38, -3.4 * 10**38),
        (float("inf"), 1),
        (1, -float("inf")),
        (3.4 * 10**38, float("inf")),
        (-3.4 * 10**38, -float("inf")),
        (11, 1),
    ],
)
def test_binary_max_fill_val_bf16(input_shapes, input_a_val, input_b_val, device):
    torch_input_a = torch.ones(input_shapes, dtype=torch.bfloat16) * input_a_val
    torch_input_b = torch.ones(input_shapes, dtype=torch.bfloat16) * input_b_val

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.maximum(tt_in_a, tt_in_b)

    result = ttnn.to_torch(tt_result)
    assert_with_pcc(golden, result, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([5, 8, 1024, 1024])),
    ),
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -300, 300),
        (-3.0 * 10**38, 3.0 * 10**38, -3.3 * 10**38, 3.3 * 10**38),
    ],
)
def test_binary_max_bf16(input_shapes, low_a, high_a, low_b, high_b, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.bfloat16)
    torch_input_a = torch_input_a[:num_elements].reshape(input_shapes)
    torch_input_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.bfloat16)
    torch_input_b = torch_input_b[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.maximum(tt_in_a, tt_in_b)
    result = ttnn.to_torch(tt_result)
    assert_with_pcc(golden, result, 0.999)


@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    [
        (torch.Size([1, 2, 32]), torch.Size([1, 2, 32])),
        (torch.Size([1]), torch.Size([1, 5, 12])),
        (torch.Size([1, 2, 32, 64, 125]), torch.Size([1, 2, 32, 1, 1])),
        (torch.Size([]), torch.Size([])),
        (torch.Size([5]), torch.Size([1])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -300, 300),
        (-1.0 * 10**38, 1.0 * 10**38, -1.7 * 10**38, 1.7 * 10**38),
    ],
)
def test_binary_max_fp32_bcast(input_shape_a, input_shape_b, low_a, high_a, low_b, high_b, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shape_a)).item()), 1)
    torch_input_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.float32)
    torch_input_a = torch_input_a[:num_elements].reshape(input_shape_a)

    num_elements = max(int(torch.prod(torch.tensor(input_shape_b)).item()), 1)
    torch_input_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.float32)
    torch_input_b = torch_input_b[:num_elements].reshape(input_shape_b)

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_result = ttnn.maximum(tt_in_a, tt_in_b, use_legacy=None)
    comp_pass = compare_equal([tt_result], [golden])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    [
        (torch.Size([1, 2, 32]), torch.Size([1, 2, 32])),
        (torch.Size([1]), torch.Size([1, 5, 12])),
        (torch.Size([1, 2, 32, 64, 125]), torch.Size([1, 2, 32, 1, 1])),
        (torch.Size([]), torch.Size([])),
        (torch.Size([5]), torch.Size([1])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -300, 300),
        (-1.0 * 10**38, 1.0 * 10**38, -1.7 * 10**38, 1.7 * 10**38),
    ],
)
def test_binary_max_bf16_bcast(input_shape_a, input_shape_b, low_a, high_a, low_b, high_b, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shape_a)).item()), 1)
    torch_input_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.bfloat16)
    torch_input_a = torch_input_a[:num_elements].reshape(input_shape_a)

    num_elements = max(int(torch.prod(torch.tensor(input_shape_b)).item()), 1)
    torch_input_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.bfloat16)
    torch_input_b = torch_input_b[:num_elements].reshape(input_shape_b)

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_result = ttnn.maximum(tt_in_a, tt_in_b, use_legacy=None)
    result = ttnn.to_torch(tt_result)
    assert_with_pcc(golden, result, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([5, 10, 1024, 1024])),
    ),
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -300, 300),
        (-1.0 * 10**38, 1.0 * 10**38, -1.7 * 10**38, 1.7 * 10**38),
    ],
)
def test_binary_max_fp32_opt(input_shapes, low_a, high_a, low_b, high_b, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.float32)
    torch_input_a = torch_input_a[:num_elements].reshape(input_shapes)
    torch_input_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.float32)
    torch_input_b = torch_input_b[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.maximum)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    output_tensor = torch.zeros(input_shapes, dtype=torch.float32)

    cq_id = 0

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = ttnn.from_torch(
        output_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn.maximum(tt_in_a, tt_in_b, output_tensor=tt_out, queue_id=cq_id)
    comp_pass = compare_equal([tt_out], [golden])
    assert comp_pass
