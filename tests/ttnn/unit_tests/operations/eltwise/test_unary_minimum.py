# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_equal
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "input_val, scalar",
    [
        (0.36719, 0.5),
        (0.0, 0.06719),
        (0.0, 0.002),
        (3.4 * 10**38, 1.0),
        (-1.0, -3.4 * 10**38),
        (3.4 * 10**38, -3.4 * 10**38),
        (float("inf"), 1.0),
        (1, -float("inf")),
        (3.4 * 10**38, float("inf")),
        (-3.4 * 10**38, -float("inf")),
        (11.0, 1.0),
    ],
)
def test_unary_min_fill_val_fp32(input_shapes, input_val, scalar, device):
    torch_input = torch.ones(input_shapes, dtype=torch.float32) * input_val

    golden_function = ttnn.get_golden_function(ttnn.minimum)
    golden = golden_function(torch_input, torch.full(input_shapes, scalar), device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.minimum(tt_in, scalar)

    comp_pass = compare_equal([tt_result], [golden])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "input_val, scalar",
    [
        (0.36719, 0.5),
        (0.0034, 0.0023),
        (0.0, 0.06719),
        (0.0, 0.002),
        (3.4 * 10**38, 1.0),
        (-1.0, -3.4 * 10**38),
        (3.4 * 10**38, -3.4 * 10**38),
        (float("inf"), 1.0),
        (1.0, -float("inf")),
        (3.4 * 10**38, float("inf")),
        (-3.4 * 10**38, -float("inf")),
        (11.0, 1.0),
    ],
)
def test_unary_min_fill_val_bf16(input_shapes, input_val, scalar, device):
    torch_input = torch.ones(input_shapes, dtype=torch.bfloat16) * input_val

    golden_function = ttnn.get_golden_function(ttnn.minimum)
    golden = golden_function(torch_input, torch.full(input_shapes, scalar), device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.minimum(tt_in, scalar)
    result = ttnn.to_torch(tt_result)
    assert_with_pcc(golden, result, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-100.0, 100.0),
        (-3.3 * 10**38, 3.3 * 10**38),
    ],
)
@pytest.mark.parametrize("scalar", [0.5, 0.0, 20.0, 3.4 * 10**38, -3.4 * 10**38])
def test_unary_min_bf16(input_shapes, low, high, scalar, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(ttnn.minimum)
    golden = golden_function(torch_input, torch.full(input_shapes, scalar), device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.minimum(tt_in, scalar)
    result = ttnn.to_torch(tt_result)
    assert_with_pcc(golden, result, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-10.0, 100.0),
        (-1.7 * 10**38, 1.7 * 10**38),
    ],
)
@pytest.mark.parametrize(
    "scalar", [0.5, 0.1, 0.0, 1.0, 10.0, 3.4 * 10**38, -3.4 * 10**38, -float("inf"), float("inf")]
)
def test_unary_min_fp32(input_shapes, low, high, scalar, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.float32)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.minimum)
    golden = golden_function(torch_input, torch.full(input_shapes, scalar), device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.minimum(tt_in, scalar)
    comp_pass = compare_equal([tt_result], [golden])
    assert comp_pass


@pytest.mark.parametrize("scalar", [-1, -2, -3, -4, -5, 3, 0, 1, 100, 10, 5, 2147483, -2147483, -16777216, 16777216])
def test_unary_min_int32_test(scalar, device):
    num_elements = torch.prod(torch.tensor(torch.Size([1, 1, 32, 32]))).item()
    torch_input = torch.linspace(-10, 10, num_elements, dtype=torch.int32)
    torch_input = torch_input[:num_elements].reshape(torch.Size([1, 1, 32, 32]))

    golden_function = ttnn.get_golden_function(ttnn.minimum)
    golden = golden_function(torch_input, torch.full(torch.Size([1, 1, 32, 32]), scalar), device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_result = ttnn.minimum(tt_in, scalar)
    comp_pass = compare_equal([tt_result], [golden])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 320, 320])),
        (torch.Size([1, 3, 1024, 1024])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-5, 5),
        (-100, 100),
        (-21474, 21474),
        (-2147483600, 2147483600),
        (-21474836, 21474836),
        (-214748364, 214748364),
        (-2147483647, 2147483647),
        (-(2**31) + 1, (2**31) - 1),
    ],
)
@pytest.mark.parametrize("scalar", [-1, -2, -3, -4, -5, 3, 0, 1, 100, 10, 5, -16777216, 16777216, -16777215, 16777215])
def test_unary_min_int32(input_shapes, low, high, scalar, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.int32)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.minimum)
    golden = golden_function(torch_input, torch.full(input_shapes, scalar), device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.minimum(tt_in, scalar)
    comp_pass = compare_equal([tt_result], [golden])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "input_val, scalar",
    [
        (-1, 1),
        (1, 0),
        (0, 0),
        (1, 1),
        (11, 53),
    ],
)
def test_unary_min_fill_val_int32(input_shapes, input_val, scalar, device):
    torch_input = torch.ones(input_shapes, dtype=torch.int32) * input_val

    golden_function = ttnn.get_golden_function(ttnn.minimum)
    golden = golden_function(torch_input, torch.full(input_shapes, scalar), device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.minimum(tt_in, scalar)
    result = ttnn.to_torch(tt_result)

    comp_pass = compare_equal([tt_result], [golden])
    assert comp_pass
