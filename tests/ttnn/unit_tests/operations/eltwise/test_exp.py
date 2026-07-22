# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose

pytestmark = pytest.mark.use_module_device


def test_exp_arange_masking(device):
    # Exp Working range - Overflow from 88.5(inf), Underflow till -87(<0)
    low = -87.0
    high = 88.5

    # Generate all possible bit patterns for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor_f32 = input_tensor.to(torch.float32)

    # masking to working range
    mask = (input_tensor_f32 >= low) & (input_tensor_f32 <= high)
    input_tensor = input_tensor[mask]

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 1)


@pytest.mark.parametrize(
    "input_shapes, sub_core_grid",
    [
        (
            (torch.Size([1, 2, 32, 960])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 7, 32, 96])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 8, 32, 128])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 17, 32, 32])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
        (
            (torch.Size([1, 1, 32, 32 * 1024])),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-5, 5),
        (-87.0, 88.5),
    ],
)
def test_exp_ULP_subcoregrid(input_shapes, sub_core_grid, low, high, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp(tt_in, sub_core_grids=sub_core_grid)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 1)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high, testing_dtype, expected_rtol, expected_atol",
    [
        (-89, -87, "bfloat16", 1e-2, 1e-3),
        (-87.3, 88.7, "float32", 1e-2, 1e-3),
    ],
)
def test_exp_atol(input_shapes, low, high, testing_dtype, expected_rtol, expected_atol, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)

    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch_dtype)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp(tt_in)
    assert_allclose(tt_result, golden, rtol=expected_rtol, atol=expected_atol)


def test_exp_fp32_accuracy(device):
    # FP32 exp finite range covered by the device implementation.
    low = -87.3
    high = 88.7

    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    bf16_promoted_inputs = all_bitpatterns.view(torch.bfloat16).to(torch.float32)

    mask = (bf16_promoted_inputs >= low) & (bf16_promoted_inputs <= high)
    bf16_promoted_inputs = bf16_promoted_inputs[mask]

    dense_fp32_inputs = torch.linspace(low, high, 4096, dtype=torch.float32)
    boundary_inputs = torch.tensor(
        [
            low,
            -80.0,
            -20.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            20.0,
            80.0,
            high,
            torch.nextafter(torch.tensor(high, dtype=torch.float32), torch.tensor(low, dtype=torch.float32)).item(),
        ],
        dtype=torch.float32,
    )
    input_tensor = torch.cat([bf16_promoted_inputs, dense_fp32_inputs, boundary_inputs]).unique()

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(golden, result, 1)


def test_exp_fp32_special_values(device):
    negative_nan = torch.tensor([-0x400000], dtype=torch.int32).view(torch.float32)
    input_tensor = torch.tensor(
        [
            -float("inf"),
            -120.0,
            -104.0,
            -0.0,
            0.0,
            1.0,
            88.7,
            89.0,
            100.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.float32,
    )
    input_tensor = torch.cat([input_tensor, negative_nan])

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(torch.isnan(result), torch.isnan(golden))
    assert torch.equal(torch.isposinf(result), torch.isposinf(golden))
    assert torch.equal(torch.isneginf(result), torch.isneginf(golden))
    assert_with_ulp(golden, result, 1, allow_nonfinite=True)
