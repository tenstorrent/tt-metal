# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
        [4, 2, 96, 192],
        [4, 7, 21, 133],
        [4, 6, 105, 245],
        [64, 64],
        [3, 128, 512],
    ],
)
def test_i1_range(device, shapes):
    torch.manual_seed(0)

    high = 10
    low = -10
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low
    torch_output_tensor = torch.special.i1(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i1(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert pcc >= 0.9999


@pytest.mark.parametrize(
    "shapes",
    [
        [4, 2, 96, 192],
        [1, 1, 64, 64],
    ],
)
def test_i1_zero(device, shapes):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.zeros(shapes, dtype=torch.float32)
    torch_output_tensor = torch.special.i1(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i1(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.9999


# Covers the asymptotic |x| > 10 branch and the ±88.5 input clamp.
# Range [-50, 50] keeps reference values within FP32 (i1(50) ≈ 2.93e20).
# BF16's ~0.4% relative precision over a 20-decade output range loosens PCC.
@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
        [4, 2, 96, 192],
    ],
)
@pytest.mark.parametrize(
    "dtype, pcc_threshold",
    [
        (ttnn.float32, 0.9999),
        (ttnn.bfloat16, 0.99),
    ],
)
def test_i1_ood(device, shapes, dtype, pcc_threshold):
    torch.manual_seed(0)

    high = 50.0
    low = -50.0
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low
    torch_output_tensor = torch.special.i1(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i1(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert pcc >= pcc_threshold


# Boundary inputs straddling the |x| > 10 branch and the ±88.5 clamp.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i1_clamp_boundary(device, dtype):
    boundaries = torch.tensor(
        [-100.0, -88.5, -88.0, -10.5, -10.0, -9.5, 9.5, 10.0, 10.5, 88.0, 88.5, 100.0],
        dtype=torch.float32,
    )
    # i1 is unbounded; out-of-clamp inputs (|x| > 88.5) return i1(±88.5).
    expected_input = torch.clamp(boundaries, min=-88.5, max=88.5)
    torch_output_tensor = torch.special.i1(expected_input)

    # Pad to a tile-aligned shape so we can run on device.
    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, : boundaries.numel()] = boundaries

    input_tensor_a = ttnn.from_torch(
        padded,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i1(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)[0, 0, 0, : boundaries.numel()]

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert pcc >= 0.9999
