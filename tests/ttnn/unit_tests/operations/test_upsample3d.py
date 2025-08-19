# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "input_shapes",
    [
        # [N, C, D, H, W] format (NCDHW for torch)
        [1, 64, 4, 8, 8],  # Small basic test
        [1, 32, 2, 16, 16],  # Different aspect ratio
        [2, 16, 3, 12, 8],  # Batch size > 1
        [1, 128, 2, 8, 8],  # More channels
        [1, 8, 6, 4, 6],  # Odd dimensions
        [3, 32, 2, 4, 4],  # Larger batch
        [1, 64, 1, 16, 16],  # Minimal depth
        [1, 256, 2, 4, 4],  # Many channels, small spatial
        [2, 64, 3, 6, 6],  # Batch + depth
    ],
)
@pytest.mark.parametrize(
    "scale_factor",
    [
        (2, 2, 2),  # Uniform scaling
        (1, 2, 2),  # No depth scaling
        (2, 1, 2),  # No height scaling
        (2, 2, 1),  # No width scaling
        (1, 3, 2),  # Mixed scaling
        (3, 1, 3),  # Different ratios
        (2, 3, 4),  # All different
        2,  # Single integer (uniform)
        3,  # Different single integer
    ],
)
def test_upsample3d_functionality(device, input_shapes, scale_factor):
    """Test 3D upsampling operation functionality with torch comparison"""
    torch.manual_seed(0)

    # Create torch input in NCDHW format
    batch_size, num_channels, depth, height, width = input_shapes
    input_ncdhw = torch.randn(input_shapes, dtype=torch.bfloat16)

    # Convert to NDHWC format for TTNN (channel-last)
    input_ndhwc = input_ncdhw.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

    tt_input_tensor = ttnn.from_torch(input_ndhwc, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # TTNN operation
    tt_output_tensor = ttnn.upsample3d(tt_input_tensor, scale_factor)
    output_ndhwc = ttnn.to_torch(tt_output_tensor)

    # Torch reference (operates on NCDHW)
    if isinstance(scale_factor, int):
        torch_scale = scale_factor
    else:
        torch_scale = scale_factor  # (scale_d, scale_h, scale_w)

    torch_upsample = nn.Upsample(scale_factor=torch_scale, mode="nearest")
    torch_result_ncdhw = torch_upsample(input_ncdhw)

    # Convert torch result to channel-last for comparison
    torch_result_ndhwc = torch_result_ncdhw.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

    # Verify shapes match
    assert list(output_ndhwc.shape) == list(
        torch_result_ndhwc.shape
    ), f"Shape mismatch: TTNN {output_ndhwc.shape} vs Torch {torch_result_ndhwc.shape}"

    # Compare results with PCC
    pcc_passed, pcc_message = assert_with_pcc(torch_result_ndhwc, output_ndhwc, pcc=0.99999)

    # Additional checks for exact match (should be identical for nearest neighbor)
    allclose = torch.allclose(output_ndhwc, torch_result_ndhwc, atol=1e-2, rtol=1e-2)
    isclose = torch.all(torch.isclose(output_ndhwc, torch_result_ndhwc, atol=1e-2, rtol=1e-2))

    assert allclose, f"Results not close: {pcc_message}"
    assert isclose, f"Some elements not close: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, scale_factor",
    [
        # Test various combinations with different tensor sizes
        ([1, 32, 3, 4, 4], [1, 2, 3]),  # Asymmetric scaling
        ([2, 64, 2, 6, 8], [3, 1, 2]),  # Batch > 1
        ([1, 16, 4, 3, 5], [2, 4, 1]),  # Different patterns
        ([1, 128, 1, 8, 8], [4, 2, 2]),  # Minimal depth
        ([3, 8, 2, 2, 2], [1, 3, 4]),  # Small tensor, large scaling
    ],
)
def test_upsample3d_asymmetric_scaling(device, input_shape, scale_factor):
    """Test asymmetric scaling patterns"""
    torch.manual_seed(0)

    # Create torch input in NCDHW format
    input_ncdhw = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ndhwc = input_ncdhw.permute(0, 2, 3, 4, 1)

    tt_input_tensor = ttnn.from_torch(input_ndhwc, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # TTNN operation
    tt_output_tensor = ttnn.upsample3d(tt_input_tensor, scale_factor)
    output_ndhwc = ttnn.to_torch(tt_output_tensor)

    # Torch reference
    if isinstance(scale_factor, int):
        torch_scale = scale_factor
    else:
        torch_scale = tuple(scale_factor)
    torch_upsample = nn.Upsample(scale_factor=torch_scale, mode="nearest")
    torch_result_ncdhw = torch_upsample(input_ncdhw)
    torch_result_ndhwc = torch_result_ncdhw.permute(0, 2, 3, 4, 1)

    # Compare results
    pcc_passed, pcc_message = assert_with_pcc(torch_result_ndhwc, output_ndhwc, pcc=0.99999)
    allclose = torch.allclose(output_ndhwc, torch_result_ndhwc, atol=1e-2, rtol=1e-2)

    assert allclose, f"Results not close for asymmetric scaling: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "mode",
    ["nearest"],  # Currently only nearest is supported
)
def test_upsample3d_mode_parameter(device, mode):
    """Test mode parameter (currently only 'nearest' supported)"""
    torch.manual_seed(0)

    input_shape = [1, 32, 2, 4, 4]  # NCDHW
    input_ncdhw = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ndhwc = input_ncdhw.permute(0, 2, 3, 4, 1)

    tt_input_tensor = ttnn.from_torch(input_ndhwc, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_output_tensor = ttnn.upsample3d(tt_input_tensor, (2, 2, 2), mode=mode)
    output_ndhwc = ttnn.to_torch(tt_output_tensor)

    # Verify output shape
    expected_shape = [1, 4, 8, 8, 32]  # NDHWC
    assert list(output_ndhwc.shape) == expected_shape


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_upsample3d_edge_cases(device):
    """Test edge cases"""
    torch.manual_seed(0)

    # Test scale factor of 1 (no upsampling)
    input_shape = [1, 16, 2, 3, 4]  # NCDHW
    input_ncdhw = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ndhwc = input_ncdhw.permute(0, 2, 3, 4, 1)

    tt_input_tensor = ttnn.from_torch(input_ndhwc, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_output_tensor = ttnn.upsample3d(tt_input_tensor, (1, 1, 1))
    output_ndhwc = ttnn.to_torch(tt_output_tensor)

    # Output should be identical to input
    assert list(output_ndhwc.shape) == list(input_ndhwc.shape)
    assert torch.allclose(output_ndhwc, input_ndhwc, atol=1e-2, rtol=1e-2)

    # Test minimal 5D tensor [1, 1, 1, 1, 1]
    minimal_input_ncdhw = torch.tensor([[[[[5.0]]]]], dtype=torch.bfloat16)
    minimal_input_ndhwc = minimal_input_ncdhw.permute(0, 2, 3, 4, 1)

    tt_minimal_tensor = ttnn.from_torch(minimal_input_ndhwc, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_minimal_output = ttnn.upsample3d(tt_minimal_tensor, 2)
    minimal_output = ttnn.to_torch(tt_minimal_output)

    expected_shape = [1, 2, 2, 2, 1]  # NDHWC
    assert list(minimal_output.shape) == expected_shape

    # All values should be approximately 5.0
    expected_output = torch.full(expected_shape, 5.0, dtype=torch.bfloat16)
    assert torch.allclose(minimal_output, expected_output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 256, 2, 8, 8],  # Many channels
        [4, 64, 3, 4, 6],  # Large batch
        [1, 32, 4, 12, 8],  # Rectangular spatial dims
        [2, 128, 1, 16, 16],  # Large spatial, minimal depth
        [1, 512, 2, 4, 4],  # Very many channels
    ],
)
@pytest.mark.parametrize("scale_factor", [2, 3, (1, 2, 3), (3, 2, 1)])
def test_upsample3d_stress_tests(device, input_shape, scale_factor):
    """Test larger tensors and various scaling patterns"""
    torch.manual_seed(0)

    # Create torch input in NCDHW format
    input_ncdhw = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ndhwc = input_ncdhw.permute(0, 2, 3, 4, 1)

    tt_input_tensor = ttnn.from_torch(input_ndhwc, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # TTNN operation
    tt_output_tensor = ttnn.upsample3d(tt_input_tensor, scale_factor)
    output_ndhwc = ttnn.to_torch(tt_output_tensor)

    # Torch reference
    if isinstance(scale_factor, int):
        torch_scale = scale_factor
    else:
        torch_scale = tuple(scale_factor)
    torch_upsample = nn.Upsample(scale_factor=torch_scale, mode="nearest")
    torch_result_ncdhw = torch_upsample(input_ncdhw)
    torch_result_ndhwc = torch_result_ncdhw.permute(0, 2, 3, 4, 1)

    # Compare results with slightly relaxed tolerance for larger tensors
    pcc_passed, pcc_message = check_with_pcc_without_tensor_printout(torch_result_ndhwc, output_ndhwc, pcc=0.9999)
    allclose = torch.allclose(output_ndhwc, torch_result_ndhwc, atol=1e-2, rtol=1e-2)

    assert allclose, f"Stress test failed: {pcc_message}"
    assert pcc_passed, f"PCC test failed: {pcc_message}"
