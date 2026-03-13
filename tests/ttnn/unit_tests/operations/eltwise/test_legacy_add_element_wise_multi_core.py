# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test file for legacy binary add operation using element-wise multi-core program factory.

This test specifically targets the legacy binary operation path that uses the
element_wise_multi_core_program_factory.cpp implementation.
"""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_test_tensors(a_shape, b_shape, device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Helper function to create test tensors"""
    torch.manual_seed(0)

    # Create PyTorch tensors
    a_pt = torch.rand(a_shape, dtype=torch.bfloat16) * 2.0 - 1.0  # Range [-1, 1]
    b_pt = torch.rand(b_shape, dtype=torch.bfloat16) * 2.0 - 1.0  # Range [-1, 1]

    # Create TTNN tensors
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)

    return a_pt, b_pt, a_tt, b_tt


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        # Basic same shape addition
        ([1, 1, 32, 32], [1, 1, 32, 32]),
        # Broadcasting scenarios
        ([1, 1, 64, 64], [1, 1, 1, 64]),
        ([1, 1, 128, 64], [1, 1, 128, 1]),
        # Multi-batch scenarios
        ([2, 4, 32, 32], [2, 4, 32, 32]),
        ([4, 1, 64, 128], [4, 1, 1, 128]),
    ],
)
def test_legacy_add_element_wise_multi_core(device, a_shape, b_shape):
    """
    Test legacy binary add operation using element-wise multi-core program factory.

    This test specifically uses use_legacy=True to target the legacy implementation
    path that uses element_wise_multi_core_program_factory.cpp
    """
    torch.manual_seed(0)

    # Create test tensors
    a_pt, b_pt, a_tt, b_tt = create_test_tensors(a_shape, b_shape, device)

    # Perform legacy add operation (this targets element_wise_multi_core_program_factory)
    output_tt = ttnn.add(a_tt, b_tt, use_legacy=True)

    # Convert back to torch for comparison
    output_pt_from_tt = ttnn.to_torch(output_tt)

    # Calculate expected result using PyTorch
    expected_pt = torch.add(a_pt, b_pt)

    # Verify results match
    assert_with_pcc(expected_pt, output_pt_from_tt, pcc=0.999)
    print(f"✓ Legacy add test passed for shapes {a_shape} + {b_shape}")


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_legacy_add_different_dtypes(device, dtype):
    """Test legacy add with different data types"""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    a_shape, b_shape = [1, 1, 64, 64], [1, 1, 64, 64]

    # Create PyTorch tensors with appropriate dtype
    torch.manual_seed(0)
    a_pt = torch.rand(a_shape, dtype=torch_dtype) * 2.0 - 1.0
    b_pt = torch.rand(b_shape, dtype=torch_dtype) * 2.0 - 1.0

    # Create TTNN tensors
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    # Perform legacy add operation
    output_tt = ttnn.add(a_tt, b_tt, use_legacy=True)
    output_pt_from_tt = ttnn.to_torch(output_tt)

    # Calculate expected result
    expected_pt = torch.add(a_pt, b_pt)

    # Verify results
    assert_with_pcc(expected_pt, output_pt_from_tt, pcc=0.999)
    print(f"✓ Legacy add test passed for dtype {dtype}")


def test_legacy_add_with_output_tensor(device):
    """Test legacy add operation with pre-allocated output tensor"""
    a_shape, b_shape = [1, 1, 32, 64], [1, 1, 32, 64]

    # Create input tensors
    a_pt, b_pt, a_tt, b_tt = create_test_tensors(a_shape, b_shape, device)

    # Create output tensor
    output_shape = torch.broadcast_shapes(torch.Size(a_shape), torch.Size(b_shape))
    output_pt_template = torch.zeros(output_shape, dtype=torch.bfloat16)
    output_tt = ttnn.from_torch(output_pt_template, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Perform legacy add with output tensor
    ttnn.add(a_tt, b_tt, output_tensor=output_tt, use_legacy=True)

    # Convert back and verify
    output_pt_from_tt = ttnn.to_torch(output_tt)
    expected_pt = torch.add(a_pt, b_pt)

    assert_with_pcc(expected_pt, output_pt_from_tt, pcc=0.999)
    print("✓ Legacy add with output tensor test passed")


def test_legacy_add_broadcasting_edge_cases(device):
    """Test legacy add with various broadcasting scenarios"""
    test_cases = [
        # Scalar broadcasting
        ([1, 1, 1, 1], [1, 1, 32, 32]),
        ([1, 1, 32, 32], [1, 1, 1, 1]),
        # Dimension broadcasting
        ([1, 1, 32, 1], [1, 1, 32, 64]),
        ([1, 1, 1, 64], [1, 1, 32, 64]),
        # Batch broadcasting
        ([1, 1, 32, 32], [4, 1, 32, 32]),
        ([2, 1, 32, 32], [1, 3, 32, 32]),
    ]

    for i, (a_shape, b_shape) in enumerate(test_cases):
        print(f"Testing broadcasting case {i+1}: {a_shape} + {b_shape}")

        # Create test tensors
        a_pt, b_pt, a_tt, b_tt = create_test_tensors(a_shape, b_shape, device)

        # Perform legacy add
        output_tt = ttnn.add(a_tt, b_tt, use_legacy=True)
        output_pt_from_tt = ttnn.to_torch(output_tt)

        # Calculate expected result
        expected_pt = torch.add(a_pt, b_pt)

        # Verify results
        assert_with_pcc(expected_pt, output_pt_from_tt, pcc=0.999)
        print(f"  ✓ Broadcasting case {i+1} passed")


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_legacy_add_different_memory_configs(device, memory_config):
    """Test legacy add with different memory configurations"""
    a_shape, b_shape = [1, 1, 32, 32], [1, 1, 32, 32]

    # Create test tensors with specified memory config
    a_pt, b_pt, a_tt, b_tt = create_test_tensors(a_shape, b_shape, device, memory_config=memory_config)

    # Perform legacy add operation
    output_tt = ttnn.add(a_tt, b_tt, memory_config=memory_config, use_legacy=True)
    output_pt_from_tt = ttnn.to_torch(output_tt)

    # Calculate expected result
    expected_pt = torch.add(a_pt, b_pt)

    # Verify results
    assert_with_pcc(expected_pt, output_pt_from_tt, pcc=0.999)
    print(f"✓ Legacy add test passed with memory config: {memory_config}")


def test_legacy_add_large_tensors(device):
    """Test legacy add with larger tensor sizes to stress the multi-core implementation"""
    # Use larger shapes that will definitely trigger multi-core processing
    test_shapes = [
        ([1, 1, 256, 256], [1, 1, 256, 256]),
        ([1, 1, 512, 128], [1, 1, 512, 128]),
        ([1, 1, 128, 512], [1, 1, 128, 512]),
    ]

    for a_shape, b_shape in test_shapes:
        print(f"Testing large tensor shapes: {a_shape} + {b_shape}")

        # Create test tensors
        a_pt, b_pt, a_tt, b_tt = create_test_tensors(a_shape, b_shape, device)

        # Perform legacy add operation
        output_tt = ttnn.add(a_tt, b_tt, use_legacy=True)
        output_pt_from_tt = ttnn.to_torch(output_tt)

        # Calculate expected result
        expected_pt = torch.add(a_pt, b_pt)

        # Verify results
        assert_with_pcc(expected_pt, output_pt_from_tt, pcc=0.999)
        print(f"  ✓ Large tensor test passed")


if __name__ == "__main__":
    """
    Example usage to run the test manually:

    import ttnn
    device = ttnn.open_device(device_id=0)
    try:
        test_legacy_add_element_wise_multi_core(device, [1, 1, 32, 32], [1, 1, 32, 32])
        test_legacy_add_different_dtypes(device, ttnn.bfloat16)
        test_legacy_add_with_output_tensor(device)
        test_legacy_add_broadcasting_edge_cases(device)
        test_legacy_add_different_memory_configs(device, ttnn.DRAM_MEMORY_CONFIG)
        test_legacy_add_large_tensors(device)
        print("All tests passed!")
    finally:
        ttnn.close_device(device)
    """
    pass
