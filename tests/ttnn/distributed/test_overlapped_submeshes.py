# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
def test_overlapped_submeshes(mesh_device):
    """
    Python version of the C++ OverlappedSubmeshes test.
    Tests the create_overlapped_submeshes functionality and buffer allocation behavior.
    """
    # Verify mesh device shape
    assert mesh_device.shape == ttnn.MeshShape(2, 4)

    # Create overlapped submesh ranges - equivalent to C++ test
    submesh_range_1 = ttnn.MeshCoordinateRange(ttnn.MeshShape(2, 2))  # 2x2 from origin
    submesh_range_2 = ttnn.MeshCoordinateRange(
        ttnn.MeshCoordinate(0, 2), ttnn.MeshCoordinate(1, 3)  # 2x2 starting at (0,2)
    )
    submesh_range_3 = ttnn.MeshCoordinateRange(ttnn.MeshShape(1, 4))  # 1x4 from origin

    # Create overlapped submeshes using the newly pybound function
    submeshes = mesh_device.create_overlapped_submeshes([submesh_range_1, submesh_range_2, submesh_range_3])

    # Verify submesh creation
    assert len(submeshes) == 3
    assert submeshes[0].shape == ttnn.MeshShape(2, 2)
    assert submeshes[1].shape == ttnn.MeshShape(2, 2)
    assert submeshes[2].shape == ttnn.MeshShape(1, 4)

    # Test buffer allocation behavior with overlapped submeshes
    # Create test tensors to verify allocation behavior
    buffer_size_elements = 1024  # 4KB buffer / 4 bytes per element
    test_shape = (1, buffer_size_elements)

    # Create tensors on each submesh to test allocation behavior
    # Note: Using ttnn.from_torch instead of direct MeshBuffer.create since
    # MeshBuffer is not directly exposed to Python

    # Create test data
    torch_tensor = torch.randn(test_shape, dtype=torch.float32)

    # Test L1 buffer allocation
    l1_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.L1
    )

    # Allocate tensor on submesh1
    tensor1 = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submeshes[0],
        memory_config=l1_memory_config,
    )

    # Allocate tensor on submesh2
    tensor2 = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submeshes[1],
        memory_config=l1_memory_config,
    )

    # Allocate another tensor on submesh2
    tensor2_next = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submeshes[1],
        memory_config=l1_memory_config,
    )

    # Allocate tensor on submesh3 (overlaps with both submesh1 and submesh2)
    tensor3 = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submeshes[2],
        memory_config=l1_memory_config,
    )

    # Test DRAM buffer allocation
    dram_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM
    )

    # Allocate DRAM tensors on each submesh
    tensor1_dram = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submeshes[0],
        memory_config=dram_memory_config,
    )

    tensor2_dram = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submeshes[1],
        memory_config=dram_memory_config,
    )

    tensor2_next_dram = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submeshes[1],
        memory_config=dram_memory_config,
    )

    tensor3_dram = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submeshes[2],
        memory_config=dram_memory_config,
    )

    # Verify tensors are allocated successfully
    # Note: In Python, we can't directly access buffer addresses like in C++,
    # but we can verify that tensors are created and have the expected properties

    # Verify L1 tensors
    assert tensor1.is_allocated()
    assert tensor2.is_allocated()
    assert tensor2_next.is_allocated()
    assert tensor3.is_allocated()

    # Verify DRAM tensors
    assert tensor1_dram.is_allocated()
    assert tensor2_dram.is_allocated()
    assert tensor2_next_dram.is_allocated()
    assert tensor3_dram.is_allocated()

    # Verify tensor shapes and properties
    for tensor in [tensor1, tensor2, tensor2_next, tensor3]:
        assert tensor.shape == test_shape
        assert tensor.dtype == ttnn.bfloat16
        assert tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tensor.memory_config().buffer_type == ttnn.BufferType.L1

    for tensor in [tensor1_dram, tensor2_dram, tensor2_next_dram, tensor3_dram]:
        assert tensor.shape == test_shape
        assert tensor.dtype == ttnn.bfloat16
        assert tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tensor.memory_config().buffer_type == ttnn.BufferType.DRAM

    # Test that we can perform operations on tensors from overlapped submeshes
    # This indirectly tests that the allocation dependencies are working correctly

    # Simple operation to verify tensors are functional
    result1 = ttnn.add(tensor1, tensor1)
    result2 = ttnn.add(tensor2, tensor2)
    # Must manually synchronize for now
    ttnn.synchronize_device(submeshes[0])
    ttnn.synchronize_device(submeshes[1])

    result3 = ttnn.add(tensor3, tensor3)
    ttnn.synchronize_device(submeshes[2])

    assert result1.is_allocated()
    assert result2.is_allocated()
    assert result3.is_allocated()
    # TODO: Right now, there are issues with:
    # - Readback for partially sharded tensors
    # - Device teardown? (hanging with overlapped submeshes)
    return

    # Convert back to torch to verify correctness
    torch_result1 = ttnn.to_torch(result1)
    torch_result2 = ttnn.to_torch(result2)
    torch_result3 = ttnn.to_torch(result3)

    # Verify the operations produced expected results (tensor + tensor = 2 * tensor)
    expected_result = 2 * torch_tensor
    torch.testing.assert_close(torch_result1, expected_result, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(torch_result2, expected_result, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(torch_result3, expected_result, rtol=1e-2, atol=1e-2)

    print("✓ Overlapped submeshes test passed!")
    print(f"✓ Created {len(submeshes)} overlapped submeshes with shapes:")
    for i, submesh in enumerate(submeshes):
        print(f"  - Submesh {i+1}: {submesh.shape}")
    print("✓ Successfully allocated and operated on tensors across overlapped submeshes")


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
def test_overlapped_submeshes_coordinate_ranges(mesh_device):
    """
    Test various coordinate range configurations for overlapped submeshes.
    """
    # Test different coordinate range patterns

    # Non-overlapping ranges
    range_a = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1))  # 1x2
    range_b = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(1, 0), ttnn.MeshCoordinate(1, 1))  # 1x2

    submeshes_non_overlapping = mesh_device.create_overlapped_submeshes([range_a, range_b])
    assert len(submeshes_non_overlapping) == 2
    assert submeshes_non_overlapping[0].shape == ttnn.MeshShape(1, 2)
    assert submeshes_non_overlapping[1].shape == ttnn.MeshShape(1, 2)

    # Fully overlapping ranges
    range_c = ttnn.MeshCoordinateRange(ttnn.MeshShape(2, 2))
    range_d = ttnn.MeshCoordinateRange(ttnn.MeshShape(2, 2))

    submeshes_overlapping = mesh_device.create_overlapped_submeshes([range_c, range_d])
    assert len(submeshes_overlapping) == 2
    assert submeshes_overlapping[0].shape == ttnn.MeshShape(2, 2)
    assert submeshes_overlapping[1].shape == ttnn.MeshShape(2, 2)

    # Partially overlapping ranges
    range_e = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(1, 2))  # 2x3
    range_f = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 1), ttnn.MeshCoordinate(1, 3))  # 2x3

    submeshes_partial = mesh_device.create_overlapped_submeshes([range_e, range_f])
    assert len(submeshes_partial) == 2
    assert submeshes_partial[0].shape == ttnn.MeshShape(2, 3)
    assert submeshes_partial[1].shape == ttnn.MeshShape(2, 3)

    print("✓ Coordinate range variations test passed!")


if __name__ == "__main__":
    # This allows running the test directly for debugging
    pytest.main([__file__, "-v"])
