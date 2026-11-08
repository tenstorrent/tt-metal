# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for issue #64100: Device operations relabel their outputs before validating the call.

This test verifies that when a device operation fails validation, the caller's tensors
retain their original topology labels instead of being mislabelled.
"""

import pytest
import torch
import ttnn


@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_copy_validation_failure_preserves_topology(device, mesh_shape):
    """
    Test that ttnn.copy preserves destination tensor topology when validation fails.
    
    Reproduces the bug from issue #64100:
    - Create sharded source and replicated destination with incompatible shapes
    - Attempt copy (should fail validation)
    - Verify destination topology is unchanged
    """
    if device.num_devices() < 2:
        pytest.skip("Requires at least 2 devices")
    
    # Create source tensor sharded across devices
    src = ttnn.from_torch(
        torch.randn(2, 1, 64, 64),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0)
    )
    
    # Create destination tensor replicated across devices with different shape
    dst = ttnn.from_torch(
        torch.zeros(1, 1, 64, 32),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    
    # Save original topology
    original_topology = dst.tensor_topology()
    original_placements = list(original_topology.placements())
    original_dist_shape = tuple(original_topology.distribution_shape())
    
    # Attempt copy - should fail validation due to shape mismatch
    with pytest.raises(RuntimeError, match="shapes do not match"):
        ttnn.copy(src, dst)
    
    # Verify topology is unchanged after failed validation
    final_topology = dst.tensor_topology()
    final_placements = list(final_topology.placements())
    final_dist_shape = tuple(final_topology.distribution_shape())
    
    assert final_placements == original_placements, (
        f"Topology placements changed after failed validation: "
        f"{original_placements} -> {final_placements}"
    )
    assert final_dist_shape == original_dist_shape, (
        f"Topology distribution shape changed after failed validation: "
        f"{original_dist_shape} -> {final_dist_shape}"
    )


@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_add_validation_failure_preserves_topology(device, mesh_shape):
    """
    Test that ttnn.add preserves output tensor topology when validation fails.
    
    Similar to test_copy_validation_failure_preserves_topology but for binary operations.
    """
    if device.num_devices() < 2:
        pytest.skip("Requires at least 2 devices")
    
    # Create sharded input tensor
    a = ttnn.from_torch(
        torch.randn(2, 1, 64, 64),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0)
    )
    
    # Create replicated input tensor with incompatible shape
    b = ttnn.from_torch(
        torch.randn(1, 1, 64, 32),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    
    # Create preallocated output tensor (replicated)
    out = ttnn.from_torch(
        torch.zeros(1, 1, 64, 32),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    
    # Save original topology
    original_topology = out.tensor_topology()
    original_placements = list(original_topology.placements())
    
    # Attempt add - should fail validation due to shape mismatch
    with pytest.raises(RuntimeError):
        ttnn.add(a, b, output_tensor=out)
    
    # Verify topology is unchanged
    final_topology = out.tensor_topology()
    final_placements = list(final_topology.placements())
    
    assert final_placements == original_placements, (
        f"Output topology changed after failed validation: "
        f"{original_placements} -> {final_placements}"
    )


def test_successful_operation_sets_topology_correctly(device):
    """
    Test that successful operations still set topology correctly.
    
    This ensures our fix doesn't break the normal path.
    """
    if device.num_devices() < 2:
        pytest.skip("Requires at least 2 devices")
    
    # Create compatible tensors
    src = ttnn.from_torch(
        torch.randn(2, 1, 64, 64),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0)
    )
    
    dst = ttnn.from_torch(
        torch.zeros(2, 1, 64, 64),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0)
    )
    
    # Copy should succeed
    ttnn.copy(src, dst)
    
    # Verify topology is set correctly (should match source)
    src_topology = src.tensor_topology()
    dst_topology = dst.tensor_topology()
    
    assert list(dst_topology.placements()) == list(src_topology.placements()), (
        "Successful copy should set destination topology to match source"
    )


if __name__ == "__main__":
    # Manual test runner for development
    pytest.main([__file__, "-v", "-s"])
