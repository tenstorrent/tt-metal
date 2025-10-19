# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}],
    indirect=True,
)
def test_distributed_matmul_replicated(mesh_device):
    """Test distributed matmul with replicated tensors on 2x4 mesh"""

    # Skip if not enough devices
    if mesh_device.get_num_devices() < 8:
        pytest.skip("Requires at least 8 devices (2x4 mesh)")

    # Input shapes as specified: [4, 1, 64, 64] x [1, 1, 64, 128]
    shape_a = [4, 1, 64, 64]
    shape_b = [1, 1, 64, 128]

    # Create random input tensors
    torch_input_a = torch.randn(shape_a, dtype=torch.bfloat16)
    torch_input_b = torch.randn(shape_b, dtype=torch.bfloat16)

    # Compute expected output using PyTorch
    torch_output = torch.matmul(torch_input_a, torch_input_b)

    # Convert to ttnn tensors with replication across the mesh
    ttnn_input_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    ttnn_input_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Call the distributed matmul operation
    ttnn_output = ttnn.distributed.matmul(ttnn_input_a, ttnn_input_b)

    # Convert back to torch for comparison
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Verify the output shape
    assert (
        ttnn_output_torch.shape == torch_output.shape
    ), f"Output shape mismatch: {ttnn_output_torch.shape} vs {torch_output.shape}"

    # Compare results with some tolerance for bfloat16
    assert torch.allclose(ttnn_output_torch, torch_output, atol=1e-2, rtol=1e-2), "Output values don't match expected"

    print(f"✓ Distributed matmul test passed!")
    print(f"  Input A shape: {shape_a}")
    print(f"  Input B shape: {shape_b}")
    print(f"  Output shape: {ttnn_output_torch.shape}")
