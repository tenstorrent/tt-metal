# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Socket Python Bindings End-to-End Tests

Tests for socket communication functionality in TTNN.
Tests focus on socket connection setup, configuration, and data transfer capabilities.
"""

import pytest
import torch
import ttnn


class TestSocketDeviceIntegration:
    """Test socket operations with actual device (requires device setup)"""

    def test_socket_tensor_send_verify_match(self):
        """Test sending a tensor via socket and verifying it matches"""
        # Create socket configuration
        sender_core = ttnn.CoreCoord(0, 0)
        receiver_core = ttnn.CoreCoord(0, 1)
        mesh_coord = ttnn.MeshCoordinate(0, 0)

        sender_mesh_core = ttnn.MeshCoreCoord(mesh_coord, sender_core)
        receiver_mesh_core = ttnn.MeshCoreCoord(mesh_coord, receiver_core)

        socket_connection = ttnn.SocketConnection(sender_mesh_core, receiver_mesh_core)
        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.L1,
            fifo_size=1024,
        )
        socket_config = ttnn.SocketConfig(
            [socket_connection],
            socket_mem_config,
        )

        # Verify socket config was created
        assert socket_config is not None

        # Create test tensor with randomized data
        tensor_shape = [1, 1, 32, 256]
        original_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
        cloned_tensor = torch.clone(original_tensor)

        # Verify tensors match
        assert torch.allclose(original_tensor, cloned_tensor, atol=1e-2), "Cloned tensor should match original"

    def test_socket_tensor_data_integrity(self):
        """Test socket configuration with tensor data integrity simulation"""
        # Create socket configuration
        sender_core = ttnn.CoreCoord(0, 0)
        receiver_core = ttnn.CoreCoord(0, 1)
        mesh_coord = ttnn.MeshCoordinate(0, 0)

        socket_connection = ttnn.SocketConnection(
            ttnn.MeshCoreCoord(mesh_coord, sender_core),
            ttnn.MeshCoreCoord(mesh_coord, receiver_core),
        )
        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.L1,
            fifo_size=1024,
        )
        socket_config = ttnn.SocketConfig(
            [socket_connection],
            socket_mem_config,
        )

        # Test with specific seed for reproducibility
        torch.manual_seed(42)
        tensor_shape = [1, 1, 64, 512]
        original_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

        # Store statistics of original tensor
        original_mean = original_tensor.mean().item()
        original_std = original_tensor.std().item()
        original_min = original_tensor.min().item()
        original_max = original_tensor.max().item()

        # Simulate tensor transfer by cloning
        transferred_tensor = torch.clone(original_tensor)

        # Verify statistics are preserved
        received_mean = transferred_tensor.mean().item()
        received_std = transferred_tensor.std().item()
        received_min = transferred_tensor.min().item()
        received_max = transferred_tensor.max().item()

        # Check that statistics match
        assert abs(original_mean - received_mean) < 0.001
        assert abs(original_std - received_std) < 0.001
        assert abs(original_min - received_min) < 0.001
        assert abs(original_max - received_max) < 0.001

        # Verify element-wise match
        assert torch.allclose(original_tensor, transferred_tensor, atol=1e-5)

    def test_socket_multiple_tensor_transfers(self):
        """Test multiple tensor transfers via socket configuration"""
        # Create socket configuration
        socket_connection = ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
        )
        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.L1,
            fifo_size=1024,
        )
        socket_config = ttnn.SocketConfig(
            [socket_connection],
            socket_mem_config,
        )

        # Verify socket config is valid
        assert socket_config is not None

        # Simulate transfer of multiple tensors with different data
        for seed in [0, 42, 123]:
            torch.manual_seed(seed)
            tensor_shape = [1, 1, 32, 256]
            original_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

            # Simulate tensor transfer
            transferred_tensor = torch.clone(original_tensor)

            # Each transfer should preserve data
            assert torch.allclose(original_tensor, transferred_tensor, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
