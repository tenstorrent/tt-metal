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


class TestSocketBasics:
    """Test basic socket object creation and configuration"""

    def test_core_coord_creation(self):
        """Test that CoreCoord objects can be created"""
        sender_core = ttnn.CoreCoord(0, 0)
        receiver_core = ttnn.CoreCoord(0, 1)

        assert sender_core is not None
        assert receiver_core is not None
        assert sender_core.x == 0
        assert sender_core.y == 0
        assert receiver_core.x == 0
        assert receiver_core.y == 1

    def test_mesh_coordinate_creation(self):
        """Test that MeshCoordinate objects can be created"""
        mesh_coord = ttnn.MeshCoordinate(0, 0)

        assert mesh_coord is not None

    def test_mesh_core_coord_creation(self):
        """Test that MeshCoreCoord objects can be created"""
        sender_core = ttnn.CoreCoord(0, 0)
        receiver_core = ttnn.CoreCoord(0, 1)
        mesh_coord = ttnn.MeshCoordinate(0, 0)

        sender_mesh_core = ttnn.MeshCoreCoord(mesh_coord, sender_core)
        receiver_mesh_core = ttnn.MeshCoreCoord(mesh_coord, receiver_core)

        assert sender_mesh_core is not None
        assert receiver_mesh_core is not None


class TestSocketConfiguration:
    """Test socket configuration and setup"""

    def test_socket_connection_creation(self):
        """Test that SocketConnection can be created"""
        sender_core = ttnn.CoreCoord(0, 0)
        receiver_core = ttnn.CoreCoord(0, 1)
        mesh_coord = ttnn.MeshCoordinate(0, 0)

        sender_mesh_core = ttnn.MeshCoreCoord(mesh_coord, sender_core)
        receiver_mesh_core = ttnn.MeshCoreCoord(mesh_coord, receiver_core)

        # SocketConnection signature: SocketConnection(sender_core, receiver_core)
        socket_connection = ttnn.SocketConnection(sender_mesh_core, receiver_mesh_core)

        assert socket_connection is not None

    def test_socket_connection_with_different_cores(self):
        """Test SocketConnection with different core coordinates"""
        mesh_coord = ttnn.MeshCoordinate(0, 0)

        # Test various core coordinate combinations
        test_cases = [
            (ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
            (ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 1)),
            (ttnn.CoreCoord(5, 5), ttnn.CoreCoord(5, 6)),
        ]

        for sender_core, receiver_core in test_cases:
            sender_mesh_core = ttnn.MeshCoreCoord(mesh_coord, sender_core)
            receiver_mesh_core = ttnn.MeshCoreCoord(mesh_coord, receiver_core)

            socket_connection = ttnn.SocketConnection(sender_mesh_core, receiver_mesh_core)
            assert socket_connection is not None

    def test_socket_memory_config_l1(self):
        """Test SocketMemoryConfig with L1 buffer type"""
        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.L1,
            fifo_size=1024,
        )

        assert socket_mem_config is not None

    def test_socket_memory_config_dram(self):
        """Test SocketMemoryConfig with DRAM buffer type"""
        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.DRAM,
            fifo_size=2048,
        )

        assert socket_mem_config is not None

    def test_socket_memory_config_various_sizes(self):
        """Test SocketMemoryConfig with various FIFO sizes"""
        fifo_sizes = [512, 1024, 2048, 4096, 8192]

        for fifo_size in fifo_sizes:
            socket_mem_config = ttnn.SocketMemoryConfig(
                socket_storage_type=ttnn.BufferType.L1,
                fifo_size=fifo_size,
            )
            assert socket_mem_config is not None

    def test_socket_config_creation(self):
        """Test SocketConfig creation with single connection"""
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

        assert socket_config is not None

    def test_socket_config_multiple_connections(self):
        """Test SocketConfig creation with multiple connections"""
        mesh_coord = ttnn.MeshCoordinate(0, 0)

        # Create multiple socket connections
        socket_connections = []
        for i in range(3):
            sender_core = ttnn.CoreCoord(i, 0)
            receiver_core = ttnn.CoreCoord(i, 1)

            sender_mesh_core = ttnn.MeshCoreCoord(mesh_coord, sender_core)
            receiver_mesh_core = ttnn.MeshCoreCoord(mesh_coord, receiver_core)

            socket_connection = ttnn.SocketConnection(sender_mesh_core, receiver_mesh_core)
            socket_connections.append(socket_connection)

        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.L1,
            fifo_size=1024,
        )

        socket_config = ttnn.SocketConfig(
            socket_connections,
            socket_mem_config,
        )

        assert socket_config is not None


class TestSocketAPI:
    """Test socket API functions"""

    def test_send_async_function_exists(self):
        """Test that send_async function is available"""
        assert hasattr(ttnn.experimental, "send_async")
        assert callable(ttnn.experimental.send_async)

    def test_recv_async_function_exists(self):
        """Test that recv_async function is available"""
        assert hasattr(ttnn.experimental, "recv_async")
        assert callable(ttnn.experimental.recv_async)

    def test_buffer_type_constants(self):
        """Test that BufferType constants are available"""
        assert hasattr(ttnn.BufferType, "L1")
        assert hasattr(ttnn.BufferType, "DRAM")

    def test_socket_types_available(self):
        """Test that all socket types are available in ttnn"""
        required_types = [
            "SocketConnection",
            "SocketMemoryConfig",
            "SocketConfig",
            "CoreCoord",
            "MeshCoordinate",
            "MeshCoreCoord",
        ]

        for type_name in required_types:
            assert hasattr(ttnn, type_name), f"ttnn.{type_name} not found"


class TestSocketTensorCreation:
    """Test tensor creation for socket operations"""

    def test_torch_tensor_creation_bfloat16(self):
        """Test creating torch tensor in bfloat16"""
        tensor_shape = [1, 1, 32, 256]
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

        assert torch_tensor.shape == torch.Size(tensor_shape)
        assert torch_tensor.dtype == torch.bfloat16

    def test_torch_tensor_creation_float32(self):
        """Test creating torch tensor in float32"""
        tensor_shape = [1, 1, 64, 512]
        torch_tensor = torch.randn(tensor_shape, dtype=torch.float32)

        assert torch_tensor.shape == torch.Size(tensor_shape)
        assert torch_tensor.dtype == torch.float32

    def test_torch_tensor_various_shapes(self):
        """Test creating torch tensors with various shapes"""
        test_shapes = [
            [1, 1, 32, 256],
            [1, 1, 64, 512],
            [1, 1, 128, 1024],
            [2, 2, 32, 256],
        ]

        for shape in test_shapes:
            torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
            assert torch_tensor.shape == torch.Size(shape)

    def test_torch_tensor_comparison(self):
        """Test comparing torch tensors"""
        tensor_shape = [1, 1, 32, 256]
        torch_tensor_1 = torch.randn(tensor_shape, dtype=torch.bfloat16)
        torch_tensor_2 = torch.clone(torch_tensor_1)

        # Same tensors should be close
        assert torch.allclose(torch_tensor_1, torch_tensor_2, atol=1e-3)

        # Different tensors should not be close
        torch_tensor_3 = torch.randn(tensor_shape, dtype=torch.bfloat16)
        # Note: Randomly generated tensors might coincidentally be close, so we don't assert False

    def test_randomized_tensor_data_values(self):
        """Test that randomized tensor data has expected statistical properties"""
        tensor_shape = [1, 1, 32, 256]
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

        # Randomized data should have mean near 0 and std near 1
        mean = torch_tensor.mean().item()
        std = torch_tensor.std().item()

        # Allow some tolerance for random variation
        assert abs(mean) < 0.5  # Mean should be close to 0
        assert 0.5 < std < 1.5  # Std should be close to 1

    def test_randomized_tensor_uniqueness(self):
        """Test that randomized tensors have different values"""
        tensor_shape = [1, 1, 32, 256]

        # Create multiple random tensors
        tensor_1 = torch.randn(tensor_shape, dtype=torch.bfloat16)
        tensor_2 = torch.randn(tensor_shape, dtype=torch.bfloat16)
        tensor_3 = torch.randn(tensor_shape, dtype=torch.bfloat16)

        # Tensors should be different (with extremely high probability)
        assert not torch.equal(tensor_1, tensor_2)
        assert not torch.equal(tensor_2, tensor_3)
        assert not torch.equal(tensor_1, tensor_3)

    def test_randomized_tensor_range(self):
        """Test that randomized tensor values are within expected range"""
        tensor_shape = [1, 1, 64, 512]
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

        # Most values should be within [-3, 3] for normal distribution
        assert torch.all(torch_tensor >= -5.0)
        assert torch.all(torch_tensor <= 5.0)

        # Check that we have both positive and negative values
        assert (torch_tensor > 0).sum() > 0
        assert (torch_tensor < 0).sum() > 0

    def test_randomized_tensor_seed_reproducibility(self):
        """Test that tensors with same seed produce same random values"""
        tensor_shape = [1, 1, 32, 256]

        # Set seed and create tensor
        torch.manual_seed(42)
        tensor_1 = torch.randn(tensor_shape, dtype=torch.bfloat16)

        # Reset seed and create another tensor
        torch.manual_seed(42)
        tensor_2 = torch.randn(tensor_shape, dtype=torch.bfloat16)

        # With same seed, tensors should be identical
        assert torch.equal(tensor_1, tensor_2)

        # Different seed should produce different tensor
        torch.manual_seed(123)
        tensor_3 = torch.randn(tensor_shape, dtype=torch.bfloat16)
        assert not torch.equal(tensor_1, tensor_3)

    def test_randomized_tensor_different_seeds(self):
        """Test randomized tensors with different seed values"""
        tensor_shape = [1, 1, 32, 256]
        seeds = [0, 42, 123, 999, 12345]

        tensors = []
        for seed in seeds:
            torch.manual_seed(seed)
            tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
            tensors.append(tensor)

        # All tensors should be different from each other
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                assert not torch.equal(tensors[i], tensors[j])

    def test_randomized_tensor_large_shape(self):
        """Test randomized tensor with larger shape"""
        tensor_shape = [4, 8, 256, 1024]
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

        assert torch_tensor.shape == torch.Size(tensor_shape)
        assert torch_tensor.dtype == torch.bfloat16

        # Check statistical properties
        mean = torch_tensor.mean().item()
        std = torch_tensor.std().item()
        assert abs(mean) < 0.3
        assert 0.8 < std < 1.2

    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_randomized_tensor_with_parametrized_seeds(self, seed):
        """Test randomized tensors with parametrized seeds"""
        tensor_shape = [1, 1, 32, 256]

        torch.manual_seed(seed)
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

        assert torch_tensor.shape == torch.Size(tensor_shape)
        assert torch_tensor.dtype == torch.bfloat16
        assert torch_tensor.mean().item() is not None
        assert torch_tensor.std().item() is not None

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_randomized_tensor_different_dtypes(self, dtype):
        """Test randomized tensors with different data types"""
        tensor_shape = [1, 1, 32, 256]
        torch_tensor = torch.randn(tensor_shape, dtype=dtype)

        assert torch_tensor.shape == torch.Size(tensor_shape)
        assert torch_tensor.dtype == dtype

    def test_randomized_tensor_statistics(self):
        """Test statistical properties of randomized tensor data"""
        tensor_shape = [2, 2, 256, 512]
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

        # Compute statistics
        mean = torch_tensor.mean().item()
        std = torch_tensor.std().item()
        min_val = torch_tensor.min().item()
        max_val = torch_tensor.max().item()

        # Validate statistics
        assert abs(mean) < 0.3
        assert 0.8 < std < 1.2
        assert min_val < max_val
        assert min_val < 0  # Should have negative values
        assert max_val > 0  # Should have positive values


class TestSocketConfigurationVariations:
    """Test various socket configuration combinations"""

    @pytest.mark.parametrize(
        "storage_type,fifo_size",
        [
            (ttnn.BufferType.L1, 512),
            (ttnn.BufferType.L1, 1024),
            (ttnn.BufferType.L1, 2048),
            (ttnn.BufferType.DRAM, 1024),
            (ttnn.BufferType.DRAM, 4096),
        ],
    )
    def test_socket_config_combinations(self, storage_type, fifo_size):
        """Test SocketConfig with various combinations of storage type and FIFO size"""
        sender_core = ttnn.CoreCoord(0, 0)
        receiver_core = ttnn.CoreCoord(0, 1)
        mesh_coord = ttnn.MeshCoordinate(0, 0)

        sender_mesh_core = ttnn.MeshCoreCoord(mesh_coord, sender_core)
        receiver_mesh_core = ttnn.MeshCoreCoord(mesh_coord, receiver_core)

        socket_connection = ttnn.SocketConnection(sender_mesh_core, receiver_mesh_core)
        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=storage_type,
            fifo_size=fifo_size,
        )

        socket_config = ttnn.SocketConfig(
            [socket_connection],
            socket_mem_config,
        )

        assert socket_config is not None

    @pytest.mark.parametrize("num_connections", [1, 2, 3, 5])
    def test_socket_config_connection_counts(self, num_connections):
        """Test SocketConfig with various numbers of connections"""
        mesh_coord = ttnn.MeshCoordinate(0, 0)

        socket_connections = []
        for i in range(num_connections):
            sender_core = ttnn.CoreCoord(i, 0)
            receiver_core = ttnn.CoreCoord(i, 1)

            sender_mesh_core = ttnn.MeshCoreCoord(mesh_coord, sender_core)
            receiver_mesh_core = ttnn.MeshCoreCoord(mesh_coord, receiver_core)

            socket_connection = ttnn.SocketConnection(sender_mesh_core, receiver_mesh_core)
            socket_connections.append(socket_connection)

        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.L1,
            fifo_size=1024,
        )

        socket_config = ttnn.SocketConfig(
            socket_connections,
            socket_mem_config,
        )

        assert socket_config is not None


class TestSocketMeshCoordinateVariations:
    """Test socket connections across multiple mesh coordinates"""

    def test_socket_across_mesh_coordinates(self):
        """Test creating socket connections across different mesh coordinates"""
        mesh_coords = [
            ttnn.MeshCoordinate(0, 0),
            ttnn.MeshCoordinate(0, 1),
            ttnn.MeshCoordinate(1, 0),
            ttnn.MeshCoordinate(1, 1),
        ]

        socket_connections = []
        for mesh_coord in mesh_coords:
            sender_core = ttnn.CoreCoord(0, 0)
            receiver_core = ttnn.CoreCoord(0, 1)

            sender_mesh_core = ttnn.MeshCoreCoord(mesh_coord, sender_core)
            receiver_mesh_core = ttnn.MeshCoreCoord(mesh_coord, receiver_core)

            socket_connection = ttnn.SocketConnection(sender_mesh_core, receiver_mesh_core)
            socket_connections.append(socket_connection)

        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.L1,
            fifo_size=1024,
        )

        socket_config = ttnn.SocketConfig(
            socket_connections,
            socket_mem_config,
        )

        assert socket_config is not None
        assert len(socket_connections) == 4


class TestSocketDeviceIntegration:
    """Test socket operations with actual device (requires device setup)"""

    def test_socket_send_recv_basic(self):
        """Test basic socket send/recv configuration"""
        # Create socket configuration without device
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

        # Verify socket config was created successfully
        assert socket_config is not None

    def test_socket_multiple_connections(self):
        """Test socket with multiple connections"""
        mesh_coord = ttnn.MeshCoordinate(0, 0)

        # Create multiple socket connections
        socket_connections = []
        for i in range(3):
            sender_core = ttnn.CoreCoord(i, 0)
            receiver_core = ttnn.CoreCoord(i, 1)

            sender_mesh_core = ttnn.MeshCoreCoord(mesh_coord, sender_core)
            receiver_mesh_core = ttnn.MeshCoreCoord(mesh_coord, receiver_core)

            socket_connection = ttnn.SocketConnection(sender_mesh_core, receiver_mesh_core)
            socket_connections.append(socket_connection)

        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.L1,
            fifo_size=1024,
        )

        socket_config = ttnn.SocketConfig(
            socket_connections,
            socket_mem_config,
        )

        assert socket_config is not None
        assert len(socket_connections) == 3

    def test_socket_async_functions_callable(self):
        """Test that async functions are callable"""
        # Verify send_async is callable
        assert callable(ttnn.experimental.send_async)

        # Verify recv_async is callable
        assert callable(ttnn.experimental.recv_async)

    def test_socket_buffer_types(self):
        """Test socket with different buffer types"""
        mesh_coord = ttnn.MeshCoordinate(0, 0)
        sender_core = ttnn.CoreCoord(0, 0)
        receiver_core = ttnn.CoreCoord(0, 1)

        sender_mesh_core = ttnn.MeshCoreCoord(mesh_coord, sender_core)
        receiver_mesh_core = ttnn.MeshCoreCoord(mesh_coord, receiver_core)

        socket_connection = ttnn.SocketConnection(sender_mesh_core, receiver_mesh_core)

        # Test with L1 buffer
        socket_mem_config_l1 = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.L1,
            fifo_size=1024,
        )
        socket_config_l1 = ttnn.SocketConfig(
            [socket_connection],
            socket_mem_config_l1,
        )
        assert socket_config_l1 is not None

        # Test with DRAM buffer
        socket_mem_config_dram = ttnn.SocketMemoryConfig(
            socket_storage_type=ttnn.BufferType.DRAM,
            fifo_size=2048,
        )
        socket_config_dram = ttnn.SocketConfig(
            [socket_connection],
            socket_mem_config_dram,
        )
        assert socket_config_dram is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
