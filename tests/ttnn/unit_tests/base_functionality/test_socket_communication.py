# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Functional tests for socket communication via nanobind bindings.

These tests verify socket functionality using:
- Python ISocket subclasses (custom socket implementations)
- Socket configuration objects

For multi-host socket tests (MeshSocket, MPISocket, BidirectionalFabricSocket),
see tests/ttnn/distributed/test_multi_mesh.py
"""

import pytest
import torch
import ttnn


# =============================================================================
# Python Loopback Socket
# =============================================================================


class PyLoopbackSocket:
    """
    A pure Python loopback socket implementation for single-host testing.

    This demonstrates how to create custom socket implementations in Python
    by subclassing the C++ ISocket interface.
    """

    # Shared buffers across all instances (keyed by channel)
    _channels = {}

    def __init__(self, channel_id: str = "default"):
        self.channel_id = channel_id
        self._rank = 0
        if channel_id not in PyLoopbackSocket._channels:
            PyLoopbackSocket._channels[channel_id] = []

    def send(self, tensor):
        """Store tensor in the channel buffer."""
        cpu_tensor = tensor.cpu()
        PyLoopbackSocket._channels[self.channel_id].append(cpu_tensor)

    def recv(self, tensor):
        """Receive tensor data into the provided tensor."""
        channel = PyLoopbackSocket._channels.get(self.channel_id, [])
        if not channel:
            raise RuntimeError(f"No data in channel '{self.channel_id}'")
        received = channel.pop(0)
        # Copy received data into the output tensor
        ttnn.copy(received.to(tensor.device()), tensor)
        return tensor

    def get_rank(self):
        return self._rank

    def get_distributed_context(self):
        return None

    @classmethod
    def clear_all_channels(cls):
        cls._channels.clear()


# =============================================================================
# Python ISocket Subclass Tests
# =============================================================================


class TestPythonISocketSubclass:
    """Tests for creating Python subclasses of ISocket."""

    @pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1_mesh")], indirect=True)
    def test_python_loopback_socket_send(self, mesh_device):
        """Test Python loopback socket send operation."""
        from ttnn._ttnn.isocket import ISocket

        class PythonLoopbackSocket(ISocket):
            _buffer = {}

            def __init__(self, channel_id: str):
                super().__init__()
                self.channel_id = channel_id

            def send(self, tensor):
                cpu_tensor = tensor.cpu()
                PythonLoopbackSocket._buffer[self.channel_id] = cpu_tensor

            def recv(self, tensor):
                pass

            def get_rank(self):
                return 0

            def get_distributed_context(self):
                return None

            def get_data(self):
                return PythonLoopbackSocket._buffer.get(self.channel_id)

        # Test send with ttnn tensor
        torch_data = torch.randn(1, 1, 32, 32, dtype=torch.float32)
        ttnn_tensor = ttnn.from_torch(
            torch_data,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )

        sender = PythonLoopbackSocket("test_py_channel")
        sender.send(ttnn_tensor)

        # Verify data was stored
        stored = sender.get_data()
        assert stored is not None
        stored_torch = ttnn.to_torch(stored)
        assert torch.allclose(stored_torch, torch_data, atol=1e-3)

    @pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1_mesh")], indirect=True)
    def test_python_loopback_send_recv(self, mesh_device):
        """Test Python loopback socket send and receive."""
        PyLoopbackSocket.clear_all_channels()

        sender = PyLoopbackSocket("device_test")
        receiver = PyLoopbackSocket("device_test")

        # Create ttnn tensor on device
        torch_input = torch.randn(1, 1, 32, 32, dtype=torch.float32)
        send_tensor = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )

        # Allocate receive buffer
        recv_tensor = ttnn.allocate_tensor_on_device(send_tensor.spec, mesh_device)

        sender.send(send_tensor)
        receiver.recv(recv_tensor)

        received_torch = ttnn.to_torch(ttnn.from_device(recv_tensor))
        assert torch.allclose(received_torch, torch_input, atol=1e-3)


# =============================================================================
# Socket Configuration Tests
# =============================================================================


class TestSocketConfig:
    """Tests for socket configuration objects."""

    def test_socket_config_creation(self):
        """Test creating socket configuration."""
        sender = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
        receiver = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1))
        conn = ttnn.SocketConnection(sender, receiver)

        mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096)
        config = ttnn.SocketConfig([conn], mem_config)

        assert config is not None

    def test_socket_config_multiple_connections(self):
        """Test socket config with multiple connections."""
        connections = []
        for i in range(4):
            sender = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(i, 0))
            receiver = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(i, 1))
            connections.append(ttnn.SocketConnection(sender, receiver))

        mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 8192)
        config = ttnn.SocketConfig(connections, mem_config)

        assert config is not None

    def test_socket_config_dram_buffer(self):
        """Test socket config with DRAM buffer type."""
        sender = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
        receiver = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1))
        conn = ttnn.SocketConnection(sender, receiver)

        mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, 16384)
        config = ttnn.SocketConfig([conn], mem_config)

        assert config is not None

    @pytest.mark.parametrize("fifo_size", [1024, 4096, 8192, 16384])
    def test_socket_config_various_fifo_sizes(self, fifo_size):
        """Test socket config with various FIFO sizes."""
        sender = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
        receiver = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1))
        conn = ttnn.SocketConnection(sender, receiver)

        mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, fifo_size)
        config = ttnn.SocketConfig([conn], mem_config)

        assert config is not None
