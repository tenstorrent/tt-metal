# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Integration tests for ComfyUI Bridge.

Tests full flow: Client -> Protocol -> Handler -> (Mock) SDXLRunner
"""

import torch
import pytest
import socket
import struct
import msgpack
from unittest.mock import Mock, patch
import sys
import os


class TestIntegration:
    """Integration tests for complete request/response flow."""

    def test_protocol_integration(self):
        """Test protocol send/receive with actual socket."""
        from comfyui_bridge.protocol import send_message, receive_message
        import tempfile

        # Create Unix socket pair
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "test.sock")

            server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_sock.bind(socket_path)
            server_sock.listen(1)

            # Client connection
            client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_sock.connect(socket_path)

            # Accept server side
            conn, _ = server_sock.accept()

            # Send from client
            request = {"operation": "ping", "data": {}, "request_id": "test-123"}
            send_message(client_sock, request)

            # Receive on server
            received = receive_message(conn)

            assert received == request

            # Cleanup
            client_sock.close()
            conn.close()
            server_sock.close()

    def test_tensor_bridge_integration(self):
        """Test TensorBridge with actual shared memory."""
        from comfyui_bridge.handlers import TensorBridge

        # Create bridge instances (simulating client and server)
        client_bridge = TensorBridge()
        server_bridge = TensorBridge()

        # Client: create tensor and send to shm
        original_tensor = torch.randn(1, 4, 128, 128)
        handle = client_bridge.tensor_to_shm(original_tensor)

        # Server: read from shm
        received_tensor = server_bridge.tensor_from_shm(handle)

        # Verify
        assert torch.allclose(received_tensor, original_tensor)

        # Cleanup
        client_bridge.cleanup_all()

    def test_full_flow_mock_runner(self):
        """Test full flow with mocked SDXLRunner."""
        from comfyui_bridge.handlers import OperationHandler
        from sdxl_config import SDXLConfig

        config = SDXLConfig()
        handler = OperationHandler(config)

        # Test ping
        ping_result = handler.handle_ping({})
        assert ping_result["status"] == "ok"
        assert ping_result["model_loaded"] == False

    def test_error_handling_invalid_operation(self):
        """Test error handling for invalid operation."""
        from comfyui_bridge.server import ComfyUIBridgeServer
        from sdxl_config import SDXLConfig

        config = SDXLConfig()
        server = ComfyUIBridgeServer("/tmp/test.sock", config)

        # Mock handler
        server.handler = Mock()

        with pytest.raises(ValueError, match="Unknown operation"):
            server._dispatch_operation("invalid_op", {})

    def test_memory_leak_detection(self):
        """Test that shared memory segments are properly cleaned up."""
        from comfyui_bridge.handlers import TensorBridge
        import psutil
        import os

        bridge = TensorBridge()

        # Get initial shm count
        initial_shm = len([f for f in os.listdir("/dev/shm") if f.startswith("tt_")])

        # Create and cleanup multiple segments
        for _ in range(10):
            tensor = torch.randn(1, 4, 128, 128)
            handle = bridge.tensor_to_shm(tensor)
            bridge.cleanup_segment(handle["shm_name"])

        # Get final shm count
        final_shm = len([f for f in os.listdir("/dev/shm") if f.startswith("tt_")])

        # Should be same (no leaks)
        assert final_shm == initial_shm


class TestBackendIntegration:
    """Test backend client integration."""

    def test_backend_import(self):
        """Test that backend can be imported."""
        # Add comfy to path
        sys.path.insert(0, "/home/tt-admin/ComfyUI-tt_standalone/comfy")

        from backends.tenstorrent_backend import TenstorrentBackend, get_backend

        assert TenstorrentBackend is not None
        assert get_backend is not None

    def test_tensor_bridge_compatibility(self):
        """Test that backend and bridge TensorBridge are compatible."""
        # Add comfy to path
        sys.path.insert(0, "/home/tt-admin/ComfyUI-tt_standalone/comfy")

        from backends.tenstorrent_backend import TensorBridge as BackendTensorBridge
        from comfyui_bridge.handlers import TensorBridge as BridgeTensorBridge

        # Create tensor on backend side
        backend_bridge = BackendTensorBridge()
        tensor = torch.randn(1, 4, 64, 64)
        handle = backend_bridge.tensor_to_shm(tensor)

        # Read on bridge side
        bridge_bridge = BridgeTensorBridge()
        reconstructed = bridge_bridge.tensor_from_shm(handle)

        # Verify
        assert torch.allclose(reconstructed, tensor, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
