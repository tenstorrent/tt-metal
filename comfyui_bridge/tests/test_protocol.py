# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Unit tests for protocol.py - message framing and serialization.
"""

import socket
import struct
import msgpack
import pytest
from io import BytesIO


# Mock socket class for testing
class MockSocket:
    """Mock socket for testing without network I/O."""

    def __init__(self):
        self.send_buffer = BytesIO()
        self.recv_buffer = BytesIO()
        self.closed = False

    def sendall(self, data):
        if self.closed:
            raise ConnectionError("Socket is closed")
        self.send_buffer.write(data)

    def recv(self, n):
        if self.closed:
            return b""
        return self.recv_buffer.read(n)

    def close(self):
        self.closed = True

    def reset_buffers(self):
        """Reset buffers for next test."""
        self.send_buffer = BytesIO()
        self.recv_buffer = BytesIO()

    def prepare_recv_data(self, data):
        """Prepare data in recv buffer."""
        self.recv_buffer = BytesIO(data)


class TestProtocol:
    """Test protocol message framing and serialization."""

    def test_send_message_basic(self):
        """Test sending a basic message."""
        from comfyui_bridge.protocol import send_message

        sock = MockSocket()
        data = {"operation": "ping", "data": {}}

        send_message(sock, data)

        # Verify sent data
        sent = sock.send_buffer.getvalue()
        assert len(sent) > 4  # Has length prefix + data

        # Parse length prefix
        length = struct.unpack(">I", sent[:4])[0]
        assert length == len(sent) - 4

        # Parse message
        msg = msgpack.unpackb(sent[4:], raw=False)
        assert msg == data

    def test_send_message_complex(self):
        """Test sending a complex message with nested data."""
        from comfyui_bridge.protocol import send_message

        sock = MockSocket()
        data = {
            "operation": "init_model",
            "data": {
                "model_type": "sdxl",
                "config": {"device_id": 0, "steps": 50},
                "nested": {"list": [1, 2, 3], "dict": {"key": "value"}},
            },
            "request_id": "test-123",
        }

        send_message(sock, data)

        sent = sock.send_buffer.getvalue()
        msg = msgpack.unpackb(sent[4:], raw=False)
        assert msg == data

    def test_receive_message_basic(self):
        """Test receiving a basic message."""
        from comfyui_bridge.protocol import receive_message

        sock = MockSocket()
        data = {"operation": "ping", "data": {}}

        # Prepare message in recv buffer
        msg_bytes = msgpack.packb(data, use_bin_type=True)
        length_bytes = struct.pack(">I", len(msg_bytes))
        sock.prepare_recv_data(length_bytes + msg_bytes)

        # Receive message
        received = receive_message(sock)
        assert received == data

    def test_receive_message_large(self):
        """Test receiving a large message (multiple recv calls)."""
        from comfyui_bridge.protocol import receive_message

        sock = MockSocket()

        # Create large message (> 4KB)
        large_data = {
            "operation": "full_denoise",
            "data": {
                "latent": {"shm_name": "test", "shape": [1, 4, 128, 128]},
                "large_field": "x" * 10000,  # 10KB string
            },
        }

        msg_bytes = msgpack.packb(large_data, use_bin_type=True)
        length_bytes = struct.pack(">I", len(msg_bytes))
        sock.prepare_recv_data(length_bytes + msg_bytes)

        received = receive_message(sock)
        assert received == large_data

    def test_receive_message_connection_closed(self):
        """Test handling of connection closed during receive."""
        from comfyui_bridge.protocol import receive_message

        sock = MockSocket()
        sock.close()

        with pytest.raises(RuntimeError, match="connection closed"):
            receive_message(sock)

    def test_receive_message_invalid_msgpack(self):
        """Test handling of invalid msgpack data."""
        from comfyui_bridge.protocol import receive_message

        sock = MockSocket()

        # Send valid length but invalid msgpack data
        invalid_data = b"this is not msgpack data"
        length_bytes = struct.pack(">I", len(invalid_data))
        sock.prepare_recv_data(length_bytes + invalid_data)

        with pytest.raises(RuntimeError, match="Invalid msgpack"):
            receive_message(sock)

    def test_receive_message_oversized(self):
        """Test rejection of oversized messages."""
        from comfyui_bridge.protocol import receive_message

        sock = MockSocket()

        # Send length > 100MB
        oversized_length = 101 * 1024 * 1024
        length_bytes = struct.pack(">I", oversized_length)
        sock.prepare_recv_data(length_bytes)

        with pytest.raises(RuntimeError, match="exceeds 100MB limit"):
            receive_message(sock)

    def test_send_error(self):
        """Test sending error response."""
        from comfyui_bridge.protocol import send_error

        sock = MockSocket()
        error_msg = "Test error message"

        send_error(sock, error_msg)

        sent = sock.send_buffer.getvalue()
        msg = msgpack.unpackb(sent[4:], raw=False)

        assert msg["status"] == "error"
        assert msg["error"] == error_msg
        assert "data" in msg

    def test_send_success(self):
        """Test sending success response."""
        from comfyui_bridge.protocol import send_success

        sock = MockSocket()
        result_data = {"model_id": "sdxl_123", "status": "ready"}

        send_success(sock, result_data)

        sent = sock.send_buffer.getvalue()
        msg = msgpack.unpackb(sent[4:], raw=False)

        assert msg["status"] == "success"
        assert msg["error"] == ""
        assert msg["data"] == result_data

    def test_round_trip(self):
        """Test full send -> receive round trip."""
        from comfyui_bridge.protocol import send_message, receive_message

        send_sock = MockSocket()
        data = {"operation": "init_model", "data": {"model_type": "sdxl"}, "request_id": "round-trip-test"}

        # Send
        send_message(send_sock, data)

        # Prepare receive with sent data
        recv_sock = MockSocket()
        recv_sock.prepare_recv_data(send_sock.send_buffer.getvalue())

        # Receive
        received = receive_message(recv_sock)

        assert received == data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
