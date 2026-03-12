# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
MessagePack protocol implementation for ComfyUI Bridge.

Implements length-prefixed binary protocol:
1. Send/receive 4-byte length prefix (big-endian unsigned int)
2. Send/receive message body (msgpack-encoded data)

Compatible with ComfyUI tenstorrent_backend.py client implementation.
"""

import struct
import socket
import msgpack
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    """
    Receive a msgpack-encoded message from socket.

    Protocol:
        [4 bytes: length (big-endian)] [N bytes: msgpack data]

    Args:
        sock: Connected socket to receive from

    Returns:
        Deserialized message dictionary

    Raises:
        RuntimeError: If connection closed or receive failed
    """
    # Receive 4-byte length prefix
    length_bytes = _recv_exactly(sock, 4)
    if len(length_bytes) < 4:
        raise RuntimeError("Failed to receive message length (connection closed)")

    # Unpack length (big-endian unsigned int)
    msg_length = struct.unpack(">I", length_bytes)[0]

    # Sanity check: reject messages > 100MB (likely corrupted)
    if msg_length > 100 * 1024 * 1024:
        raise RuntimeError(f"Message length {msg_length} exceeds 100MB limit")

    # Receive message body
    msg_bytes = _recv_exactly(sock, msg_length)
    if len(msg_bytes) < msg_length:
        raise RuntimeError(f"Connection closed while receiving message (got {len(msg_bytes)}/{msg_length} bytes)")

    # Deserialize with msgpack
    try:
        message = msgpack.unpackb(msg_bytes, raw=False)
        logger.debug(f"Received message: operation={message.get('operation')}, size={msg_length} bytes")
        return message
    except Exception as e:
        logger.error(f"Failed to deserialize message: {e}")
        raise RuntimeError(f"Invalid msgpack data: {e}")


def send_message(sock: socket.socket, data: Dict[str, Any]) -> None:
    """
    Send a msgpack-encoded message to socket.

    Protocol:
        [4 bytes: length (big-endian)] [N bytes: msgpack data]

    Args:
        sock: Connected socket to send to
        data: Dictionary to serialize and send

    Raises:
        RuntimeError: If send failed
    """
    try:
        # Serialize with msgpack (use_bin_type=True for compatibility)
        msg_bytes = msgpack.packb(data, use_bin_type=True)
        msg_length = len(msg_bytes)

        # Pack length as big-endian unsigned int
        length_bytes = struct.pack(">I", msg_length)

        # Send length + message
        sock.sendall(length_bytes + msg_bytes)
        logger.debug(f"Sent message: size={msg_length} bytes")

    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise RuntimeError(f"Send failed: {e}")


def send_error(sock: socket.socket, error_msg: str) -> None:
    """
    Send an error response to client.

    Response format:
        {
            "status": "error",
            "error": "<error message>",
            "data": {}
        }

    Args:
        sock: Connected socket
        error_msg: Error message to send
    """
    response = {"status": "error", "error": str(error_msg), "data": {}}
    logger.warning(f"Sending error response: {error_msg}")
    send_message(sock, response)


def send_success(sock: socket.socket, data: Optional[Dict[str, Any]] = None) -> None:
    """
    Send a success response to client.

    Response format:
        {
            "status": "success",
            "error": "",
            "data": <result data>
        }

    Args:
        sock: Connected socket
        data: Result data dictionary (default: {})
    """
    response = {"status": "success", "error": "", "data": data or {}}
    logger.debug(f"Sending success response with {len(data or {})} data fields")
    send_message(sock, response)


def _recv_exactly(sock: socket.socket, n_bytes: int) -> bytes:
    """
    Receive exactly n_bytes from socket.

    Helper function that handles partial receives.

    Args:
        sock: Socket to receive from
        n_bytes: Number of bytes to receive

    Returns:
        Exactly n_bytes of data (or less if connection closed)
    """
    data = b""
    while len(data) < n_bytes:
        chunk = sock.recv(min(4096, n_bytes - len(data)))
        if not chunk:
            # Connection closed
            break
        data += chunk
    return data
