#!/usr/bin/env python3
"""
Simple script to request the allocation server to dump remaining buffers.
Usage: python3 dump_remaining_buffers.py
"""

import socket
import struct
import sys

SOCKET_PATH = "/tmp/tt_allocation_server.sock"

# Message types
DUMP_REMAINING = 5


def dump_remaining_buffers():
    """Request the server to dump all remaining allocated buffers."""
    try:
        # Connect to server
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)

        # Create DUMP_REMAINING message
        # struct format: B 3x i Q B 3x i Q Q 4Q (72 bytes total)
        msg = struct.pack(
            "B3xiQB3xiQQ4Q",
            DUMP_REMAINING,  # type
            0,  # device_id (unused for dump)
            0,  # size (unused)
            0,  # buffer_type (unused)
            0,  # process_id (unused)
            0,  # buffer_id (unused)
            0,  # timestamp (unused)
            0,
            0,
            0,
            0,  # response fields (unused)
        )

        sock.send(msg)
        print("✓ Dump request sent to allocation server")
        print("  Check the server output for remaining buffers\n")

        sock.close()

    except FileNotFoundError:
        print(f"❌ Error: Allocation server socket not found at {SOCKET_PATH}")
        print("   Make sure the allocation server is running.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    dump_remaining_buffers()
