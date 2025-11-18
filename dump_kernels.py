#!/usr/bin/env python3
"""
Simple script to dump detailed kernel information from the allocation server.
Shows which kernels are currently loaded, their sizes, and which process owns them.
"""

import socket
import struct
import sys

TT_ALLOC_SERVER_SOCKET = "/tmp/tt_alloc_server.sock"

# Message format matches C++ AllocMessage struct
# B = uint8_t, I = uint32_t, Q = uint64_t, i = int32_t
MESSAGE_FORMAT = "B3xi Q B3x i Q Q" + "Q" * 8  # 104 bytes total


def send_dump_kernels():
    """Send DUMP_KERNELS command to the allocation server"""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(TT_ALLOC_SERVER_SOCKET)

        # Create DUMP_KERNELS message (type = 12)
        message = struct.pack(
            MESSAGE_FORMAT,
            12,  # DUMP_KERNELS type
            -1,  # device_id (unused)
            0,  # size (unused)
            0,  # buffer_type (unused)
            0,  # process_id (unused)
            0,  # buffer_id (unused)
            0,  # timestamp (unused)
            # 8 response fields (all zeros)
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

        sock.send(message)
        sock.close()

        print("✓ DUMP_KERNELS command sent to server")
        print("  Check the server log (out.log) for detailed kernel information")

    except FileNotFoundError:
        print("❌ Error: Allocation server socket not found")
        print("   Make sure the allocation server is running:")
        print("   ./build/programming_examples/allocation_server_poc > out.log 2>&1 &")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    send_dump_kernels()
