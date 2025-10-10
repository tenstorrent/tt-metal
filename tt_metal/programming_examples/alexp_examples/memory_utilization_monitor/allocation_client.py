#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Python client for Allocation Server

This demonstrates how Python applications can report allocations
to the allocation server for cross-process tracking.
"""

import socket
import struct
import os
import time
from enum import IntEnum


class MessageType(IntEnum):
    """Message types for allocation server protocol"""

    ALLOC = 1
    FREE = 2
    QUERY = 3
    RESPONSE = 4
    DUMP_REMAINING = 5
    DEVICE_INFO_QUERY = 6
    DEVICE_INFO_RESPONSE = 7


class BufferType(IntEnum):
    """Buffer types matching TT-Metal BufferType enum"""

    DRAM = 0
    L1 = 1
    L1_SMALL = 2
    TRACE = 3


class AllocationClient:
    """
    Client for reporting allocations to the allocation server.

    Usage:
        client = AllocationClient()
        buffer_id = client.allocate(device_id=0, size=1024*1024, buffer_type=BufferType.L1)
        # ... use buffer ...
        client.deallocate(buffer_id)
    """

    SOCKET_PATH = "/tmp/tt_allocation_server.sock"

    # Message format: Type(u8), device_id(i32), size(u64), buffer_type(u8),
    #                 process_id(i32), buffer_id(u64), timestamp(u64),
    #                 [response fields: 4x u64 + device info fields: 2x u64 + 6x u32]
    # Total: 1 + 3(pad) + 4 + 8 + 1 + 3(pad) + 4 + 8 + 8 + 32 + 16 + 24 = 112 bytes
    MESSAGE_FORMAT = "=BxxxiQBxxxiQQQQQQQQIIIIII"  # Little-endian

    def __init__(self):
        """Connect to allocation server"""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.SOCKET_PATH)
            print(f"✓ Connected to allocation server")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to allocation server: {e}\n" f"Make sure allocation_server_poc is running!"
            )

        self.next_buffer_id = 1
        self.process_id = os.getpid()

    def __del__(self):
        """Close connection"""
        if hasattr(self, "sock"):
            self.sock.close()

    def allocate(self, device_id: int, size: int, buffer_type: BufferType) -> int:
        """
        Report an allocation to the server.

        Args:
            device_id: Device ID (0-7)
            size: Size in bytes
            buffer_type: Type of buffer (DRAM, L1, etc.)

        Returns:
            buffer_id: Unique identifier for this allocation
        """
        buffer_id = self.next_buffer_id
        self.next_buffer_id += 1

        timestamp = int(time.time() * 1e9)  # nanoseconds

        msg = struct.pack(
            self.MESSAGE_FORMAT,
            MessageType.ALLOC,  # type
            device_id,  # device_id
            size,  # size
            buffer_type,  # buffer_type
            self.process_id,  # process_id
            buffer_id,  # buffer_id
            timestamp,  # timestamp
            0,
            0,
            0,
            0,  # response fields (unused)
            0,
            0,  # device info: total_dram_size, total_l1_size
            0,
            0,
            0,
            0,
            0,
            0,  # device info: arch_type, num_dram_channels, dram_size_per_channel, l1_size_per_core, is_available, num_devices
        )

        self.sock.send(msg)
        return buffer_id

    def deallocate(self, buffer_id: int):
        """
        Report a deallocation to the server.

        Args:
            buffer_id: The buffer ID returned from allocate()
        """
        msg = struct.pack(
            self.MESSAGE_FORMAT,
            MessageType.FREE,  # type
            0,  # device_id (unused)
            0,  # size (unused)
            0,  # buffer_type (unused)
            self.process_id,  # process_id
            buffer_id,  # buffer_id
            0,  # timestamp (unused)
            0,
            0,
            0,
            0,  # response fields (unused)
            0,
            0,  # device info fields (unused)
            0,
            0,
            0,
            0,
            0,
            0,  # device info fields (unused)
        )

        self.sock.send(msg)

    def query_device(self, device_id: int) -> dict:
        """
        Query memory statistics for a device.

        Args:
            device_id: Device ID to query

        Returns:
            dict with keys: dram_allocated, l1_allocated, l1_small_allocated,
                          trace_allocated, total_buffers
        """
        msg = struct.pack(
            self.MESSAGE_FORMAT,
            MessageType.QUERY,  # type
            device_id,  # device_id
            0,
            0,
            0,
            0,
            0,  # unused fields
            0,
            0,
            0,
            0,  # response fields (will be filled)
            0,
            0,  # device info fields (will be filled)
            0,
            0,
            0,
            0,
            0,
            0,  # device info fields (will be filled)
        )

        self.sock.send(msg)

        # Receive response
        response = self.sock.recv(struct.calcsize(self.MESSAGE_FORMAT))
        data = struct.unpack(self.MESSAGE_FORMAT, response)

        return {
            "device_id": data[1],
            "dram_allocated": data[7],
            "l1_allocated": data[8],
            "l1_small_allocated": data[9],
            "trace_allocated": data[10],
        }

    def get_num_devices(self) -> int:
        """
        Query the number of available devices.

        Returns:
            Number of devices detected by the server
        """
        msg = struct.pack(
            self.MESSAGE_FORMAT,
            MessageType.DEVICE_INFO_QUERY,  # type
            -1,  # device_id = -1 means query for device count
            0,
            0,
            0,
            0,
            0,  # unused fields
            0,
            0,
            0,
            0,  # response fields
            0,
            0,  # device info fields (will be filled)
            0,
            0,
            0,
            0,
            0,
            0,  # device info fields (will be filled)
        )

        self.sock.send(msg)

        # Receive response
        response = self.sock.recv(struct.calcsize(self.MESSAGE_FORMAT))
        data = struct.unpack(self.MESSAGE_FORMAT, response)

        # num_devices is at index 17 (last field)
        return data[17]

    def query_device_info(self, device_id: int) -> dict:
        """
        Query detailed device information.

        Args:
            device_id: Device ID to query (or -1 for device count)

        Returns:
            dict with device capabilities: arch_type, total_dram_size, total_l1_size,
            num_dram_channels, dram_size_per_channel, l1_size_per_core, is_available
        """
        msg = struct.pack(
            self.MESSAGE_FORMAT,
            MessageType.DEVICE_INFO_QUERY,  # type
            device_id,  # device_id
            0,
            0,
            0,
            0,
            0,  # unused fields
            0,
            0,
            0,
            0,  # response fields
            0,
            0,  # device info fields (will be filled)
            0,
            0,
            0,
            0,
            0,
            0,  # device info fields (will be filled)
        )

        self.sock.send(msg)

        # Receive response
        response = self.sock.recv(struct.calcsize(self.MESSAGE_FORMAT))
        data = struct.unpack(self.MESSAGE_FORMAT, response)

        arch_names = {0: "Invalid", 1: "Grayskull", 2: "Wormhole_B0", 3: "Blackhole", 4: "Quasar"}

        return {
            "device_id": data[1],
            "is_available": bool(data[16]),
            "arch_type": data[12],
            "arch_name": arch_names.get(data[12], "Unknown"),
            "total_dram_size": data[11],
            "total_l1_size": data[10],  # Note: indices shifted for Q vs I
            "num_dram_channels": data[13],
            "dram_size_per_channel": data[14],
            "l1_size_per_core": data[15],
            "num_devices": data[17],
        }


def format_bytes(bytes_val: int) -> str:
    """Format bytes in human-readable form"""
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    size = float(bytes_val)

    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1

    return f"{size:.2f} {units[unit_idx]}"


def demo():
    """Demo showing how to use the allocation client"""
    print("\n🧪 Python Allocation Client Demo")
    print(f"   PID: {os.getpid()}")
    print(f"   Watch the server and monitor output!\n")

    try:
        client = AllocationClient()
        buffers = []

        # Step 1: Allocate some L1
        print("[Step 1] Allocating 4MB of L1...")
        for i in range(4):
            buffer_id = client.allocate(device_id=0, size=1024 * 1024, buffer_type=BufferType.L1)
            buffers.append(buffer_id)
            print(f"  Allocated buffer {buffer_id}")
        time.sleep(3)

        # Step 2: Allocate some DRAM
        print("\n[Step 2] Allocating 100MB of DRAM...")
        for i in range(4):
            buffer_id = client.allocate(device_id=0, size=25 * 1024 * 1024, buffer_type=BufferType.DRAM)
            buffers.append(buffer_id)
            print(f"  Allocated buffer {buffer_id}")
        time.sleep(3)

        # Step 3: Query stats
        print("\n[Step 3] Querying current statistics...")
        stats = client.query_device(device_id=0)
        print(f"  Device 0:")
        print(f"    DRAM: {format_bytes(stats['dram_allocated'])}")
        print(f"    L1: {format_bytes(stats['l1_allocated'])}")
        time.sleep(3)

        # Step 4: Free half
        print("\n[Step 4] Freeing half the buffers...")
        half = len(buffers) // 2
        for buffer_id in buffers[:half]:
            client.deallocate(buffer_id)
            print(f"  Freed buffer {buffer_id}")
        buffers = buffers[half:]
        time.sleep(3)

        # Step 5: Free all
        print("\n[Step 5] Freeing all remaining buffers...")
        for buffer_id in buffers:
            client.deallocate(buffer_id)
            print(f"  Freed buffer {buffer_id}")
        buffers.clear()
        time.sleep(2)

        print("\n✅ Demo complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(demo())
