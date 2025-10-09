#!/usr/bin/env python3
"""
Dump TRACE buffer contents for inspection.
This script reads trace buffers and decodes their command structure.
"""

import ttnn
import torch
import sys
from pathlib import Path


def decode_trace_commands(data, start_idx=0, count=100):
    """
    Decode trace buffer commands (simplified decoder).
    Actual format is device-specific and complex.
    """
    print(f"\n{'='*80}")
    print(f"TRACE BUFFER COMMAND DECODE (starting at word {start_idx})")
    print(f"{'='*80}\n")

    idx = start_idx
    displayed = 0

    while idx < len(data) and displayed < count:
        word = data[idx]

        # Simple heuristic - actual format is more complex
        if word == 0:
            idx += 1
            continue

        print(f"[{idx:08x}] 0x{word:08x}", end="")

        # Try to identify command types (this is simplified)
        if (word & 0xFFFF0000) == 0xDEAD0000:
            print(" <- Possible kernel launch command")
        elif (word & 0xFF000000) == 0x01000000:
            print(" <- Possible data movement")
        elif (word & 0xFF000000) == 0x02000000:
            print(" <- Possible sync/barrier")
        else:
            print()

        idx += 1
        displayed += 1

    print()


def dump_trace_buffer_info(device, trace_id=None):
    """
    Dump information about trace buffers on a device.
    NOTE: This requires internal API access that may not be publicly available.
    """
    print(f"\n{'='*80}")
    print(f"TRACE BUFFER INFORMATION - Device {device.id()}")
    print(f"{'='*80}\n")

    # Get trace buffer size from device
    trace_size = device.get_trace_buffers_size()
    print(f"Total trace buffers size: {trace_size:,} bytes ({trace_size / (1024**2):.2f} MB)")

    # Get allocator config
    allocator = device.allocator()
    config = allocator.get_config()
    trace_region_size = config.trace_region_size

    print(f"Trace region size (reserved): {trace_region_size:,} bytes ({trace_region_size / (1024**2):.2f} MB)")
    print(f"Trace region usage: {trace_size / trace_region_size * 100:.1f}%")

    # Get trace buffer statistics
    trace_stats = allocator.get_statistics(ttnn.BufferType.TRACE)
    print(f"\nTRACE Allocator Statistics:")
    print(f"  Total allocated: {trace_stats.total_allocated_bytes:,} bytes")
    print(f"  Total allocatable: {trace_stats.total_allocatable_bytes:,} bytes")
    print(f"  Largest free block: {trace_stats.largest_free_block_bytes:,} bytes")


def dump_all_device_traces(mesh_device):
    """Dump trace info for all devices in mesh"""
    num_devices = mesh_device.get_num_devices()
    devices = mesh_device.get_devices()

    print(f"\n{'='*80}")
    print(f"TRACE BUFFERS ACROSS ALL {num_devices} DEVICES")
    print(f"{'='*80}\n")

    for device_id, device in enumerate(devices):
        dump_trace_buffer_info(device, device_id)
        print()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dump trace buffer contents")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID")
    parser.add_argument("--all", action="store_true", help="Dump all devices")
    args = parser.parse_args()

    # This is just a template - actual implementation needs device context
    print("This is a template script showing how to access trace buffer information.")
    print("To use it, integrate into your existing test that has active devices.")
    print("\nExample integration:")
    print(
        """
    # In your test:
    mesh_device = ttnn.open_mesh_device(...)

    # ... run your model with trace ...

    # Then dump trace info:
    from dump_trace_contents import dump_all_device_traces
    dump_all_device_traces(mesh_device)
    """
    )
