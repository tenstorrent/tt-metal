#!/usr/bin/env python3
"""
Add this to your simple_text_demo.py to dump trace buffer information
"""

import ttnn


def dump_trace_buffer_stats(mesh_device, stage=""):
    """
    Dump trace buffer statistics for all devices in the mesh.
    Call this at different stages of your test to see trace allocation.
    """
    num_devices = mesh_device.get_num_devices()
    devices = mesh_device.get_devices()

    print(f"\n{'='*80}")
    print(f"TRACE BUFFER STATISTICS {stage}")
    print(f"{'='*80}\n")

    total_trace_bytes = 0
    total_trace_region = 0

    for device_id, device in enumerate(devices):
        # Get current trace buffer size
        trace_size = device.get_trace_buffers_size()

        # Get allocator and config
        allocator = device.allocator()
        config = allocator.get_config()
        trace_region_size = config.trace_region_size

        # Get detailed statistics
        trace_stats = allocator.get_statistics(ttnn.BufferType.TRACE)

        total_trace_bytes += trace_size
        total_trace_region += trace_region_size

        print(f"Device {device_id}:")
        print(f"  Trace buffers size:    {trace_size:12,} bytes ({trace_size / (1024**2):7.2f} MB)")
        print(f"  Trace region size:     {trace_region_size:12,} bytes ({trace_region_size / (1024**2):7.2f} MB)")
        print(f"  Usage:                 {trace_size / trace_region_size * 100:6.1f}%")
        print(f"  Allocated:             {trace_stats.total_allocated_bytes:12,} bytes")
        print(f"  Allocatable:           {trace_stats.total_allocatable_bytes:12,} bytes")
        print(f"  Largest free:          {trace_stats.largest_free_block_bytes:12,} bytes")
        print()

    print(f"Total across all devices:")
    print(f"  Total trace buffers:   {total_trace_bytes:12,} bytes ({total_trace_bytes / (1024**2):7.2f} MB)")
    print(f"  Total trace region:    {total_trace_region:12,} bytes ({total_trace_region / (1024**2):7.2f} MB)")
    print(f"  Overall usage:         {total_trace_bytes / total_trace_region * 100:6.1f}%")
    print(f"{'='*80}\n")


def dump_trace_buffer_memory_blocks(device, device_id=0):
    """
    Dump detailed memory block information for TRACE region.
    This shows the actual allocation map.
    """
    print(f"\n{'='*80}")
    print(f"TRACE MEMORY BLOCKS - Device {device_id}")
    print(f"{'='*80}\n")

    allocator = device.allocator()

    # Get memory block table for TRACE
    blocks = allocator.get_memory_block_table(ttnn.BufferType.TRACE)

    print(f"Total blocks: {len(blocks)}")
    print(f"\n{'Address':<12} {'Size':<12} {'Status':<10}")
    print("-" * 40)

    for block in blocks[:20]:  # Show first 20 blocks
        addr = block.address
        size = block.size
        allocated = "ALLOCATED" if block.allocated else "FREE"
        print(f"0x{addr:08x}   {size:10,}   {allocated}")

    if len(blocks) > 20:
        print(f"... ({len(blocks) - 20} more blocks)")

    print()


# Example integration into simple_text_demo.py:
"""
Add these calls at key points in your test:

1. After device initialization:
   dump_trace_buffer_stats(mesh_device, stage="[AFTER INIT]")

2. After trace capture:
   trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
   output = model(input)
   ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
   dump_trace_buffer_stats(mesh_device, stage="[AFTER TRACE CAPTURE]")

3. After execution:
   for _ in range(10):
       ttnn.execute_trace(mesh_device, trace_id, blocking=False)
   dump_trace_buffer_stats(mesh_device, stage="[AFTER EXECUTION]")

4. Before cleanup:
   dump_trace_buffer_stats(mesh_device, stage="[BEFORE CLEANUP]")
   ttnn.release_trace(mesh_device, trace_id)
   dump_trace_buffer_stats(mesh_device, stage="[AFTER RELEASE]")
"""
