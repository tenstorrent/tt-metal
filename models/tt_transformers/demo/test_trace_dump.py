#!/usr/bin/env python3
"""
Quick test to dump trace buffer information.
Run this with your existing test environment.
"""

import pytest

import ttnn


@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000, "num_command_queues": 2}], indirect=True)
def test_dump_trace_info(mesh_device, reset_seeds):
    """Simple test to inspect trace buffer allocation"""

    print("\n" + "=" * 80)
    print("TRACE BUFFER INSPECTION TEST")
    print("=" * 80 + "\n")

    num_devices = mesh_device.get_num_devices()
    print(f"Testing with {num_devices} devices\n")

    # Stage 1: Check initial state
    print("STAGE 1: Initial state (no traces)")
    print("-" * 80)

    # Get all devices from mesh
    devices = mesh_device.get_devices()

    for device_id, device in enumerate(devices):
        trace_size = device.get_trace_buffers_size()
        config = device.allocator().get_config()

        print(f"Device {device_id}:")
        print(f"  Trace region reserved: {config.trace_region_size / (1024**2):.2f} MB")
        print(f"  Trace buffers used:    {trace_size / (1024**2):.2f} MB")
        print(f"  Address of trace buffer: Check your allocation monitor!")
        print()

    # Stage 2: Create a simple trace
    print("\nSTAGE 2: Creating a test trace")
    print("-" * 80)

    # Create a simple tensor operation to trace
    device_0 = devices[0]

    import torch

    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Capture trace
    print("Capturing trace...")
    trace_id = ttnn.begin_trace_capture(device_0, cq_id=0)
    output = ttnn.add(input_tensor, input_tensor)
    ttnn.end_trace_capture(device_0, trace_id, cq_id=0)
    print(f"Trace captured with ID: {trace_id}")

    # Check trace buffer size
    trace_size_after = device_0.get_trace_buffers_size()
    trace_stats = device_0.allocator().get_statistics(ttnn.BufferType.TRACE)

    print(f"\nDevice 0 after trace capture:")
    print(f"  Trace buffers used:    {trace_size_after / (1024**2):.2f} MB")
    print(f"  Allocated bytes:       {trace_stats.total_allocated_bytes:,}")
    print(f"  Largest free block:    {trace_stats.largest_free_block_bytes / (1024**2):.2f} MB")

    # Execute trace a few times
    print("\nExecuting trace 10 times...")
    for i in range(10):
        ttnn.execute_trace(device_0, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device_0)
    print("Execution complete")

    # Memory should be stable
    trace_size_after_exec = device_0.get_trace_buffers_size()
    print(f"Trace buffers after execution: {trace_size_after_exec / (1024**2):.2f} MB")
    print("(Should be same as after capture - trace buffers are persistent)")

    # Cleanup
    print("\nSTAGE 3: Releasing trace")
    print("-" * 80)
    ttnn.release_trace(device_0, trace_id)

    trace_size_after_release = device_0.get_trace_buffers_size()
    print(f"Trace buffers after release: {trace_size_after_release / (1024**2):.2f} MB")
    print("(Should go back to 0 if no other traces exist)")

    # Cleanup tensor
    input_tensor.deallocate()

    print("\n" + "=" * 80)
    print("TEST COMPLETE - Check your allocation monitor for buffer addresses!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run with pytest
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
