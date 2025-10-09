#!/usr/bin/env python3
"""
Test script to demonstrate real-time memory tracking.

This script allocates and deallocates buffers to show how
the memory monitor tracks changes in real-time.

Run the memory monitor in one terminal:
    ./memory_monitor -r 500

Then run this script in another terminal:
    python test_runtime_tracking.py

Watch the monitor update as memory is allocated and freed!
"""

import ttnn
import time
import torch


def print_step(step_num, description):
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {description}")
    print(f"{'='*60}")
    print("Check the memory monitor to see the changes!")
    time.sleep(3)  # Give time to observe in monitor


def main():
    print("=" * 60)
    print("Real-Time Memory Tracking Test")
    print("=" * 60)
    print("\nMake sure you have the memory monitor running:")
    print("  ./memory_monitor -r 500")
    print("\nThis script will allocate and free memory to demonstrate")
    print("real-time tracking. Watch the monitor window!")
    print("\nPress Enter to start...")
    input()

    # Step 1: Open device
    print_step(1, "Opening device (minimal memory)")
    device = ttnn.open_device(device_id=0)

    # Step 2: Allocate a small buffer
    print_step(2, "Allocating 1MB L1 buffer")
    shape = (1, 1, 32, 32)  # Small tensor
    buffer1 = ttnn.from_torch(
        torch.randn(shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Step 3: Allocate more L1 memory
    print_step(3, "Allocating 10MB L1 buffer")
    shape = (1, 1, 256, 512)  # Larger tensor
    buffer2 = ttnn.from_torch(
        torch.randn(shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Step 4: Allocate DRAM buffer
    print_step(4, "Allocating 100MB DRAM buffer")
    shape = (1, 1, 1024, 2048)  # Large tensor
    buffer3 = ttnn.from_torch(
        torch.randn(shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Step 5: Free first buffer
    print_step(5, "Deallocating 1MB L1 buffer")
    buffer1.deallocate()
    del buffer1

    # Step 6: Free second buffer
    print_step(6, "Deallocating 10MB L1 buffer")
    buffer2.deallocate()
    del buffer2

    # Step 7: Allocate multiple buffers
    print_step(7, "Allocating 5 small L1 buffers")
    buffers = []
    for i in range(5):
        shape = (1, 1, 64, 64)
        buf = ttnn.from_torch(
            torch.randn(shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        buffers.append(buf)

    # Step 8: Free DRAM buffer
    print_step(8, "Deallocating 100MB DRAM buffer")
    buffer3.deallocate()
    del buffer3

    # Step 9: Free all remaining buffers
    print_step(9, "Deallocating all remaining buffers")
    for buf in buffers:
        buf.deallocate()
    buffers.clear()

    # Step 10: Close device
    print_step(10, "Closing device (memory should be ~0%)")
    ttnn.close_device(device)

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\nYou should have seen memory utilization:")
    print("  - Increase as buffers were allocated")
    print("  - Decrease as buffers were deallocated")
    print("  - Return to near-zero at the end")
    print("\nThis demonstrates real-time memory tracking!")


if __name__ == "__main__":
    main()
