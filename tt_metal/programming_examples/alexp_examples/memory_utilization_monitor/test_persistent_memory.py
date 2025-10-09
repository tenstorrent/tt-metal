#!/usr/bin/env python3
"""
Persistent memory allocation test for real-time monitoring.

This script keeps buffers allocated for an extended period so you can
observe the memory utilization in the monitor.

IMPORTANT: Run this script AFTER starting the memory monitor, and keep
both running simultaneously to see real-time tracking.

Usage:
  Terminal 1: ./memory_monitor -r 500
  Terminal 2: python test_persistent_memory.py
"""

import ttnn
import time
import torch
import sys


def print_banner(message):
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70)


def main():
    print_banner("Persistent Memory Allocation Test")
    print("\n⚠️  IMPORTANT: Make sure memory_monitor is running in another terminal!")
    print("   Terminal 1: ./memory_monitor -r 500")
    print("\nThis script will allocate memory and KEEP IT ALLOCATED")
    print("so you can see the memory utilization in the monitor.\n")

    try:
        # Step 1: Get device ID (reuse existing device if monitor opened it)
        print("Opening device 0...")
        device = ttnn.open_device(device_id=0)
        print("✓ Device opened\n")
        time.sleep(2)

        # Step 2: Allocate some L1 memory
        print_banner("Step 1: Allocating L1 Memory")
        print("Allocating 4MB of L1 memory...")
        l1_buffers = []

        for i in range(4):
            shape = (1, 1, 128, 256)  # About 1MB each
            buf = ttnn.from_torch(
                torch.randn(shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            l1_buffers.append(buf)
            print(f"  Buffer {i+1}/4 allocated")

        print(f"\n✓ Allocated {len(l1_buffers)} L1 buffers (~4MB total)")
        print("📊 CHECK MONITOR: L1 usage should now be > 0%")
        print("\nWaiting 10 seconds...")
        time.sleep(10)

        # Step 3: Allocate DRAM memory
        print_banner("Step 2: Allocating DRAM Memory")
        print("Allocating 200MB of DRAM memory...")
        dram_buffers = []

        for i in range(4):
            shape = (1, 1, 1024, 1280)  # About 50MB each
            buf = ttnn.from_torch(
                torch.randn(shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            dram_buffers.append(buf)
            print(f"  Buffer {i+1}/4 allocated")

        print(f"\n✓ Allocated {len(dram_buffers)} DRAM buffers (~200MB total)")
        print("📊 CHECK MONITOR: DRAM usage should now be > 0%")
        print("\nWaiting 10 seconds...")
        time.sleep(10)

        # Step 4: Allocate more L1
        print_banner("Step 3: Allocating More L1 Memory")
        print("Allocating another 8MB of L1 memory...")

        for i in range(8):
            shape = (1, 1, 128, 256)  # About 1MB each
            buf = ttnn.from_torch(
                torch.randn(shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            l1_buffers.append(buf)
            print(f"  Buffer {i+1}/8 allocated")

        print(f"\n✓ Now have {len(l1_buffers)} L1 buffers total (~12MB)")
        print("📊 CHECK MONITOR: L1 usage should have increased")
        print("\nWaiting 10 seconds...")
        time.sleep(10)

        # Step 5: Free half the L1 buffers
        print_banner("Step 4: Deallocating Half of L1 Memory")
        half = len(l1_buffers) // 2
        print(f"Deallocating {half} L1 buffers...")

        for i in range(half):
            l1_buffers[i].deallocate()
        l1_buffers = l1_buffers[half:]

        print(f"✓ Deallocated, {len(l1_buffers)} L1 buffers remain (~6MB)")
        print("📊 CHECK MONITOR: L1 usage should have decreased")
        print("\nWaiting 10 seconds...")
        time.sleep(10)

        # Step 6: Free all DRAM
        print_banner("Step 5: Deallocating All DRAM Memory")
        print(f"Deallocating all {len(dram_buffers)} DRAM buffers...")

        for buf in dram_buffers:
            buf.deallocate()
        dram_buffers.clear()

        print("✓ All DRAM deallocated")
        print("📊 CHECK MONITOR: DRAM usage should be back to ~0%")
        print("\nWaiting 10 seconds...")
        time.sleep(10)

        # Step 7: Final cleanup
        print_banner("Step 6: Final Cleanup")
        print("Deallocating remaining L1 buffers...")

        for buf in l1_buffers:
            buf.deallocate()
        l1_buffers.clear()

        print("✓ All memory deallocated")
        print("📊 CHECK MONITOR: All usage should be back to ~0%")
        print("\nWaiting 5 seconds before closing...")
        time.sleep(5)

        # Close device
        print("\nClosing device...")
        ttnn.close_device(device)

        print_banner("Test Complete!")
        print("\n✅ If you saw memory utilization change in the monitor,")
        print("   then real-time tracking is working correctly!\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Cleaning up...")
        try:
            for buf in l1_buffers:
                buf.deallocate()
            for buf in dram_buffers:
                buf.deallocate()
            ttnn.close_device(device)
        except:
            pass
        print("Done.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
