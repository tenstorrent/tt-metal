#!/usr/bin/env python3
"""
Test to check if memory accumulates across successive mesh device open/close cycles
"""
import os
import sys
import time

# Set up environment
os.environ["TT_ALLOC_TRACKING_ENABLED"] = "1"

import ttnn


def run_single_test():
    """Open mesh device, create a small tensor, close mesh device"""
    print("Opening mesh device...")
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))

    print("Creating small tensor...")
    import torch

    tensor = torch.randn(2, 2, 128, 128)
    ttnn_tensor = ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=ttnn.MeshShape(2, 4)),
    )

    print("Deallocating tensor...")
    ttnn_tensor.deallocate()
    del ttnn_tensor

    print("Closing mesh device...")
    ttnn.close_mesh_device(mesh_device)
    del mesh_device

    print("Waiting for cleanup...")
    time.sleep(2)
    print()


def main():
    print("=" * 70)
    print("Testing for memory accumulation across successive runs")
    print("=" * 70)
    print()
    print("Watch the allocation_monitor_client to see if DRAM increases")
    print("after each run. It should return to baseline (~0 or small amount)")
    print()

    for i in range(3):
        print(f"\n{'='*70}")
        print(f"RUN {i+1}")
        print(f"{'='*70}\n")
        run_single_test()
        print(f"✓ Run {i+1} complete. Check monitor for memory state.")
        print("  Expected: Memory should return to baseline (~0-100KB)")
        time.sleep(3)

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
    print("\nIf DRAM is still increasing after each run, there's a leak.")
    print("If DRAM returns to baseline, the fix is working correctly.")


if __name__ == "__main__":
    main()
