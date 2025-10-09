#!/usr/bin/env python3
"""
Simple test to verify if 36KB DRAM leak accumulates across iterations.
Watch the allocation_server_poc output to see if "Active allocations" increases.
"""
import os
import time

os.environ["TT_ALLOC_TRACKING_ENABLED"] = "1"

import ttnn

print("=" * 70)
print("LEAK ACCUMULATION TEST")
print("=" * 70)
print("Watch the allocation_server_poc output:")
print("  - If 'Active allocations' increases: REAL LEAK (physical memory not freed)")
print("  - If 'Active allocations' stays same: TRACKING BUG (double-counting)")
print("=" * 70)
print()

for iteration in range(5):
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration + 1}/5")
    print(f"{'='*70}")

    print("Opening mesh device...")
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))

    print("Closing mesh device...")
    ttnn.close_mesh_device(mesh_device)
    del mesh_device

    print(f"✓ Iteration {iteration + 1} complete")
    print("Waiting 3 seconds for cleanup to propagate...")
    time.sleep(3)

    print(f"\nCheck allocation_server_poc now:")
    print(f"  Expected if NO leak: ~16-32 active allocations")
    print(f"  Expected if LEAK:    {16 * (iteration + 1)} active allocations")
    print()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("\nFinal check:")
print("  - Look at allocation_server_poc 'Active allocations' count")
print("  - If it's ~16-32: Just a tracking issue, physical memory is freed")
print("  - If it's 80 (16×5): REAL LEAK, physical memory is NOT freed")
print("=" * 70)
