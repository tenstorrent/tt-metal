#!/usr/bin/env python3
"""
Simple test to verify that disable_and_clear_program_cache()
actually reports buffer deallocations to the allocation server.
"""

import os
import time
import torch
import ttnn

# Enable tracking
os.environ["TT_ALLOC_TRACKING_ENABLED"] = "1"

print("=" * 80)
print("Testing Program Cache Clearing")
print("=" * 80)
print()

print(f"PID: {os.getpid()}")
print()

# Step 1: Open mesh device
print("[1] Opening 8-device mesh...")
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
print(f"✓ Mesh opened: {mesh_device}")
time.sleep(2)

# Step 2: Create a simple tensor and perform operations
print("\n[2] Creating tensor and performing operations...")
torch_a = torch.randn(8, 8, 256, 256, dtype=torch.bfloat16)
torch_b = torch.randn(8, 8, 256, 256, dtype=torch.bfloat16)

ttnn_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
ttnn_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

mapper = ttnn.create_mesh_mapper(
    mesh_device,
    ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(0), ttnn.PlacementShard(1)],
        ttnn.MeshShape(2, 4),
    ),
)

dist_a = ttnn.distribute_tensor(ttnn_a, mapper, mesh_device)
dist_b = ttnn.distribute_tensor(ttnn_b, mapper, mesh_device)

print("  Computing: add...")
result_add = ttnn.add(dist_a, dist_b)
print("  ✓ Add complete")

print("  Computing: matmul...")
result_matmul = ttnn.matmul(dist_a, dist_b)
print("  ✓ MatMul complete")

print("\n⚠️  CHECK SERVER: Should see ~36KB cached program buffers")
time.sleep(3)

# Step 3: Deallocate tensors
print("\n[3] Deallocating tensors...")
ttnn.deallocate(result_add)
ttnn.deallocate(result_matmul)
ttnn.deallocate(dist_a)
ttnn.deallocate(dist_b)
print("✓ Tensors deallocated")

print("\n⚠️  CHECK SERVER: Tensors freed, but ~36KB cached buffers remain")
time.sleep(3)

# Step 4: Clear program cache - THIS IS THE KEY TEST
print("\n[4] Clearing program cache...")
print("    Calling: mesh_device.disable_and_clear_program_cache()")

mesh_device.disable_and_clear_program_cache()

print("✓ Program cache cleared")

print("\n🔍 CHECK SERVER NOW: Did the 36KB get deallocated?")
print("    If YES: Deallocations are being tracked correctly ✓")
print("    If NO:  Something is wrong with MeshBuffer deallocation tracking ✗")
time.sleep(5)

# Step 5: Close mesh device
print("\n[5] Closing mesh device...")
ttnn.close_mesh_device(mesh_device)
print("✓ Mesh closed")

time.sleep(2)

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
print()
print("Summary:")
print("  1. Created tensors and ran add/matmul (creates 36KB cached programs)")
print("  2. Deallocated tensors (36KB cached programs remain)")
print("  3. Called disable_and_clear_program_cache()")
print("  4. ⚠️  DID THE SERVER SEE DEALLOCATIONS FOR THE 36KB?")
print()
