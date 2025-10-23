#!/usr/bin/env python3
"""
Debug script to verify L1 buffer tracking is working
Creates explicit L1 buffers and checks if they're tracked
"""

import torch
import ttnn
import os

# Enable tracking
os.environ["TT_ALLOC_TRACKING_ENABLED"] = "1"

print("=" * 60)
print("L1 Tracking Debug Test")
print("=" * 60)
print()

# Initialize device
device = ttnn.open_device(device_id=0)
print(f"✓ Opened device 0")
print()

# Create some L1 buffers of various sizes
print("Creating L1 buffers...")
print()

# Test 1: Small L1 buffer (1MB)
print("1. Creating 1MB L1 buffer...")
tensor_1mb = torch.randn(1, 1, 32, 8192, dtype=torch.bfloat16)  # ~512KB
tt_tensor_1mb = ttnn.from_torch(tensor_1mb, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
print(f"   ✓ Created L1 tensor: {tt_tensor_1mb.shape}")
print()

# Test 2: Medium L1 buffer (4MB)
print("2. Creating 4MB L1 buffer...")
tensor_4mb = torch.randn(1, 1, 64, 16384, dtype=torch.bfloat16)  # ~2MB
tt_tensor_4mb = ttnn.from_torch(tensor_4mb, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
print(f"   ✓ Created L1 tensor: {tt_tensor_4mb.shape}")
print()

# Test 3: Large L1 buffer (16MB)
print("3. Creating 16MB L1 buffer...")
tensor_16mb = torch.randn(1, 1, 128, 32768, dtype=torch.bfloat16)  # ~8MB
tt_tensor_16mb = ttnn.from_torch(tensor_16mb, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
print(f"   ✓ Created L1 tensor: {tt_tensor_16mb.shape}")
print()

print("=" * 60)
print("Check allocation server output now!")
print("You should see 3 L1 allocations with sizes:")
print("  - ~500KB-1MB")
print("  - ~2-4MB")
print("  - ~8-16MB")
print()
print("If you DON'T see these, there's an issue with buffer tracking.")
print()
print("Press Enter to deallocate and close device...")
input()

# Cleanup
print("Deallocating buffers...")
del tt_tensor_1mb
del tt_tensor_4mb
del tt_tensor_16mb
print("✓ Deallocated")
print()

ttnn.close_device(device)
print("✓ Closed device")
