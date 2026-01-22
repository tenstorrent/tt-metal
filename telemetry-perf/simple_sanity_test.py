#!/usr/bin/env python3
"""
Simple sanity test for single-device workload with telemetry
"""
import time
import ttnn
import torch

print("=== Simple Single-Device Sanity Test ===\n")

# Open single device properly
print("Opening device 0...")
device = ttnn.open_device(device_id=0)
print(f"Device opened: {device}\n")

# Create small tensors
print("Creating 512x512 tensor...")
shape = (1, 1, 512, 512)
input_tensor = torch.randn(shape)

# Transfer to device
print("Transferring to device...")
tt_input = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

# Run a few simple ops and measure time
print("\nRunning 5 iterations of add + multiply ops...")
times = []

for i in range(5):
    start = time.perf_counter()

    # Simple ops
    result = ttnn.add(tt_input, tt_input)
    result = ttnn.multiply(result, tt_input)
    result = ttnn.add(result, tt_input)

    # Force completion
    _ = ttnn.to_torch(result)

    elapsed = (time.perf_counter() - start) * 1000  # ms
    times.append(elapsed)
    print(f"  Iteration {i+1}: {elapsed:.2f}ms")

    # Cleanup
    ttnn.deallocate(result)

# Stats
import statistics

print(f"\nResults:")
print(f"  Mean: {statistics.mean(times):.2f}ms")
print(f"  Stdev: {statistics.stdev(times):.2f}ms")
print(f"  Min: {min(times):.2f}ms")
print(f"  Max: {max(times):.2f}ms")

# Cleanup
ttnn.deallocate(tt_input)
ttnn.close_device(device)

print("\nTest completed successfully!")
