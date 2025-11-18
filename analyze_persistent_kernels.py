#!/usr/bin/env python3
"""Analyze which kernels are persistent (never unloaded)"""

import re

# Track all kernel loads with their sizes
kernel_loads = []  # List of (line_num, device, size_mb, total_after)
kernel_unloads = []  # List of (line_num, device, size_mb, total_after)

load_pattern = re.compile(r"KERNEL_LOAD.*Device (\d+): \+([0-9.]+) MB \(Total: ([0-9.]+) MB\)")
unload_pattern = re.compile(r"KERNEL_UNLOAD.*Device (\d+): -([0-9.]+) MB \(Total: ([0-9.]+) MB\)")

with open("out.log", "r") as f:
    for i, line in enumerate(f, 1):
        load_match = load_pattern.search(line)
        if load_match:
            device = int(load_match.group(1))
            size = float(load_match.group(2))
            total = float(load_match.group(3))
            kernel_loads.append((i, device, size, total))

        unload_match = unload_pattern.search(line)
        if unload_match:
            device = int(unload_match.group(1))
            size = float(unload_match.group(2))
            total = float(unload_match.group(3))
            kernel_unloads.append((i, device, size, total))

print("=" * 80)
print("FIRST 10 KERNEL LOADS (potential persistent system kernels)")
print("=" * 80)
for i, (line_num, device, size, total) in enumerate(kernel_loads[:10], 1):
    print(f"{i}. Line {line_num}: Device {device}, Size: {size:.6f} MB, Running total: {total:.6f} MB")

print("\n" + "=" * 80)
print(f"SUMMARY:")
print(f"  Total kernel loads: {len(kernel_loads)}")
print(f"  Total kernel unloads: {len(kernel_unloads)}")
print(f"  Orphaned kernels: {len(kernel_loads) - len(kernel_unloads)}")
print("=" * 80)

# Find unique kernel sizes
from collections import Counter

sizes = [size for _, _, size, _ in kernel_loads]
size_counts = Counter(sizes)
print("\nKERNEL SIZE DISTRIBUTION:")
for size, count in sorted(size_counts.items(), reverse=True)[:10]:
    print(f"  {size:.6f} MB: {count} instances")

# Check if the first 4 loads per device are never unloaded
print("\n" + "=" * 80)
print("FIRST 4 LOADS PER DEVICE (likely persistent system kernels):")
print("=" * 80)
for dev in range(4):
    dev_loads = [(ln, sz, tot) for ln, d, sz, tot in kernel_loads if d == dev]
    print(f"\nDevice {dev}:")
    for i, (line_num, size, total) in enumerate(dev_loads[:4], 1):
        print(f"  Kernel {i}: Line {line_num}, Size: {size:.6f} MB ({size * 1024:.2f} KB)")
