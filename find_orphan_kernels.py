#!/usr/bin/env python3
"""Find kernel IDs that were loaded but never unloaded"""

import re
from collections import defaultdict

# Track kernel loads and unloads by (device_id, kernel_id)
kernels_loaded = defaultdict(lambda: defaultdict(int))  # device -> kernel_id -> count
kernels_unloaded = defaultdict(lambda: defaultdict(int))

load_pattern = re.compile(r"KERNEL_LOAD.*Device (\d+).*kernel_id=(\w+)")
unload_pattern = re.compile(r"KERNEL_UNLOAD.*Device (\d+).*kernel_id=(\w+)")

with open("out.log", "r") as f:
    for line in f:
        load_match = load_pattern.search(line)
        if load_match:
            device = int(load_match.group(1))
            kernel_id = load_match.group(2)
            kernels_loaded[device][kernel_id] += 1

        unload_match = unload_pattern.search(line)
        if unload_match:
            device = int(unload_match.group(1))
            kernel_id = unload_match.group(2)
            kernels_unloaded[device][kernel_id] += 1

print("=" * 70)
print("ORPHANED KERNELS (loaded but not unloaded)")
print("=" * 70)

total_orphans = 0
for device in sorted(kernels_loaded.keys()):
    device_orphans = []
    for kernel_id in kernels_loaded[device]:
        loaded = kernels_loaded[device][kernel_id]
        unloaded = kernels_unloaded[device].get(kernel_id, 0)
        if loaded > unloaded:
            device_orphans.append((kernel_id, loaded - unloaded))
            total_orphans += loaded - unloaded

    if device_orphans:
        print(f"\nDevice {device}:")
        for kernel_id, count in device_orphans:
            print(f"  Kernel 0x{kernel_id}: {count} orphaned instance(s)")

print(f"\n{'=' * 70}")
print(f"Total orphaned kernels: {total_orphans}")
print(f"{'=' * 70}")
