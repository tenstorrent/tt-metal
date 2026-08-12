#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Read-only L2CPU/X280 probe (no writes). Safe on shared Galaxy.

Checks: exalens open, enabled L2CPUs, reset state, LIM readability, WayEnable.
"""
import os
import sys

sys.path.insert(0, "/data/ucheema/tt-llm-engine/x280/host")

from device import open_chip  # noqa: E402
import loader  # noqa: E402

dev = int(os.environ.get("TT_DEVICE", "0"))

print(f"=== opening device {dev} via exalens (read-only) ===")
chip = open_chip(backend="exalens", device_id=dev)
print("chip opened")

print("\n=== 1. telemetry: which L2CPUs are enabled? ===")
try:
    enabled = loader.detect_enabled_l2cpu(chip)
    print(f"enabled_l2cpu bitmask = 0x{enabled:x}")
    for i in range(4):
        print(
            f"  L2CPU{i} @ NOC{loader.L2CPU_TILE_MAPPING[i]}: "
            f"{'ENABLED' if (enabled >> i) & 1 else 'harvested/absent'}"
        )
except Exception as e:
    print(f"telemetry read failed: {type(e).__name__}: {e}")
    enabled = None

print("\n=== 2. L2CPU reset register (read only) ===")
try:
    val = chip.axi_read32(loader.L2CPU_RESET_REG)
    print(f"L2CPU_RESET_REG (0x{loader.L2CPU_RESET_REG:x}) = 0x{val:08x}")
    for i in range(4):
        held = not ((val >> (i + 4)) & 1)
        print(f"  L2CPU{i}: {'HELD IN RESET' if held else 'released'}")
except Exception as e:
    print(f"reset reg read failed: {type(e).__name__}: {e}")

print("\n=== 3. LIM readback (read only) ===")
for idx in range(4):
    if enabled is not None and not ((enabled >> idx) & 1):
        continue
    noc_x, noc_y = loader.L2CPU_TILE_MAPPING[idx]
    try:
        buf = bytearray(32)
        chip.noc_read(0, noc_x, noc_y, loader.LIM_BASE, buf)
        words = [int.from_bytes(buf[i : i + 8], "little") for i in range(0, 32, 8)]
        print(f"  L2CPU{idx} NOC({noc_x},{noc_y}) LIM+0x0 : " + " ".join(f"0x{w:016x}" for w in words))
        sent = loader.read_u64_noc(chip, noc_x, noc_y, loader.SENTINEL_ADDR)
        print(f"  L2CPU{idx} sentinel slot 0x{loader.SENTINEL_ADDR:08x} = 0x{sent:016x}")
    except Exception as e:
        print(f"  L2CPU{idx}: LIM read failed: {type(e).__name__}: {e}")

print("\n=== 4. L3 WayEnable (read only) -- 0 means LIM is SRAM, 0xF means LIM was converted to cache ===")
for idx in range(4):
    if enabled is not None and not ((enabled >> idx) & 1):
        continue
    noc_x, noc_y = loader.L2CPU_TILE_MAPPING[idx]
    try:
        we = chip.noc_read32(0, noc_x, noc_y, loader.L3_CACHE_WAYENABLE)
        print(f"  L2CPU{idx}: WayEnable = 0x{we:x}" f"{'  (LIM intact)' if we == 0 else '  (LIM converted to cache!)'}")
    except Exception as e:
        print(f"  L2CPU{idx}: WayEnable read failed: {type(e).__name__}: {e}")

print("\nprobe complete -- no writes were performed")
