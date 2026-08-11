#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Run hello_x280_lim.bin on a real SiFive X280 in a Blackhole L2CPU tile.

Deliberately the LEAST invasive sequence that can work, chosen after a
read-only probe showed the tile is idle, held in reset, and WayEnable == 0
(LIM still SRAM):

  * NO WayEnable=0xF   -- that is a one-way door; it converts L3/LIM to cache
                          and is only undone by an ASIC reset. boot_idle_x280.py
                          does it because it runs firmware from DRAM; we run
                          from LIM at the binary's real link address instead.
  * NO arc_msg(0x52)   -- the AICLK ramp is explicitly flagged as an open
                          hardware-policy question on Galaxy chassis.
  * NO PLL programming -- clock.set_l2cpu_pll changes a chip-wide clock. The
                          core runs at its reset default; we only need it to
                          execute, not to be fast.

Everything written here is either LIM scratch SRAM on an idle tile, the
per-hart reset vectors, or the L2CPU reset register -- all restored by
re-asserting reset at the end.

ECC: after ASIC reset the LIM cells hold random data with uninitialised ECC,
and narrow (byte) writes from the core would then fault. We pre-write the whole
region we use with full-width NOC writes first, which lays down valid ECC, so
the firmware's byte-granular console writes land on good lines.
"""
import os
import sys
import time

sys.path.insert(0, "/data/ucheema/tt-llm-engine/x280/host")

from device import open_chip  # noqa: E402
import loader  # noqa: E402

DEV = int(os.environ.get("TT_DEVICE", "0"))
L2CPU = int(os.environ.get("X280_L2CPU", "0"))
FW = os.environ.get(
    "X280_FW",
    "/data/sanjaysundaram/x280-fabric-wt/tests/tt_metal/tt_metal/"
    "x280_freedom_e_sdk/l2cpu/build/out/hello_x280_lim.bin",
)

# Must match l2cpu/src/x280_lim_console.h
CONSOLE_ADDR = 0x08101000
CONSOLE_MAGIC = 0x2800C0FFEE000280
CONSOLE_CAPACITY = 3072
SENTINEL_ADDR = loader.SENTINEL_ADDR  # 0x08100000
SENTINEL_VALUE = loader.SENTINEL_VALUE  # 0xDEADBEEFCAFEBABE
ACTIVE_FW = loader.ACTIVE_FW_LOAD_ADDR  # 0x08001000

# Image + heap + 4 harts x 32 KiB of stack sits well inside 256 KiB.
PREZERO_BYTES = 0x40000

noc_x, noc_y = loader.L2CPU_TILE_MAPPING[L2CPU]
print(f"device {DEV}, L2CPU{L2CPU} at NOC ({noc_x},{noc_y})")
print(f"firmware: {FW} ({os.path.getsize(FW)} bytes)")

chip = open_chip(backend="exalens", device_id=DEV)

# --- 0. refuse to touch a tile that is already running -----------------------
we = chip.noc_read32(0, noc_x, noc_y, loader.L3_CACHE_WAYENABLE)
if we != 0:
    sys.exit(f"ABORT: WayEnable=0x{we:x} on this tile; LIM is not SRAM here.")
rst = chip.axi_read32(loader.L2CPU_RESET_REG)
if (rst >> (L2CPU + 4)) & 1:
    sys.exit(f"ABORT: L2CPU{L2CPU} is NOT in reset (reg=0x{rst:08x}); something may be using it.")
print(f"pre-flight ok: WayEnable=0, L2CPU{L2CPU} held in reset")

# --- 1. hold in reset (idempotent; establishes a known state) ----------------
loader.assert_l2cpu_reset(chip, L2CPU)

# --- 2. lay down valid ECC over everything the firmware will touch -----------
print(f"zeroing LIM 0x{ACTIVE_FW:08x} + {PREZERO_BYTES//1024} KiB (full-width writes -> valid ECC)")
chip.noc_write(0, noc_x, noc_y, ACTIVE_FW, bytes(PREZERO_BYTES))
print(f"zeroing sentinel + console block at 0x{SENTINEL_ADDR:08x} / 0x{CONSOLE_ADDR:08x}")
chip.noc_write(0, noc_x, noc_y, SENTINEL_ADDR, bytes(0x2000))

# --- 3. load the firmware ----------------------------------------------------
with open(FW, "rb") as f:
    fw = f.read()
if len(fw) % 4:
    fw += b"\x00" * (4 - len(fw) % 4)
chip.noc_write(0, noc_x, noc_y, ACTIVE_FW, fw)

# read back the first bytes and compare -- catches a silently dropped NOC write
buf = bytearray(16)
chip.noc_read(0, noc_x, noc_y, ACTIVE_FW, buf)
if bytes(buf) != fw[:16]:
    sys.exit(f"ABORT: LIM readback mismatch: {bytes(buf).hex()} != {fw[:16].hex()}")
print(f"firmware loaded at 0x{ACTIVE_FW:08x}, readback verified")

# --- 4. point all four harts at the image and release ------------------------
loader.set_reset_vectors(chip, noc_x, noc_y, entry_addr=ACTIVE_FW)
print(f"reset vectors -> 0x{ACTIVE_FW:08x}")

print("releasing L2CPU from reset ...")
loader.release_l2cpu_reset(chip, L2CPU)

# --- 5. wait for the firmware's sentinel ------------------------------------
ok, val = loader.poll_flag(chip, noc_x, noc_y, SENTINEL_ADDR, SENTINEL_VALUE, timeout=30)
print(
    f"sentinel @0x{SENTINEL_ADDR:08x} = 0x{val:016x} "
    f"(expected 0x{SENTINEL_VALUE:016x}) -> {'MATCH' if ok else 'NO MATCH'}"
)

# --- 6. read the console block back over the NOC -----------------------------
hdr = bytearray(16)
chip.noc_read(0, noc_x, noc_y, CONSOLE_ADDR, hdr)
magic = int.from_bytes(hdr[0:8], "little")
length = int.from_bytes(hdr[8:12], "little")
dropped = int.from_bytes(hdr[12:16], "little")
print(f"console magic = 0x{magic:016x} (expected 0x{CONSOLE_MAGIC:016x})")
print(f"console len   = {length}, dropped = {dropped}")

text = ""
if magic == CONSOLE_MAGIC and 0 < length <= CONSOLE_CAPACITY:
    n = (length + 3) & ~3
    data = bytearray(n)
    chip.noc_read(0, noc_x, noc_y, CONSOLE_ADDR + 16, data)
    text = data[:length].decode("utf-8", errors="replace")
    print("\n" + "=" * 70)
    print("X280 CONSOLE OUTPUT, READ BACK FROM LIM OVER THE NOC")
    print("=" * 70)
    print(text)
    print("=" * 70)

# --- 7. trap diagnostics, if it died instead -------------------------------
if not ok:
    print("\ntrap diagnostics (freedom-metal early_trap_vector spins; x280.ld pins these):")
    for name, addr in (("mcause", 0x0811FFE0), ("mepc", 0x0811FFE8), ("mtval", 0x0811FFF0)):
        try:
            print(f"  {name} = 0x{loader.read_u64_noc(chip, noc_x, noc_y, addr):016x}")
        except Exception as e:
            print(f"  {name}: {e}")

# --- 8. leave the tile exactly as we found it -------------------------------
loader.assert_l2cpu_reset(chip, L2CPU)
we_after = chip.noc_read32(0, noc_x, noc_y, loader.L3_CACHE_WAYENABLE)
print(f"\nL2CPU{L2CPU} put back in reset; WayEnable still 0x{we_after:x}")

sys.exit(0 if (ok and "Hello, World!" in text) else 1)
