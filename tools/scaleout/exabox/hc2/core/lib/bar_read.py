#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""lib/bar_read.py: read-only BAR0 read of a Blackhole chip's ARC boot registers, for chips whose firmware
is not ready: UMD's device open waits 300 s for their firmware, this reads the registers it polls without it.

    python3 -I lib/bar_read.py <bdf> [sysfs root]

mmaps one page of /sys/bus/pci/devices/<bdf>/resource0 per register, PROT_READ on an O_RDONLY fd (nothing is written),
and does one 32-bit load each. Offsets from tt-umd device/api/umd/device/arch/blackhole_implementation.hpp:
ARC_APB_BAR0_XBAR_OFFSET_START 0x1FF00000 + ARC_RESET_SCRATCH_OFFSET 0x30060 (postcode), + SCRATCH_RAM_2 0x30408
(BOOT_STATUS; ready when & 0x7 == 0x5, BlackholeDeviceFirmware::wait_firmware_ready), + SCRATCH_RAM_4 0x30410
(ERROR_STATUS0). UMD reads ARC over BAR0 only when the chip's PCIe tile is at NOC x 11 (BlackholeArcApb::
is_arc_available_over_axi: NIU_CFG_NOC0_BAR_PCIE_ADDR 0x1FD04000 + NOC_NODE_ID_OFFSET 0x44, & 0x3F), so that is read
first and the ARC registers only when it is 11 (or reads all ones: a chip off the bus).
Prints one line: "pcie_x=<n> postcode=0x... boot_status=0x... error_status0=0x..." ("-" for a register not read).
Exit 0 read, 2 resource0 missing or not mappable (no root, no BAR).
"""
import ctypes
import mmap
import os
import sys

ARC_APB = 0x1FF00000
POSTCODE, BOOT_STATUS, ERROR_STATUS0 = ARC_APB + 0x30060, ARC_APB + 0x30408, ARC_APB + 0x30410
NODE_ID = 0x1FD04000 + 0x44
ONES = 0xFFFFFFFF

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def rd32(fd, off):
    page = mmap.PAGESIZE
    base = off & ~(page - 1)
    p = libc.mmap(None, page, mmap.PROT_READ, mmap.MAP_SHARED, fd, base)
    if p in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap resource0 at 0x%x" % base)
    try:
        return ctypes.c_uint32.from_address(p + off - base).value  # one aligned 32-bit load
    finally:
        libc.munmap(p, page)


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    bdf = sys.argv[1] if sys.argv[1].count(":") == 2 else "0000:" + sys.argv[1]
    res = os.path.join(sys.argv[2] if len(sys.argv) > 2 else "/sys", "bus/pci/devices", bdf, "resource0")
    try:
        fd = os.open(res, os.O_RDONLY | os.O_SYNC)
    except OSError as e:
        print("bar_read: %s: %s" % (res, e.strerror), file=sys.stderr)
        return 2
    try:
        size = os.fstat(fd).st_size
        if size and size <= ERROR_STATUS0:
            print("bar_read: %s: BAR0 is 0x%x bytes, too small for 0x%x" % (res, size, ERROR_STATUS0), file=sys.stderr)
            return 2
        x = rd32(fd, NODE_ID)
        v = {"pcie_x": "0x%x" % x if x == ONES else str(x & 0x3F)}
        arc = x == ONES or (x & 0x3F) == 11
        for k, o in (("postcode", POSTCODE), ("boot_status", BOOT_STATUS), ("error_status0", ERROR_STATUS0)):
            v[k] = "0x%08x" % rd32(fd, o) if arc else "-"
    except OSError as e:
        print("bar_read: %s: %s" % (res, e.strerror or e), file=sys.stderr)
        return 2
    finally:
        os.close(fd)
    print(" ".join("%s=%s" % kv for kv in v.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
