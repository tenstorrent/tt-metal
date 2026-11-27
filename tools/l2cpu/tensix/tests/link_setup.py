# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Shared setup of the link tests: open the device, allocate the channel (bank-5 slice of an interleaved DRAM buffer =
the local GDDR of L2CPU tile 0), boot the test responder on the x280 harts through the bring-up component.

Run every test as ONE process from a fresh chip reset (x280 harts leave reset once per chip reset), with the watcher
off. The device must stay open for the whole test: the L2CPU clock runs only while a process holds the chip.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # tools/l2cpu (package `tensix`)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "host"))  # package `l2cpu`

from l2cpu.bringup import bringup_ttnn, region_base_pa  # noqa: E402
from l2cpu.hw import REGION_ALIGN, L2cpuHw, TtnnClusterBackend  # noqa: E402
from tensix import ops  # noqa: E402

RESPONDER = os.path.join(os.path.dirname(HERE), "responder", "build", "responder.bin")
CHANNEL_PAGE_BYTES = (32 << 20) + REGION_ALIGN  # per bank; the base is rounded up inside the bank-5 page
DATA_OFF = 0x100000  # push destination zone (after the channel lines and the responder image)


def open_link(log=print, trace_region_size=64 << 20, core_mhz=1750):
    import ttnn

    assert (
        os.environ.get("TT_METAL_WATCHER") is None
    ), "the watcher flags NoC accesses to the L2CPU tile: run without it"
    dev = ttnn.open_device(device_id=0, trace_region_size=trace_region_size)
    chan = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 8, CHANNEL_PAGE_BYTES // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    base = region_base_pa(chan.buffer_address())
    hw = L2cpuHw(TtnnClusterBackend(0), log=log)
    hw.pa_write(base, bytes(ops.LINK_SIZE))  # sequence words start at 0
    if not os.path.exists(RESPONDER):
        # The responder is built from source on first use (riscv64-unknown-elf-gcc, see the README).
        subprocess.run(["make", "-C", os.path.dirname(os.path.dirname(RESPONDER))], check=True)
    image = open(RESPONDER, "rb").read()

    def ready(h):
        return h.pa_read32(base + ops.LINK_OFF_RESP_STATUS) == ops.LINK_RESP_MAGIC

    t0 = time.time()
    bringup_ttnn(image, chan.buffer_address(), fw_offset=ops.LINK_IMAGE_OFFSET, high_mhz=core_mhz, ready=ready, log=log)
    log(f"responder serving at channel PA 0x{base:x} ({time.time() - t0:.2f} s)")
    return dev, hw, ops.Link(dev, chan, base)


def stop_responder(hw, link):
    hw.pa_write32(link.base + ops.LINK_OFF_STOP, 1)
