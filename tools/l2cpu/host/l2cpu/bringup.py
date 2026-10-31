# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Cold bring-up of an L2CPU tile: image + boot record + reset vectors, slow clock, release, fast clock, prefetchers.

Precondition: a fresh chip reset (the harts of a tile leave reset ONCE per chip reset: a second release is not
supported by the silicon, see ../../README.md). Sequence (tt-bh-linux boot.py order):
  preflight (NoC node id, reset bit clear) -> CCACHE0_WAYENABLE=15 -> image + read-back -> boot record (scratch)
  -> 4 reset vectors -> pre_release hook (resident page, RNMI handlers) -> PLL 200 MHz -> L2CPU_RESET |= 1 << (4 + tile)
  -> PLL high -> L2 prefetchers -> ready() poll (bounded).
"""
from __future__ import annotations

import time

from . import layout as A
from .hw import BOOT_MAGIC, MEMPORT_BASE, REGION_ALIGN, L2cpuHw, make_backend


def bringup(
    image: bytes,
    load_pa: int,
    hw: L2cpuHw | None = None,
    region_pa: int | None = None,
    high_mhz=1750,
    ready=None,
    ready_timeout=5.0,
    log=print,
    pre_release=None,
):
    """pre_release(hw): called after the image, the scratch record and the reset vectors are written and before the
    release (ctl writes the resident page and the RNMI handler addresses there)."""
    hw = hw or L2cpuHw(make_backend(), log=log)
    t0 = time.time()
    if load_pa % REGION_ALIGN:
        raise ValueError("load address must be 64 KiB aligned")
    nid = hw.node_id()
    if nid != (hw.x, hw.y):
        raise RuntimeError(f"L2CPU NIU node id {nid} != {(hw.x, hw.y)}")
    r = hw.read_l2cpu_reset()
    if (r >> (4 + hw.tile)) & 1:
        raise RuntimeError(f"L2CPU_RESET=0x{r:08x}: tile already released; reset all chips first")
    if hasattr(hw.b, "telemetry"):
        t = hw.b.telemetry()
        if not (t.enabled_l2cpu & 1) or not ((t.enabled_gddr >> 5) & 1):
            raise RuntimeError(
                f"tile 0 or D5 harvested: enabled_l2cpu=0x{t.enabled_l2cpu:x} enabled_gddr=0x{t.enabled_gddr:x}"
            )
    we = hw.set_wayenable(15)
    log(f"WAYENABLE -> {we}")
    n = hw.load_image(image, load_pa)
    log(f"image {n} bytes at PA 0x{load_pa:x} (D5 offset 0x{load_pa - MEMPORT_BASE:x}), readback ok")
    hw.write_scratch_u64(0x00, load_pa)
    hw.write_scratch_u64(0x08, BOOT_MAGIC)
    hw.write_scratch_u64(0x10, region_pa if region_pa is not None else load_pa - A.L2CPU_OFF_FW)
    for off in range(0x18, 0x40, 8):
        hw.write_scratch_u64(off, 0)
    log(f"scratch {hw.read_scratch().hex()}")
    hw.set_reset_vectors(load_pa)
    log(f"reset vectors -> 0x{load_pa:x}")
    pre = pre_release(hw) if pre_release is not None else None
    r2 = hw.release(low_mhz=200, high_mhz=high_mhz)
    pf = hw.set_prefetchers()
    log(f"prefetchers {[(hex(a), hex(b)) for a, b in pf]}")
    res = None
    if ready is not None:
        t1 = time.time()
        while True:
            res = ready(hw)
            if res:
                break
            if time.time() - t1 > ready_timeout:
                raise RuntimeError(f"firmware not ready after {ready_timeout} s; scratch={hw.read_scratch().hex()}")
            time.sleep(0.005)
    log(f"bringup done in {time.time() - t0:.2f} s, L2CPU_RESET=0x{r2:08x}, hart status 0x{hw.hart_status():04x}")
    return dict(load_pa=load_pa, reset=r2, ready=res, pre_release=pre)


def region_base_pa(buffer_address: int) -> int:
    """x280 cached PA of the region base for a ttnn buffer whose bank-5 slice starts at buffer_address."""
    return (MEMPORT_BASE + buffer_address + REGION_ALIGN - 1) & ~(REGION_ALIGN - 1)


def bringup_ttnn(
    image: bytes,
    buffer_address: int,
    fw_offset: int = A.L2CPU_OFF_FW,
    high_mhz=1750,
    ready=None,
    ready_timeout=5.0,
    log=print,
    guard=False,
    pre_release=None,
):
    """In a tt-metal process (device open, logical id 0): the region is an interleaved DRAM buffer whose bank-5 slice
    (the L2CPU tile's local DRAM, measured: bank 5 = D5) starts at `buffer_address`. Loads the image at
    region + fw_offset and releases the harts through ttnn.cluster."""
    from .hw import TtnnClusterBackend

    hw = L2cpuHw(TtnnClusterBackend(0), log=log, guard=guard)
    # ttnn DRAM buffers are only 64 B aligned; the region base is rounded up to 64 KiB inside the buffer, so allocate
    # at least 64 KiB more than the region needs. Host, Tensix and x280 must all use this base.
    region_pa = region_base_pa(buffer_address)
    info = bringup(
        image,
        region_pa + fw_offset,
        hw=hw,
        region_pa=region_pa,
        high_mhz=high_mhz,
        ready=ready,
        ready_timeout=ready_timeout,
        log=log,
        pre_release=pre_release,
    )
    info["hw"] = hw
    info["region_pa"] = region_pa
    return info
