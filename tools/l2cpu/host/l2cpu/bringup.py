# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Cold bring-up of L2CPU tiles: image + boot record + reset vectors, slow clock, release, fast clock, prefetchers.

Precondition: a fresh chip reset (the harts of a tile leave reset ONCE per chip reset: a second release is not
supported by the silicon, see ../../README.md). Sequence (tt-bh-linux boot.py order), per tile:
  preflight (NoC node id, reset bit clear, tile and its DRAM not harvested) -> CCACHE0_WAYENABLE=15 -> image +
  read-back -> boot record (scratch) -> 4 reset vectors -> pre_release hook (resident page, RNMI handlers)
then once for all tiles: PLL 200 MHz -> L2CPU_RESET |= sum(1 << (4 + tile)) (one write) -> PLL high, and per tile:
  L2 prefetchers -> ready() poll (bounded). bringup() is the one-tile case, bringup_tiles() several tiles.
"""
from __future__ import annotations

import time

from . import layout as A
from .hw import BOOT_MAGIC, L2CPU_TILE_BANK, MEMPORT_BASE, REGION_ALIGN, L2cpuHw, make_backend


def prepare(image: bytes, load_pa: int, hw: L2cpuHw, region_pa: int | None = None, log=print, pre_release=None):
    """Everything before the release, for one tile: preflight, image, boot record, reset vectors, pre_release(hw)
    (ctl writes the resident page and the RNMI handler addresses there). Returns the hook's result."""
    if load_pa % REGION_ALIGN:
        raise ValueError("load address must be 64 KiB aligned")
    nid = hw.node_id()
    if nid != (hw.x, hw.y):
        raise RuntimeError(f"L2CPU NIU node id {nid} != {(hw.x, hw.y)}")
    r = hw.read_l2cpu_reset()
    if (r >> (4 + hw.tile)) & 1:
        raise RuntimeError(f"L2CPU_RESET=0x{r:08x}: tile {hw.tile} already released; reset all chips first")
    if hasattr(hw.b, "telemetry"):
        t = hw.b.telemetry()
        d = L2CPU_TILE_BANK[hw.tile]
        if not (t.enabled_l2cpu >> hw.tile) & 1 or not (t.enabled_gddr >> d) & 1:
            raise RuntimeError(
                f"tile {hw.tile} or its D{d} harvested: enabled_l2cpu=0x{t.enabled_l2cpu:x} "
                f"enabled_gddr=0x{t.enabled_gddr:x}"
            )
    we = hw.set_wayenable(15)
    log(f"WAYENABLE -> {we}")
    n = hw.load_image(image, load_pa)
    log(f"image {n} bytes at PA 0x{load_pa:x} (local DRAM offset 0x{load_pa - MEMPORT_BASE:x}), readback ok")
    hw.write_scratch_u64(0x00, load_pa)
    hw.write_scratch_u64(0x08, BOOT_MAGIC)
    hw.write_scratch_u64(0x10, region_pa if region_pa is not None else load_pa - A.L2CPU_OFF_FW)
    for off in range(0x18, 0x40, 8):
        hw.write_scratch_u64(off, 0)
    log(f"scratch {hw.read_scratch().hex()}")
    hw.set_reset_vectors(load_pa)
    log(f"reset vectors -> 0x{load_pa:x}")
    return pre_release(hw) if pre_release is not None else None


def _wait_ready(hw, ready, ready_timeout):
    t1 = time.time()
    while True:
        res = ready(hw)
        if res:
            return res
        if time.time() - t1 > ready_timeout:
            raise RuntimeError(
                f"tile {hw.tile}: firmware not ready after {ready_timeout} s; scratch={hw.read_scratch().hex()}"
            )
        time.sleep(0.005)


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
    pre = prepare(image, load_pa, hw, region_pa=region_pa, log=log, pre_release=pre_release)
    r2 = hw.release(low_mhz=200, high_mhz=high_mhz)
    pf = hw.set_prefetchers()
    log(f"prefetchers {[(hex(a), hex(b)) for a, b in pf]}")
    res = _wait_ready(hw, ready, ready_timeout) if ready is not None else None
    log(f"bringup done in {time.time() - t0:.2f} s, L2CPU_RESET=0x{r2:08x}, hart status 0x{hw.hart_status():04x}")
    return dict(load_pa=load_pa, reset=r2, ready=res, pre_release=pre)


def bringup_tiles(specs, high_mhz=1750, ready_timeout=5.0, log=print):
    """Several tiles of one chip, released together. specs: list of dicts with hw (L2cpuHw of that tile), image,
    load_pa, region_pa, and optional pre_release(hw) / ready(hw). Every tile is prepared first, then ONE write of
    L2CPU_RESET sets all their bits inside one PLL 200 MHz -> high dance (tt-bh-linux reset_x280 for a list of
    tiles; a tile's bit goes 0 -> 1 once per chip reset, so a failed preparation releases nothing). Returns one
    info dict per spec."""
    t0 = time.time()
    tiles = [s["hw"].tile for s in specs]
    if len(set(tiles)) != len(tiles):
        raise ValueError(f"tile listed twice: {tiles}")
    pres = [
        prepare(
            s["image"], s["load_pa"], s["hw"], region_pa=s.get("region_pa"), log=log, pre_release=s.get("pre_release")
        )
        for s in specs
    ]
    r2 = specs[0]["hw"].release(low_mhz=200, high_mhz=high_mhz, tiles=tiles)
    out = []
    for s in specs:
        s["hw"].set_prefetchers()
    for s, pre in zip(specs, pres):
        res = _wait_ready(s["hw"], s["ready"], ready_timeout) if s.get("ready") is not None else None
        out.append(dict(load_pa=s["load_pa"], reset=r2, ready=res, pre_release=pre))
    log(f"bringup of tiles {tiles} done in {time.time() - t0:.2f} s, L2CPU_RESET=0x{r2:08x}")
    return out


def region_base_pa(buffer_address: int) -> int:
    """x280 cached PA of the region base for a ttnn interleaved buffer whose page in the tile's local bank (bank 5 for
    tile 0, 6 / 7 / 7 for tiles 1 / 2 / 3; bank base 0) starts at buffer_address. Same value for every tile: each
    sees its own bank there. Tiles 2 and 3 share bank 7, so they need two different buffers."""
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
