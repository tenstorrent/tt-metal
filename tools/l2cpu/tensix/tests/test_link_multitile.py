#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Links to several L2CPU tiles at once: the test responder on tiles 0-3 (one release), one channel per tile in the
tile's local DRAM (tile 3 in a second buffer: tiles 2 and 3 share bank 7).
  1. split push: 32 rows of 303,872 B split over 1 / 2 / 4 tiles in one program (one core per tile), uncached
     alias; every row's bytes checked in its tile; trace-timed.
  2. round trip per step, trace-timed: [notify (one core per tile), wait] with the wait as ONE kernel polling every
     tile (wait_all_program) or as one wait program per tile; and the same after a split push.
  3. one tile dead: its responder is stopped, then [notify all, wait] replays: the step ends at the bound, only
     that tile's wait status word is set (0xDEAD0000 | req), the other tiles keep serving (done_seq advances).
"""
import argparse
import os
import sys
import time

from link_setup import CHANNEL_PAGE_BYTES, DATA_OFF, RESPONDER, ops  # noqa: F401  (sys.path set up there)

from l2cpu.bringup import bringup_tiles, region_base_pa
from l2cpu.hw import L2cpuHw, TtnnClusterBackend
from l2cpu.monitor import install_lock

ROW = 303_872
UC_OFF = DATA_OFF + 0xC00000  # uncached zone of each channel (never touched through the coherent alias)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--timeout-us", type=int, default=20_000)
    ap.add_argument("--dead-iters", type=int, default=10)
    a = ap.parse_args()
    import torch
    import ttnn

    assert os.environ.get("TT_METAL_WATCHER") is None
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
    bufs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 8, CHANNEL_PAGE_BYTES // 4]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            dev,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    ]
    backend = TtnnClusterBackend(0)
    install_lock(type("H", (), {"b": backend})())
    hws, links = [], []
    image = open(RESPONDER, "rb").read()
    specs = []
    for t in range(4):
        buf = bufs[1] if t == 3 else bufs[0]
        base = region_base_pa(buf.buffer_address())
        hw = L2cpuHw(backend, tile=t, guard=True, log=None)
        hw.pa_write(base, bytes(ops.LINK_SIZE))
        hws.append(hw)
        links.append(ops.Link(dev, buf, base, xy=ops.L2CPU_TILES[t]))
        specs.append(
            dict(
                hw=hw,
                image=image,
                load_pa=base + ops.LINK_IMAGE_OFFSET,
                region_pa=base,
                ready=lambda h, b=base: h.pa_read32(b + ops.LINK_OFF_RESP_STATUS) == ops.LINK_RESP_MAGIC,
            )
        )
    infos = bringup_tiles(specs, log=lambda *x: None)
    print(
        f"responders on tiles 0-3 serving, L2CPU_RESET 0x{infos[0]['reset']:08x}, channels "
        f"{[hex(L.base) for L in links]}",
        flush=True,
    )
    ok = True

    def timed(progs, iters, out=None):
        for p in progs:
            links[0].run(p)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for p in progs:
            links[0].run(p)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        t0 = time.perf_counter()
        for _ in range(iters):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / iters
        ttnn.release_trace(dev, tid)
        return dt * 1e3

    # ---- 1. split push
    torch.manual_seed(0)
    src = torch.randn(1, 1, 32, ROW // 2).to(torch.bfloat16)
    tsrc = ttnn.from_torch(
        src, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ref = src.view(torch.int16).numpy().tobytes()
    for hw, L in zip(hws, links):
        hw.pa_write(L.base + ops.LINK_OFF_PUSH_SRC, L.push_source_table(tsrc, ROW))
    print("| tiles | rows per tile | push ms (32 rows) | GB/s | bytes identical |\n|---|---|---|---|---|")
    for nt in (1, 2, 4):
        blocks = [(i * 32 // nt, 32 // nt) for i in range(nt)]
        prog = ops.push_split_program(links[:nt], blocks, ROW, UC_OFF, uncached=True)
        for hw, L in zip(hws[:nt], links[:nt]):
            hw.pa_write(L.base + UC_OFF - ops.UNCACHED_DELTA, bytes(32 // nt * ROW))
        links[0].run(prog)
        ttnn.synchronize_device(dev)
        same = all(
            hws[i].pa_read(links[i].base + UC_OFF - ops.UNCACHED_DELTA, n * ROW) == ref[r0 * ROW : (r0 + n) * ROW]
            for i, (r0, n) in enumerate(blocks)
        )
        ms = timed([prog], 50)
        ok &= same
        print(f"| {nt} | {32 // nt} | {ms:.3f} | {32 * ROW / ms / 1e6:.1f} | {same} |", flush=True)

    # ---- 2. round trip per step
    print("\n| tiles | step | wait as | ms per step |\n|---|---|---|---|")
    for nt in (1, 2, 4):
        L = links[:nt]
        variants = {
            "one kernel": [ops.wait_all_program(L, timeout_us=a.timeout_us)],
            "one program per tile": [x.wait_program(timeout_us=a.timeout_us) for x in L],
        }
        push = ops.push_split_program(L, [(i * 32 // nt, 32 // nt) for i in range(nt)], ROW, UC_OFF, uncached=True)
        for name, waits in variants.items():
            ms = timed([ops.notify_all_program(L)] + waits, a.iters)
            print(f"| {nt} | notify + wait | {name} | {ms:.4f} |", flush=True)
            ms = timed([push, ops.notify_all_program(L)] + waits, 50)
            print(f"| {nt} | split push 32 rows + notify + wait | {name} | {ms:.4f} |", flush=True)
    st = [hw.pa_read32(L.base + ops.LINK_OFF_WAIT_STATUS) for hw, L in zip(hws, links)]
    if any(st):
        print(f"FAIL: wait status words {[hex(x) for x in st]} after healthy runs")
        ok = False

    # ---- 3. one dead tile
    dead = 2
    hws[dead].pa_write32(links[dead].base + ops.LINK_OFF_STOP, 1)
    t0 = time.time()
    while hws[dead].pa_read32(links[dead].base + ops.LINK_OFF_RESP_STATUS) != 0:
        if time.time() - t0 > 2.0:
            print("FAIL: responder did not stop")
            sys.exit(1)
    for name, waits in (
        ("one kernel", [ops.wait_all_program(links, timeout_us=a.timeout_us)]),
        ("one program per tile", [x.wait_program(timeout_us=a.timeout_us) for x in links]),
    ):
        for hw, L in zip(hws, links):
            hw.pa_write32(L.base + ops.LINK_OFF_WAIT_STATUS, 0)
        done0 = [hw.pa_read32(L.base + ops.LINK_OFF_DONE_SEQ) for hw, L in zip(hws, links)]
        ms = timed([ops.notify_all_program(links)] + waits, a.dead_iters)
        req = [hw.pa_read32(L.base + ops.LINK_OFF_REQ_SEQ) for hw, L in zip(hws, links)]
        done = [hw.pa_read32(L.base + ops.LINK_OFF_DONE_SEQ) for hw, L in zip(hws, links)]
        st = [hw.pa_read32(L.base + ops.LINK_OFF_WAIT_STATUS) for hw, L in zip(hws, links)]
        good = (
            st[dead] == ops.WAIT_STATUS_TIMEOUT | (req[dead] & 0xFFFF)
            and done[dead] == done0[dead]
            and all(st[t] == 0 and done[t] == req[t] for t in range(4) if t != dead)
        )
        ok &= good
        print(
            f"dead tile {dead}, wait as {name}: {a.dead_iters + 1} steps, {ms:.2f} ms per step (bound "
            f"{a.timeout_us / 1e3:.1f} ms); wait status {[hex(x) for x in st]}; req {req} done {done}: "
            f"{'PASS' if good else 'FAIL'}",
            flush=True,
        )
    print("PASS" if ok else "FAIL", flush=True)
    for t in range(4):
        if t != dead:
            hws[t].pa_write32(links[t].base + ops.LINK_OFF_STOP, 1)
    ttnn.close_device(dev)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
