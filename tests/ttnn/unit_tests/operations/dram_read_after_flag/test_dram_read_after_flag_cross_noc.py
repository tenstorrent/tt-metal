# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DRAM read-after-flag across NoCs (Blackhole, single chip, no fabric): an acked DRAM write is not yet
visible to a read through the other NoC.

Pattern (per stream: one writer core and one reader core on the same chip; kernels/ has the detail):
  writer (BRISC, NoC W) : seq = 0, 1, 2, ...; non-posted noc_async_write of a 64 B page into page
                          (seq % 2) of an interleaved DRAM buffer, noc_async_write_barrier (acked),
                          then noc_semaphore_inc of a flag in the reader core's L1.
  reader (NCRISC, NoC R): waits flag >= 1, noc_async_read of that page (address resolved for its own
                          NoC), read barrier, invalidate_l1_cache, checks every word == seq;
                          flag -= 1, credit += 1 on the writer (at most one page in flight).
  hammer                : every free Tensix core streams 8 KB DRAM writes (BRISC, NOC0) and reads
                          (NCRISC, NOC1) over a separate buffer on every bank (needed to reproduce).
Each NoC reaches a DRAM channel through its own endpoint (port) of that channel.

Expected: flag seen => page visible, so every read returns seq.

Observed (8x4 Blackhole Galaxy, every chip running this independently, 64 B, hammer, 2026-11-10):
  writer NOC1, reader NOC0 (different ports): ~9 stale reads per 1e8, ONLY on DRAM channels at
      translated (x=18, y=12..17) = banks 4 and 5 of an 8-bank interleaved buffer
  writer NOC0, reader NOC1 (different ports): ~3 per 1e8, seen on banks 1, 2, 5, 7 (6 events)
  writer and reader on the same NoC (same port): 0 in 2.8e8 (NOC0) and 2.55e9 (NOC1)
  writer NOC0, reader NOC1 reading through the NOC0 endpoint (same port, other NoC): 0 in 1.32e9
  (the reverse cannot be tried: the NOC1 endpoints' NOC0 NIUs are in DRISC stream mode, reads return zeros)
Every stale read: the WHOLE page from 2 iterations earlier (the previous write to it), flag exactly 1,
one re-read returns the new page (1k-18k cycles from the bad read to the good one, including the check
itself: an upper bound, not the visibility lag). Seen on 28 of 32 chips (and on a second Galaxy with a
fabric router as the writer). Rate falls with page size (~0.3e-8 at 576 B, none seen at 5.6 KB).

Run (from TT_METAL_HOME; RAF_SECONDS = run time per test, default 600):
  one chip:  pytest tests/ttnn/unit_tests/operations/dram_read_after_flag/test_dram_read_after_flag_cross_noc.py \
                 -k "1chip and w1r0"
  Galaxy:    ... -k "galaxy_8x4 and w1r0"   (all 32 chips at once: ~30x the rate)
  RAF_BANKS (default "4,5"): banks the two pages of every stream land on; "all" spreads the streams
  over every bank (use with w0r1). RAF_STREAMS (default 8) writer/reader pairs per chip.

Expected failure output:
  DRAM_RAF STALE chip=(2,2) stream=2 seq=0x23137 page=13 (bank 5) read w0=0x23135 w1=0xfffdceca
      kind=whole page from 2 iterations earlier flag_at_wait=1 rereads=1 cycles_to_good_read<=1038
  AssertionError: 6 reads returned stale data after the flag was seen
"""

import os
import time

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
SECONDS = float(os.environ.get("RAF_SECONDS", "600"))
STREAMS = int(os.environ.get("RAF_STREAMS", "8"))
BANKS = os.environ.get("RAF_BANKS", "4,5")
NUM_BANKS = int(os.environ.get("RAF_NUM_BANKS", "8"))  # DRAM banks of an interleaved buffer (8 on an unharvested BH)
PAGE = int(os.environ.get("RAF_PAGE", "64"))  # bytes, a multiple of 64
# experiment knobs (defaults = the configuration that reproduces)
HAMMER = os.environ.get("RAF_HAMMER", "rw")  # rw | w (writes only) | r (reads only) | none
HAMMER_NOCS = os.environ.get("RAF_HAMMER_NOCS", "01")  # NoC of the hammer's writer, reader
HAMMER_CORES = int(os.environ.get("RAF_HAMMER_CORES", "0"))  # 0: every free core
HAMMER_BANK = int(os.environ.get("RAF_HAMMER_BANK", "-1"))  # -1: every bank; else only that bank
FLAG_DELAY = int(os.environ.get("RAF_FLAG_DELAY", "0"))  # writer: cycles between the ack and the flag
FENCE = int(os.environ.get("RAF_FENCE", "0"))  # writer: read the page back through its own NoC before the flag
READ_DELAY = int(os.environ.get("RAF_READ_DELAY", "0"))  # reader: cycles between seeing the flag and the read
TIMEOUT_CYCLES = 500_000_000  # per wait; a stuck launch still ends well inside the 5 s dispatch timeout
HAMMER_PAGE, HAMMER_PAGES_PER_CORE = 8192, 64
ROW = 128  # uint32 words of per-core results (one L1 shard per core)
PAYLOAD_CB, HAMMER_W_CB, HAMMER_R_CB = 0, 1, 2
HAMMER_BATCH = 4  # outstanding hammer transfers per core (raf_hammer.cpp); the reader has a slot per transfer


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _endpoint(lo, hi):
    a = (int(hi) << 32) | int(lo)
    return f"(x={(a >> 36) & 0x3F},y={(a >> 42) & 0x3F})"


@pytest.mark.timeout(SECONDS + 600)
@pytest.mark.parametrize(
    "mesh_device", [pytest.param((1, 1), id="1chip"), pytest.param((8, 4), id="galaxy_8x4")], indirect=True
)
@pytest.mark.parametrize(
    "writer_noc, reader_noc",
    [(1, 0), (0, 1), (0, 0), (1, 1)],
    ids=["w1r0", "w0r1", "w0r0_control", "w1r1_control"],
)
def test_dram_read_after_flag_cross_noc(mesh_device, writer_noc, reader_noc):
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]  # = results shard order
    grid_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    virt = mesh_device.worker_core_from_logical_core
    nS = STREAMS
    assert 2 * nS < len(cores)
    w_core, r_core = cores[0 : 2 * nS : 2], cores[1 : 2 * nS : 2]
    hammer = [] if HAMMER == "none" else cores[2 * nS :]
    hammer = hammer[:HAMMER_CORES] if HAMMER_CORES > 0 else hammer
    h_mult, h_add = (1, 0) if HAMMER_BANK < 0 else (NUM_BANKS, HAMMER_BANK)

    # page (2 s + j) * NUM_BANKS + bank: page j of stream s sits in bank banks[(2 s + j) % len(banks)]
    banks = list(range(NUM_BANKS)) if BANKS == "all" else [int(b) for b in BANKS.split(",")]
    pages = [[(2 * s + j) * NUM_BANKS + banks[(2 * s + j) % len(banks)] for j in range(2)] for s in range(nS)]

    def tensor(shape, mem):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=mem,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )

    buf = tensor([1, 1, 2 * nS * NUM_BANKS, PAGE // 4], ttnn.DRAM_MEMORY_CONFIG)  # page = row
    ham_pages = max(1, len(hammer)) * HAMMER_PAGES_PER_CORE * h_mult
    ham = tensor([1, 1, ham_pages, HAMMER_PAGE // 4], ttnn.DRAM_MEMORY_CONFIG)
    shard = ttnn.ShardSpec(grid_crs, [1, ROW], ttnn.ShardOrientation.ROW_MAJOR)
    res = tensor(
        [1, 1, len(cores), ROW], ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
    )
    sems = [ttnn.create_global_semaphore(mesh_device, grid_crs, 0) for _ in range(5)]
    flag, credit, state, stop_w, stop_r = [ttnn.get_global_semaphore_address(s) for s in sems]
    ttnn.synchronize_device(mesh_device)
    buf_ct = list(ttnn.TensorAccessorArgs(buf).get_compile_time_args())
    ham_ct = list(ttnn.TensorAccessorArgs(ham).get_compile_time_args())
    nocs = (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
    RV0, RV1 = ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1

    def cb(idx, size, cs):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.uint32, page_size=size)
        return ttnn.CBDescriptor(total_size=size, core_ranges=_crs(cs), format_descriptors=[fmt])

    def kernel(name, cs, ct, rt, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNELS, name),
            core_ranges=_crs(cs),
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )

    def build(n):
        w_rt, r_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for s in range(nS):
            wc, rc = w_core[s], r_core[s]
            rv, wv = virt(rc), virt(wc)
            common = [flag, credit, state, res.buffer_address(), TIMEOUT_CYCLES]
            w_rt[wc.x][wc.y] = [buf.buffer_address(), *pages[s], n, rv.x, rv.y] + common + [FLAG_DELAY, FENCE]
            stops = [len(hammer), stop_w, stop_r] + [v for h in hammer for v in (virt(h).x, virt(h).y)]
            r_rt[rc.x][rc.y] = [buf.buffer_address(), *pages[s], n, wv.x, wv.y] + common + [READ_DELAY] + stops
        hw_rt, hr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()  # each hammer core runs until every reader signals
        for i, h in enumerate(hammer):
            slice_ = [ham.buffer_address(), i * HAMMER_PAGES_PER_CORE, HAMMER_PAGES_PER_CORE]
            tail = [nS, 8 * TIMEOUT_CYCLES, h_mult, h_add, res.buffer_address()]
            hw_rt[h.x][h.y] = slice_ + [stop_w] + tail
            hr_rt[h.x][h.y] = slice_ + [stop_r] + tail
        kernels = [
            kernel("raf_writer.cpp", w_core, [PAGE, PAYLOAD_CB] + buf_ct, w_rt, dm(RV0, nocs[writer_noc])),
            kernel("raf_reader.cpp", r_core, [PAGE, PAYLOAD_CB] + buf_ct, r_rt, dm(RV1, nocs[reader_noc])),
        ]
        hw_noc, hr_noc = nocs[int(HAMMER_NOCS[0])], nocs[int(HAMMER_NOCS[1])]
        if hammer and "w" in HAMMER:
            hct = [HAMMER_PAGE, 1, HAMMER_W_CB] + ham_ct
            kernels.append(kernel("raf_hammer.cpp", hammer, hct, hw_rt, dm(RV0, hw_noc)))
        if hammer and "r" in HAMMER:
            hct = [HAMMER_PAGE, 0, HAMMER_R_CB] + ham_ct
            kernels.append(kernel("raf_hammer.cpp", hammer, hct, hr_rt, dm(RV1, hr_noc)))
        cbs = [cb(PAYLOAD_CB, 2 * PAGE, w_core + r_core)]  # (the writer's second slot: the FENCE read-back)
        if hammer:
            cbs += [cb(HAMMER_W_CB, HAMMER_PAGE, hammer), cb(HAMMER_R_CB, HAMMER_BATCH * HAMMER_PAGE, hammer)]
        pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        mpd = ttnn.MeshProgramDescriptor()
        mpd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows - 1, cols - 1))] = pd
        return mpd

    def results():  # [chip][core index] -> row
        return [
            (np.asarray(ttnn.to_torch(t).reshape(-1, ROW).numpy()).astype(np.int64) & 0xFFFFFFFF)
            for t in ttnn.get_device_tensors(res)
        ]

    def totals(all_rows):
        rd = [rr[cores.index(r_core[s])] for rr in all_rows for s in range(nS)]
        wr = [rr[cores.index(w_core[s])] for rr in all_rows for s in range(nS)]
        return dict(
            txns=sum(int(v[0]) | int(v[1]) << 32 for v in rd),
            written=sum(int(v[0]) | int(v[1]) << 32 for v in wr),
            mism=sum(int(v[2]) for v in rd),
            whole_old=sum(int(v[3]) for v in rd),
            other=sum(int(v[4]) for v in rd),
            unresolved=sum(int(v[5]) for v in rd),
            max_rereads=max(int(v[6]) for v in rd),
            max_cycles=max(int(v[7]) for v in rd),
            flag_over=sum(int(v[9]) for v in rd),
            final_flag=sum(int(v[10]) for v in rd),
            timeouts=sum(int(v[8]) for v in rd) + sum(int(v[2]) for v in wr),
        )

    logger.info(
        f"DRAM_RAF mesh={rows}x{cols} writer=NOC{writer_noc} reader=NOC{reader_noc} page={PAGE}B streams/chip={nS} "
        f"banks={banks} hammer={HAMMER} hammer_nocs(w,r)={HAMMER_NOCS} hammer_cores/chip={len(hammer)} "
        f"hammer_bank={HAMMER_BANK} flag_delay={FLAG_DELAY} fence={FENCE} read_delay={READ_DELAY} seconds={SECONDS}"
    )
    io = [buf, ham, res]
    t_start = time.time()  # (the calibration launches count too)
    n = 2000  # calibrate: then each launch ~1 s (the dispatch timeout is 5 s)
    mpd = build(n)
    for _ in range(2):
        t0 = time.time()
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
    n = int(max(2000, min(2**30, n / max(time.time() - t0, 1e-3))))
    mpd = build(n)
    seen = (0, 0)
    while time.time() - t_start < SECONDS:
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
        t = totals(results())
        if (t["mism"], t["timeouts"]) != seen:
            seen = (t["mism"], t["timeouts"])
            logger.warning(f"DRAM_RAF after {time.time() - t_start:.0f}s: {t}")
        if t["timeouts"] or t["unresolved"]:
            break
    wall = time.time() - t_start

    all_rows = results()
    r0 = all_rows[0]
    for s in range(nS):  # where the pages actually resolve (chip 0; other chips listed only if different)
        w, r = r0[cores.index(w_core[s])], r0[cores.index(r_core[s])]
        for i, rr in enumerate(all_rows[1:], 1):
            wi, ri = rr[cores.index(w_core[s])], rr[cores.index(r_core[s])]
            if list(wi[12:16]) != list(w[12:16]) or list(ri[12:16]) != list(r[12:16]):
                print(
                    f"DRAM_RAF MAP chip={divmod(i, cols)} stream={s} differs: "
                    f"writer {list(wi[12:16])} reader {list(ri[12:16])}"
                )
        print(
            f"DRAM_RAF MAP stream={s} pages={pages[s]} writer NOC{writer_noc} -> {_endpoint(w[12], w[13])} "
            f"{_endpoint(w[14], w[15])}, reader NOC{reader_noc} -> {_endpoint(r[12], r[13])} {_endpoint(r[14], r[15])}"
        )
    for i, rr in enumerate(all_rows):
        chip = divmod(i, cols)
        for s in range(nS):
            row = rr[cores.index(r_core[s])]
            for j in range(int(row[11])):
                seq, page, w0, w1, fv, rereads, cyc, bad = (int(v) for v in row[16 + 8 * j : 24 + 8 * j])
                whole_old, bad = bool(bad & 0x80000000), bad & 0x7FFFFFFF
                kind = "whole page from 2 iterations earlier" if whole_old else f"partial ({bad} words not of seq)"
                rereads = "never resolved" if rereads == 0xFFFFFFFF else rereads
                print(
                    f"DRAM_RAF STALE chip={chip} stream={s} seq={seq:#x} page={page} (bank {page % NUM_BANKS}) "
                    f"read w0={w0:#x} w1={w1:#x} kind={kind} flag_at_wait={fv} rereads={rereads} "
                    f"cycles_to_good_read<={cyc}"
                )
    t = totals(all_rows)
    print(
        f"DRAM_RAF TOTAL txns={t['txns']} mismatches={t['mism']} (whole_old={t['whole_old']} other={t['other']}) "
        f"unresolved={t['unresolved']} max_rereads={t['max_rereads']} max_cycles={t['max_cycles']} "
        f"flag_over={t['flag_over']} final_flag_nonzero={t['final_flag']} timeouts={t['timeouts']} "
        f"wall={wall:.0f}s rate={t['txns'] / max(wall, 1e-3) / 1e6:.2f} Mtxn/s"
    )
    hrows = [rr[cores.index(h)] for rr in all_rows for h in hammer]
    hw = sum(int(v[0]) | int(v[1]) << 32 for v in hrows) * HAMMER_PAGE / max(wall, 1e-3) / 1e9 / len(all_rows)
    hr = sum(int(v[4]) | int(v[5]) << 32 for v in hrows) * HAMMER_PAGE / max(wall, 1e-3) / 1e9 / len(all_rows)
    print(
        f"DRAM_RAF HAMMER write={hw:.1f} GB/s/chip read={hr:.1f} GB/s/chip "
        f"safety_timeouts={sum(int(v[2]) + int(v[6]) for v in hrows)}"
    )
    assert t["timeouts"] == 0, "a stream stopped making progress (wait timed out)"
    assert t["unresolved"] == 0, "a page never became visible"
    assert t["txns"] == t["written"], "readers and writers disagree on the transaction count"
    assert t["flag_over"] == 0 and t["final_flag"] == 0, "flag counted more increments than writes"
    assert t["mism"] == 0, f"{t['mism']} reads returned stale data after the flag was seen"
