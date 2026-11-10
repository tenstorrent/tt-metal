# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fabric DRAM read-after-flag: a flushed fabric atomic increment is seen before the data it orders.

Pattern (one probe stream per fabric neighbour pair and link, in +X and +Y; kernels/ has the detail):
  sender (BRISC, chip A)    : seq = 0, 1, 2, ...; ONE fabric packet per seq = fused unicast write of a
                              576 B page into page (seq % 2) of an interleaved DRAM buffer on neighbour B
                              + atomic inc of a flag in the receiver core's L1, flush=true. The DRAM
                              address is resolved for the router's NoC (linear::addrgen_detail::
                              get_noc_address -> NOC1). At most 1 page unacknowledged (P = 2).
  router (EDM, chip B)      : non-posted DRAM write with a transaction id, waits until that id has
                              nothing outstanding (flush), then does the atomic inc (on NOC1).
  receiver (NCRISC, chip B) : waits flag >= 1, noc_async_read of the page on NOC0 (address resolved for
                              NOC0), read barrier, invalidate_l1_cache, checks the page's first / last
                              words == seq; flag -= 1; credit back to the sender over the fabric.
  hammer (optional)         : every free Tensix core streams 8 KB DRAM writes (NOC0) and reads (NOC1).

Expected: flag seen => page visible, so every read returns seq.

Observed on an 8x4 Blackhole Galaxy (torus XY, 576 B, Tensix hammer, job 191618): 5 stale reads in
215M transactions. The flag was exactly 1 every time (no extra increment), one re-read always
returned the new page, 650-9975 cycles after the first read. 4 were the whole page from 2 iterations
earlier (the previous write to that page); 1 was torn (w0 new, w1 and the last word old). All 5 on the
+Y probe, link 0, on different chips. The 5632 B variants were clean at ~2e8 transactions each.

Run (from TT_METAL_HOME; RAF_SECONDS = total run time per test, default 600):
  QuietBox (BH, 2x2):   pytest tests/ttnn/unit_tests/operations/fabric/test_fabric_dram_read_after_flag.py -k quietbox_2x2
  LoudBox  (BH, 2x4):   pytest tests/ttnn/unit_tests/operations/fabric/test_fabric_dram_read_after_flag.py -k loudbox_2x4
  Galaxy   (BH, 8x4):   pytest tests/ttnn/unit_tests/operations/fabric/test_fabric_dram_read_after_flag.py -k galaxy_8x4
  The configuration that reproduced: -k "galaxy_8x4 and 576B and bg_hammer".

Expected failure output (one line per logged mismatch, then the assert):
  DRAM_RAF STALE chip=(7,3) <- (6,3) axis=y link=0 seq=0x... read w0=0x... w1=0x... tail=0x...
      kind=torn(head new, tail old) flag_at_wait=1 rereads=1 cycles=650
  DRAM_RAF TOTAL txns=... mismatches=5 (head=4 torn=1) unresolved=0 ...
  AssertionError: 5 reads returned stale data after the flag was seen
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
P = 2  # pages per stream: page (seq % P) is rewritten every P iterations
TIMEOUT_CYCLES = 1_000_000_000  # per wait; far below the 5 s dispatch timeout
HAMMER_PAGE, HAMMER_PAGES_PER_CORE = 8192, 64
ROW = 128  # uint32 words of per-core results (one L1 shard per core)
PAYLOAD_CB, HAMMER_CB = 0, 1


def _device_params(fabric_config):
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = 12352  # one fabric packet per page up to 12 KB (as in the model)
    return {"fabric_config": fabric_config, "fabric_router_config": rc}


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


@pytest.mark.timeout(SECONDS + 600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((8, 4), _device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY), id="galaxy_8x4"),
        pytest.param((2, 4), _device_params(ttnn.FabricConfig.FABRIC_2D), id="loudbox_2x4"),
        pytest.param((2, 2), _device_params(ttnn.FabricConfig.FABRIC_2D), id="quietbox_2x2"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("payload", [576, 5632], ids=["576B", "5632B"])
@pytest.mark.parametrize("hammer", [True, False], ids=["bg_hammer", "bg_none"])
def test_fabric_dram_read_after_flag(mesh_device, payload, hammer):
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    torus = ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    assert payload <= ttnn.get_tt_fabric_max_payload_size_bytes()
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]  # = results shard order
    grid_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    fid = lambda rc: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*rc))
    virt = mesh_device.worker_core_from_logical_core

    def nbr(r, c, step, sign):
        rr, cc = r + sign * step[0], c + sign * step[1]
        if torus:
            return rr % rows, cc % cols
        return (rr, cc) if 0 <= rr < rows and 0 <= cc < cols else None

    # Probe streams: (axis step, link) for every link that connects all neighbour pairs along that axis in
    # both directions. Data goes +step, credits -step: one worker per router link.
    streams = []
    for name, step in (("x", (0, 1)), ("y", (1, 0))):
        pairs = [((r, c), nbr(r, c, step, +1)) for r in range(rows) for c in range(cols)]
        pairs = [(a, b) for a, b in pairs if b is not None and b != a]
        if not pairs:
            continue
        links = set.intersection(
            *[set(ttnn.get_forwarding_link_indices(mesh_device, fid(a), fid(b))) for a, b in pairs],
            *[set(ttnn.get_forwarding_link_indices(mesh_device, fid(b), fid(a))) for a, b in pairs],
        )
        streams += [(name, step, l) for l in sorted(links)]
    nS = len(streams)
    assert nS > 0, "no fabric neighbours"
    send_core, recv_core = cores[0 : 2 * nS : 2], cores[1 : 2 * nS : 2]
    hammer_cores = cores[2 * nS :] if hammer else []

    def tensor(shape, mem):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=mem,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )

    pages = tensor([1, 1, nS * P, payload // 4], ttnn.DRAM_MEMORY_CONFIG)  # page = row
    ham = tensor([1, 1, max(1, len(hammer_cores)) * HAMMER_PAGES_PER_CORE, HAMMER_PAGE // 4], ttnn.DRAM_MEMORY_CONFIG)
    shard = ttnn.ShardSpec(grid_crs, [1, ROW], ttnn.ShardOrientation.ROW_MAJOR)
    res = tensor(
        [1, 1, len(cores), ROW], ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
    )
    sems = [ttnn.create_global_semaphore(mesh_device, grid_crs, 0) for _ in range(5)]
    flag, credit, state, stop_w, stop_r = [ttnn.get_global_semaphore_address(s) for s in sems]
    ttnn.synchronize_device(mesh_device)
    pages_ct = list(ttnn.TensorAccessorArgs(pages).get_compile_time_args())
    ham_ct = list(ttnn.TensorAccessorArgs(ham).get_compile_time_args())
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
    RV0, RV1, N0, N1 = (
        ttnn.DataMovementProcessor.RISCV_0,
        ttnn.DataMovementProcessor.RISCV_1,
        ttnn.NOC.NOC_0,
        ttnn.NOC.NOC_1,
    )

    def cb(idx, size, cs):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.uint32, page_size=size)
        return ttnn.CBDescriptor(total_size=size, core_ranges=_crs(cs), format_descriptors=[fmt])

    def kernel(name, cs, ct, rt, cfg):
        path = os.path.join(KERNELS, name)
        return ttnn.KernelDescriptor(
            kernel_source=path, core_ranges=_crs(cs), compile_time_args=ct, runtime_args=rt, config=cfg
        )

    def build(n):
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(rows):
            for c in range(cols):
                me = fid((r, c))
                tx = [(s, nbr(r, c, st, +1)) for s, (_, st, _) in enumerate(streams) if nbr(r, c, st, +1)]
                rx = [(s, nbr(r, c, st, -1)) for s, (_, st, _) in enumerate(streams) if nbr(r, c, st, -1)]
                probe = [send_core[s] for s, _ in tx] + [recv_core[s] for s, _ in rx]
                hams = hammer_cores if rx else []  # this chip's receivers stop them
                cbs = [cb(PAYLOAD_CB, payload, probe)] + ([cb(HAMMER_CB, HAMMER_PAGE, hams)] if hams else [])
                pd = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=cbs)
                kernels = []
                tx_rt, rx_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for s, dn in tx:
                    core, rv, peer = send_core[s], virt(recv_core[s]), fid(dn)
                    args = [pages.buffer_address(), s * P, n, rv.x, rv.y, flag, credit, state, res.buffer_address()]
                    args += [int(peer.mesh_id), peer.chip_id, TIMEOUT_CYCLES]
                    tx_rt[core.x][core.y] = args + list(
                        ttnn.setup_fabric_connection(mesh_device, me, peer, streams[s][2], pd, core)
                    )
                for s, up in rx:
                    core, sv, peer = recv_core[s], virt(send_core[s]), fid(up)
                    args = [pages.buffer_address(), s * P, n, sv.x, sv.y, flag, credit, state, res.buffer_address()]
                    args += [int(peer.mesh_id), peer.chip_id, TIMEOUT_CYCLES, len(hams), stop_w, stop_r]
                    args += [v for h in hams for v in (virt(h).x, virt(h).y)]
                    rx_rt[core.x][core.y] = args + list(
                        ttnn.setup_fabric_connection(mesh_device, me, peer, streams[s][2], pd, core)
                    )
                if tx:
                    sc = [send_core[s] for s, _ in tx]
                    kernels.append(
                        kernel("dram_raf_sender.cpp", sc, [payload, P, PAYLOAD_CB] + pages_ct, tx_rt, dm(RV0, N0))
                    )
                if rx:
                    rc_ = [recv_core[s] for s, _ in rx]
                    kernels.append(
                        kernel("dram_raf_receiver.cpp", rc_, [payload, P, PAYLOAD_CB] + pages_ct, rx_rt, dm(RV1, N0))
                    )
                for is_writer, stop, cfg in ((1, stop_w, dm(RV0, N0)), (0, stop_r, dm(RV1, N1))):
                    if not hams:
                        break
                    h_rt = ttnn.RuntimeArgs()
                    for i, h in enumerate(hams):
                        h_rt[h.x][h.y] = [
                            ham.buffer_address(),
                            i * HAMMER_PAGES_PER_CORE,
                            HAMMER_PAGES_PER_CORE,
                            stop,
                            len(rx),
                            4 * TIMEOUT_CYCLES,
                        ]
                    kernels.append(
                        kernel("dram_raf_hammer.cpp", hams, [HAMMER_PAGE, is_writer, HAMMER_CB] + ham_ct, h_rt, cfg)
                    )
                pd.kernels = kernels
                mpd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = pd
        return mpd

    def results():  # [chip][core index] -> row
        return [
            (np.asarray(ttnn.to_torch(t).reshape(-1, ROW).numpy()).astype(np.int64) & 0xFFFFFFFF)
            for t in ttnn.get_device_tensors(res)
        ]

    def totals(all_rows):
        recv = [rr[cores.index(recv_core[s])] for rr in all_rows for s in range(nS)]
        send = [rr[cores.index(send_core[s])] for rr in all_rows for s in range(nS)]
        t = dict(
            txns=sum(int(v[0]) | int(v[1]) << 32 for v in recv),
            sent=sum(int(v[0]) for v in send),
            mism=sum(int(v[2]) for v in recv),
            head=sum(int(v[3]) for v in recv),
            torn=sum(int(v[4]) for v in recv),
            unresolved=sum(int(v[5]) for v in recv),
            max_rereads=max(int(v[6]) for v in recv),
            max_cycles=max(int(v[7]) for v in recv),
            flag_over=sum(int(v[9]) for v in recv),
            final_flag=sum(int(v[10]) for v in recv),
            timeouts=sum(int(v[8]) for v in recv) + sum(int(v[1]) for v in send),
        )
        return t

    logger.info(
        f"DRAM_RAF mesh={rows}x{cols} torus={torus} payload={payload}B P={P} streams/chip={nS} "
        f"{[(a, l) for a, _, l in streams]} hammer_cores/chip={len(hammer_cores)} seconds={SECONDS}"
    )
    io = [pages, ham, res]
    n = 2000  # calibrate: then each launch ~1 s (the dispatch timeout is 5 s)
    mpd = build(n)
    for _ in range(2):
        t0 = time.time()
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
    n = int(max(2000, min(2**30, n / max(time.time() - t0, 1e-3))))
    mpd = build(n)
    t_start, seen = time.time(), (0, 0)
    while time.time() - t_start < SECONDS:
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
        t = totals(results())
        if (t["mism"], t["timeouts"]) != seen:
            seen = (t["mism"], t["timeouts"])
            logger.warning(f"DRAM_RAF after {time.time() - t_start:.0f}s: {t}")
        if t["timeouts"]:
            break
    wall = time.time() - t_start

    all_rows = results()
    for i, rr in enumerate(all_rows):
        chip = divmod(i, cols)
        for s, (axis, step, link) in enumerate(streams):
            row = rr[cores.index(recv_core[s])]
            for j in range(int(row[11])):
                seq, w0, w1, wt, fv, rereads, cyc, age = (int(v) for v in row[16 + 8 * j : 24 + 8 * j])
                head_new, tail_new = w0 == seq and w1 == (~seq & 0xFFFFFFFF), wt == seq ^ 0xA5A5A5A5
                if not head_new and not tail_new and w1 == (~w0 & 0xFFFFFFFF) and wt == w0 ^ 0xA5A5A5A5:
                    kind = f"stale(whole page from {age} iterations earlier)"
                elif head_new:
                    kind = "torn(head new, tail old)"
                elif tail_new:
                    kind = "torn(head old, tail new)"
                else:
                    kind = "garbage"
                print(
                    f"DRAM_RAF STALE chip={chip} <- {nbr(*chip, step, -1)} axis={axis} link={link} seq={seq:#x} "
                    f"read w0={w0:#x} w1={w1:#x} tail={wt:#x} kind={kind} flag_at_wait={fv} rereads={rereads} "
                    f"cycles={cyc}"
                )
    t = totals(all_rows)
    print(
        f"DRAM_RAF TOTAL txns={t['txns']} mismatches={t['mism']} (head={t['head']} torn={t['torn']}) "
        f"unresolved={t['unresolved']} max_rereads={t['max_rereads']} max_cycles={t['max_cycles']} "
        f"flag_over={t['flag_over']} final_flag_nonzero={t['final_flag']} timeouts={t['timeouts']} "
        f"wall={wall:.0f}s rate={t['txns'] / max(wall, 1e-3) / 1e6:.1f} Mtxn/s"
    )
    assert t["timeouts"] == 0, "a probe stream stopped making progress (wait timed out)"
    assert t["txns"] == t["sent"], "receivers and senders disagree on the transaction count"
    assert t["flag_over"] == 0 and t["final_flag"] == 0, "flag counted more increments than packets"
    assert t["mism"] == 0, f"{t['mism']} reads returned stale data after the flag was seen"
