# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""x-download probe: deliver x [M, H] (bfp8) from DRAM into the L1 of all 64 compute cores of the streamed-expert
layout, no weights, no compute. 16 bank readers (test_stream_matmul._pick_cores) each read their share of x from their
bank (kernels/stream_mm/sm_reader.cpp, large contiguous reads) and their NCRISC sends it to every compute core
(xdl_send.cpp): MODE mcast = multicast to the grid rectangles, unicast = one write per compute core (64x replication
traffic), col = unicast to one head per grid column, which multicasts down its column (xdl_head.cpp; column
multicasts run ~49 GB/s each and do not overlap), colchain = readers send each chunk once to the first column head
and the heads form a chain, each multicasting down its column and forwarding to the next (xdl_chead.cpp), halves =
readers send each chunk to one sender per physical column run, which multicasts linked bursts into its run
(xdl_hsend.cpp; linked multicast ~67 GB/s into a 7x10 rectangle), rot = the readers multicast their own chunks as
linked bursts, one reader at a time per column-run rectangle, passing a token around (xdl_rot.cpp). Receivers only take the bytes into a ring. Tag ``xdl_{mode}_M{M}``; "weight_bytes" = the unique x bytes, so
the analyzer's GB/s is unique x bandwidth (x64 for bytes delivered)."""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.mimo_v2_d_p.tests.perf.test_stream_matmul import BF8_TILE, _crs, _pick_cores

STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))
H = int(os.environ.get("MIMO_XDL_H", "7168"))
CHUNK = 16  # tiles per chunk (17 KB)
SLOTS = 32  # receiver ring, chunks
ROUND = int(os.environ.get("MIMO_XDL_ROUND", "4"))  # chunks per reader per token hold (rot)
ROT_RECTS = os.environ.get("MIMO_XDL_ROT_RECTS", "cols")  # rot: one token ring per column, or per column run
AVOID = int(os.environ.get("MIMO_XDL_AVOID_READERS", "0"))
RD_SWAP = int(os.environ.get("MIMO_XDL_RD_SWAP", "0"))  # readers: DRAM reads on NOC1, sends on NOC0
BYR = int(
    os.environ.get("MIMO_XDL_BY_READER", "0")
)  # halves: reader r feeds relay r % RELAYS (else chunk c -> c % RELAYS)
HS_NOC = int(os.environ.get("MIMO_XDL_HS_NOC", "0"))  # NoC of the half-grid senders' multicasts


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("mode", os.environ.get("MIMO_XDL_MODES", "mcast,unicast").split(","))
@pytest.mark.parametrize("m", [int(v) for v in os.environ.get("MIMO_XDL_M", "512").split(",")], ids=lambda m: f"M{m}")
def test_x_download(device, m, mode):
    banks = device.dram_grid_size().x
    readers, receivers, phys = _pick_cores(device)
    if AVOID:  # compute cores only in columns without readers; multicast rectangles then never touch a reader
        grid0 = device.compute_with_storage_grid_size()
        rcols = sorted({c.x for c in readers})
        receivers = [ttnn.CoreCoord(x, y) for x in range(grid0.x) if x not in rcols for y in range(grid0.y)][:64]
    n_rd = len(readers)
    tiles = (m // 32) * (H // 32)
    per = tiles // n_rd
    assert per % CHUNK == 0, (tiles, n_rd)
    nch = per // CHUNK
    halves = n_rd // banks
    host = torch.cat([torch.randn(halves * per * 32, 32) for _ in range(banks)], dim=1)
    grid_dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    x_dev = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid_dram, (halves * per * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    cc_crs = _crs(receivers)
    ring = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(receivers) * SLOTS * CHUNK * 32, 32]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cc_crs, (SLOTS * CHUNK * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    pk = lambda c: (phys(c).x << 16) | phys(c).y
    grid = device.compute_with_storage_grid_size()
    px = [phys(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
    runs, start = [], 0
    for x in range(1, grid.x + 1):
        if x == grid.x or px[x] != px[x - 1] + 1:
            runs.append((start, x - 1))
            start = x
    if AVOID:  # rectangles = maximal runs of physically contiguous columns without readers
        cols = [x for x in range(grid.x) if x not in rcols]
        runs, cur = [], [cols[0]]
        for x in cols[1:]:
            if x == cur[-1] + 1 and px[x] == px[cur[-1]] + 1:
                cur.append(x)
            else:
                runs.append((cur[0], cur[-1]))
                cur = [x]
        runs.append((cur[0], cur[-1]))
        RC = int(os.environ.get("MIMO_XDL_RECT_COLS", "0"))  # split each run into rectangles this many columns wide
        if RC:
            runs = [(x, min(x + RC - 1, x1)) for x0, x1 in runs for x in range(x0, x1 + 1, RC)]
    DONE, ARR, TOK = 0, 1, 2  # TOK + k: rectangle k's token
    rd_crs = _crs(readers)
    # column heads: one core per grid column that is neither a reader nor a compute core where possible
    busy = {(c.x, c.y) for c in readers + receivers}
    heads = []
    for x in range(grid.x):
        col = [ttnn.CoreCoord(x, y) for y in range(grid.y)]
        heads.append(
            next(
                (c for c in col if (c.x, c.y) not in busy),
                next(c for c in col if (c.x, c.y) not in {(r.x, r.y) for r in readers}),
            )
        )
    head_xy = [pk(h) for h in heads]
    # half-grid senders: a free core in each physical column run
    RPG = int(os.environ.get("MIMO_XDL_RELAYS", "1"))  # halves: relays per column run (chunk c -> relay c % RPG)
    hsend = []
    idle = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in busy]
    for x0, x1 in runs:  # nearest idle cores (inside the run if possible; outside is fine for a multicast sender)
        # NOC1 multicasts travel -x: a sender just east of the rectangle enters it directly (west would wrap)
        side = (lambda c: 0 if c.x > x1 else 1) if HS_NOC == 1 else (lambda c: 0)
        cand = sorted(
            idle, key=lambda c: (0 if x0 <= c.x <= x1 else 1, side(c), min(abs(c.x - x0), abs(c.x - x1)), c.y)
        )[:RPG]
        hsend += cand
        idle = [c for c in idle if c not in cand]
    print("relays", [(c.x, c.y) for c in hsend], "runs", runs)
    order = sorted(range(len(heads)), key=lambda x: -phys(heads[x]).x)  # head chain order (NOC1 runs -x)
    rd_rt, sd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    recv_xy = [pk(c) for c in receivers]
    for r, c in enumerate(readers):
        rd_rt[c.x][c.y] = [x_dev.buffer_address(), r % banks, (r // banks) * per * BF8_TILE]
        rects = []
        for x0, x1 in runs:
            hi, lo = ttnn.CoreCoord(x1, grid.y - 1), ttnn.CoreCoord(x0, 0)  # NOC1: high corner first
            rects += [pk(hi), pk(lo), (x1 - x0 + 1) * grid.y - int(x0 <= c.x <= x1)]
        if mode == "rot":
            rr = []
            for x0, x1 in runs if ROT_RECTS == "runs" else [(x, x) for x in range(grid.x)]:
                lo, hi = ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x1, grid.y - 1)
                rr += [pk(hi), pk(lo), (x1 - x0 + 1) * grid.y - int(x0 <= c.x <= x1)]  # NOC1: high corner first
            sd_rt[c.x][c.y] = (
                [ring.buffer_address(), r, n_rd, pk(readers[(r + 1) % n_rd]), r * nch] + rr + [len(recv_xy)] + recv_xy
            )
            continue
        tgt = (
            head_xy
            if mode == "col"
            else [pk(heads[order[0]])]
            if mode == "colchain"
            else [pk(h) for h in hsend]
            if mode == "halves"
            else recv_xy
        )
        sd_rt[c.x][c.y] = [ring.buffer_address(), r * nch, len(tgt)] + tgt + [len(runs)] + rects
    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
    FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
    K = "models/demos/mimo_v2_d_p/tests/perf/kernels/stream_mm"
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{K}/sm_reader.cpp",
            source_type=FP,
            core_ranges=rd_crs,
            compile_time_args=[0, CHUNK, BF8_TILE, nch, 2],
            runtime_args=rd_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1 if RD_SWAP else ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{K}/xdl_rot.cpp" if mode == "rot" else f"{K}/xdl_send.cpp",
            source_type=FP,
            core_ranges=rd_crs,
            compile_time_args=(
                [0, CHUNK, BF8_TILE, nch, ROUND, SLOTS, TOK, len(runs) if ROT_RECTS == "runs" else grid.x, DONE, 0]
                if mode == "rot"
                else [
                    0,
                    CHUNK,
                    BF8_TILE,
                    nch,
                    {"unicast": 0, "mcast": 1, "col": 2, "colchain": 2, "halves": 2, "self": 0}[mode],
                    SLOTS,
                    DONE,
                    ARR,
                    RPG if mode == "halves" else 1,
                    BYR,
                ]
            ),
            runtime_args=sd_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0 if RD_SWAP else ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{K}/xdl_recv.cpp",
            source_type=FP,
            core_ranges=cc_crs,
            compile_time_args=[DONE, 1 if mode in ("col", "colchain", "self") else RPG if mode == "halves" else n_rd],
            runtime_args=[],
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
    ]
    if mode == "col":
        hd_rt = ttnn.RuntimeArgs()
        for x, h in enumerate(heads):
            col_recv = [pk(c) for c in receivers if c.x == x]
            lo, hi = ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, grid.y - 1)
            hd_rt[h.x][h.y] = [
                ring.buffer_address(),
                ring.buffer_address(),
                pk(hi),
                pk(lo),
                grid.y - 1,
                len(col_recv),
            ] + col_recv
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{K}/xdl_head.cpp",
                source_type=FP,
                core_ranges=_crs(heads),
                compile_time_args=[
                    n_rd * nch,
                    CHUNK * BF8_TILE,
                    SLOTS,
                    ARR,
                    DONE,
                    int(os.environ.get("MIMO_XDL_PROBE", "0")),
                ],
                runtime_args=hd_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            )
        )
    if mode == "self":  # the relays read x themselves (no bank readers involved)
        kernels = [k for k in kernels if "xdl_recv" in k.kernel_source]
        s_rt, m_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, h in enumerate(hsend):
            x0, x1 = runs[i // RPG]
            rc = [pk(c) for c in receivers if x0 <= c.x <= x1] if i % RPG == 0 else []  # one relay per run signals
            lo, hi = ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x1, grid.y - 1)
            s_rt[h.x][h.y] = [x_dev.buffer_address(), per * BF8_TILE, i % RPG]
            m_rt[h.x][h.y] = [
                ring.buffer_address(),
                pk(hi),
                pk(lo),
                (x1 - x0 + 1) * grid.y - int(x0 <= h.x <= x1),
                len(rc),
            ] + rc
        B = int(os.environ.get("MIMO_XDL_SBATCH", "4"))
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{K}/xdl_selfread.cpp",
                source_type=FP,
                core_ranges=_crs(hsend),
                compile_time_args=[0, CHUNK * BF8_TILE, n_rd * nch, n_rd, banks, B, CHUNK, RPG],
                runtime_args=s_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{K}/xdl_mcpush.cpp",
                source_type=FP,
                core_ranges=_crs(hsend),
                compile_time_args=[0, CHUNK, CHUNK * BF8_TILE, n_rd * nch // RPG, SLOTS, DONE],
                runtime_args=m_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            )
        )
    if mode == "halves":
        hs_rt = ttnn.RuntimeArgs()
        for (x0, x1), h in [(runs[i // RPG], h) for i, h in enumerate(hsend)]:
            rc = [pk(c) for c in receivers if x0 <= c.x <= x1]
            lo, hi = ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x1, grid.y - 1)
            a, b = (lo, hi) if HS_NOC == 0 else (hi, lo)
            hs_rt[h.x][h.y] = [
                ring.buffer_address(),
                pk(a),
                pk(b),
                (x1 - x0 + 1) * grid.y - int(x0 <= h.x <= x1),
                len(rc),
            ] + rc
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{K}/xdl_hsend.cpp",
                source_type=FP,
                core_ranges=_crs(hsend),
                compile_time_args=[
                    n_rd * nch // RPG,
                    CHUNK * BF8_TILE,
                    SLOTS,
                    ARR,
                    DONE,
                ],  # both splits: 1/RPG of the chunks
                runtime_args=hs_rt,
                defines=[("BURST", os.environ.get("MIMO_XDL_BURST", "8"))]
                + ([("NO_MC", "1")] if os.environ.get("MIMO_XDL_NO_MC") else []),
                config=dm(
                    ttnn.DataMovementProcessor.RISCV_0 if HS_NOC == 0 else ttnn.DataMovementProcessor.RISCV_1,
                    ttnn.NOC.NOC_0 if HS_NOC == 0 else ttnn.NOC.NOC_1,
                ),
            )
        )
    if mode == "colchain":
        hd_rt = ttnn.RuntimeArgs()
        for i, x in enumerate(order):
            h = heads[x]
            col_recv = [pk(c) for c in receivers if c.x == x]
            lo, hi = ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, grid.y - 1)
            nxt = pk(heads[order[i + 1]]) if i + 1 < len(order) else 0
            hd_rt[h.x][h.y] = [ring.buffer_address(), pk(hi), pk(lo), grid.y - 1, nxt, len(col_recv)] + col_recv
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{K}/xdl_chead.cpp",
                source_type=FP,
                core_ranges=_crs(heads),
                compile_time_args=[n_rd * nch, CHUNK * BF8_TILE, SLOTS, ARR, DONE],
                runtime_args=hd_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            )
        )
    cbs = [
        ttnn.CBDescriptor(
            total_size=(2 * ROUND if mode == "rot" else int(os.environ.get("MIMO_XDL_CBCH", "8"))) * CHUNK * BF8_TILE,
            core_ranges=_crs(hsend) if mode == "self" else rd_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat8_b, page_size=BF8_TILE)
            ],
        )
    ]
    sem_cores = {(c.x, c.y): c for c in readers + receivers + heads + hsend}
    sems = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=_crs(list(sem_cores.values())), initial_value=0)
        for i in [DONE, ARR] + list(range(TOK, TOK + 12))
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    tag = f"xdl_{mode}_M{m}_p{os.environ.get('MIMO_XDL_PROBE', '0')}"
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(json.dumps({"tag": tag, "E": 1, "weight_bytes": tiles * BF8_TILE, "flops": 0}) + "\n")
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *a, **k: None
    for it in range(4):
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_start")
        ttnn.generic_op([x_dev, ring], program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
