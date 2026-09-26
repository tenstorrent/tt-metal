# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Spatially pipelined streamed routed expert (SwiGLU), one Blackhole chip.

    y = (silu(x @ Wg) * (x @ Wu)) @ Wd        x [M, H], Wg / Wu [H, I], Wd [I, H], M a multiple of 128

gate/up and down run on different cores, so all of them compute at once:
  * 64 gate/up cores, in adjacent pairs (as test_stream_expert_pair.py): M-group g = left / right core of a pair takes
    rows [64 g, 64 g + 64) of each 128-row sub-block, each core owns NP = 2 gate/up pairs; the gate/up weights (only)
    come through the bank readers / forwarders into core A and are pulled by core B, and stay resident per expert in a
    ring of RING_EXPERTS experts; x travels down NCH chains per group. Each core writes its h slice straight into the
    down cores' chain heads (kernels/stream_mm/se3_compute.cpp and se5_recv.cpp built with SE_GU_ONLY).
  * ND down cores, PCD = Ht / ND output tile columns each: their down weights are read from DRAM by their own NCRISC
    into a ring of two experts (se6_dw.cpp); h (both groups, 128 rows) arrives through 2 chains in pieces
    (se6_drecv.cpp); y = h @ Wd accumulates in DST one row tile at a time (se6_dcompute.cpp).
L1 buffers are placed in one arena tensor laid out differently on the two kinds of core (a tensor takes its size on
every core, so separate per-role tensors would not fit).

Weights bfp4 / bfp8 (MIMO_SP_WDTYPE), x and h bfp8, y bf16, LoFi. Tag ``streamsp_H{H}_M{M}_E{E}_w{dtype}``.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mimo_v2_d_p.tests.perf.test_dram_read_fwd import noc_hops
from models.demos.mimo_v2_d_p.tests.perf.test_stream_expert_pair import _pick
from models.demos.mimo_v2_d_p.tests.perf.test_stream_matmul import BF8_TILE, R, _crs

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


def _env_list(name, default, cast=str):
    return [cast(v) for v in os.environ.get(name, default).split(",") if v]


KDIR = "models/demos/mimo_v2_d_p/tests/perf/kernels/stream_mm"
MS = _env_list("MIMO_SP_M", "128,256,512", int)
EXPERTS = int(os.environ.get("MIMO_SP_EXPERTS", "4"))
ITERS = int(os.environ.get("MIMO_SP_ITERS", "3"))
H = int(os.environ.get("MIMO_SP_H", "7168"))
I = int(os.environ.get("MIMO_SP_I", "2048"))
X_SLOTS = int(os.environ.get("MIMO_SP_X_SLOTS", "8"))
NCH = int(os.environ.get("MIMO_SP_CHAINS", "2"))  # x chains (trees) per M-group
X_TREE = int(os.environ.get("MIMO_SP_X_TREE", "0"))  # 1: x goes down binary trees (depth ~log2), 0: chains
READERS = int(os.environ.get("MIMO_SP_READERS", "16"))  # 16 / 8 bank readers + forwarders feed core A; 0: A reads DRAM
ND = int(os.environ.get("MIMO_SP_DOWN", "28" if READERS == 16 else "32"))  # down cores
D_CHAINS = int(os.environ.get("MIMO_SP_DOWN_CHAINS", "2"))
X_PIECES = int(os.environ.get("MIMO_SP_X_PIECES", "4"))  # x blocks travel the chains cut-through in pieces
H_PIECES = int(os.environ.get("MIMO_SP_H_PIECES", "16"))
RING_EXPERTS = float(os.environ.get("MIMO_SP_RING", "2"))  # gate/up weight ring, in experts
HBUF = int(os.environ.get("MIMO_SP_HBUF", "2"))  # h buffers on the down cores (and h slots on the gate/up cores)
READ_BATCH = 1
STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))
KBLK, MT, NP, G = 8, 2, 2, 2
W_DTYPES = {"bf8": (ttnn.bfloat8_b, BF8_TILE), "bf4": (ttnn.bfloat4_b, 576)}
WDTYPES = _env_list("MIMO_SP_WDTYPE", "bf4")
H_TILE = BF8_TILE


def _bank_sharded(regions, banks, dtype, device):
    """Per-core weight regions (equal-size [tiles, 32, 32]) -> one width-sharded DRAM tensor: core i's region is
    contiguous in bank i % banks at byte offset (i // banks) * region bytes."""
    per = -(-len(regions) // banks)
    regions = list(regions) + [torch.zeros_like(regions[0])] * (per * banks - len(regions))  # pad the last row of banks
    host = torch.cat(
        [torch.cat([regions[b + h * banks] for h in range(per)]).reshape(-1, 32) for b in range(banks)], dim=1
    )
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, (host.shape[0], 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


def _chains(cores, n, phys):
    """Greedy NOC0-nearest path through `cores` (indices into a coordinate list via phys), cut into n chains."""
    left = list(cores)
    path = [min(left, key=lambda c: (c[1].y, c[1].x))]
    left.remove(path[0])
    while left:
        nxt = min(left, key=lambda c: noc_hops(phys(path[-1][1]), phys(c[1]), 0))
        path.append(nxt)
        left.remove(nxt)
    L = len(path) // n
    return [path[i * L : (i + 1) * L] if i < n - 1 else path[i * L :] for i in range(n)]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("wdtype", WDTYPES)
@pytest.mark.parametrize("m", MS, ids=lambda m: f"M{m}")
def test_stream_expert_spatial(device, m, wdtype):
    w_dtype, w_tile = W_DTYPES[wdtype]
    banks = device.dram_grid_size().x
    if READERS:
        readers, gu, _, _, phys, grid = _pick(device, READERS)
    else:  # no dedicated readers: 32 horizontally adjacent free pairs, left core A = M-group 0
        grid = device.compute_with_storage_grid_size()
        phys = lambda c: device.worker_core_from_logical_core(c)
        px = [phys(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
        readers, gu = [], []
        for y in range(grid.y):
            x = 0
            while x + 1 < grid.x and len(gu) < 64:
                if px[x + 1] == px[x] + 1:
                    gu += [ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + 1, y)]
                    x += 2
                else:
                    x += 1
    taken = {(c.x, c.y) for c in readers + gu}
    free = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
    assert len(free) >= ND, (len(free), ND)
    down = free[:ND]
    n_rd, ngu = len(readers), len(gu)
    Ht, It, E = H // 32, I // 32, EXPERTS
    n_cg = ngu // G
    assert m % (G * MT * 32) == 0 and It == n_cg * NP and Ht % ND == 0, (m, Ht, It, ngu, ND)
    S = m // (G * MT * 32)
    V = E * S
    nk_gu = Ht // KBLK
    slot = KBLK * 2 * NP  # gate/up block tiles
    ring_g = int(round(RING_EXPERTS * nk_gu))
    pcd = Ht // ND
    kd = max(k for k in (8, 4, 2, 1) if k * pcd <= 16 and It % k == 0)
    nblk = It // kd
    slot_d = kd * pcd
    ring_d = int(round(float(os.environ.get("MIMO_SP_DRING", "2")) * nblk))  # down weight ring, in experts
    assert ring_d % 2 == 0 and ring_d >= nblk
    x_blk = MT * KBLK
    half = n_cg * NP * MT * H_TILE  # one group's h per sub-block
    h_all_tiles = G * half // H_TILE
    mt_all = G * MT
    out_tiles = mt_all * pcd
    cg_of = lambda ci: 2 * (ci // R) + (ci % R) // 2
    g_of = lambda ci: ci % 2

    # ---- arena layout (bytes), 2 KB aligned so the bf16 output reads back as whole tiles ----
    al = lambda b: (b + 2047) // 2048 * 2048
    X_OFF = al(ring_g * slot * w_tile)
    gu_bytes = X_OFF + al(X_SLOTS * x_blk * BF8_TILE)
    H_OFF = al(ring_d * slot_d * w_tile)
    O_OFF = H_OFF + al(HBUF * h_all_tiles * H_TILE)
    D_OFF = O_OFF + al(2 * out_tiles * 2048)  # two output slots; y goes to DRAM
    dn_bytes = D_OFF + 2048  # the coordinator's per-down-core done words
    arena_tiles = max(gu_bytes, dn_bytes) // 2048
    logger.info(
        f"H {H} M {m}: {S} sub-blocks; gate/up ring {ring_g} blocks ({ring_g * slot * w_tile >> 10} KB), down "
        f"{ND} x {pcd} cols (K-block {kd}, ring {ring_d * slot_d * w_tile >> 10} KB); arena {arena_tiles * 2} KB "
        f"(gu {gu_bytes >> 10} KB, down {dn_bytes >> 10} KB)"
    )

    torch.manual_seed(0)
    Wg, Wu, Wd = torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02
    xs = torch.randn(V, G * MT * 32, H)
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=w_dtype, layout=ttnn.TILE_LAYOUT)).float()
    x_last = xs[(E - 1) * S :].reshape(m, H)
    ref = (torch.nn.functional.silu(x_last @ Wg) * (x_last @ Wu)) @ Wd
    ref_q = (torch.nn.functional.silu(x_last @ q(Wg)) * (x_last @ q(Wu))) @ q(Wd)

    # ---- gate/up weights through the bank readers: per reader, per expert, per K-block, its two column groups ----
    tiles = lambda w: w.view(w.shape[0] // 32, 32, w.shape[1] // 32, 32).permute(0, 2, 1, 3)
    Wg_t, Wu_t, Wd_t = tiles(Wg), tiles(Wu), tiles(Wd)
    if not READERS:  # core A of column group cg reads its blocks itself: per cg, per expert, per K-block
        wg = []
        for cg in range(n_cg):
            blocks = [
                torch.stack(
                    [
                        t
                        for p in range(NP)
                        for t in (
                            Wg_t[c * KBLK : (c + 1) * KBLK, cg * NP + p],
                            Wu_t[c * KBLK : (c + 1) * KBLK, cg * NP + p],
                        )
                    ],
                    dim=1,
                ).reshape(-1, 32, 32)
                for c in range(nk_gu)
            ]
            wg.append(torch.cat(blocks).repeat(E, 1, 1))
        wg_dev = _bank_sharded(wg, banks, w_dtype, device)
        wg_region = wg[0].shape[0] * w_tile
    per_reader = []
    RA = n_cg // max(n_rd, 1)  # A cores (column groups) per reader
    for r in range(n_rd):
        blocks = []
        for c in range(nk_gu):
            ks = slice(c * KBLK, (c + 1) * KBLK)
            for cg in range(r * RA, (r + 1) * RA):
                cols = [t for p in range(NP) for t in (Wg_t[ks, cg * NP + p], Wu_t[ks, cg * NP + p])]
                blocks.append(torch.stack(cols, dim=1).reshape(-1, 32, 32))
        per_reader.append(torch.cat(blocks).repeat(E, 1, 1))
    if READERS:
        halves = max(n_rd // banks, 1)
        w_host = torch.cat(
            [torch.cat([per_reader[b + h * banks] for h in range(halves)]).reshape(-1, 32) for b in range(banks)], dim=1
        )
        t_bank = w_host.shape[0] // 32
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        w_dev = ttnn.from_torch(
            w_host,
            dtype=w_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, (t_bank * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        region_bytes = t_bank // halves * w_tile

    # ---- down weights, one interleaved DRAM tensor: per down core, per expert, per K-block, [kd x pcd] tiles ----
    dt = []
    for d in range(ND):
        per_e = torch.cat(
            [Wd_t[c * kd : (c + 1) * kd, d * pcd : (d + 1) * pcd].reshape(-1, 32, 32) for c in range(nblk)]
        )
        dt.append(per_e.repeat(E, 1, 1))
    wd_dev = _bank_sharded(dt, banks, w_dtype, device)
    wd_region = dt[0].shape[0] * w_tile

    # ---- x in DRAM: per virtual expert, K-block, M-group: [64 x 256] ----
    rows = lambda v, g: xs[v][g * MT * 32 : (g + 1) * MT * 32]
    x_host = torch.cat(
        [rows(v, g)[:, b * KBLK * 32 : (b + 1) * KBLK * 32] for v in range(V) for b in range(nk_gu) for g in range(G)]
    )
    x_dram = ttnn.from_torch(
        x_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    y_dram = ttnn.allocate_tensor_on_device(
        ttnn.Shape([V * mt_all * 32, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    gu_crs, dn_crs = _crs(gu), _crs(down)
    rd_crs = _crs(readers) if readers else None
    both = gu + down
    arena = ttnn.from_torch(  # zeroed: the coordinator's done words must start at 0 (it re-zeroes them after a run)
        torch.zeros(len(both) * arena_tiles * 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_crs(both), (arena_tiles * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    base = arena.buffer_address()
    land_addr, x_ring_addr, h_all_addr = base, base + X_OFF, base + H_OFF

    sem_crs = _crs(readers + both)
    AVAIL, BCOPY, DATA, HARR, GO, DONE, GATH, XARR, SFREE, HFREE, HSFREE, SFREE2 = (
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
    )
    GATH1, GATH2 = 14, 15  # slices of h(v) are counted per down-core h buffer v % HBUF
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=sem_crs, initial_value=0) for i in range(16)]
    pk = lambda c: (phys(c).x << 16) | phys(c).y

    # x chains among each group's gate/up cores; h chains among the down cores
    x_pred, x_succ, x_succ2, x_pred_id, x_heads = {}, {}, {}, {}, []
    for g in range(G):
        for seg in _chains([(ci, gu[ci]) for ci in range(ngu) if g_of(ci) == g], NCH, phys):
            nodes = [c for c, _ in seg]
            x_heads.append((nodes[0], g))
            for i, c in enumerate(nodes):
                kids = [2 * i + 1, 2 * i + 2] if X_TREE else [i + 1]  # heap over the NOC0-greedy path
                for k, kid in enumerate(j for j in kids if j < len(nodes)):
                    (x_succ if k == 0 else x_succ2)[c] = nodes[kid]
                    x_pred[nodes[kid]] = c
                    x_pred_id[nodes[kid]] = SFREE if k == 0 else SFREE2
    d_pred, d_succ, d_heads = {}, {}, []
    for seg in _chains([(d, down[d]) for d in range(ND)], D_CHAINS, phys):
        d_heads.append(seg[0][0])
        for a, b in zip(seg, seg[1:]):
            d_succ[a[0]], d_pred[b[0]] = b[0], a[0]
    coord = down[0]
    xy_or0 = lambda lst, i: pk(lst[i]) if i is not None else 0

    rd_rt, fw_rt, gu_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    head_xy = [pk(down[d]) for d in d_heads]
    for r, c in enumerate(readers):
        rd_rt[c.x][c.y] = [w_dev.buffer_address(), r % banks, (r // banks) * region_bytes, 0]
        fw_rt[c.x][c.y] = [land_addr] + [pk(gu[2 * (r * RA + a)]) for a in range(RA)] + [0] * RA
    RR = 2 * RA if readers else R  # gate/up cores per reader
    for r in range(ngu // RR):
        c = readers[r] if readers else gu[r * RR]
        for j in range(RR):
            ci = r * RR + j
            rc, g = gu[ci], g_of(ci)
            gu_rt[rc.x][rc.y] = [
                pk(c),
                j // 2,
                cg_of(ci),
                len(head_xy),
                0,
                pk(coord),
                GATH,
                x_ring_addr,
                0,
                xy_or0(gu, x_pred.get(ci)),
                xy_or0(gu, x_succ.get(ci)),
                0,
                HSFREE,
                h_all_addr,
                g,
                pk(gu[ci ^ 1]),
                AVAIL,
                BCOPY,
                xy_or0(gu, x_succ2.get(ci)),
                x_pred_id.get(ci, SFREE),
                SFREE2,
            ] + head_xy
    xh_rt, bn_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    head_set = {ci for ci, _ in x_heads}
    for ci, g in x_heads:
        if g == 0 and READERS:
            xh_rt[gu[ci].x][gu[ci].y] = [x_dram.buffer_address(), x_ring_addr, g]
    an_rt = ttnn.RuntimeArgs()
    for ci in range(ngu):
        if g_of(ci) == 0 and not READERS:
            cg = cg_of(ci)
            an_rt[gu[ci].x][gu[ci].y] = [
                pk(gu[ci ^ 1]),
                wg_dev.buffer_address(),
                cg % banks,
                int(ci in head_set),
                x_dram.buffer_address(),
                x_ring_addr,
                0,
                (cg // banks) * wg_region,
            ]
    for ci in range(ngu):
        if g_of(ci) == 1:
            bn_rt[gu[ci].x][gu[ci].y] = [
                pk(gu[ci ^ 1]),
                land_addr,
                int(ci in head_set),
                x_dram.buffer_address(),
                x_ring_addr,
                1,
            ]
    dr_rt, dw_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    gu_xy = [pk(c) for c in gu]
    for d, dc in enumerate(down):
        dr_rt[dc.x][dc.y] = [
            h_all_addr,
            xy_or0(down, d_pred.get(d)),
            xy_or0(down, d_succ.get(d)),
            pk(coord),
            int(d == 0),
            ngu,
            y_dram.buffer_address(),
            d * pcd,
            base + D_OFF,
            d,
        ] + gu_xy
        dw_rt[dc.x][dc.y] = [wd_dev.buffer_address(), d % banks, (d // banks) * wd_region]

    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
    FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
    zones = [("SE_ZONES", "1")] if os.environ.get("MIMO_SE_ZONES") else []
    zones += [("SE_WAITZ", os.environ["MIMO_SE_WAITZ"])] if os.environ.get("MIMO_SE_WAITZ") else []
    gu_only = [("SE_GU_ONLY", "1")]
    kernels = (
        []
        if not READERS
        else [
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se_reader.cpp",
                source_type=FP,
                core_ranges=rd_crs,
                compile_time_args=[0, w_tile, RA * slot, READ_BATCH, nk_gu, 0, 1, E, RA * slot, 0],
                runtime_args=rd_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se_forward.cpp",
                source_type=FP,
                core_ranges=rd_crs,
                compile_time_args=[
                    0,
                    RA,
                    w_tile,
                    RA * slot,
                    slot,
                    ring_g,
                    0,
                    DATA,
                    KBLK,
                    nk_gu,
                    0,
                    1,
                    E,
                    READ_BATCH,
                    0,
                    1,
                    slot,
                    0,
                ],
                runtime_args=fw_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se4_xhead.cpp",
                source_type=FP,
                core_ranges=_crs([gu[ci] for ci, g in x_heads if g == 0]),
                compile_time_args=[x_blk, BF8_TILE, V * nk_gu, X_SLOTS, XARR, HFREE, G, X_PIECES],
                runtime_args=xh_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            ),
        ]
    )
    if not READERS:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se7_anc.cpp",
                source_type=FP,
                core_ranges=_crs([gu[ci] for ci in range(ngu) if g_of(ci) == 0]),
                compile_time_args=[
                    1,
                    slot,
                    w_tile,
                    E * nk_gu,
                    ring_g,
                    AVAIL,
                    BCOPY,
                    x_blk,
                    BF8_TILE,
                    V * nk_gu,
                    X_SLOTS,
                    XARR,
                    HFREE,
                    G,
                    X_PIECES,
                    int(wdtype == "bf8"),
                    2,
                ],
                runtime_args=an_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            )
        )
    kernels += [
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se5_bnc.cpp",
            source_type=FP,
            core_ranges=_crs([gu[ci] for ci in range(ngu) if g_of(ci) == 1]),
            compile_time_args=[
                1,
                slot,
                w_tile,
                E * nk_gu,
                ring_g,
                AVAIL,
                BCOPY,
                x_blk,
                BF8_TILE,
                V * nk_gu,
                X_SLOTS,
                XARR,
                HFREE,
                G,
                X_PIECES,
            ],
            runtime_args=bn_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se5_recv.cpp",
            source_type=FP,
            core_ranges=gu_crs,
            compile_time_args=[
                0,
                x_blk,
                1,
                slot,
                E * nk_gu,
                V,
                ring_g,
                16,
                1,
                3,
                2,
                MT,
                H_TILE,
                ngu,
                DATA,
                HARR,
                GO,
                DONE,
                KBLK,
                HARR,
                SFREE,
                HBUF,
                XARR,
                HFREE,
                X_SLOTS,
                x_blk * BF8_TILE,
                S,
                NP,
                HSFREE,
                H_PIECES,
                nk_gu,
                GATH1,
                GATH2,
                X_PIECES,
                G,
            ],
            defines=zones + gu_only + ([] if READERS else [("SE_W_NCRISC", "1")]),
            runtime_args=gu_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se3_compute.cpp",
            source_type=FP,
            core_ranges=gu_crs,
            compile_time_args=[KBLK, MT, nk_gu, 0, 1, E, S, slot, 1, 1, NP, 0, ring_g],
            runtime_args=[],
            defines=zones + gu_only,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se6_drecv.cpp",
            source_type=FP,
            core_ranges=dn_crs,
            compile_time_args=[
                2,
                h_all_tiles,
                H_TILE,
                H_PIECES,
                16,
                out_tiles,
                V,
                S,
                ngu,
                ND,
                HARR,
                HSFREE,
                GATH,
                DONE,
                GO,
                HBUF,
                mt_all,
                pcd,
                Ht,
                GATH,
                GATH1,
                GATH2,
            ],
            runtime_args=dr_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se6_dw.cpp",
            source_type=FP,
            core_ranges=dn_crs,
            compile_time_args=[1, slot_d, w_tile, E * nblk, 2, int(wdtype == "bf8")],
            runtime_args=dw_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se6_dcompute.cpp",
            source_type=FP,
            core_ranges=dn_crs,
            compile_time_args=[MT, G, It, kd, pcd, E, S, slot_d, ring_d],
            runtime_args=[],
            defines=zones,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
        ),
    ]

    fmt = lambda i, d_, page: [ttnn.CBFormatDescriptor(buffer_index=i, data_format=d_, page_size=page)]

    def arena_cb(idx, off, size, crs, d_, page):
        cb = ttnn.cb_descriptor_from_sharded_tensor(idx, arena, address_offset=off, total_size=size, core_ranges=crs)
        cb.format_descriptors = fmt(idx, d_, page)
        return cb

    cbs = (
        []
        + (
            []
            if not READERS
            else [
                ttnn.CBDescriptor(
                    total_size=2 * READ_BATCH * RA * slot * w_tile,
                    core_ranges=rd_crs,
                    format_descriptors=fmt(0, w_dtype, w_tile),
                )
            ]
        )
        + [
            arena_cb(1, 0, ring_g * slot * w_tile, gu_crs, w_dtype, w_tile),
            arena_cb(0, X_OFF, X_SLOTS * x_blk * BF8_TILE, gu_crs, ttnn.bfloat8_b, BF8_TILE),
            ttnn.CBDescriptor(
                total_size=HBUF * MT * NP * H_TILE,
                core_ranges=gu_crs,
                format_descriptors=fmt(3, ttnn.bfloat8_b, H_TILE),
            ),
            # gate/up compute configures its packer from CB 16 (bf16), unused here
            ttnn.CBDescriptor(total_size=2048, core_ranges=gu_crs, format_descriptors=fmt(16, ttnn.bfloat16, 2048)),
            arena_cb(1, 0, ring_d * slot_d * w_tile, dn_crs, w_dtype, w_tile),
            arena_cb(2, H_OFF, HBUF * h_all_tiles * H_TILE, dn_crs, ttnn.bfloat8_b, H_TILE),
            arena_cb(16, O_OFF, 2 * out_tiles * 2048, dn_crs, ttnn.bfloat16, 2048),
        ]
    )
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    tag = f"streamsp_H{H}_M{m}_E{E}_w{wdtype}"
    w_bytes = E * 3 * H * I * w_tile / 1024
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(
            json.dumps(
                {"tag": tag, "M": m, "E": E, "wdtype": wdtype, "weight_bytes": w_bytes, "flops": 6 * E * m * H * I}
            )
            + "\n"
        )
    order = ttnn.corerange_to_cores(_crs(both), None, True)
    shard_of = {(c.x, c.y): i for i, c in enumerate(order)}
    for it in range(1 + ITERS):
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_start")
        ttnn.generic_op(([w_dev] if READERS else [wg_dev]) + [wd_dev, x_dram, arena, y_dram], program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
        if it == 0:
            got = ttnn.to_torch(y_dram).float()[(E - 1) * S * mt_all * 32 :]  # the last expert's rows
            ok, pcc_q = comp_pcc(ref_q, got, 0.99)
            _, pcc = comp_pcc(ref, got, 0.0)
            logger.info(f"{tag}: PCC {pcc_q} vs quantized-weight reference, {pcc} vs fp32")
            assert ok, pcc_q
    logger.info(f"ran {tag}: {w_bytes / 1e6:.0f} MB of weights streamed per run")
