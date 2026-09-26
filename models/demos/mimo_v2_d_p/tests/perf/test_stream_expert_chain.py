# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chained-x M-split big-M streamed routed expert (SwiGLU), one Blackhole chip.

    y = (silu(x @ Wg) * (x @ Wu)) @ Wd        x [M, H], Wg / Wu [H, I], Wd [I, H], M a multiple of 128

Like test_stream_expert_bigm.py (weights resident per expert, the expert's rows run as S = M / 128 sub-blocks), but
the 64 compute cores form two M-groups, one per run of physically contiguous worker columns (one multicast
rectangle each): group g takes rows [64 g, 64 g + 64) of every 128-row sub-block. Each core owns NP = 2 gate/up pairs
and I-tile / H-tile columns accordingly (down: 7 of 224 output tile columns at H 7168), so a group covers all of N
and needs only its own rows of x and its own h. x travels by unicast down NCH chains per group (a chain head's NCRISC
reads the group's rows from DRAM, kernels/stream_mm/se4_xhead.cpp; every core forwards each block to its successor,
se4_recv.cpp): a single core multicasting into a ~70-core rectangle only moves ~12-15 GB/s, a unicast stream ~30.
h is still gathered and multicast once per group by a relay core (se3_bcast.cpp). Each weight block is forwarded to
the two cores (one per group) that share its columns.

Weights bfp4 / bfp8 (MIMO_SM_WDTYPE), x and h bfp8, y bf16, LoFi. Tag ``streamch_H{H}_M{M}_E{E}_w{dtype}_c{NCH}``;
weight GB/s = E * 3 * H * I * tile_bytes / 1024 / time.
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
from models.demos.mimo_v2_d_p.tests.perf.test_stream_matmul import BF8_TILE, R, _crs

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


def _env_list(name, default, cast=str):
    return [cast(v) for v in os.environ.get(name, default).split(",") if v]


KDIR = "models/demos/mimo_v2_d_p/tests/perf/kernels/stream_mm"
MS = _env_list("MIMO_SM_M", "128,256,512", int)
EXPERTS = int(os.environ.get("MIMO_SM_EXPERTS", "4"))
ITERS = int(os.environ.get("MIMO_SM_ITERS", "3"))
H = int(os.environ.get("MIMO_SM_H", "7168"))
I = int(os.environ.get("MIMO_SM_I", "2048"))
X_SLOTS = int(os.environ.get("MIMO_SM_X_SLOTS", "4"))
XS_SLOTS = int(os.environ.get("MIMO_SM_XS_SLOTS", "4"))
PIECE = int(os.environ.get("MIMO_SM_PIECE", "16384"))
READ_BATCH = int(os.environ.get("MIMO_SM_READ_BATCH", "1"))
NCH = int(os.environ.get("MIMO_SM_CHAINS", "4"))  # x chains per M-group
STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))
KBLK, MT, NP, G = 8, 2, 2, 2  # gate/up K-block, sub-block rows per group (tiles), pairs per core, M-groups
W_DTYPES = {"bf8": (ttnn.bfloat8_b, BF8_TILE), "bf4": (ttnn.bfloat4_b, 576)}
WDTYPES = _env_list("MIMO_SM_WDTYPE", "bf4")
H_TILE = BF8_TILE


def _pick(device):
    """16 bank readers (as test_stream_matmul) and, per reader, one core per M-group for each of its two column groups
    (receiver j: column group 2r + j // 2, M-group j % 2), nearest by NOC1 hops within the group's column run."""
    grid = device.compute_with_storage_grid_size()
    phys = lambda c: device.worker_core_from_logical_core(c)
    opt = list(device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0))
    readers = opt + [ttnn.CoreCoord(c.x + 1, c.y) for c in opt]
    px = [phys(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
    runs, start = [], 0
    for x in range(1, grid.x + 1):
        if x == grid.x or px[x] != px[x - 1] + 1:
            runs.append((start, x - 1))
            start = x
    assert len(runs) == G, runs
    taken = {(c.x, c.y) for c in readers}
    receivers = []
    for r in readers:
        got = {}
        for g in range(G):
            x0, x1 = runs[g]
            cand = sorted(
                (ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(x0, x1 + 1) if (x, y) not in taken),
                key=lambda f: (noc_hops(phys(r), phys(f), 1), f.y, f.x),
            )[:2]
            assert len(cand) == 2, "not enough free cores"
            taken |= {(f.x, f.y) for f in cand}
            got[g] = cand
        receivers += [got[0][0], got[1][0], got[0][1], got[1][1]]  # j = 2 * (column group) + M-group
    free = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
    relay = min(free, key=lambda c: (c.x - grid.x / 2) ** 2 + (c.y - grid.y / 2) ** 2)
    return readers, receivers, relay, runs, phys, grid


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("wdtype", WDTYPES)
@pytest.mark.parametrize("m", MS, ids=lambda m: f"M{m}")
def test_stream_expert_chain(device, m, wdtype):
    w_dtype, w_tile = W_DTYPES[wdtype]
    banks = device.dram_grid_size().x
    readers, receivers, relay, runs, phys, grid = _pick(device)
    n_rd, ncc = len(readers), len(receivers)
    Ht, It, E = H // 32, I // 32, EXPERTS
    n_cg = ncc // G  # column groups
    assert m % (G * MT * 32) == 0 and It == n_cg * NP and Ht % n_cg == 0, (m, Ht, It, ncc)
    S = m // (G * MT * 32)
    V = E * S
    nk_gu = Ht // KBLK
    pcd = Ht // n_cg
    slot = KBLK * 2 * NP
    kd = max(k for k in (8, 4, 2, 1) if k * pcd <= slot and It % k == 0)
    nk_dd = It // kd
    bpe = nk_gu + nk_dd
    rt_d = max(r for r in (4, 2, 1) if MT % r == 0 and r * pcd <= 8)
    gu_chunk, d_chunk = 2 * slot, 2 * kd * pcd  # a chunk holds the blocks of the reader's two column groups
    rd_slot = max(gu_chunk, d_chunk)
    x_blk = MT * KBLK
    half = n_cg * NP * MT * H_TILE  # one group's h per sub-block
    cg_of = lambda ci: 2 * (ci // R) + (ci % R) // 2
    g_of = lambda ci: ci % 2
    logger.info(
        f"H {H} M {m}: {S} sub-blocks of {G} x {MT * 32} rows, weights {bpe * slot * w_tile >> 10} KB resident, "
        f"down {pcd} cols (K-block {kd}, row passes of {rt_d}), relay {(relay.x, relay.y)}"
    )

    torch.manual_seed(0)
    Wg, Wu, Wd = torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02
    xs = torch.randn(V, G * MT * 32, H)  # every sub-block of every expert has its own rows
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=w_dtype, layout=ttnn.TILE_LAYOUT)).float()
    x_last = xs[(E - 1) * S :].reshape(m, H)
    ref = (torch.nn.functional.silu(x_last @ Wg) * (x_last @ Wu)) @ Wd
    ref_q = (torch.nn.functional.silu(x_last @ q(Wg)) * (x_last @ q(Wu))) @ q(Wd)

    # ---- weights: per reader, per expert: gu blocks [KBLK x (g, u) x NP] per column group, then down [kd x pcd] ----
    tiles = lambda w: w.view(w.shape[0] // 32, 32, w.shape[1] // 32, 32).permute(0, 2, 1, 3)
    Wg_t, Wu_t, Wd_t = tiles(Wg), tiles(Wu), tiles(Wd)
    per_reader = []
    for r in range(n_rd):
        blocks = []
        for c in range(nk_gu):
            ks = slice(c * KBLK, (c + 1) * KBLK)
            for cg in (2 * r, 2 * r + 1):
                cols = [t for p in range(NP) for t in (Wg_t[ks, cg * NP + p], Wu_t[ks, cg * NP + p])]
                blocks.append(torch.stack(cols, dim=1).reshape(-1, 32, 32))
        for c in range(nk_dd):
            for cg in (2 * r, 2 * r + 1):
                blocks.append(Wd_t[c * kd : (c + 1) * kd, cg * pcd : (cg + 1) * pcd].reshape(-1, 32, 32))
        per_reader.append(torch.cat(blocks).repeat(E, 1, 1))
    halves = n_rd // banks
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

    # ---- x in DRAM: per virtual expert, K-block, M-group: [64 x 256] (16 consecutive tiles, row-major) ----
    rows = lambda v, g: xs[v][g * MT * 32 : (g + 1) * MT * 32]
    x_host = torch.cat(
        [rows(v, g)[:, b * KBLK * 32 : (b + 1) * KBLK * 32] for v in range(V) for b in range(nk_gu) for g in range(G)]
    )
    x_dram = ttnn.from_torch(
        x_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    cc_crs = _crs(receivers)
    cc_order = ttnn.corerange_to_cores(cc_crs, None, True)
    rl_crs = _crs([relay])
    hs = lambda h, w: ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cc_crs, (h, w), ttnn.ShardOrientation.ROW_MAJOR),
    )
    alloc = lambda rows_, cols, dt, mc: ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows_, cols]), dt, ttnn.TILE_LAYOUT, device, mc
    )
    x_ring = alloc(ncc * X_SLOTS * MT * 32, KBLK * 32, ttnn.bfloat8_b, hs(X_SLOTS * MT * 32, KBLK * 32))
    land = alloc(ncc * bpe * slot * 32, 32, w_dtype, hs(bpe * slot * 32, 32))
    # h region: the compute cores' h_all is its first half; the relay (L1 buffers share addresses on every core) keeps
    # one gathered buffer per M-group in it.
    h_tiles = half // H_TILE
    h_reg = alloc(ncc * G * h_tiles * 32, 32, ttnn.bfloat8_b, hs(G * h_tiles * 32, 32))
    out = alloc(ncc * S * MT * 32, pcd * 32, ttnn.bfloat16, hs(S * MT * 32, pcd * 32))
    # x chains: per group, a greedy NOC0-nearest path through the group's cores, cut into NCH chains.
    chain_pred, chain_succ, heads = {}, {}, []
    for g in range(G):
        left = [ci for ci in range(ncc) if g_of(ci) == g]
        path = [min(left, key=lambda ci: (receivers[ci].y, receivers[ci].x))]
        left.remove(path[0])
        while left:
            nxt = min(left, key=lambda ci: noc_hops(phys(receivers[path[-1]]), phys(receivers[ci]), 0))
            path.append(nxt)
            left.remove(nxt)
        L = len(path) // NCH
        for c in range(NCH):
            seg = path[c * L : (c + 1) * L] if c < NCH - 1 else path[c * L :]
            heads.append((seg[0], g))
            for a, b in zip(seg, seg[1:]):
                chain_succ[a], chain_pred[b] = b, a

    rd_crs, all_crs = _crs(readers), _crs(readers + receivers + [relay])
    DATA, HARR, GO, DONE, HARR1 = R, R + 1, R + 2, R + 3, R + 4
    GATH, XARR, SFREE, HFREE = R + 5, R + 7, R + 8, R + 9  # GATH: one per M-group
    # The relay's counter source words (per group, x and h) reuse ids it has no other use for (at most 16 semaphores).
    XW, HW = [DATA, GO], [DONE, HARR1]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in range(R + 10)]
    xy_or0 = lambda ci: pk(receivers[ci]) if ci is not None else 0
    pk = lambda c: (phys(c).x << 16) | phys(c).y
    peers = [pk(c) for c in receivers]

    rd_rt, fw_rt, rv_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for r, c in enumerate(readers):
        rd_rt[c.x][c.y] = [w_dev.buffer_address(), r % banks, (r // banks) * region_bytes, d_chunk]
        fw_rt[c.x][c.y] = [land.buffer_address()] + [pk(receivers[r * R + j]) for j in range(R)] + [kd * pcd] * R
        for j in range(R):
            ci = r * R + j
            rc, g = receivers[ci], g_of(ci)
            rv_rt[rc.x][rc.y] = [
                pk(c),
                j,
                cg_of(ci),
                pk(relay),
                h_reg.buffer_address() + g * half,
                pk(receivers[0]),
                GATH + g,
                x_ring.buffer_address(),
                int(ci == 0),
                xy_or0(chain_pred.get(ci)),
                xy_or0(chain_succ.get(ci)),
            ] + peers
    rl_rt, xr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    args = [x_ring.buffer_address(), h_reg.buffer_address(), h_reg.buffer_address()]
    for g, (x0, x1) in enumerate(runs):
        lo, hi = ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x1, grid.y - 1)
        n = (x1 - x0 + 1) * grid.y - int(x0 <= relay.x <= x1)
        args += [pk(lo), pk(hi), n, GATH + g, 0, XW[g], HW[g]]
    rl_rt[relay.x][relay.y] = args
    xh_rt = ttnn.RuntimeArgs()
    for ci, g in heads:
        xh_rt[receivers[ci].x][receivers[ci].y] = [x_dram.buffer_address(), x_ring.buffer_address(), g]
    xh_crs = _crs([receivers[ci] for ci, _ in heads])

    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
    FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
    zones = [("SE_ZONES", "1")] if os.environ.get("MIMO_SE_ZONES") else []
    total_x = V * nk_gu * G
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se_reader.cpp",
            source_type=FP,
            core_ranges=rd_crs,
            compile_time_args=[0, w_tile, rd_slot, READ_BATCH, nk_gu, nk_dd, 1, E, gu_chunk, 0],
            runtime_args=rd_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se_forward.cpp",
            source_type=FP,
            core_ranges=rd_crs,
            compile_time_args=[
                0,
                R,
                w_tile,
                rd_slot,
                slot,
                bpe,
                0,
                DATA,
                KBLK,
                nk_gu,
                nk_dd,
                1,
                E,
                READ_BATCH,
                0,
                2,
                slot,
                int(os.environ.get("MIMO_SM_PAIR", "0")),
            ],
            runtime_args=fw_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se3_bcast.cpp",
            source_type=FP,
            core_ranges=rl_crs,
            compile_time_args=[0, x_blk, BF8_TILE, 0, X_SLOTS, V, n_cg, half, PIECE, HARR, XARR, G],
            runtime_args=rl_rt,
            defines=[("SE_NO_LINK", "1")] if os.environ.get("MIMO_SM_NO_LINK") else [],
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se4_xhead.cpp",
            source_type=FP,
            core_ranges=xh_crs,
            compile_time_args=[x_blk, BF8_TILE, V * nk_gu, X_SLOTS, XARR, HFREE, G],
            runtime_args=xh_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se4_recv.cpp",
            source_type=FP,
            core_ranges=cc_crs,
            compile_time_args=[
                0,
                x_blk,
                1,
                slot,
                E * bpe,
                V,
                bpe,
                16,
                MT * pcd,
                3,
                2,
                MT,
                H_TILE,
                ncc,
                DATA,
                HARR,
                GO,
                DONE,
                KBLK,
                HARR1,
                SFREE,
                1,
                XARR,
                HFREE,
                X_SLOTS,
                x_blk * BF8_TILE,
                S,
                NP,
            ],
            runtime_args=rv_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se3_compute.cpp",
            source_type=FP,
            core_ranges=cc_crs,
            compile_time_args=[KBLK, MT, nk_gu, It, kd, E, S, slot, pcd, rt_d, NP, 0, 0],
            runtime_args=[],
            defines=zones,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
        ),
    ]
    fmt = lambda i, dt, page: [ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)]
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * READ_BATCH * rd_slot * w_tile, core_ranges=rd_crs, format_descriptors=fmt(0, w_dtype, w_tile)
        ),
        ttnn.CBDescriptor(
            total_size=x_blk * BF8_TILE,
            core_ranges=rl_crs,  # unused (no x on the relay)
            format_descriptors=fmt(0, ttnn.bfloat8_b, BF8_TILE),
        ),
        ttnn.cb_descriptor_from_sharded_tensor(0, x_ring),
        ttnn.cb_descriptor_from_sharded_tensor(1, land),
        ttnn.cb_descriptor_from_sharded_tensor(2, h_reg, total_size=half),
        ttnn.CBDescriptor(
            total_size=MT * NP * H_TILE, core_ranges=cc_crs, format_descriptors=fmt(3, ttnn.bfloat8_b, H_TILE)
        ),
        ttnn.cb_descriptor_from_sharded_tensor(16, out),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    tag = f"streamch_H{H}_M{m}_E{E}_w{wdtype}_c{NCH}"
    w_bytes = E * 3 * H * I * w_tile / 1024
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(
            json.dumps(
                {"tag": tag, "M": m, "E": E, "wdtype": wdtype, "weight_bytes": w_bytes, "flops": 6 * E * m * H * I}
            )
            + "\n"
        )
    for it in range(1 + ITERS):
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_start")
        ttnn.generic_op([w_dev, x_dram, x_ring, land, h_reg, out], program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
        if it == 0:
            # Each core's out ring: slot s = sub-block s of the last expert, [MT x pcd] tiles row-major.
            got_sh = ttnn.to_torch(out).float().view(ncc, S, MT * 32, pcd * 32)
            got = torch.zeros(m, H)
            ci_of = {(c.x, c.y): i for i, c in enumerate(receivers)}
            for s_, core in enumerate(cc_order):
                ci = ci_of[(core.x, core.y)]
                g, cg = g_of(ci), cg_of(ci)
                for sb in range(S):
                    r0 = sb * G * MT * 32 + g * MT * 32
                    got[r0 : r0 + MT * 32, cg * pcd * 32 : (cg + 1) * pcd * 32] = got_sh[s_, sb]
            ok, pcc_q = comp_pcc(ref_q, got, 0.99)
            if not ok or os.environ.get("MIMO_SM_DEBUG"):
                bad = []
                for ci in range(ncc):
                    g, cg = g_of(ci), cg_of(ci)
                    for sb in range(S):
                        r0 = sb * G * MT * 32 + g * MT * 32
                        blk = lambda t: t[r0 : r0 + MT * 32, cg * pcd * 32 : (cg + 1) * pcd * 32]
                        e = ((blk(got) - blk(ref_q)).norm() / blk(ref_q).norm()).item()
                        if e > 0.2:
                            bad.append((ci, (receivers[ci].x, receivers[ci].y), g, sb, round(e, 2)))
                logger.warning(f"{len(bad)} bad core blocks: {bad[:8]}")
                # h of the last virtual expert, per group, from a core of each group: [K-blocks][MT][8] tiles
                hr = ttnn.to_torch(h_reg).float().view(ncc, G, It // KBLK, MT, KBLK, 32, 32)
                shard_of = {(c.x, c.y): i for i, c in enumerate(cc_order)}
                xv = xs[V - 1]
                for g in range(G):
                    sh = shard_of[(receivers[g].x, receivers[g].y)]
                    got_h = hr[sh, 0].permute(1, 3, 0, 2, 4).reshape(MT * 32, It * 32)
                    xr_ = xv[g * MT * 32 : (g + 1) * MT * 32]
                    h_ref = torch.nn.functional.silu(xr_ @ q(Wg)) * (xr_ @ q(Wu))
                    errs = [
                        (
                            (got_h[:, k * 32 : (k + 1) * 32] - h_ref[:, k * 32 : (k + 1) * 32]).norm()
                            / h_ref[:, k * 32 : (k + 1) * 32].norm()
                        ).item()
                        for k in range(It)
                    ]
                    badk = [k for k, e in enumerate(errs) if e > 0.45]
                    logger.warning(
                        f"  group {g}: h err median {sorted(errs)[It // 2]:.3f} max {max(errs):.3f}; bad h K-tiles {badk[:8]} (column groups {sorted({k // NP for k in badk})}, readers {sorted({k // NP // 2 for k in badk})})"
                    )
            _, pcc = comp_pcc(ref, got, 0.0)
            logger.info(f"{tag}: PCC {pcc_q} vs quantized-weight reference, {pcc} vs fp32")
            assert ok, pcc_q
    logger.info(f"ran {tag}: {w_bytes / 1e6:.0f} MB of weights streamed per run")
