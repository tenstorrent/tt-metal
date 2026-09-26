# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Big-M streamed routed expert (SwiGLU), one Blackhole chip: weights read from DRAM once per expert, reused across
the expert's M rows.

    y = (silu(x @ Wg) * (x @ Wu)) @ Wd        x [M, H], Wg / Wu [H, I], Wd [I, H], M a multiple of 128

Same weight stream as test_stream_expert.py (16 readers: BRISC reads the bank on NOC0, NCRISC forwards on NOC1 to 4
compute cores each, credit-gated), but each compute core's in1 ring holds its whole weight slice of one expert
(gate/up blocks [8 x 2], down blocks [KBLK_D x PCD]) and the expert's M rows run as S = M / 128 sub-blocks
("virtual experts") that all reuse it (kernels/stream_mm/se2_compute.cpp); a block is freed after its last use so the
next expert streams in behind the current one.

x is honest here: every sub-block of every expert has its own rows, read from an interleaved DRAM tensor and
multicast K-block by K-block into each compute core's x ring by a dedicated broadcaster core (se2_xread.cpp on NCRISC,
se2_bcast.cpp on BRISC), which also gathers and multicasts h (the only multicast sender in the program). The output
keeps the last expert's M rows. Weights bfp4 / bfp8 (MIMO_SB_WDTYPE), x and h bfp8, y bf16, LoFi.
Tag ``streambig_H{H}_M{M}_E{E}_w{dtype}``; weight GB/s = E * 3 * H * I * tile_bytes / 1024 / time.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mimo_v2_d_p.tests.perf.test_stream_matmul import BF8_TILE, R, _crs, _pick_cores

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


def _env_list(name, default, cast=str):
    return [cast(v) for v in os.environ.get(name, default).split(",") if v]


KDIR = "models/demos/mimo_v2_d_p/tests/perf/kernels/stream_mm"
MS = _env_list("MIMO_SB_M", "128,256,512", int)
EXPERTS = int(os.environ.get("MIMO_SB_EXPERTS", "4"))
ITERS = int(os.environ.get("MIMO_SB_ITERS", "3"))
H = int(os.environ.get("MIMO_SB_H", "7168"))
I = int(os.environ.get("MIMO_SB_I", "2048"))
HBUF = os.environ.get("MIMO_SB_HBUF", "auto")
X_SLOTS = int(os.environ.get("MIMO_SB_X_SLOTS", "3"))  # compute cores' x ring (K-blocks)
XS_SLOTS = int(os.environ.get("MIMO_SB_XS_SLOTS", "3"))  # broadcaster staging (K-blocks)
PIECE = int(os.environ.get("MIMO_SB_PIECE", "16384"))
BANDS = int(os.environ.get("MIMO_SB_BANDS", "1"))  # bands of grid rows, each with its own x / h broadcaster
# "cols": one broadcaster per run of physically contiguous columns (each sends one copy into one rectangle);
# "rows": BANDS bands of grid rows (each broadcaster covers every column run of its rows)
SPLIT = os.environ.get("MIMO_SB_SPLIT", "cols")
H_SINGLE = int(os.environ.get("MIMO_SB_H_SINGLE", "0"))  # 1: broadcaster 0 serves h to the whole grid
READ_BATCH = 2
STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))
KBLK, MT = 8, 4  # gate/up K-block; sub-block rows (tiles)
W_DTYPES = {"bf8": (ttnn.bfloat8_b, BF8_TILE), "bf4": (ttnn.bfloat4_b, 576)}
WDTYPES = _env_list("MIMO_SB_WDTYPE", "bf4")
H_TILE = BF8_TILE


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("wdtype", WDTYPES)
@pytest.mark.parametrize("m", MS, ids=lambda m: f"M{m}")
def test_stream_expert_bigm(device, m, wdtype):
    w_dtype, w_tile = W_DTYPES[wdtype]
    banks = device.dram_grid_size().x
    readers, receivers, phys = _pick_cores(device)
    n_rd, ncc = len(readers), len(receivers)
    Ht, It, E = H // 32, I // 32, EXPERTS
    assert m % (MT * 32) == 0 and It == ncc and Ht % n_rd == 0, (m, Ht, It, ncc)
    S = m // (MT * 32)
    V = E * S
    nk_gu = Ht // KBLK
    per_rd = Ht // n_rd
    pcd_of = [per_rd // R + (j < per_rd % R) for j in range(R)] * n_rd
    col0 = [sum(pcd_of[:ci]) for ci in range(ncc)]
    pcd_max = max(pcd_of)
    slot = KBLK * 2
    kd = max(k for k in (8, 4, 2, 1) if k * pcd_max <= slot and It % k == 0)  # down K-block
    nk_dd = It // kd
    bpe = nk_gu + nk_dd
    rt_of = lambda g: max(r for r in (4, 2, 1) if MT % r == 0 and r * g <= 8)
    gu_chunk = R * slot
    d_chunk = [kd * sum(pcd_of[r * R + j] for j in range(R)) for r in range(n_rd)]
    rd_slot = max([gu_chunk] + d_chunk)
    x_blk = MT * KBLK
    h_bytes = ncc * MT * H_TILE
    fixed = bpe * slot * w_tile + X_SLOTS * x_blk * BF8_TILE + S * MT * pcd_max * 2048
    hbuf = int(HBUF) if HBUF != "auto" else (2 if fixed + 2 * h_bytes < 1250 * 1024 else 1)
    logger.info(
        f"H {H} M {m} ({S} sub-blocks): weights {bpe * slot * w_tile >> 10} KB resident ({bpe} blocks, down "
        f"K-block {kd}), h_all {hbuf} x {h_bytes >> 10} KB, L1 ~{(fixed + hbuf * h_bytes) >> 10} KB"
    )

    # Bands, each served by its own broadcaster (a free core of the band nearest its centre) whose multicasts cover
    # exactly the band, so the broadcasters' destinations never overlap.
    grid = device.compute_with_storage_grid_size()
    taken = {(c.x, c.y) for c in readers + receivers}
    free = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
    px = [phys(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
    col_runs, start = [], 0
    for x in range(1, grid.x + 1):
        if x == grid.x or px[x] != px[x - 1] + 1:  # a run of physically contiguous columns
            col_runs.append((start, x - 1))
            start = x
    if SPLIT == "cols":
        bands = [[(x0, x1, 0, grid.y - 1)] for x0, x1 in col_runs]
    else:
        bands = [
            [(x0, x1, b * grid.y // BANDS, (b + 1) * grid.y // BANDS - 1) for x0, x1 in col_runs] for b in range(BANDS)
        ]
    inside = lambda c, rs: any(x0 <= c.x <= x1 and y0 <= c.y <= y1 for x0, x1, y0, y1 in rs)
    band_of = lambda c: next(b for b, rs in enumerate(bands) if inside(c, rs))
    bcs = []
    for rs in bands:
        cand = [c for c in free if inside(c, rs)]
        assert cand, f"no free core in band {rs}"
        cx = sum(x0 + x1 for x0, x1, _, _ in rs) / (2 * len(rs))
        cy = sum(y0 + y1 for _, _, y0, y1 in rs) / (2 * len(rs))
        bcs.append(min(cand, key=lambda c: (c.x - cx) ** 2 + (c.y - cy) ** 2))
    logger.info(f"{len(bands)} broadcaster band(s) ({SPLIT}): {[(c.x, c.y) for c in bcs]}")

    torch.manual_seed(0)
    Wg, Wu, Wd = torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02
    xs = torch.randn(V, MT * 32, H)  # every sub-block of every expert has its own rows
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=w_dtype, layout=ttnn.TILE_LAYOUT)).float()
    x_last = xs[(E - 1) * S :].reshape(m, H)
    ref = (torch.nn.functional.silu(x_last @ Wg) * (x_last @ Wu)) @ Wd
    ref_q = (torch.nn.functional.silu(x_last @ q(Wg)) * (x_last @ q(Wu))) @ q(Wd)

    # ---- weights: per reader, per expert: gu blocks (K-block major, receiver minor), then down blocks ----
    tiles = lambda w: w.view(w.shape[0] // 32, 32, w.shape[1] // 32, 32).permute(0, 2, 1, 3)
    Wg_t, Wu_t, Wd_t = tiles(Wg), tiles(Wu), tiles(Wd)
    per_reader = []
    for r in range(n_rd):
        blocks = []
        for c in range(nk_gu):
            for j in range(R):
                ci = r * R + j
                ks = slice(c * KBLK, (c + 1) * KBLK)
                blocks.append(torch.stack([Wg_t[ks, ci], Wu_t[ks, ci]], dim=1).reshape(-1, 32, 32))
        for c in range(nk_dd):
            for j in range(R):
                ci = r * R + j
                blocks.append(Wd_t[c * kd : (c + 1) * kd, col0[ci] : col0[ci] + pcd_of[ci]].reshape(-1, 32, 32))
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

    # ---- x in DRAM: per virtual expert, per K-block, [128 x 256] (32 consecutive tiles, row-major) ----
    x_host = torch.cat([xs[v][:, b * KBLK * 32 : (b + 1) * KBLK * 32] for v in range(V) for b in range(nk_gu)])
    x_dram = ttnn.from_torch(
        x_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    cc_crs = _crs(receivers)
    cc_order = ttnn.corerange_to_cores(cc_crs, None, True)
    bc_crs = _crs(bcs)
    hs = lambda crs, h, w: ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, (h, w), ttnn.ShardOrientation.ROW_MAJOR),
    )
    alloc = lambda rows, cols, dt, mc: ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, cols]), dt, ttnn.TILE_LAYOUT, device, mc
    )
    x_ring = alloc(ncc * X_SLOTS * MT * 32, KBLK * 32, ttnn.bfloat8_b, hs(cc_crs, X_SLOTS * MT * 32, KBLK * 32))
    land = alloc(ncc * bpe * slot * 32, 32, w_dtype, hs(cc_crs, bpe * slot * 32, 32))
    h_all = alloc(ncc * hbuf * ncc * MT * 32, 32, ttnn.bfloat8_b, hs(cc_crs, hbuf * ncc * MT * 32, 32))
    out = alloc(ncc * S * MT * 32, pcd_max * 32, ttnn.bfloat16, hs(cc_crs, S * MT * 32, pcd_max * 32))
    # L1 buffers get the same address on every core, so h_all's range is free on the broadcasters too: they gather there.

    rd_crs, all_crs = _crs(readers), _crs(readers + receivers + bcs)
    DATA, HARR, GO, DONE, HARR1, GATH, XARR, XCRED = range(R, R + 8)
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in range(R + 8)]
    pk = lambda c: (phys(c).x << 16) | phys(c).y
    peers = [pk(c) for c in receivers]

    def band_rects(b, rs=None):
        args = []
        for x0, x1, y0, y1 in bands[b] if rs is None else rs:
            args += [
                pk(ttnn.CoreCoord(x0, y0)),
                pk(ttnn.CoreCoord(x1, y1)),
                (x1 - x0 + 1) * (y1 - y0 + 1) - int(inside(bcs[b], [(x0, x1, y0, y1)])),
            ]
        return [len(args) // 3] + args

    rd_rt, fw_rt, rv_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for r, c in enumerate(readers):
        rd_rt[c.x][c.y] = [w_dev.buffer_address(), r % banks, (r // banks) * region_bytes, d_chunk[r]]
        fw_rt[c.x][c.y] = (
            [land.buffer_address()]
            + [pk(receivers[r * R + j]) for j in range(R)]
            + [kd * pcd_of[r * R + j] for j in range(R)]
        )
        for j in range(R):
            ci = r * R + j
            rc = receivers[ci]
            h_bcs = bcs[:1] if H_SINGLE else bcs  # broadcasters that gather h
            rv_rt[rc.x][rc.y] = (
                [pk(c), j, ci, pk(bcs[band_of(rc)]), h_all.buffer_address(), pk(receivers[0]), len(h_bcs)]
                + [pk(b) for b in h_bcs]
                + peers
            )
    bc_rt, xr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for b, bc in enumerate(bcs):
        band_ncc = sum(1 for c in receivers if band_of(c) == b)
        h_rects = band_rects(b) if not H_SINGLE else (band_rects(b, [r for rs in bands for r in rs]) if b == 0 else [0])
        bc_rt[bc.x][bc.y] = (
            [x_ring.buffer_address(), h_all.buffer_address(), h_all.buffer_address(), band_ncc]
            + band_rects(b)
            + h_rects
        )
        xr_rt[bc.x][bc.y] = [x_dram.buffer_address()]

    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
    FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
    zones = [("SE_ZONES", "1")] if os.environ.get("MIMO_SE_ZONES") else []
    groups = sorted(set(pcd_of))
    grp_crs = {g: _crs([c for ci, c in enumerate(receivers) if pcd_of[ci] == g]) for g in groups}
    total_x = V * nk_gu
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
                1,
                slot,
                0,
            ],
            runtime_args=fw_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se2_bcast.cpp",
            source_type=FP,
            core_ranges=bc_crs,
            compile_time_args=[
                0,
                x_blk,
                BF8_TILE,
                total_x,
                X_SLOTS,
                V,
                ncc,
                h_bytes,
                hbuf,
                GATH,
                XCRED,
                HARR,
                HARR1,
                XARR,
                PIECE,
            ],
            runtime_args=bc_rt,
            defines=zones,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se2_xread.cpp",
            source_type=FP,
            core_ranges=bc_crs,
            compile_time_args=[0, x_blk, BF8_TILE, total_x],
            runtime_args=xr_rt,
            defines=zones,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
    ]
    for g in groups:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se2_recv.cpp",
                source_type=FP,
                core_ranges=grp_crs[g],
                compile_time_args=[
                    0,
                    x_blk,
                    1,
                    slot,
                    E * bpe,
                    V,
                    bpe,
                    16,
                    MT * g,
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
                    GATH,
                    hbuf,
                    XARR,
                    XCRED,
                    X_SLOTS,
                    total_x,
                    S,
                ],
                runtime_args=rv_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se2_compute.cpp",
                source_type=FP,
                core_ranges=grp_crs[g],
                compile_time_args=[KBLK, MT, nk_gu, It, kd, E, S, slot, g, rt_of(g)],
                runtime_args=[],
                defines=zones,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
            )
        )
    fmt = lambda i, dt, page: [ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)]
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * READ_BATCH * rd_slot * w_tile, core_ranges=rd_crs, format_descriptors=fmt(0, w_dtype, w_tile)
        ),
        ttnn.CBDescriptor(
            total_size=XS_SLOTS * x_blk * BF8_TILE,
            core_ranges=bc_crs,
            format_descriptors=fmt(0, ttnn.bfloat8_b, BF8_TILE),
        ),
        ttnn.cb_descriptor_from_sharded_tensor(0, x_ring),
        ttnn.cb_descriptor_from_sharded_tensor(1, land),
        ttnn.cb_descriptor_from_sharded_tensor(2, h_all),
        ttnn.CBDescriptor(
            total_size=MT * H_TILE, core_ranges=cc_crs, format_descriptors=fmt(3, ttnn.bfloat8_b, H_TILE)
        ),
    ] + [
        ttnn.cb_descriptor_from_sharded_tensor(16, out, total_size=S * MT * g * 2048, core_ranges=grp_crs[g])
        for g in groups
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    tag = f"streambig_H{H}_M{m}_E{E}_w{wdtype}_{SPLIT}{len(bands)}{'h1' if H_SINGLE else ''}"
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
        ttnn.generic_op([w_dev, x_dram, x_ring, land, h_all, out], program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
        if it == 0:
            # Each core's out ring: slot s = sub-block s of the last expert, [MT x PCD] tiles row-major, packed linearly.
            got_sh = ttnn.to_torch(out).float().view(ncc, S * MT, 32, pcd_max, 32).permute(0, 1, 3, 2, 4)
            got_sh = got_sh.reshape(ncc, S * MT * pcd_max, 32, 32)
            got = torch.zeros(S * MT, 32, Ht, 32)
            ci_of = {(c.x, c.y): i for i, c in enumerate(receivers)}
            for s_, core in enumerate(cc_order):
                ci = ci_of[(core.x, core.y)]
                g = pcd_of[ci]
                for sb in range(S):
                    for mi in range(MT):
                        for c in range(g):
                            got[sb * MT + mi, :, col0[ci] + c] = got_sh[s_, sb * MT * g + mi * g + c]
            got = got.reshape(m, H)
            ok, pcc_q = comp_pcc(ref_q, got, 0.99)
            if os.environ.get("MIMO_SB_DEBUG"):
                err = (got - ref_q).view(S * MT, 32, Ht, 32).norm(dim=(1, 3)) / ref_q.view(S * MT, 32, Ht, 32).norm(
                    dim=(1, 3)
                )
                bad = (err > 0.2).nonzero().tolist()
                owner = {}
                for ci in range(ncc):
                    for c in range(pcd_of[ci]):
                        owner[col0[ci] + c] = ci
                cores = sorted({owner[c] for _, c in bad})
                logger.warning(
                    f"{len(bad)} bad (row-tile, col-tile) of {S * MT * Ht}; rows {sorted({r for r, _ in bad})}; "
                    f"owner cores {[(receivers[ci].x, receivers[ci].y, band_of(receivers[ci])) for ci in cores][:20]}"
                )
            if not ok:
                logger.warning(
                    f"got |mean| {got.abs().mean():.4f} ref {ref_q.abs().mean():.4f}; zero tiles "
                    f"{int((got.view(m // 32, 32, Ht, 32).abs().sum(dim=(1, 3)) == 0).sum())}/{m // 32 * Ht}"
                )
                xr = ttnn.to_torch(x_ring).float().view(ncc, X_SLOTS, MT * 32, KBLK * 32)
                for gx in range(total_x - X_SLOTS, total_x):
                    v, b = divmod(gx, nk_gu)
                    want = xs[v][:, b * KBLK * 32 : (b + 1) * KBLK * 32]
                    by_band = {}
                    for sh, core in enumerate(cc_order):
                        by_band.setdefault(band_of(core), []).append(
                            round(comp_pcc(want, xr[sh, gx % X_SLOTS], 0)[1], 4)
                        )
                    logger.warning(
                        f"  x block {gx}: min PCC per band {[(b, min(v)) for b, v in sorted(by_band.items())]}"
                    )
                lr = ttnn.to_torch(land).float().view(ncc, bpe, slot, 32, 32)
                for ci in (0, 5):
                    want = torch.stack([Wg_t[0:KBLK, ci], Wu_t[0:KBLK, ci]], dim=1).reshape(-1, 32, 32)
                    logger.warning(
                        f"  land core{ci} slot0 (gu blk0): PCC {comp_pcc(q(want.reshape(-1, 32)), lr[ci, 0].reshape(-1, 32), 0)[1]}"
                    )
                hr = ttnn.to_torch(h_all).float().view(ncc, hbuf, It // 8, MT, 8, 32, 32).permute(0, 1, 3, 5, 2, 4, 6)
                hr = hr.reshape(ncc, hbuf, MT * 32, It * 32)
                xv = xs[V - 1]
                h_ref = torch.nn.functional.silu(xv @ q(Wg)) * (xv @ q(Wu))
                hb = {}
                for sh, core in enumerate(cc_order):
                    hb.setdefault(band_of(core), []).append(
                        round(comp_pcc(h_ref, hr[sh, (V - 1) % hbuf].nan_to_num(0, 0, 0), 0)[1], 4)
                    )
                logger.warning(f"  h_all last v: min PCC per band {[(b, min(v)) for b, v in sorted(hb.items())]}")
                got_h = hr[5, (V - 1) % hbuf]
                for j in (0, 1, 7, 8, 63):
                    gj, rj = got_h[:, j * 32 : (j + 1) * 32], h_ref[:, j * 32 : (j + 1) * 32]
                    logger.warning(
                        f"  h col {j}: rows-inf {int(torch.isinf(gj).any(dim=1).sum())}, PCC {comp_pcc(rj, gj.nan_to_num(0, 0, 0), 0)[1]}"
                        f" row-tiles ok {[round(comp_pcc(rj[t*32:(t+1)*32], gj[t*32:(t+1)*32].nan_to_num(0,0,0), 0)[1], 3) for t in range(MT)]}"
                    )
                logger.warning(f"  h_all (last v, core 5) PCC {comp_pcc(h_ref, got_h, 0)[1]}, |h| {got_h.abs().mean()}")
                # which x rows / h does the output match best? try the other experts' x
                for v in range(V):
                    xv = xs[v]
                    yv = (torch.nn.functional.silu(xv @ q(Wg)) * (xv @ q(Wu))) @ q(Wd)
                    logger.warning(f"  vs x of v{v}: PCC {comp_pcc(yv, got[:MT * 32], 0)[1]}")
            _, pcc = comp_pcc(ref, got, 0.0)
            logger.info(f"{tag}: PCC {pcc_q} vs quantized-weight reference, {pcc} vs fp32")
            assert ok, pcc_q
    logger.info(f"ran {tag}: {w_bytes / 1e6:.0f} MB of weights streamed per run")
