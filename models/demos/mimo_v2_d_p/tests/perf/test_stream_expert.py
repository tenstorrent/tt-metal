# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Streamed full routed expert (SwiGLU) prototype at MiMo-V2 shapes, one Blackhole chip:

    y = (silu(x @ Wg) * (x @ Wu)) @ Wd        x [M, 4096], Wg / Wu [4096, 2048], Wd [2048, 4096], M <= 128

Same weight stream as test_stream_matmul.py (16 readers: BRISC reads the bank on NOC0, NCRISC forwards on NOC1 to 4
compute cores each, credit-gated), carrying per expert 16 gate/up K-blocks and 8 down K-blocks, all [8 x 2] tiles, in the pipelined order
gu(0), gu(1), d(0), gu(2), d(1), ..., d(E - 1) (compute runs gate/up of e + 1 while h of e is exchanged);
for gate/up the two columns are the core's gate and up tile column, for down its two output tile columns.

64 compute cores (kernels/stream_mm/se_recv.cpp + se_compute.cpp):
  gate/up  accumulate in DST over K, silu(gate) * up in DST -> the core's h slice [M x 32]
  h        gather + broadcast: each core writes its slice (Mt tiles) into the coordinator's h_all (row-major within
           each K-block, like x); once all 64 slices are in, the coordinator multicasts that h_all half to the worker
           grid and bumps every core's arrival counter. h_all is double-buffered by expert parity, and the h of expert e
           goes out only once every core has finished expert e - 2 (the coordinator's "go" count)
  down     accumulate in DST over the 64 h K-tiles -> the core's 2 output tile columns

NUM_EXPERTS copies of the expert are streamed back to back (distinct DRAM bytes); the output keeps the last one.
Weights bfp8 or bfp4 (MIMO_SE_WDTYPE), x and h bfp8, y bf16, LoFi. Tag ``streamexp_M{M}_E{E}_w{dtype}``;
weight GB/s = E * 3 * 4096 * 2048 * tile_bytes / 1024 / time.
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
MS = _env_list("MIMO_SE_M", "32,64,128", int)
EXPERTS = int(os.environ.get("MIMO_SE_EXPERTS", "4"))
SLOTS = int(os.environ.get("MIMO_SE_SLOTS", "3"))
READ_BATCH = int(os.environ.get("MIMO_SE_READ_BATCH", "2"))
ITERS = int(os.environ.get("MIMO_SE_ITERS", "3"))
STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))
COMPUTE_ONLY = int(os.environ.get("MIMO_SE_COMPUTE_ONLY", "0"))  # profiling: compute on L1-resident data only, no check
H = int(os.environ.get("MIMO_SE_H", "4096"))  # hidden (gate/up K, down N): 4096 MiMo-V2, 7168 Kimi K2
I = int(os.environ.get("MIMO_SE_I", "2048"))  # expert intermediate (one gate + one up tile column per compute core)
KBLK = 8
HBUF = os.environ.get("MIMO_SE_HBUF", "auto")  # h_all buffers: 2 (by expert parity), 1, or auto (2 if it fits in L1)
# Weight dtypes: bfp8 (1088 B tiles) or bfp4 (576 B tiles); h (the SwiGLU output, down's in0) and x are bfp8.
W_DTYPES = {"bf8": (ttnn.bfloat8_b, BF8_TILE), "bf4": (ttnn.bfloat4_b, 576)}
WDTYPES = _env_list("MIMO_SE_WDTYPE", "bf8,bf4")
H_DTYPE, H_TILE = ttnn.bfloat8_b, BF8_TILE


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("wdtype", WDTYPES)
@pytest.mark.parametrize("m", MS, ids=lambda m: f"M{m}")
def test_stream_expert(device, m, wdtype):
    w_dtype, w_tile = W_DTYPES[wdtype]
    banks = device.dram_grid_size().x
    readers, receivers, phys = _pick_cores(device)
    n_rd, ncc = len(readers), len(receivers)
    Ht, It, Mt, E = H // 32, I // 32, m // 32, EXPERTS
    assert It == ncc and Ht % n_rd == 0, (Ht, It, ncc, n_rd)
    # Down output columns: Ht / n_rd per reader (keeps the DRAM banks balanced), split over its R receivers as evenly as
    # possible (H 4096: 2 each; H 7168: 4, 4, 3, 3). Down runs in passes of W columns so MT * W accumulators fit in DST.
    per_rd = Ht // n_rd
    pcd_of = [per_rd // R + (j < per_rd % R) for j in range(R)] * n_rd  # by compute core index
    col0 = [sum(pcd_of[:ci]) for ci in range(ncc)]
    pcd_max = max(pcd_of)
    W = min(8 // Mt, pcd_max)
    passes = -(-pcd_max // W)
    assert all(-(-p // W) == passes for p in pcd_of), "every core needs the same number of down passes"
    pass_w = lambda ci, p: min(W, pcd_of[ci] - p * W)
    nk_gu, nk_d = Ht // KBLK, It // KBLK
    slot_tiles = KBLK * max(2, W)  # landing slot: a gate/up block [KBLK x 2] or a down pass block [KBLK x <= W]
    blocks_per_expert = nk_gu + passes * nk_d
    gu_chunk = R * KBLK * 2
    d_chunk = [[KBLK * sum(pass_w(r * R + j, p) for j in range(R)) for p in range(passes)] for r in range(n_rd)]
    rd_slot = max([gu_chunk] + [t for d in d_chunk for t in d])
    x_bytes, h_bytes = nk_gu * Mt * KBLK * BF8_TILE, ncc * Mt * H_TILE
    hbuf = int(HBUF) if HBUF != "auto" else (2 if x_bytes + 2 * h_bytes < 1200 * 1024 else 1)
    logger.info(
        f"H {H} M {m}: x {x_bytes >> 10} KB, h_all {hbuf} x {h_bytes >> 10} KB, down cols {sorted(set(pcd_of))}"
        f" in {passes} pass(es) of <= {W}, landing slot {slot_tiles} tiles"
    )

    torch.manual_seed(0)
    Wg, Wu, Wd = torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02
    x = torch.randn(m, H)
    ref = (torch.nn.functional.silu(x @ Wg) * (x @ Wu)) @ Wd
    # Kernel check: the same expert with its weights round-tripped through the on-device format (bf4 alone costs ~0.02
    # PCC against fp32 on Gaussian weights, which says nothing about the kernel).
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=w_dtype, layout=ttnn.TILE_LAYOUT)).float()
    ref_q = (torch.nn.functional.silu(x @ q(Wg)) * (x @ q(Wu))) @ q(Wd)

    tiles = lambda w: w.view(w.shape[0] // 32, 32, w.shape[1] // 32, 32).permute(0, 2, 1, 3)  # [kt, nt, 32, 32]
    Wg_t, Wu_t, Wd_t = tiles(Wg), tiles(Wu), tiles(Wd)
    per_reader = []
    for r in range(n_rd):
        gu, dn = [], []
        for c in range(nk_gu):
            for j in range(R):
                ci = r * R + j
                ks = slice(c * KBLK, (c + 1) * KBLK)
                gu.append(torch.stack([Wg_t[ks, ci], Wu_t[ks, ci]], dim=1).reshape(-1, 32, 32))  # [k][gate, up]
        for p in range(passes):
            for c in range(nk_d):
                for j in range(R):
                    ci = r * R + j
                    cs = slice(col0[ci] + p * W, col0[ci] + p * W + pass_w(ci, p))
                    dn.append(Wd_t[c * KBLK : (c + 1) * KBLK, cs].reshape(-1, 32, 32))
        gu, dn = torch.cat(gu), torch.cat(dn)
        # Pipelined consumption order: gu(0), then gu(e + 1), d(e) for every e (the last one has no gu(e + 1)).
        per_reader.append(torch.cat([gu] + [torch.cat([gu, dn])] * (E - 1) + [dn]))
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

    cc_crs = _crs(receivers)
    cc_order = ttnn.corerange_to_cores(cc_crs, None, True)
    hs = lambda h, w: ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cc_crs, (h, w), ttnn.ShardOrientation.ROW_MAJOR),
    )
    x_blk = torch.cat([x[:, c * KBLK * 32 : (c + 1) * KBLK * 32] for c in range(nk_gu)])
    x_dev = ttnn.from_torch(
        x_blk.repeat(ncc, 1),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=hs(nk_gu * m, KBLK * 32),
    )
    land = ttnn.allocate_tensor_on_device(
        ttnn.Shape([ncc * SLOTS * slot_tiles * 32, 32]),
        w_dtype,
        ttnn.TILE_LAYOUT,
        device,
        hs(SLOTS * slot_tiles * 32, 32),
    )
    h_all = ttnn.allocate_tensor_on_device(
        ttnn.Shape([ncc * hbuf * ncc * m, 32]), H_DTYPE, ttnn.TILE_LAYOUT, device, hs(hbuf * ncc * m, 32)
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([ncc * m, pcd_max * 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, hs(m, pcd_max * 32)
    )

    rd_crs, all_crs = _crs(readers), _crs(readers + receivers)
    DATA, HARR, GO, DONE, HARR1, GATH = R, R + 1, R + 2, R + 3, R + 4, R + 5
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in range(R + 6)]
    pk = lambda c: (phys(c).x << 16) | phys(c).y
    peers = [pk(c) for c in receivers]
    # h_all broadcast rectangles (coordinator only): the full worker grid, split wherever the physical columns jump (NOC0 order: low -> high)
    grid = device.compute_with_storage_grid_size()
    px = [phys(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
    col_runs, start = [], 0
    for x in range(1, grid.x + 1):
        if x == grid.x or px[x] != px[x - 1] + 1:
            col_runs.append((start, x - 1))
            start = x
    rects = []
    for x0, x1 in col_runs:
        lo, hi = ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x1, grid.y - 1)
        rects.append((x0, x1, pk(lo), pk(hi), (x1 - x0 + 1) * grid.y))

    def mcast_args(c):
        args = [len(rects)]
        for x0, x1, lo, hi, n in rects:
            args += [lo, hi, n - int(x0 <= c.x <= x1), 0]  # the coordinator already holds h_all: skip self
        return args

    rd_rt, fw_rt, rv_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for r, c in enumerate(readers):
        rd_rt[c.x][c.y] = [w_dev.buffer_address(), r % banks, (r // banks) * region_bytes] + d_chunk[r]
        fw_rt[c.x][c.y] = (
            [land.buffer_address()]
            + [pk(receivers[r * R + j]) for j in range(R)]
            + [KBLK * pass_w(r * R + j, p) for j in range(R) for p in range(passes)]
        )
        for j in range(R):
            ci = r * R + j
            rc = receivers[ci]
            rv_rt[rc.x][rc.y] = [pk(c), j, ci, h_all.buffer_address(), pk(receivers[0])] + peers + mcast_args(c)
    batch = READ_BATCH
    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
    FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
    x_tiles = nk_gu * Mt * KBLK
    groups = sorted(set(pcd_of))  # one recv / compute kernel per down width (CT differs)
    grp_crs = {g: _crs([c for ci, c in enumerate(receivers) if pcd_of[ci] == g]) for g in groups}
    zones = [("SE_ZONES", "1")] if os.environ.get("MIMO_SE_ZONES") else []
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se_reader.cpp",
            source_type=FP,
            core_ranges=rd_crs,
            compile_time_args=[0, w_tile, rd_slot, batch, nk_gu, nk_d, passes, E, gu_chunk, 1],
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
                slot_tiles,
                SLOTS,
                0,
                DATA,
                KBLK,
                nk_gu,
                nk_d,
                passes,
                E,
                batch,
                1,
                1,
                KBLK * 2,
                0,
            ],
            runtime_args=fw_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
    ]
    for g in groups:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se_recv.cpp",
                source_type=FP,
                core_ranges=grp_crs[g],
                compile_time_args=[
                    0,
                    x_tiles,
                    1,
                    slot_tiles,
                    blocks_per_expert,
                    E,
                    SLOTS,
                    16,
                    Mt * g,
                    3,
                    2,
                    Mt,
                    H_TILE,
                    ncc,
                    DATA,
                    HARR,
                    GO,
                    DONE,
                    KBLK,
                    HARR1,
                    COMPUTE_ONLY,
                    GATH,
                    hbuf,
                ],
                defines=zones
                + [(k[5:], os.environ[k]) for k in ("MIMO_SE_BC_PIECE", "MIMO_SE_BC_PER_PASS") if k in os.environ],
                runtime_args=rv_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            )
        )
    for g in groups:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se_compute.cpp",
                source_type=FP,
                core_ranges=grp_crs[g],
                compile_time_args=[KBLK, Mt, nk_gu, nk_d, E, slot_tiles, g, W],
                runtime_args=[],
                defines=zones + ([("SE_NO_ACT", "1")] if os.environ.get("MIMO_SE_NO_ACT") else []),
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
            )
        )
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * batch * rd_slot * w_tile,
            core_ranges=rd_crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=w_dtype, page_size=w_tile)],
        ),
        ttnn.cb_descriptor_from_sharded_tensor(0, x_dev),
        ttnn.cb_descriptor_from_sharded_tensor(1, land),
        ttnn.cb_descriptor_from_sharded_tensor(2, h_all),
        ttnn.CBDescriptor(
            total_size=Mt * H_TILE,
            core_ranges=cc_crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=3, data_format=H_DTYPE, page_size=H_TILE)],
        ),
    ] + [  # out CB per down width: each expert pushes MT * PCD tiles, so the ring must be exactly that big to wrap cleanly
        ttnn.cb_descriptor_from_sharded_tensor(16, out, total_size=Mt * g * 2048, core_ranges=grp_crs[g])
        for g in groups
    ]
    if COMPUTE_ONLY:
        kernels = kernels[2:]  # no readers / forwarders: compute runs on whatever is in L1
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    tag = f"streamexp_{'' if H == 4096 else f'H{H}_'}M{m}_E{E}_w{wdtype}{'_computeonly' if COMPUTE_ONLY else ''}"
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
        ttnn.generic_op([w_dev, x_dev, land, h_all, out], program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
        if it == 0 and not COMPUTE_ONLY:
            # Each core's shard holds its MT * PCD output tiles pass-major ([pass][m][column of the pass]) in tile order.
            got_sh = ttnn.to_torch(out).float().view(ncc, Mt, 32, pcd_max, 32).permute(0, 1, 3, 2, 4)
            got_sh = got_sh.reshape(ncc, Mt * pcd_max, 32, 32)
            got = torch.zeros(Mt, 32, Ht, 32)
            ci_of = {(c.x, c.y): i for i, c in enumerate(receivers)}
            for s, core in enumerate(cc_order):
                ci = ci_of[(core.x, core.y)]
                t = 0
                for p in range(passes):
                    for mi in range(Mt):
                        for w in range(pass_w(ci, p)):
                            got[mi, :, col0[ci] + p * W + w] = got_sh[s, t]
                            t += 1
            got = got.reshape(m, H)
            ok, pcc_q = comp_pcc(ref_q, got, 0.99)
            if not ok:
                err = ((got - ref_q).view(m, Ht, 32).norm(dim=(0, 2)) / ref_q.view(m, Ht, 32).norm(dim=(0, 2))).tolist()
                bad = [t for t, e in enumerate(err) if e > 0.1]
                logger.warning(f"{tag}: {len(bad)}/{Ht} bad output tile columns: {bad[:40]}")
            _, pcc = comp_pcc(ref, got, 0.0)
            logger.info(f"{tag}: PCC {pcc_q} vs quantized-weight reference, {pcc} vs fp32")
            assert ok, pcc_q
    logger.info(f"ran {tag}: {w_bytes / 1e6:.0f} MB of weights streamed per run")
