# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paired + chained M-split big-M streamed routed expert (SwiGLU), one Blackhole chip.

    y = (silu(x @ Wg) * (x @ Wu)) @ Wd        x [M, H], Wg / Wu [H, I], Wd [I, H], M a multiple of 128

Like test_stream_expert_bigm.py (weights resident per expert, the expert's rows run as S = M / 128 sub-blocks), but
the 64 compute cores form two M-groups, one per run of physically contiguous worker columns (one multicast
rectangle each): group g takes rows [64 g, 64 g + 64) of every 128-row sub-block. Each core owns NP = 2 gate/up pairs
and I-tile / H-tile columns accordingly (down: 7 of 224 output tile columns at H 7168), so a group covers all of N
and needs only its own rows of x and its own h. The two cores sharing a column group sit side by side, so each weight
block is forwarded once, into the left core A, and the right core B's NCRISC pulls it from A one hop away
(se5_bnc.cpp; a multicast, even to 2 cores, only moves ~12-15 GB/s per sender). Both activations travel by unicast down
NCH chains per group: x from a chain head's DRAM reader (se4_xhead.cpp), h from a relay core that gathers the group's
slices and writes the assembled h into the chain heads (se5_relay.cpp); every core forwards each x block and each h to
its successor (se5_recv.cpp). No multicast carries activations.

Weights bfp4 / bfp8 (MIMO_SM_WDTYPE), x and h bfp8, y bf16, LoFi. Tag ``streampr_H{H}_M{M}_E{E}_w{dtype}_c{NCH}``;
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
NCH = int(os.environ.get("MIMO_SM_CHAINS", "2"))  # x / h chains per M-group
H_PIECES = int(os.environ.get("MIMO_SM_H_PIECES", "8"))  # h travels the chains in this many pieces
STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))
KBLK, MT, NP, G = 8, 2, 2, 2  # gate/up K-block, sub-block rows per group (tiles), pairs per core, M-groups
W_DTYPES = {"bf8": (ttnn.bfloat8_b, BF8_TILE), "bf4": (ttnn.bfloat4_b, 576)}
WDTYPES = _env_list("MIMO_SM_WDTYPE", "bf4")
H_TILE = BF8_TILE


def _pick(device, n_readers=16):
    """16 bank readers (as test_stream_matmul) and, per reader, two pairs of horizontally adjacent free cores (one pair per
    column group; left core = M-group 0, right = M-group 1), nearest by NOC1 hops. Receiver j: column group
    2r + j // 2, M-group j % 2."""
    grid = device.compute_with_storage_grid_size()
    phys = lambda c: device.worker_core_from_logical_core(c)
    opt = list(device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0))
    readers = opt + [ttnn.CoreCoord(c.x + 1, c.y) for c in opt] if n_readers == 16 else opt
    px = [phys(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
    taken = {(c.x, c.y) for c in readers}
    receivers = []
    for r in readers:
        for _ in range(32 // len(readers)):  # pairs per reader
            pairs = [
                (ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + 1, y))
                for y in range(grid.y)
                for x in range(grid.x - 1)
                if px[x + 1] == px[x] + 1 and (x, y) not in taken and (x + 1, y) not in taken
            ]
            assert pairs, "not enough free core pairs"
            a, b = min(pairs, key=lambda p: (noc_hops(phys(r), phys(p[0]), 1), p[0].y, p[0].x))
            taken |= {(a.x, a.y), (b.x, b.y)}
            receivers += [a, b]
    free = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
    relay = min(free, key=lambda c: (c.x - grid.x / 2) ** 2 + (c.y - grid.y / 2) ** 2)
    return readers, receivers, relay, None, phys, grid


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("wdtype", WDTYPES)
@pytest.mark.parametrize("m", MS, ids=lambda m: f"M{m}")
def test_stream_expert_pair(device, m, wdtype):
    w_dtype, w_tile = W_DTYPES[wdtype]
    banks = device.dram_grid_size().x
    readers, receivers, relay, _, phys, grid = _pick(device)
    n_rd, ncc = len(readers), len(receivers)
    Ht, It, E = H // 32, I // 32, EXPERTS
    n_cg = ncc // G  # column groups
    assert m % (G * MT * 32) == 0 and It == n_cg * NP and Ht % n_cg == 0, (m, Ht, It, ncc)
    S = m // (G * MT * 32)
    V = E * S
    pipe = int(S == 1)  # one sub-block per expert: stream the weights in consumption order
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
        gu, dn = torch.cat(blocks[: 2 * nk_gu]), torch.cat(blocks[2 * nk_gu :])
        # Stream order = consumption order: plain per expert, or pipelined gu(0), gu(1), d(0), ... for one sub-block.
        per_reader.append(
            torch.cat([gu] + [torch.cat([gu, dn])] * (E - 1) + [dn]) if pipe else torch.cat([gu, dn]).repeat(E, 1, 1)
        )
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
    rd_crs, all_crs = _crs(readers), _crs(readers + receivers + [relay])
    DATA, HARR, GO, DONE, HARR1 = R, R + 1, R + 2, R + 3, R + 4
    GATH, XARR, SFREE, HFREE, HSFREE = R + 5, R + 7, R + 8, R + 9, R + 10  # GATH: one per M-group
    # The relay's counter source words (per group, x and h) reuse ids it has no other use for (at most 16 semaphores).
    AVAIL, BCOPY = 2, 3  # the forwarder only uses credit semaphores 0 and 1 (two A receivers)
    RELAY_WORDS = [DATA, GO, DONE, HARR1]  # the heads' h-free words on the relay (ids it has no other use for)
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in range(R + 11)]
    # x chains: per group, a greedy NOC0-nearest path through the group's cores, cut into NCH chains.
    chain_pred, chain_succ, heads, head_word = {}, {}, [], {}
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

    if os.environ.get("MIMO_SM_PRINT_CHAINS"):
        for g in range(G):
            order = [ci for ci in range(ncc) if g_of(ci) == g and ci not in chain_pred]
            for h0 in order:
                seq, cur = [], h0
                while cur is not None:
                    seq.append((receivers[cur].x, receivers[cur].y))
                    cur = chain_succ.get(cur)
                hops = [noc_hops(phys(ttnn.CoreCoord(*a)), phys(ttnn.CoreCoord(*b)), 0) for a, b in zip(seq, seq[1:])]
                logger.info(f"chain g{g}: {seq} NOC0 hops {hops}")
    xy_or0 = lambda ci: pk(receivers[ci]) if ci is not None else 0
    pk = lambda c: (phys(c).x << 16) | phys(c).y
    peers = [pk(c) for c in receivers]

    rd_rt, fw_rt, rv_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for r, c in enumerate(readers):
        rd_rt[c.x][c.y] = [w_dev.buffer_address(), r % banks, (r // banks) * region_bytes, d_chunk]
        a_cores = [receivers[r * R + j] for j in (0, 2)]  # the forwarder only feeds the A (left) cores
        fw_rt[c.x][c.y] = [land.buffer_address()] + [pk(a) for a in a_cores] + [kd * pcd] * 2
        for j in range(R):
            ci = r * R + j
            rc, g = receivers[ci], g_of(ci)
            partner = receivers[ci ^ 1]
            g_heads = [pk(receivers[h]) for h, hg in heads if hg == g]
            rv_rt[rc.x][rc.y] = (
                [
                    pk(c),
                    j // 2,
                    cg_of(ci),
                    len(g_heads),
                    0,
                    pk(receivers[0]),
                    GATH + g,
                    x_ring.buffer_address(),
                    int(ci == 0),
                    xy_or0(chain_pred.get(ci)),
                    xy_or0(chain_succ.get(ci)),
                    xy_or0(chain_pred.get(ci)),
                    HSFREE,
                    h_reg.buffer_address(),
                    g,
                    pk(partner),
                    AVAIL,
                    BCOPY,
                    0,
                    SFREE,
                    SFREE,
                ]
                + g_heads
                + peers
            )
    # NCRISC: A-core chain heads read x (se4_xhead.cpp); every B core pulls its weights from A and, if it is a chain
    # head, reads x too (se5_bnc.cpp).
    xh_rt, bn_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    head_set = {ci for ci, _ in heads}
    for ci, g in heads:
        if g == 0:
            xh_rt[receivers[ci].x][receivers[ci].y] = [x_dram.buffer_address(), x_ring.buffer_address(), g]
    for ci in range(ncc):
        if g_of(ci) == 1:
            rc = receivers[ci]
            bn_rt[rc.x][rc.y] = [
                pk(receivers[ci ^ 1]),
                land.buffer_address(),
                int(ci in head_set),
                x_dram.buffer_address(),
                x_ring.buffer_address(),
                1,
            ]
    xh_crs = _crs([receivers[ci] for ci, g in heads if g == 0])
    bn_crs = _crs([receivers[ci] for ci in range(ncc) if g_of(ci) == 1])

    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
    FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
    zones = [("SE_ZONES", "1")] if os.environ.get("MIMO_SE_ZONES") else []
    zones += [("SE_WAITZ", os.environ["MIMO_SE_WAITZ"])] if os.environ.get("MIMO_SE_WAITZ") else []
    total_x = V * nk_gu * G
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se_reader.cpp",
            source_type=FP,
            core_ranges=rd_crs,
            compile_time_args=[0, w_tile, rd_slot, READ_BATCH, nk_gu, nk_dd, 1, E, gu_chunk, pipe],
            runtime_args=rd_rt,
            defines=zones,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se_forward.cpp",
            source_type=FP,
            core_ranges=rd_crs,
            compile_time_args=[
                0,
                2,
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
                pipe,
                1,
                slot,
                0,
            ],
            runtime_args=fw_rt,
            defines=zones,
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se4_xhead.cpp",
            source_type=FP,
            core_ranges=xh_crs,
            compile_time_args=[x_blk, BF8_TILE, V * nk_gu, X_SLOTS, XARR, HFREE, G, 1],
            runtime_args=xh_rt,
            defines=[("SE_FAKE_X", "1")] if os.environ.get("MIMO_SM_FAKE_X") else [],
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se5_bnc.cpp",
            source_type=FP,
            core_ranges=bn_crs,
            compile_time_args=[
                1,
                slot,
                w_tile,
                E * bpe,
                bpe,
                AVAIL,
                BCOPY,
                x_blk,
                BF8_TILE,
                V * nk_gu,
                X_SLOTS,
                XARR,
                HFREE,
                G,
                1,
            ],
            runtime_args=bn_rt,
            defines=[("SE_FAKE_X", "1")] if os.environ.get("MIMO_SM_FAKE_X") else [],
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se5_recv.cpp",
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
                HSFREE,
                H_PIECES,
                nk_gu,
                0,
                0,
                1,
                G,
            ],
            defines=zones,
            runtime_args=rv_rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/se3_compute.cpp",
            source_type=FP,
            core_ranges=cc_crs,
            compile_time_args=[KBLK, MT, nk_gu, It, kd, E, S, slot, pcd, rt_d, NP, pipe, 0],
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

    tag = f"streampr_H{H}_M{m}_E{E}_w{wdtype}_c{NCH}"
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
            assert ok or os.environ.get("MIMO_SM_FAKE_X"), pcc_q
    logger.info(f"ran {tag}: {w_bytes / 1e6:.0f} MB of weights streamed per run")
