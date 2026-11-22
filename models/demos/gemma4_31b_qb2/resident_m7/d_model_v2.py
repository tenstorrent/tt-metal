# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident DFlash2 drafter program v2 (M6b fixes, M6B_TIMING_1.md §5; d_model.py is the verified phase-1 version):
fix 1 (d_slice_compute_v2.cpp: conv_in's base part before the coefficients, conv_out's expansion after xin), fix 2
(d_streamer_writer_v2.cpp / d_slice_dm_v2.cpp: each slice receives and counts only its coefficient tiles), optional
timeline marks (timeline=True: L1 word arrays on the streamers and slices, read by timeline()).

Phase 1: the D-program, M6b phase 1; bench/resident/m6/M6A_DESIGN.md + addenda): one
program per chip that runs the drafter's layers on R = 32 rows (U user blocks of Mb rows) and the LM head, on the
target's core roles (model.py) with the drafter's kernels (kernels/d_*):
  - 32 streamer cores stream their column slices of each drafter layer's BF8 weights (the conv coefficient
    projections, q|k|v, o, gate|up, down) and of the target's LM head (H option (b): the target's resident head
    tensor, read by both programs) through the weight ring, as the target's streamers (matmul_block at HiFi2);
  - K slice cores own the drafter's residual by hidden columns: the norms (per-row partial sums of squares exchanged
    as the target's), the conv coefficients expanded per channel, the two block convs per sublayer, the residual;
  - the hub (hub_dm.cpp, the target's, unchanged: 2 layers all-reduce rounds, no attention cores, feed mode none)
    all-reduces the o / down partials over fabric and reduces the per-row argmax of the head.
Attention is the o = q stub until K2a's units land (the writer gathers each chip's q columns as the o_proj input);
inject mode replaces each layer's attention output (the y of its attention round) by host rows: the teacher-forced
check against the recorded ttnn drafter. A program runs drafter layers [layer_offset, layer_offset + layers)."""
import math
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_31b_qb2.resident_m7 import PKG_DIR
from models.demos.gemma4_31b_qb2.resident_m7.common import (
    BF8_TILE,
    E_HEAD,
    HIDDEN,
    HT,
    IT,
    N_CHIPS,
    PER_BANK,
    RING_BYTES,
    TILE,
    VOCAB,
    block_geometry,
    entry_core_cols,
    f32_bits,
    rect_cover,
    streamer_cores,
)
from models.demos.gemma4_31b_qb2.resident_m7.drafter_common import (
    D_DOWN,
    D_ENTRIES,
    D_GU,
    D_KP_ATTN,
    D_KP_MLP,
    D_LAYERS,
    D_O,
    D_Q,
    D_QKV,
    KP_COLS,
    d_core_cols,
    d_geometry,
)
from models.demos.gemma4_31b_qb2.resident_m7.model import tile_order, tile_unorder

KDIR = f"{PKG_DIR}/kernels/"
SEM_SLOTS, SEM_O, SEM_ACT, SEM_GATHER, SEM_REC, SEM_EXCH, SEM_HADDR, SEM_STAGE, SEM_SLOTADDR = range(9)
SEM_COEF, SEM_COEFADDR = 9, 10
# the streamers' entries in d_streamer_common.hpp order (kKpA, kQkv, kO, kKpM, kGU, kDown, kHead): drafter entries, then
# the target's LM head
STREAM = [D_KP_ATTN, D_QKV, D_O, D_KP_MLP, D_GU, D_DOWN, None]
K_KPA, K_QKV, K_O, K_KPM, K_GU, K_DOWN, K_HEAD = range(7)
OT = D_Q // TILE  # 64 q tiles per chip: the o_proj input
COEF_TILES = KP_COLS // TILE  # 48 (the streamers' coefficient columns)
COEF_LOCAL = 12  # a slice's coefficient tiles: 4 sections x 3 column tiles (fix 2)
CONST_TILES = 20
FEED_NONE = 2
OUT_WORDS = 80
R = 32
TB = 64 * R


class DrafterWeightsOnDevice:
    """the drafter's per-layer entries (every layer, back to back per entry: a fixed per-bank stride) and the target's
    LM head, width-sharded over the DRAM banks in the streamers' bank layout"""

    def __init__(self, mesh, dcache, head_cache):
        from models.demos.gemma4_31b_qb2.resident_m7 import drafter_weights
        from models.demos.gemma4_31b_qb2.resident_m7.weights import load_host as target_load_host

        n, banks = mesh.get_num_devices(), mesh.dram_grid_size().x
        self.banks = banks
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})

        def place(host, Kt, cols):
            width = PER_BANK * max(len(c) for bank in cols for c in bank)
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, [Kt * TILE, width * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            )
            t = ttnn.to_device(ttnn.from_host_shards(host, ttnn.MeshShape(1, n)), mesh, memory_config=mc)
            ttnn.synchronize_device(mesh)
            return t

        t0 = time.time()
        self.entries, self.keep, self.bytes = [], [], 0
        for e in STREAM:
            if e is None:
                cols = entry_core_cols(E_HEAD, banks)
                t = place(target_load_host(head_cache, 0, E_HEAD), HT, cols)
                self.keep.append(t)
                self.entries.append((t.buffer_address(), 0, 1))
                self.bytes += HT * (VOCAB // n // TILE) * BF8_TILE
                continue
            _, Kt, N = D_ENTRIES[e]
            ts = [place(drafter_weights.load_host(dcache, e, l), Kt, d_core_cols(e, banks)) for l in range(D_LAYERS)]
            addrs = [t.buffer_address() for t in ts]
            stride = (addrs[1] - addrs[0]) & 0xFFFFFFFF
            assert all(((b - a) & 0xFFFFFFFF) == stride for a, b in zip(addrs, addrs[1:])), (e, addrs)
            self.keep += ts
            self.entries.append((addrs[0], stride, D_LAYERS))
            self.bytes += D_LAYERS * Kt * N * BF8_TILE
        logger.info(f"drafter weights + head on device: {self.bytes / 1e9:.2f} GB / chip, {time.time() - t0:.0f} s")


class DrafterProgram:
    def __init__(
        self,
        mesh,
        weights,
        vectors,
        layers=D_LAYERS,
        layer_offset=0,
        rows=32,
        block=8,
        slices=8,
        inject=False,
        dump=False,
        timeline=False,
        bf8_partials=False,
        links=int(os.environ.get("GRES_LINKS", "2")),
        packet_bytes=8192,
        ring_bytes=RING_BYTES,
        l1_top=int(os.environ.get("GRES_D_L1_TOP", "1330000")),
        fidelity=ttnn.MathFidelity.HiFi2,
    ):
        """weights: DrafterWeightsOnDevice; vectors: the drafter cache's vectors.pt (norms, conv bases, eps); rows M
        (<= 32) in user blocks of `block` rows; inject: the attention outputs come from set_inject (teacher forcing);
        dump: every layer's h, xin, x after attention, h2, m_in, x after the MLP to DRAM (the final h always)"""
        assert 0 <= layer_offset and layer_offset + layers <= D_LAYERS
        self.mesh, self.layers, self.offset, self.inject, self.dump = mesh, layers, layer_offset, inject, dump
        n = self.n = mesh.get_num_devices()
        assert n == N_CHIPS
        banks = self.banks = weights.banks
        M = self.M = rows
        assert 1 <= M <= R
        Mb = self.Mb = block
        assert M % Mb == 0
        K = self.K = slices
        assert banks % K == 0 and HT % K == 0
        T = self.T = HT // K
        V = self.V = T
        # fix 3: the o / down partials (streamers -> slices and the hub) as BF8 tiles (1088 B) instead of bf16 (2048 B)
        self.bf8_partials = bf8_partials
        pb, pdt = (BF8_TILE, ttnn.bfloat8_b) if bf8_partials else (TB, ttnn.bfloat16)
        P = -(-T * pb // packet_bytes)
        rounds = 2 * layers

        # ---- cores: as model.py (streamers 4 per bank, the hub and K slices from the free cores of their bounding box)
        groups, used = streamer_cores(mesh)
        cores = [c for grp in groups for c in grp]
        S = self.S = len(cores)
        xs, ys = [c.x for c in cores], [c.y for c in cores]
        free = [
            ttnn.CoreCoord(x, y)
            for y in range(min(ys), max(ys) + 1)
            for x in range(min(xs), max(xs) + 1)
            if (x, y) not in used
        ]
        hub, slices_c = free[0], free[1 : 1 + K]
        assert len(slices_c) == K
        C = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])  # noqa: E731
        grid, hub_grid, slice_grid = C(cores), C([hub]), C(slices_c)
        phys = lambda c: mesh.worker_core_from_logical_core(c)  # noqa: E731
        sphys, hub_p, slice_p = [phys(c) for c in cores], phys(hub), [phys(c) for c in slices_c]
        self.cores, self.hub, self.slices = cores, hub, slices_c
        logger.info(f"D-program: layers [{layer_offset}, {layer_offset + layers}), M {M} (blocks of {Mb}), {S} "
                    f"streamers, hub {hub}, {K} slices of {T} tiles, {'inject' if inject else 'o = q stub'}")

        rep = ttnn.ReplicateTensorToMesh(mesh)
        dram = ttnn.DRAM_MEMORY_CONFIG

        def settled(t):
            ttnn.synchronize_device(mesh)
            return t

        def u32(grid_, ncores, words):
            return settled(
                ttnn.from_torch(
                    torch.zeros(ncores, words, dtype=torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    mesh_mapper=rep,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(grid_, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
                    ),
                )
            )

        def dram_t(t, dtype):
            return settled(ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=rep))

        # ---- columns per core
        cols = [d_core_cols(e, banks) if e is not None else entry_core_cols(E_HEAD, banks) for e in STREAM]
        per_core = lambda e, i: cols[e][i // PER_BANK][i % PER_BANK]  # noqa: E731
        self.per_core = per_core
        nq_max = max(len(per_core(K_QKV, i)) for i in range(S))
        nd_max = max(len(per_core(K_DOWN, i)) for i in range(S))
        ng_max = max(len(per_core(K_GU, i)) // 2 for i in range(S))
        nv_max = max(len(per_core(K_HEAD, i)) for i in range(S))
        nc_max = max(len(per_core(K_KPA, i)) for i in range(S))
        kblk = max(nq_max, ng_max)
        kact = max(IT, OT, nv_max)
        self.nv_max = nv_max
        # fix 2: slice k's coefficient column tiles per section are g0 .. g1 (g = channel tile / 16); its senders are the
        # streamers holding any of them
        g_rng = [((k * T) // 16, (k * T + T - 1) // 16) for k in range(K)]
        assert all(g1 - g0 < 3 for g0, g1 in g_rng)
        senders = [sum(any(g0 <= c % 12 <= g1 for c in per_core(K_KPA, i)) for i in range(S)) for g0, g1 in g_rng]
        assert all(per_core(K_KPA, i) == per_core(K_KPM, i) for i in range(S)) and all(senders)
        self.coef_senders = senders
        geo = [d_geometry(e) if e is not None else block_geometry(HT, BF8_TILE) for e in STREAM]
        kts = [D_ENTRIES[e][1] if e is not None else HT for e in STREAM]

        act_l1 = TB * (HT + kact + 4 * kblk + nd_max + nc_max)
        l1_base = ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1)
        ring_bytes = min(ring_bytes, l1_top - l1_base - act_l1 - 1024)
        self.ring_bytes = ring_bytes = (ring_bytes // BF8_TILE) * BF8_TILE
        assert ring_bytes >= max(g[3] for g in geo) * 4, ring_bytes
        logger.info(f"weight ring {ring_bytes} B per streamer, activations {act_l1} B")

        # ---- L1 tensors (hub; timeline word arrays)
        self.timeline_on = timeline
        if timeline:
            words = lambda cores_, n_: settled(ttnn.from_torch(  # noqa: E731
                torch.zeros(len(cores_), n_, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh, mesh_mapper=rep,
                memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                ttnn.ShardSpec(C(cores_), [1, n_], ttnn.ShardOrientation.ROW_MAJOR))))
            self.st_t, self.sl_t = words(cores, layers * 16), words(slices_c, 64)
        self.rec_t = u32(hub_grid, 1, S * (-(-R * 8 // 16) * 4))
        self.cand_t = u32(hub_grid, 1, n * (-(-R * 8 // 16) * 4))

        # ---- DRAM tensors
        self.row_elems = K * V * 1024
        z = lambda rows_: torch.zeros(rows_, self.row_elems, dtype=torch.bfloat16)  # noqa: E731
        self.out_t = dram_t(torch.zeros(1, OUT_WORDS, dtype=torch.int32), ttnn.uint32)
        self.x0_t = dram_t(z(1), ttnn.bfloat16)
        self.wnorm_t = dram_t(self._norm_rows(vectors), ttnn.bfloat16)
        self.base_t = dram_t(self._base_rows(vectors), ttnn.bfloat16)
        self.inj_t = dram_t(z(layers), ttnn.bfloat16)
        self.dump_rows_n = 6 * layers + 1
        self.dump_t = dram_t(z(self.dump_rows_n), ttnn.bfloat16)
        self.logits_t = dram_t(torch.zeros(S, nv_max * TB // 2), ttnn.bfloat16)
        # the hub's serving rows (hub_dm.cpp): feed mode none, every block valid
        row32 = lambda vals: dram_t(torch.tensor([vals], dtype=torch.int32), ttnn.uint32)  # noqa: E731
        self.tokens_t, self.positions_t, self.slots_t = row32([0] * 32), row32([0] * 32), row32([0] * 32)
        self.control_t, self.status_t = row32([FEED_NONE] + [0] * 31), row32([0] * 32)

        self.hub_sems = [ttnn.create_global_semaphore(mesh, hub_grid, 0) for _ in range(2)]  # cand, launch
        self.slice_sems = [ttnn.create_global_semaphore(mesh, slice_grid, 0) for _ in range(3)]  # slot 0/1, launch
        hsem = [ttnn.get_global_semaphore_address(s) for s in self.hub_sems]
        ssem = [ttnn.get_global_semaphore_address(s) for s in self.slice_sems]

        self.links = 1
        if links > 1:
            node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, c))  # noqa: E731
            trained = min(
                len(ttnn.get_forwarding_link_indices(mesh, node(c), node(c + s)))
                for c in range(n)
                for s in (1, -1)
                if 0 <= c + s < n
            )
            self.links = max(1, min(links, trained))

        # ---- program
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        NCRISC, BRISC = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
        NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default

        def compute_cfg(fid, unpack_fp32=()):
            c = ttnn.ComputeConfigDescriptor()
            c.math_fidelity = fid
            c.fp32_dest_acc_en = True
            c.dst_full_sync_en = True
            if unpack_fp32:
                modes = type(c.unpack_to_dest_mode)([ttnn.UnpackToDestMode.Default] * 64)
                for i in unpack_fp32:
                    modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
                c.unpack_to_dest_mode = modes
            return c

        def fmt(idx, dt=ttnn.bfloat16, page=TB):
            return ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)

        def cb(idxs, grid_, pages, dt=ttnn.bfloat16, page=TB):
            idxs = idxs if isinstance(idxs, tuple) else (idxs,)
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid_, format_descriptors=[fmt(i, dt, page) for i in idxs])

        f32 = (ttnn.float32, 4096)
        hdrs = 2 * K * P * 2 + 2
        cbs = [
            # streamers
            ttnn.CBDescriptor(
                total_size=ring_bytes, core_ranges=grid,
                format_descriptors=[fmt(e, ttnn.bfloat8_b, BF8_TILE) for e in range(len(STREAM))],
            ),
            cb(7, grid, 1, ttnn.uint32, 16),
            cb(8, grid, HT),
            cb(9, grid, kblk),
            cb((10, 18, 19), grid, kact),  # o_in | act | logits (as model.py)
            cb(11, grid, nd_max, pdt, pb),
            cb(12, grid, kblk),
            cb(14, grid, kblk),
            cb(16, grid, kblk),
            cb(20, grid, nc_max),
            # hub
            cb(0, hub_grid, 2 * HT, pdt, pb),
            cb(6, hub_grid, 1, ttnn.uint32, 512),
            cb(7, hub_grid, 1, ttnn.uint32, hdrs * 160),
            # slices (d_slice_compute.cpp: x0 | hk | y and base | c share their buffers)
            cb(0, slice_grid, 2 * n * V, pdt, pb),
            cb(2, slice_grid, 2 * V),
            cb((3, 5, 12), slice_grid, V),
            cb(4, slice_grid, V),
            cb(6, slice_grid, CONST_TILES),
            cb(7, slice_grid, 1, *f32),
            cb(8, slice_grid, 2 * K, *f32),
            cb(9, slice_grid, 1, *f32),
            cb(10, slice_grid, 2, *f32),
            cb(11, slice_grid, V),
            cb(13, slice_grid, COEF_LOCAL),
            cb((14, 15), slice_grid, 4 * V),
            cb(17, slice_grid, V),  # Sh = S @ h (fix 1)
            cb(18, slice_grid, V),  # B: conv_in's base part (fix 1)
        ] + ([cb(16, slice_grid, V)] if inject else [])
        owners = {
            SEM_SLOTS: grid,
            SEM_O: grid,
            SEM_ACT: grid,
            SEM_GATHER: hub_grid,
            SEM_REC: hub_grid,
            SEM_EXCH: slice_grid,
            SEM_HADDR: slice_grid,
            SEM_STAGE: grid,
            SEM_SLOTADDR: grid.merge(hub_grid),
            SEM_COEF: slice_grid,
            SEM_COEFADDR: grid,
        }
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=g, initial_value=0) for i, g in owners.items()]

        ct = [layers, S, HT, IT, 0, ring_bytes, ng_max, nd_max, nq_max, nv_max, kblk, kact, SEM_SLOTS, SEM_O, SEM_ACT]
        ct += [OT, nc_max, R, M, K]
        for Kt, (sb, pages, page, blk) in zip(kts, geo):
            ct += [Kt, sb, pages, page, blk]
        ct += [SEM_COEF, nc_max, COEF_TILES, pb]  # 58: a partial tile's bytes (fix 3)
        slot_bytes = V * 2048
        self.slot_bytes = slot_bytes
        part_slot = V * pb  # a chip's slot on a slice (partials)
        banks_per_slice = banks // K
        peers = [v for p in sphys for v in (p.x, p.y)]
        slice_xy = [v for p in slice_p for v in (p.x, p.y)]
        rects = rect_cover(mesh, cores, False)
        kernel = lambda src, cr_, ct_, rt_, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=KDIR + src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cr_,
            compile_time_args=ct_,
            runtime_args=rt_,
            config=cfg,
        )
        mesh_pd = ttnn.MeshProgramDescriptor()
        self.programs = []
        for chip in range(n):
            coord = ttnn.MeshCoordinate(0, chip)
            sct = list(ct)
            sct[4] = chip
            rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i, c in enumerate(cores):
                bank, j = divmod(i, PER_BANK)
                ent = []
                for e, (addr, stride, copies) in enumerate(weights.entries):
                    if STREAM[e] is not None:
                        addr, copies = addr + layer_offset * stride, copies - layer_offset
                    ent += [len(per_core(e, i)), addr & 0xFFFFFFFF, stride, copies]
                qs, o, g, d, v, kc = (per_core(e, i) for e in (K_QKV, K_O, K_GU, K_DOWN, K_HEAD, K_KPA))
                assert o == d and kc == per_core(K_KPM, i)
                ng = len(g) // 2
                k = bank // banks_per_slice
                assert not d or k * T <= d[0] and d[-1] < (k + 1) * T, (i, d, k)
                first_index = chip * (VOCAB // n) + (v[0] * TILE if v else 0)
                ts_addr = self.st_t.buffer_address() if timeline else 0
                rr[c.x][c.y] = [bank, i & 0x3, j, PER_BANK] + ent + [
                    ts_addr, hub_p.x, hub_p.y, self.rec_t.buffer_address(), SEM_REC, len(v), first_index, i,
                ]
                wr[c.x][c.y] = [
                    ng,
                    g[0] if ng else 0,
                    len(d),
                    d[0] if d else 0,
                    hub_p.x,
                    hub_p.y,
                    len(kc),
                    self.rec_t.buffer_address(),
                    SEM_GATHER,
                    SEM_REC,
                    len(qs),
                    qs[0] if qs else 0,
                    kc[0] if kc else 0,
                    SEM_COEFADDR,
                    self.logits_t.buffer_address(),
                    len(v),
                    first_index,
                    i,
                    ts_addr,
                    slice_p[k].x,
                    slice_p[k].y,
                    SEM_STAGE,
                    ssem[0],
                    ssem[1],
                    (d[0] if d else 0) - k * T,
                    part_slot,
                    SEM_SLOTADDR,
                    SEM_HADDR,
                ] + peers + slice_xy
                cr[c.x][c.y] = [ng, len(d), len(qs), len(v), len(kc)]
            hub_rt = ttnn.RuntimeArgs()
            hub_rt[hub.x][hub.y] = [
                SEM_STAGE,
                SEM_SLOTADDR,
                ssem[0],
                ssem[1],
                part_slot,
                self.rec_t.buffer_address(),
                self.cand_t.buffer_address(),
                hsem[0],
                hsem[1],
                self.tokens_t.buffer_address(),
                self.positions_t.buffer_address(),
                self.slots_t.buffer_address(),
                self.out_t.buffer_address(),
                self.control_t.buffer_address(),
                self.status_t.buffer_address(),
                1 << 30,
                R,
                0,
            ] + rects + slice_xy
            hub_ct = [n, chip, R, M, HT, K, T, packet_bytes, rounds, S, SEM_GATHER, SEM_REC, self.links, 0, Mb, pb]
            kernels = [
                kernel("d_streamer_reader.cpp", grid, sct, rr, dm(NCRISC, NOC0)),
                kernel("d_streamer_writer_v2.cpp", grid, sct, wr, dm(BRISC, NOC1)),
                kernel("d_streamer_compute.cpp", grid, sct, cr, compute_cfg(fidelity)),
                kernel("d_hub_dm_v2.cpp", hub_grid, hub_ct, hub_rt, dm(BRISC, NOC0)),
            ]
            sc_rt = ttnn.RuntimeArgs()
            for k, c in enumerate(slices_c):
                sdm_rt = ttnn.RuntimeArgs()
                sdm_rt[c.x][c.y] = [
                    SEM_HADDR,
                    ssem[0],
                    ssem[1],
                    ssem[2],
                    self.x0_t.buffer_address(),
                    self.wnorm_t.buffer_address(),
                    self.base_t.buffer_address(),
                    self.inj_t.buffer_address(),
                    self.dump_t.buffer_address(),
                    SEM_SLOTADDR,
                    SEM_COEFADDR,
                    hub_p.x | (hub_p.y << 16),
                ] + rects + slice_xy + [self.sl_t.buffer_address() if timeline else 0, senders[k]]
                sc_ct = [n, chip, k, K, T, V, layers, PER_BANK * banks_per_slice, P, SEM_SLOTS, SEM_EXCH, int(dump)]
                sc_ct += [int(inject), SEM_COEF, S, Mb, COEF_LOCAL, pb]
                kernels.append(kernel("d_slice_dm_v2.cpp", C([c]), sc_ct, sdm_rt, dm(BRISC, NOC0)))
                sc_rt[c.x][c.y] = [k * T]
            kernels.append(
                kernel(
                    "d_slice_compute_v2.cpp",
                    slice_grid,
                    [n, V, layers, K, f32_bits(vectors["eps"]), f32_bits(1 / HIDDEN), int(inject)],
                    sc_rt,
                    compute_cfg(ttnn.MathFidelity.HiFi4, unpack_fp32=(7, 8)),
                )
            )
            program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            args = program.kernels[3].runtime_args[hub.x][hub.y]
            for link in range(self.links):
                for nb in (chip + 1, chip - 1):
                    if 0 <= nb < n:
                        args.append(1)
                        args.extend(
                            ttnn.setup_fabric_connection(
                                mesh,
                                mesh.get_fabric_node_id(coord),
                                mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, nb)),
                                link,
                                program,
                                hub,
                            )
                        )
                    else:
                        args.append(0)
            mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program
            self.programs.append(program)
        self.mesh_pd = mesh_pd
        self.io = [self.rec_t, self.cand_t, self.out_t, self.x0_t, self.wnorm_t, self.base_t, self.inj_t, self.dump_t]
        self.io += [self.logits_t, self.tokens_t, self.positions_t, self.slots_t, self.control_t, self.status_t]
        self.io += weights.keep
        self.io += [self.st_t, self.sl_t] if timeline else []
        self.trace_id = None

    # ---- host-side rows (each slice's V views of R rows, in tile order: model.py's _view_rows)
    def view_rows(self, mats):
        """[len(mats), K V 1024] bf16 from [R or 1, HIDDEN] matrices (a single row is replicated on all R rows)"""
        K, T, V = self.K, self.T, self.V
        out = torch.zeros(len(mats), self.row_elems, dtype=torch.bfloat16)
        for i, w in enumerate(mats):
            w = w.reshape(-1, HIDDEN).bfloat16()
            w = w.expand(R, -1) if w.shape[0] == 1 else torch.nn.functional.pad(w, (0, 0, 0, R - w.shape[0]))
            flat = tile_order(w.contiguous())
            for k in range(K):
                seg = flat[k * T * TB // 2 : (k + 1) * T * TB // 2]
                out[i, k * V * 1024 : k * V * 1024 + seg.numel()] = seg
        return out

    def _norm_rows(self, vectors):
        """row 2 l: layer l's input norm, 2 l + 1 its post-attention norm, 2 layers: the final norm"""
        vecs = []
        for l in range(self.offset, self.offset + self.layers):
            vecs += [vectors["layers"][l]["input_norm"], vectors["layers"][l]["post_norm"]]
        return self.view_rows(vecs + [vectors["final_norm"]])

    def _base_rows(self, vectors):
        """row 4 j + s: sublayer j's conv base section s (io, tap), row-replicated"""
        vecs = []
        for l in range(self.offset, self.offset + self.layers):
            for key in ("attn_base", "mlp_base"):
                vecs += list(vectors["layers"][l][key].reshape(4, HIDDEN))
        return self.view_rows(vecs)

    def _write(self, host, dev, dtype):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)), dev)

    def set_x0(self, x):
        """x [M, HIDDEN]: the block rows' input residual"""
        self._write(self.view_rows([x]), self.x0_t, ttnn.bfloat16)

    def set_inject(self, attn):
        """attn [layers, M, HIDDEN]: each program layer's attention output (after o_proj, before the out conv)"""
        assert self.inject
        self._write(self.view_rows(list(attn)), self.inj_t, ttnn.bfloat16)

    def reset(self):
        for s in self.hub_sems + self.slice_sems:
            ttnn.reset_global_semaphore_value(s, 0)
        ttnn.synchronize_device(self.mesh)

    def launch(self):
        if self.trace_id is not None:
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
        else:
            ttnn.generic_op(self.io, self.mesh_pd)

    # ---- read back
    def _per_chip(self, t):
        return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]

    def dump_rows(self, row):
        """[n, M, HIDDEN] of dump row `row`"""
        out = []
        for d in self._per_chip(self.dump_t):
            d = d.reshape(self.dump_rows_n, self.K, self.V * 1024)[row]
            flat = torch.cat([d[k, : self.T * TB // 2] for k in range(self.K)])
            out.append(tile_unorder(flat, R)[: self.M].float())
        return torch.stack(out)

    def layer_dumps(self, l):
        """program layer l's {h, xin, x_attn, h2, m_in, x_out}: [n, M, HIDDEN] each (dump)"""
        keys = ("h", "xin", "x_attn", "h2", "m_in", "x_out")
        return {key: self.dump_rows(6 * l + i) for i, key in enumerate(keys)}

    def final_h(self):
        return self.dump_rows(6 * self.layers)

    def logits(self):
        """[M, VOCAB] (bf16 values as float) from every chip's streamers"""
        out = torch.zeros(self.M, VOCAB)
        for chip, raw in enumerate(self._per_chip(self.logits_t)):
            raw = raw.reshape(self.S, -1)
            for i in range(self.S):
                v = self.per_core(K_HEAD, i)
                if v:
                    lg = tile_unorder(raw[i].float(), R)[: self.M]
                    c0 = chip * (VOCAB // self.n) + v[0] * TILE
                    out[:, c0 : c0 + len(v) * TILE] = lg[:, : len(v) * TILE]
        return out

    def out(self):
        """per chip: ([M argmax ids], [M keys])"""
        res = []
        for o in self._per_chip(self.out_t):
            w = o.reshape(-1).to(torch.int64) & 0xFFFFFFFF
            res.append(([int(w[2 * r]) for r in range(self.M)], [int(w[2 * r + 1]) for r in range(self.M)]))
        return res

    def cb_tops(self):
        """per role (streamers, hub, slices): the highest L1 address its CBs occupy (engine.cb_tops' rule)"""
        base = ttnn.get_allocator_base_address(self.mesh, ttnn.BufferType.L1)
        roles = {"streamers": {(c.x, c.y) for c in self.cores}, "hub": {(self.hub.x, self.hub.y)},
                 "slices": {(c.x, c.y) for c in self.slices}}
        per_core = {}
        for d in self.programs[0].cbs:
            for cr in d.core_ranges.ranges():
                for x in range(cr.start.x, cr.end.x + 1):
                    for y in range(cr.start.y, cr.end.y + 1):
                        per_core[(x, y)] = per_core.get((x, y), 0) + -(-d.total_size // 16) * 16
        return {role: base + max(per_core[xy] for xy in xys) for role, xys in roles.items()}
