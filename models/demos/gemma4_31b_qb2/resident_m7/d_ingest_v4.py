# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""v4 (M6b, for R64 Phase C (ii): bench/resident/r64/R64_PHASE_C_ANSWERS_m5b.md, R64_TAPS_BUFFER_SPEC.md):
v3 (below) with
  (1) the taps from r64b's DRAM taps buffers instead of the lazy ring: two buffers (the target launch's parity), each
      per chip [1, 1, 64, n x 5376] bf16 TILE (every chip all columns; chip c's fc input reads column tiles
      [42 c, 42 c + 42) of each tap); each launch row names its source row: parity 64 + row (row = 32 lane + r, the
      verify's launch row), or -1 (kernels/d_ingest_slice_dm_v3.cpp: 64 B-aligned span reads, no row staging);
  (2) per-row slots: set_rows(src_rows, positions, slots, taps_launch, engine_launches): up to 32 committed rows of
      any users; the taps buffer's liveness is checked (launch k's buffer is rewritten by launch k + 2);
  (3) the M8 XCHIP rule (LEAD_DECISIONS_M6, XCHIP_ALTERNATION_AUDIT 25bdcc89): every slice bumps the hub's record
      semaphore after its last ring write and the hub (kernels/d_ingest_hub_dm.cpp) waits for 2 S + K before its
      cross-chip candidate exchange, so no chip finishes (and starts its next program) while a peer still writes the
      ring.
Without `taps` the program allocates its own buffer pair (tests: fill_taps writes host rows into it).
The per-layer k|v semaphores (v3's race fix) are unchanged.

v3 (M6b GI1 fixes; d_ingest.py is C1's, frozen): (1) one k | v semaphore per layer on the slices (SEM_KV + l,
ids 11 .. 15; d_ingest_writer_v2.cpp / d_ingest_slice_dm_v2.cpp): C1's cumulative count let a streamer running a layer
ahead stand in for a slow one, so a slice could pack a layer before all its tiles landed (gi1_ringrep2.json);
(2) the lazy tap tensor for every ring slot (C1 kept slot 0's rows only, so a launch for slot s > 0 read past it;
d_ingest_v2.py IngestProgramSlots, folded in here).

C1: Resident DFlash2 drafter, ingest mode (M6b C1; M6A_DESIGN.md + addendum 1 §2, §4.5): one program per chip that
turns up to 32 committed rows' target taps into the drafter's context K / V ring rows, every drafter layer.
  - the K = 8 slices gather their K-tile range of the taps from the lazy rows (ROW_MAJOR, chip c's 1344 columns of
    each tap) into tiles and multicast them; the 32 streamers run the input-sharded fc (W_fc, 210 x 168 per chip);
    the hub all-reduces the partials (round 0, hub_dm.cpp unchanged); the slices apply hidden_norm and multicast ctx;
  - per drafter layer the streamers project k | v = ctx @ W_kv (2 column tiles each: bank b's 8 tiles are head block
    b: k head b for b < 4, v head b - 4), each head block goes to its slice (slice b), which applies k_norm + RoPE
    (k heads), packs BF8 and read-modify-writes the ring pages its rows land on.
The ring (DrafterRing): per (slot, layer, kv head) 72 pages of 32 positions, position p at page (p / 32) mod 72, row
p mod 32; K and V separate DRAM TILE BF8 tensors [slots x 5 x 4 x 72 x 32, 256] per chip (each chip its 4 kv heads)."""
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_31b_qb2.resident_m7 import PKG_DIR
from models.demos.gemma4_31b_qb2.resident_m7.common import (
    BF8_TILE,
    HIDDEN,
    HT,
    N_CHIPS,
    PER_BANK,
    RING_BYTES,
    TILE,
    f32_bits,
    rect_cover,
    split,
    streamer_cores,
)
from models.demos.gemma4_31b_qb2.resident_m7.drafter_common import (
    D_ENTRIES,
    D_FC,
    D_HEAD_DIM,
    D_KV_E,
    D_LAYERS,
    RING_PAGES,
    TAP_COLS,
    D_TAPS,
    d_core_cols,
    d_geometry,
)
from models.demos.gemma4_31b_qb2.resident_m7.model import tile_order, tile_unorder

KDIR = f"{PKG_DIR}/kernels/"
SEM_IN, SEM_GATHER, SEM_REC, SEM_EXCH, SEM_HADDR, SEM_STAGE, SEM_SLOTADDR, SEM_KV, SEM_KVADDR = 0, 3, 4, 5, 6, 7, 8, 11, 10
# (v3) the slices' k | v semaphores: SEM_KV + l per drafter layer (ids 11 .. 15)
KIN = D_TAPS * TAP_COLS // TILE  # 210
TAPS_ROWS = 64  # (v4) the r64b taps buffer's rows: 32 lane + r
STAGE_BYTES = 42 * 2 * 64  # (v4) d_ingest_slice_dm_v3.cpp kStage
KV_TILES = 2
HEAD_TILES = D_HEAD_DIM // TILE  # 8
FEED_NONE = 2
R = 32
TB = 64 * R


class IngestWeightsOnDevice:
    """W_fc (input-sharded, one copy) and W_kv of every drafter layer (back to back: a fixed per-bank stride)"""

    def __init__(self, mesh, dcache):
        from models.demos.gemma4_31b_qb2.resident_m7 import drafter_weights

        n, banks = mesh.get_num_devices(), mesh.dram_grid_size().x
        self.banks = banks
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})

        def place(host, e):
            cols = d_core_cols(e, banks)
            width = PER_BANK * max(len(c) for bank in cols for c in bank)
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, [D_ENTRIES[e][1] * TILE, width * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            )
            t = ttnn.to_device(ttnn.from_host_shards(host, ttnn.MeshShape(1, n)), mesh, memory_config=mc)
            ttnn.synchronize_device(mesh)
            return t

        t0 = time.time()
        fc = place(drafter_weights.load_host(dcache, D_FC), D_FC)
        kv = [place(drafter_weights.load_host(dcache, D_KV_E, l), D_KV_E) for l in range(D_LAYERS)]
        addrs = [t.buffer_address() for t in kv]
        stride = (addrs[1] - addrs[0]) & 0xFFFFFFFF
        assert all(((b - a) & 0xFFFFFFFF) == stride for a, b in zip(addrs, addrs[1:])), addrs
        self.keep = [fc] + kv
        self.entries = [(fc.buffer_address(), 0, 1), (addrs[0], stride, D_LAYERS)]
        logger.info(f"ingest weights on device: {time.time() - t0:.0f} s")


class DrafterRing:
    """the drafter's context K / V rings (BF8, DRAM, the same address on every chip; each chip's own 4 kv heads)"""

    def __init__(self, mesh, slots=1):
        self.mesh, self.slots = mesh, slots
        self.rows = slots * D_LAYERS * 4 * RING_PAGES * TILE
        z = torch.zeros(self.rows, D_HEAD_DIM)
        self.k, self.v = (self._dev(z) for _ in range(2))

    def _dev(self, host):
        t = ttnn.from_torch(host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        ttnn.synchronize_device(self.mesh)
        return t

    def fill(self, k_host, v_host):
        """every chip's ring from host [rows, 256] (replicated)"""
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for host, dev in ((k_host, self.k), (v_host, self.v)):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep), dev)
        ttnn.synchronize_device(self.mesh)

    def read(self):
        """(k, v): [chips, rows, 256] float"""
        return tuple(torch.stack([ttnn.to_torch(x).float().reshape(-1, D_HEAD_DIM) for x in ttnn.get_device_tensors(t)])
                     for t in (self.k, self.v))

    @staticmethod
    def row(slot, layer, head, pos):
        """the ring row of (slot, layer, kv head of the chip, position)"""
        return (((slot * D_LAYERS + layer) * 4 + head) * RING_PAGES + (pos // TILE) % RING_PAGES) * TILE + pos % TILE


class IngestProgram:
    def __init__(
        self,
        mesh,
        weights,
        vectors,
        ring,
        rows=32,
        slices=8,
        dump=True,
        links=int(os.environ.get("GRES_LINKS", "2")),
        packet_bytes=8192,
        ring_bytes=RING_BYTES,
        l1_top=int(os.environ.get("GRES_D_L1_TOP", "1330000")),
        fidelity=ttnn.MathFidelity.HiFi2,
        taps=None,
        n_taps=D_TAPS,
    ):
        """taps: (taps buffer 0, taps buffer 1) of the r64b engine (device tensors, the same address on every chip),
        whose tap layers are the drafter's (n_taps of them, ascending); None: a pair of this program's own"""
        self.mesh, self.ring = mesh, ring
        n = self.n = mesh.get_num_devices()
        self._slots = ring.slots
        self.n_taps = n_taps
        assert n_taps == D_TAPS, "the taps buffer's tap layers must be the drafter's (tap index = fc tap order)"
        assert n == N_CHIPS
        banks = self.banks = weights.banks
        M = self.M = rows
        K = self.K = slices
        assert K == banks == 8, "slice k = bank k = head block k"
        T = self.T = HT // K
        V = self.V = T
        P = -(-T * TB // packet_bytes)
        layers = self.layers = D_LAYERS

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
        C = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])  # noqa: E731
        grid, hub_grid, slice_grid = C(cores), C([hub]), C(slices_c)
        phys = lambda c: mesh.worker_core_from_logical_core(c)  # noqa: E731
        hub_p, slice_p = phys(hub), [phys(c) for c in slices_c]
        self.cores, self.hub, self.slices = cores, hub, slices_c
        logger.info(f"ingest program: M {M}, {S} streamers, hub {hub}, {K} slices")

        rep = ttnn.ReplicateTensorToMesh(mesh)
        dram = ttnn.DRAM_MEMORY_CONFIG

        def settled(t):
            ttnn.synchronize_device(mesh)
            return t

        def dram_t(t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
            return settled(ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=dram, mesh_mapper=rep))

        def u32(grid_, words):
            return settled(
                ttnn.from_torch(
                    torch.zeros(1, words, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh, mesh_mapper=rep,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                        ttnn.ShardSpec(grid_, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
                    ),
                )
            )

        cols = [d_core_cols(D_FC, banks), d_core_cols(D_KV_E, banks)]
        per_core = lambda e, i: cols[e][i // PER_BANK][i % PER_BANK]  # noqa: E731
        nd_max = max(len(per_core(0, i)) for i in range(S))
        assert all(len(per_core(1, i)) == KV_TILES for i in range(S))
        geo = [d_geometry(D_FC), d_geometry(D_KV_E)]
        kts = [D_ENTRIES[D_FC][1], D_ENTRIES[D_KV_E][1]]
        assert kts[0] == KIN

        act_l1 = TB * (KIN + nd_max + 2 * KV_TILES)
        l1_base = ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1)
        ring_bytes = min(ring_bytes, l1_top - l1_base - act_l1 - 1024)
        self.ring_bytes = ring_bytes = (ring_bytes // BF8_TILE) * BF8_TILE
        logger.info(f"ingest weight ring {ring_bytes} B per streamer")

        self.rec_t = u32(hub_grid, S * (-(-R * 8 // 16) * 4))
        self.cand_t = u32(hub_grid, n * (-(-R * 8 // 16) * 4))
        self.row_elems = K * V * 1024
        self.out_t = dram_t(torch.zeros(1, 80, dtype=torch.int32), ttnn.uint32)
        # (v4) the taps buffers (r64b layout) and the launch rows' source rows
        if taps is None:
            taps = tuple(dram_t(torch.zeros(1, 1, TAPS_ROWS, n_taps * HIDDEN), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                         for _ in range(2))
        self.taps = taps
        self.wnorm_t = dram_t(self.view_rows([vectors["hidden_norm"]]), ttnn.bfloat16)
        row32 = lambda vals: dram_t(torch.tensor([vals], dtype=torch.int32), ttnn.uint32)  # noqa: E731
        self.pos_t, self.slot_t = row32([-1] * 32), row32([0] * 32)
        self.src_t = row32([-1] * 32)
        self.rope_t = dram_t(torch.zeros(2 * R, D_HEAD_DIM), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        kn = torch.stack([vectors["layers"][l]["k_norm"].float().expand(R, -1) for l in range(layers)]).reshape(-1, D_HEAD_DIM)
        self.knorm_t = dram_t(kn, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.pdump_t = dram_t(torch.zeros(K * layers * R, D_HEAD_DIM), ttnn.bfloat8_b, ttnn.TILE_LAYOUT) if dump else None
        self.cdump_t = dram_t(torch.zeros(1, self.row_elems), ttnn.bfloat16) if dump else None
        self.tokens_t, self.positions_t, self.slots_t = row32([0] * 32), row32([0] * 32), row32([0] * 32)
        self.control_t, self.status_t = row32([FEED_NONE] + [0] * 31), row32([0] * 32)
        self.theta = vectors["theta"]

        self.hub_sems = [ttnn.create_global_semaphore(mesh, hub_grid, 0) for _ in range(2)]
        self.slice_sems = [ttnn.create_global_semaphore(mesh, slice_grid, 0) for _ in range(3)]
        hsem = [ttnn.get_global_semaphore_address(s) for s in self.hub_sems]
        ssem = [ttnn.get_global_semaphore_address(s) for s in self.slice_sems]
        self.links = 1
        if links > 1:
            node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, c))  # noqa: E731
            trained = min(len(ttnn.get_forwarding_link_indices(mesh, node(c), node(c + s)))
                          for c in range(n) for s in (1, -1) if 0 <= c + s < n)
            self.links = max(1, min(links, trained))

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
        q_split = split(KIN, K)
        nq_max = max(c for c, _ in q_split)
        scr_bytes = 384 + 64 + STAGE_BYTES + HEAD_TILES * BF8_TILE  # (v4) d_ingest_slice_dm_v3.cpp's scratch
        cbs = [
            ttnn.CBDescriptor(total_size=ring_bytes, core_ranges=grid,
                              format_descriptors=[fmt(e, ttnn.bfloat8_b, BF8_TILE) for e in range(2)]),
            cb(7, grid, 1, ttnn.uint32, 16),
            cb((8, 9), grid, KIN),
            cb(11, grid, nd_max),
            cb(12, grid, 2 * KV_TILES),
            cb(0, hub_grid, 2 * HT),
            cb(6, hub_grid, 1, ttnn.uint32, 512),
            cb(7, hub_grid, 1, ttnn.uint32, hdrs * 160),
            cb(0, slice_grid, n * V),
            cb(3, slice_grid, V),
            cb(4, slice_grid, V),
            cb(6, slice_grid, 3),
            cb(7, slice_grid, 1, *f32),
            cb(8, slice_grid, K, *f32),
            cb(9, slice_grid, 1, *f32),
            cb(10, slice_grid, 2, *f32),
            cb(11, slice_grid, V),
            cb(12, slice_grid, layers * HEAD_TILES),
            cb(13, slice_grid, HEAD_TILES),
            cb(14, slice_grid, 2 * HEAD_TILES),
            cb(15, slice_grid, HEAD_TILES),
            cb(16, slice_grid, HEAD_TILES, ttnn.bfloat8_b, BF8_TILE),
            cb(17, slice_grid, nq_max),
            cb(18, slice_grid, 1, ttnn.uint32, -(-scr_bytes // 16) * 16),
            cb(19, slice_grid, 1, *f32),  # the head norm's partial (compute only)
        ]
        owners = {SEM_IN: grid, SEM_GATHER: hub_grid, SEM_REC: hub_grid, SEM_EXCH: slice_grid, SEM_HADDR: slice_grid,
                  SEM_STAGE: grid, SEM_SLOTADDR: grid.merge(hub_grid), SEM_KVADDR: grid}
        owners.update({SEM_KV + l: slice_grid for l in range(layers)})
        assert SEM_KV + layers <= 16, "semaphore ids"

        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=g, initial_value=0) for i, g in owners.items()]

        ct = [layers, S, HT, KIN, 0, ring_bytes, nd_max, K, SEM_IN, SEM_KV, M]
        for Kt, (sb, pages, page, blk) in zip(kts, geo):
            ct += [Kt, sb, pages, page, blk]
        slot_bytes = self.slot_bytes = V * 2048
        slice_xy = [v for p in slice_p for v in (p.x, p.y)]
        rects = rect_cover(mesh, cores, False)
        kernel = lambda src, cr_, ct_, rt_, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=KDIR + src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cr_, compile_time_args=ct_, runtime_args=rt_, config=cfg,
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
                    ent += [len(per_core(e, i)), addr & 0xFFFFFFFF, stride, copies]
                d, kvc = per_core(0, i), per_core(1, i)
                assert bank * T <= d[0] and d[-1] < (bank + 1) * T and kvc == [8 * bank + 2 * j, 8 * bank + 2 * j + 1]
                rr[c.x][c.y] = [bank, i & 0x3, j, PER_BANK] + ent + [hub_p.x, hub_p.y, SEM_REC]
                wr[c.x][c.y] = [
                    len(d), d[0], hub_p.x, hub_p.y, SEM_GATHER, SEM_REC, i, slice_p[bank].x, slice_p[bank].y,
                    SEM_STAGE, ssem[0], ssem[1], d[0] - bank * T, slot_bytes, SEM_SLOTADDR, SEM_HADDR, SEM_KVADDR, j,
                ] + slice_xy
                cr[c.x][c.y] = [len(d), KV_TILES]
            hub_rt = ttnn.RuntimeArgs()
            hub_rt[hub.x][hub.y] = [
                SEM_STAGE, SEM_SLOTADDR, ssem[0], ssem[1], slot_bytes, self.rec_t.buffer_address(),
                self.cand_t.buffer_address(), hsem[0], hsem[1], self.tokens_t.buffer_address(),
                self.positions_t.buffer_address(), self.slots_t.buffer_address(), self.out_t.buffer_address(),
                self.control_t.buffer_address(), self.status_t.buffer_address(), 1 << 30, R, 0,
            ] + rects + slice_xy
            hub_ct = [n, chip, R, M, HT, K, T, packet_bytes, 1, S, SEM_GATHER, SEM_REC, self.links, 0, M, K]
            kernels = [
                kernel("d_ingest_reader.cpp", grid, sct, rr, dm(NCRISC, NOC0)),
                kernel("d_ingest_writer_v2.cpp", grid, sct, wr, dm(BRISC, NOC1)),
                kernel("d_ingest_compute.cpp", grid, sct, cr, compute_cfg(fidelity)),
                kernel("d_ingest_hub_dm.cpp", hub_grid, hub_ct, hub_rt, dm(BRISC, NOC0)),
            ]
            sc_rt = ttnn.RuntimeArgs()
            for k, c in enumerate(slices_c):
                nq, qa = q_split[k]
                sdm_rt = ttnn.RuntimeArgs()
                sdm_rt[c.x][c.y] = [
                    SEM_HADDR, ssem[0], ssem[2], self.taps[0].buffer_address(), self.wnorm_t.buffer_address(),
                    self.pos_t.buffer_address(), self.slot_t.buffer_address(), self.rope_t.buffer_address(),
                    self.knorm_t.buffer_address(), (ring.k if k < 4 else ring.v).buffer_address(),
                    self.pdump_t.buffer_address() if dump else 0, self.cdump_t.buffer_address() if dump else 0,
                    SEM_SLOTADDR, SEM_KVADDR, hub_p.x | (hub_p.y << 16),
                ] + rects + slice_xy + [self.taps[1].buffer_address(), self.src_t.buffer_address()]
                sc_ct = [n, chip, k, K, T, V, layers, PER_BANK * (banks // K), P, SEM_IN, SEM_EXCH, SEM_KV, M, qa, nq,
                         RING_PAGES, D_LAYERS, ring.slots, n_taps, SEM_REC]
                kernels.append(kernel("d_ingest_slice_dm_v3.cpp", C([c]), sc_ct, sdm_rt, dm(BRISC, NOC0)))
                sc_rt[c.x][c.y] = [int(k < 4)]
            kernels.append(kernel(
                "d_ingest_slice_compute.cpp", slice_grid,
                [n, V, layers, K, f32_bits(vectors["eps"]), f32_bits(1 / HIDDEN), f32_bits(1 / D_HEAD_DIM)],
                sc_rt, compute_cfg(ttnn.MathFidelity.HiFi4, unpack_fp32=(7, 8, 19)),
            ))
            program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            args = program.kernels[3].runtime_args[hub.x][hub.y]
            for link in range(self.links):
                for nb in (chip + 1, chip - 1):
                    if 0 <= nb < n:
                        args.append(1)
                        args.extend(ttnn.setup_fabric_connection(
                            mesh, mesh.get_fabric_node_id(coord), mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, nb)),
                            link, program, hub))
                    else:
                        args.append(0)
            mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program
            self.programs.append(program)
        self.mesh_pd = mesh_pd
        self.io = [self.rec_t, self.cand_t, self.out_t, self.taps[0], self.taps[1], self.src_t, self.wnorm_t, self.pos_t,
                   self.slot_t, self.rope_t,
                   self.knorm_t, ring.k, ring.v, self.tokens_t, self.positions_t, self.slots_t, self.control_t,
                   self.status_t] + [t for t in (self.pdump_t, self.cdump_t) if t is not None] + weights.keep

    # ---- host-side inputs
    def view_rows(self, mats):
        """d_model.DrafterProgram.view_rows: [len(mats), K V 1024] bf16 from [R or 1, HIDDEN] matrices"""
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

    def fill_taps(self, parity, rows):
        """(tests) rows {buffer row (32 lane + r): taps [5, HIDDEN]} into this program's own taps buffer `parity`
        (the r64b layout: [1, 1, 64, n x 5376], tap-major columns), every chip the same; other rows zero"""
        host = torch.zeros(1, 1, TAPS_ROWS, self.n_taps * HIDDEN, dtype=torch.bfloat16)
        for r, taps in rows.items():
            host[0, 0, r] = taps.reshape(-1).bfloat16()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)), self.taps[parity])

    @staticmethod
    def taps_live(taps_launch, engine_launches):
        """the verify launch `taps_launch` (0-based) wrote taps_buffer(taps_launch % 2); launch taps_launch + 2 rewrites
        it, so it is live while the engine has run at most taps_launch + 2 launches"""
        return taps_launch < engine_launches <= taps_launch + 2

    def set_rows(self, src_rows, positions, slots, taps_launch, engine_launches):
        """up to 32 launch rows: row r ingests buffer row src_rows[r] (32 lane + r of verify `taps_launch`) into ring
        slot slots[r] at position positions[r]; engine_launches: the r64b engine's launch count now (the ordering
        check). Rows past len(src_rows) are inert"""
        rows = len(src_rows)
        assert rows == len(positions) == len(slots) <= self.M, (rows, len(positions), len(slots))
        assert self.taps_live(taps_launch, engine_launches), \
            f"taps of launch {taps_launch} not live at engine launch count {engine_launches}"
        parity = taps_launch % 2
        assert all(0 <= r < TAPS_ROWS for r in src_rows) and all(0 <= s < self._slots for s in slots)
        keys = [(s, p) for s, p in zip(slots, positions)]
        assert len(set(keys)) == len(keys), "one row per (slot, position)"
        src = [parity * TAPS_ROWS + r for r in src_rows] + [-1] * (R - rows)
        pos = list(positions) + [-1] * (R - rows)
        slt = list(slots) + [0] * (R - rows)
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        w = lambda host, dev, dt, layout=ttnn.ROW_MAJOR_LAYOUT: ttnn.copy_host_to_device_tensor(  # noqa: E731
            ttnn.from_torch(host, dtype=dt, layout=layout, mesh_mapper=rep), dev)
        w(torch.tensor([src], dtype=torch.int32), self.src_t, ttnn.uint32)
        w(torch.tensor([pos], dtype=torch.int32), self.pos_t, ttnn.uint32)
        w(torch.tensor([slt], dtype=torch.int32), self.slot_t, ttnn.uint32)
        inv = 1.0 / (self.theta ** (torch.arange(0, D_HEAD_DIM, 2, dtype=torch.float32) / D_HEAD_DIM))
        f = torch.outer(torch.tensor([max(p, 0) for p in pos], dtype=torch.float32), inv)
        emb = torch.cat([f, f], -1)
        live = torch.tensor([p >= 0 for p in pos], dtype=torch.float32)[:, None]
        w(torch.cat([emb.cos() * live, emb.sin() * live]), self.rope_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(self.mesh)

    def reset(self):
        for s in self.hub_sems + self.slice_sems:
            ttnn.reset_global_semaphore_value(s, 0)
        ttnn.synchronize_device(self.mesh)

    def launch(self):
        ttnn.generic_op(self.io, self.mesh_pd)

    # ---- read back
    def ctx(self):
        """[n, M, HIDDEN] ctx rows (dump)"""
        out = []
        for d in (ttnn.to_torch(x) for x in ttnn.get_device_tensors(self.cdump_t)):
            d = d.reshape(self.K, self.V * 1024)
            flat = torch.cat([d[k, : self.T * TB // 2] for k in range(self.K)])
            out.append(tile_unorder(flat, R)[: self.M].float())
        return torch.stack(out)

    def packed(self):
        """[n, K, layers, 32, 256] the slices' BF8 tiles of every layer (dump), as values"""
        return torch.stack([ttnn.to_torch(x).float().reshape(self.K, self.layers, R, D_HEAD_DIM)
                            for x in ttnn.get_device_tensors(self.pdump_t)])

    def cb_tops(self):
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
