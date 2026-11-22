# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""K2a (M5 option B): the R = 32 attention for U = 32 / M users of M rows, on 8 head cores and 60 unit cores per chip
(kernels/k2a_*; design bench/resident/m5/K2A_DESIGN_v2.md + addenda 1-2). The host side: placement (the engine's
attention cores: the 4 group leaders + 4 spare full-only cores become head cores, the 28 group non-leaders + 32
full-only cores unit cores), CB / semaphore descriptors per role, the DRAM staging buffers, kernel descriptors.
Interface as attention.Attention's where model.py uses it (cbs, semaphores, kernels, xy, heads for the streamers,
unit cores for the hub's error words)."""
import os
import struct

import torch

import ttnn
from models.demos.gemma4_31b_qb2.resident_m5_14.common import EPS, f32_bits, rect_cover, seam_split
from models.demos.gemma4_31b_qb2.resident_m5_14.k2a_sched import pack

GROUPS, GROUP_CORES = 4, 8
HEADS, UNITS, PER_HEAD = 8, 60, 15
DTS, DTF = 8, 16
BF8, BF16, F32 = 1088, 2048, 4096
# CB indices (k2a_common.hpp)
(CB_Q, CB_K, CB_V, CB_MASK, CB_ONE, CB_S, CB_P, CB_M, CB_L, CB_OWN, CB_PARTS, CB_OUT, CB_SC, CB_DIAG, CB_CP, CB_RA,
 CB_RB, CB_SCHED, CB_GIN) = range(19)  # (K2b step 3) CB_GIN: the group nodes' inputs
CB_NEWK, CB_NEWV, CB_TAB, CB_TS = 24, 25, 26, 27
(HCB_RX, HCB_W, HCB_CS, HCB_ONE, HCB_SQ, HCB_RINV, HCB_XN, HCB_QO, HCB_KR, HCB_VN, HCB_KV8, HCB_OG) = range(12)
HCB_RX8, HCB_O8 = 12, 13  # (M5 D16) R = 8: the streamers' 8-row tiles in, O out (k2a_common.hpp hcb_rx8 / hcb_o8)
HCB_FEED = 24
# (K2b step 1) MAX_Q bounds the per-kind key lists only (Q streams one key at a time: CQ = 2 slots of DTF pages);
# MAX_UNITS 40 (k2a_common.hpp kMaxUnits)
MAX_Q, MAX_UNITS, NP, NV_S, NV_F, PT_F = 8, 40, 150, 8, 8, 18  # (m5_14 F1) NV_S 9 -> 8: no straddling tail tile
# full units per user <= 8 (k2a_common.hpp kMaxChunksF; K2b step 1: the flat merge of <= 8 partials fits cb_parts);
# (K2b step 3) longer users (<= 128 chunks: kMaxChunksT) take the group-node tree: positions < 128 * 2048.
# The R = 32 program's capacity is capped there for every K2a core and the hub alike, so a user past it is flagged
# (F_POSITION) and inert everywhere
MAX_POS = 128 * 2048
FLAT_POS = 8 * 2048  # (K2b) the flat merge's reach: a launch with every position below it never uses the tree
PART_SLOTS, MAX_CHUNKS_T = 32, 128  # (K2b step 3) DRAM partials: [user x half (<= 32), chunk, PT_F pages]
TL_HEAD, TL_UNIT = 6, 2  # (timing) words per layer and core (k2a_common.hpp kTlHead: + D16 pad / repack, kTlUnit)
TAB_W = 64
L1_BASE, L1_TOP = 112_576, 1_458_880  # the CB tops' convention: the L1 floor - 32 KB (K2A_DESIGN_v2 addendum 2 (3a))
KR = int(os.environ.get("GRES_K2A_KR", "240"))  # cb_k pages (multiple of 16)
VR = int(os.environ.get("GRES_K2A_VR", "216"))  # cb_v pages (multiple of 8: m5_14 NV_S = NV_F = 8; m5_13 72 = 9 x 8)
assert KR % 16 == 0 and VR % 8 == 0, (KR, VR)
# (diagnostics) the unit compute's fidelity and DEST accumulation (default: the engine's M1L workers': HiFi2, bf16)
UNIT_FID = os.environ.get("GRES_K2A_UNIT_FID", "")
UNIT_FP32 = os.environ.get("GRES_K2A_UNIT_FP32", "0") == "1"


def tl_tensor(mesh, cores, words):
    """(CP1 timing, m5_3_1) one L1 shard of `words` u32 per core of `cores` (the same address on each), zeroed"""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    t = ttnn.from_torch(torch.zeros(len(cores), words, dtype=torch.int32), dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                        memory_config=ttnn.MemoryConfig(
                            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(grid, [1, words], ttnn.ShardOrientation.ROW_MAJOR)))
    ttnn.synchronize_device(mesh)
    return t


def tl_read(t, n, layers, words):
    """per chip [n cores, layers, words] wall-clock words (int64)"""
    return torch.stack([ttnn.to_torch(c).reshape(n, layers, words).to(torch.int64) & 0xFFFFFFFF
                        for c in ttnn.get_device_tensors(t)])


class K2aAttention:
    def __init__(self, mesh, M, layers, capacity, used, num_streamers, layer_group, timeline=False, R=32,
                 parts_dram=None):
        assert M in (1, 2, 4, 8), M
        # (M5 D16) R = 8: U = 8 / M users (M < 8); the head cores pad the streamers' 8-row tiles to 32 rows
        assert R == 32 or (R == 8 and M < 8), (R, M)
        self.R = R
        self.mesh, self.M, self.U, self.layers = mesh, M, R // M, layers
        self.capacity = min(capacity, MAX_POS)  # (the R = 32 program's: see MAX_POS)
        self.num_streamers = num_streamers
        grid = mesh.compute_with_storage_grid_size()
        free = {(x, y) for x in range(grid.x) for y in range(grid.y)} - set(used)
        # the engine's attention placement (attention.Attention): 4 groups of 8, then the full-only cores
        groups = []
        for w, h in ((4, 2), (2, 4)):
            for y in range(grid.y - h + 1):
                for x in range(grid.x - w + 1):
                    cells = [(x + dx, y + dy) for dy in range(h) for dx in range(w)]
                    if len(groups) < GROUPS and all(c in free for c in cells):
                        groups.append(cells)
                        free -= set(cells)
        assert len(groups) == GROUPS, "no room for the attention groups"
        spare = sorted(free, key=lambda c: (c[1], c[0]))
        assert len(spare) >= 36 + 1, f"K2a needs 36 more cores, has {len(spare)}"
        C = ttnn.CoreCoord
        leaders = [C(*grp[0]) for grp in groups]
        extra = [C(*c) for c in spare[:37]]
        self.heads = leaders + extra[32:36]
        # unit cores, kv head g's kPerHead sliding cores contiguous: group g's 7 non-leaders + 8 full-only cores
        self.units = [c for g in range(GROUPS) for c in [C(*x) for x in groups[g][1:]] + extra[8 * g : 8 * g + 8]]
        assert len(self.units) == UNITS and len(self.heads) == HEADS
        self.spare = extra[36:37]
        self.flat = self.units  # (the hub's error words: the unit cores)
        self.leaders = self.heads  # (the streamers' q|k|v and O partners)
        CR = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])  # noqa: E731
        self.grid_u, self.grid_h = CR(self.units), CR(self.heads)
        self.phys = lambda c: mesh.worker_core_from_logical_core(c)  # noqa: E731
        # layer groups (as attention.Attention): sliding (bits 0..5) / full (8..13)
        sl = {g for l, g in enumerate(layer_group[:layers]) if l % 6 != 5}
        fu = {g for l, g in enumerate(layer_group[:layers]) if l % 6 == 5}
        assert not sl & fu and all(0 <= g < 6 for g in sl | fu), (sl, fu)
        assert 16 + max(len(sl) * 3, len(fu) * 17) <= TAB_W, "a unit's table region"
        self.group_kinds = sum(1 << g for g in sl) | sum(1 << (8 + g) for g in fu)
        self.CQ = 2 * DTF  # (K2b step 1) the Q ring: 2 slots of kQSlot = DTF pages
        # (K2b step 3) the DRAM partials of long users ([user x half, chunk, PT_F pages] of 2 KB): one buffer per engine,
        # shared by every block model (model.py passes the first one's); a single page when no position can reach the
        # tree (capacity <= FLAT_POS)
        if parts_dram is None:
            pages = PART_SLOTS * MAX_CHUNKS_T * PT_F if self.capacity > FLAT_POS else 1
            parts_dram = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32 * pages]), ttnn.bfloat16,
                                                        ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        self.parts_dram = parts_dram
        # (K2b K1) every unit core's schedule records (its cb_sched page) after each launch's build: [60, 1024] u32
        self.sched_dump = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, len(self.units), 1024]), ttnn.uint32,
                                                         ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        # DRAM staging: Q (bf16 tiles: per kind (g * tiles + T) * Dt + d), K and V (BF8: g * Dt + d)
        q_pages = max(4 * pack(False, M, R)[4] * DTS, pack(True, M, R)[4] * DTF)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.q_stage = ttnn.from_torch(torch.zeros(1, 1, 32, 32 * q_pages), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=mesh, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.k_stage, self.v_stage = (
            ttnn.from_torch(torch.zeros(1, 1, 32, 32 * 4 * DTS), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                            device=mesh, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(2)
        )
        self.io = [self.q_stage, self.k_stage, self.v_stage]
        # (CP1 timing, m5_3_1) per core timeline words (k2a_common.hpp K2A_MARK: head kTlHead, unit kTlUnit per layer)
        self.tl_h = tl_tensor(mesh, self.heads, layers * TL_HEAD) if timeline else None
        self.tl_u = tl_tensor(mesh, self.units, layers * TL_UNIT) if timeline else None
        self.io += [t for t in (self.tl_h, self.tl_u) if t is not None]

    def read_sched(self):
        """(K2b K1) per chip [unit cores, 1024] u32: each unit core's schedule page as its last launch built it"""
        return [ttnn.to_torch(t).reshape(len(self.units), 1024).to(torch.int64) & 0xFFFFFFFF
                for t in ttnn.get_device_tensors(self.sched_dump)]

    def timelines(self):
        """(CP1) {"head": [chips, 8, layers, 4] (rx done, READY sent, O complete, sem_o sent), "unit": [chips, 60,
        layers, 2] (READY seen, last root's O sent)}, raw wall-clock words; None with the timeline off"""
        if self.tl_h is None:
            return None
        return {"head": tl_read(self.tl_h, HEADS, self.layers, TL_HEAD),
                "unit": tl_read(self.tl_u, UNITS, self.layers, TL_UNIT)}

    def xy(self, c):
        p = self.phys(c)
        return p.x | (p.y << 16)

    # ---- descriptors
    def ct(self):
        return [self.layers, self.M, self.num_streamers, f32_bits(EPS), f32_bits(1 / 256), f32_bits(1 / 512),
                self.group_kinds, KR, VR, NP, UNITS, PER_HEAD, self.CQ, self.R]  # (D16) 13: R

    def unit_cbs(self):
        bf16, bf8, u32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint32
        return [  # (index, pages, dtype, page bytes)
            (CB_Q, self.CQ, bf16, BF16), (CB_K, KR, bf8, BF8), (CB_V, VR, bf8, BF8), (CB_MASK, 4, bf16, BF16),
            (CB_ONE, 2, bf16, BF16), (CB_S, NV_S, bf16, BF16), (CB_P, NV_S, bf16, BF16), (CB_M, 2, bf16, BF16),
            (CB_L, 2, bf16, BF16), (CB_OWN, 2 * PT_F, bf16, BF16), (CB_PARTS, NP, bf16, BF16),
            (CB_OUT, 32, bf16, BF16), (CB_SC, 8, bf16, BF16), (CB_DIAG, 8, bf16, BF16), (CB_CP, PT_F, bf16, BF16),
            (CB_RA, PT_F, bf16, BF16), (CB_RB, PT_F, bf16, BF16), (CB_SCHED, 1, u32, 4096), (CB_GIN, 2 * PT_F, bf16, BF16),
            (CB_NEWK, DTF, bf8, BF8), (CB_NEWV, DTF, bf8, BF8), (CB_TAB, MAX_UNITS, u32, 4 * TAB_W), (CB_TS, 1, u32, 512),
        ]

    def head_cbs(self):
        bf16, bf8, f32, u32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint32
        return [
            (HCB_RX, 32, bf16, BF16), (HCB_W, 32, bf16, BF16), (HCB_CS, 48, bf16, BF16), (HCB_ONE, 1, bf16, BF16),
            (HCB_SQ, 16, f32, F32), (HCB_RINV, 1, f32, F32), (HCB_XN, 16, bf16, BF16), (HCB_QO, 16, bf16, BF16),
            (HCB_KR, 16, bf16, BF16), (HCB_VN, 16, bf16, BF16), (HCB_KV8, 16, bf8, BF8), (HCB_OG, 16, bf16, BF16),
            (HCB_FEED, 1, u32, 512),
        ] + ([(HCB_RX8, 32, bf16, 64 * self.R), (HCB_O8, 16, bf16, 64 * self.R)] if self.R < 32 else [])  # (D16)

    def l1_bytes(self):
        tot = lambda cbs: sum(-(-p * b // 16) * 16 for _, p, _, b in cbs)  # noqa: E731
        return {"unit": tot(self.unit_cbs()), "unit_top": L1_BASE + tot(self.unit_cbs()), "head": tot(self.head_cbs()),
                "head_top": L1_BASE + tot(self.head_cbs()), "limit": L1_TOP}

    def cbs(self):
        def make(lst, grid):
            out = []
            for idx, pages, dt, page in lst:
                fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)
                out.append(ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid, format_descriptors=[fd]))
            return out

        l1 = self.l1_bytes()
        assert l1["unit_top"] <= L1_TOP and l1["head_top"] <= L1_TOP, l1
        return make(self.unit_cbs(), self.grid_u) + make(self.head_cbs(), self.grid_h)

    def semaphores(self):
        # (K2b step 3) unit ids 14 / 15: the group nodes' lane counters; head id 6: H_SCHED (the schedule status)
        return [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.grid_u, initial_value=0) for i in range(16)] + [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=self.grid_h, initial_value=0) for i in range(7)]

    def kernels(self, kernel, dm, compute_cfg, streamer_cores, serve, rope, w_addr, kv_addrs, sem_o, sem_rxaddr,
                fid_head=ttnn.MathFidelity.HiFi4, fid_unit=ttnn.MathFidelity.HiFi2):
        """reader / writer / compute descriptors of one chip (the same on every chip); rope: cos_s, sin_s, cos_f, sin_f
        addresses; kv_addrs: per layer (K, V)"""
        NCRISC, BRISC = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
        NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default
        gmap = [0] * 8
        for l, grp in enumerate(serve.layer_group[: self.layers]):
            gmap[l // 8] |= grp << (4 * (l % 8))
        pools = [serve.pool_pages.get(g_, 0) for g_ in range(6)]
        kv = [a for pair in kv_addrs[: self.layers] for a in pair]
        hxy = [self.xy(c) for c in self.heads]
        uxy = [self.xy(c) for c in self.units]
        pmax = -(-serve.capacity // 128)
        # unit cores: common args (k2a_unit_reader.cpp) + the lists they index
        head = 17
        ucom = [serve.positions.buffer_address(), serve.slots.buffer_address(), self.capacity, serve.max_users,
                serve.tables.buffer_address(), serve.scratch_t.buffer_address(), pmax]
        o_gmap = head
        o_pool = o_gmap + 8
        o_kv = o_pool + 6
        o_hxy = o_kv + len(kv)
        o_uxy = o_hxy + HEADS
        ucom += [o_gmap, o_pool, o_kv, self.q_stage.buffer_address(), self.k_stage.buffer_address(),
                 self.v_stage.buffer_address(), o_hxy, o_uxy, self.parts_dram.buffer_address(),  # (K2b step 3) 15
                 self.sched_dump.buffer_address()]  # (K2b K1) 16
        assert len(ucom) == head
        ucom += gmap + pools + kv + hxy + uxy
        # head cores
        seam = seam_split(32)
        srects = rect_cover(self.mesh, streamer_cores, seam)
        sxy = [self.xy(c) for c in streamer_cores]
        hhead = 14
        o_rect = hhead
        o_sxy = o_rect + len(srects)
        o_huxy = o_sxy + len(sxy)
        hcom = [serve.positions.buffer_address(), serve.slots.buffer_address(), self.capacity, serve.max_users] + list(rope) \
            + [w_addr, sem_o, sem_rxaddr, o_rect, o_sxy, o_huxy]
        assert len(hcom) == hhead
        hcom += srects + sxy + uxy
        hwcom = [self.q_stage.buffer_address(), self.k_stage.buffer_address(), self.v_stage.buffer_address(), 4] + uxy
        ur, uw, uc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.units):
            ur[c.x][c.y] = [i]
            uw[c.x][c.y] = [i]
            uc[c.x][c.y] = []
        hr, hw, hc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for h, c in enumerate(self.heads):
            hr[c.x][c.y] = [h]
            hw[c.x][c.y] = [h]
            hc[c.x][c.y] = [h]
        ct = self.ct()
        out = []
        for src, grid, rt, com, cfg in (
            ("k2a_unit_reader.cpp", self.grid_u, ur, ucom, dm(NCRISC, NOC1)),
            ("k2a_unit_writer.cpp", self.grid_u, uw, ucom, dm(BRISC, NOC0)),
            ("k2a_unit_compute.cpp", self.grid_u, uc, [], compute_cfg(
                getattr(ttnn.MathFidelity, UNIT_FID) if UNIT_FID else fid_unit, UNIT_FP32, True)),
            ("k2a_head_reader.cpp", self.grid_h, hr, hcom, dm(NCRISC, NOC1)),
            ("k2a_head_writer.cpp", self.grid_h, hw, hwcom, dm(BRISC, NOC0)),
            ("k2a_head_compute.cpp", self.grid_h, hc, [], compute_cfg(fid_head, True, True)),
        ):
            kd = kernel(src, grid, ct, rt, cfg)
            if com:
                kd.common_runtime_args = com
            tl = self.tl_h if grid is self.grid_h else self.tl_u
            if tl is not None and "compute" not in src:  # (CP1 timing: the DM kernels' marks)
                kd.defines = [("K2A_TL", str(tl.buffer_address()))]
            out.append(kd)
        return out
