# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""(v3, M7 D v10, branch (B) R4: LEAD_DECISIONS_M7 M270 / M275, bench/resident/m7/m71/D_OMCAST_DESIGN.md + addendum 1;
d_k2a_v2.py frozen) d_k2a_v2.py with the head cores' reader kernels/d_k2a_head_reader_v2.cpp: the O tiles and the sem_o
increments go to the streamers as unicasts (one per streamer, the sxy list) instead of one multicast per streamer
rectangle, on the same RISC and NoC (NCRISC, NoC 1). Everything else, args included, is v2's: the head common args
keep the rectangle list (unread by v2's kernel), so hcom is byte-equal to v2's (gates G0 (iv)). Below, v2's docstring.

(v2, M7 D v8: bench/resident/m7/M7_BUILD_PREREG_v2_ADDENDUM_2.md s2) d_k2a.py with an optional `rope=(cos, sin)`:
the RoPE tables of another owner (the target engine's sliding pair, bitwise the drafter's: host/rope_alias_check.py) in
place of this program's own [capacity, 256] pair (2 x capacity x 512 B per chip: ~268 MB at 262K). Kernels unchanged.
Below, d_k2a.py's docstring.

M6b integration (bench/resident/m6/M6B_INTEGRATION_PLAN.md + review): the DFlash2 drafter's attention on K2a's head
and unit cores, in drafter mode (kernels/d_k2a_*: copies of resident_m5_3's K2a kernels, frozen, with kDraftW > 0).
Host side, copied from resident_m5_3/k2a.py with the drafter deltas:
  - placement: 8 head cores and 60 unit cores from the cores the D-program leaves free (kv head g's 15 unit cores
    contiguous in the list);
  - every layer sliding (5 Qwen3 layers: 8 q / 4 kv heads x 256 per chip, GQA 2, RoPE theta 1e4, as Gemma sliding);
  - the window: rows of a user block at p .. p + M - 1 see [p - W, p + M - 1], W = sliding_window (2048);
  - K / V: the DrafterRing (d_ingest.py), slot-major, no page tables (no cb_tab / cb_ts);
  - roots per core <= 2, partials per root <= 9: cb_parts NP = 2 x 9 x 10 pages, cb_sc / cb_diag 9 pages (GI0:
    d_k2a_sched.py);
  - head prep: q = rope(rms(q) w_q / 16) (the 1 / sqrt(256) score scale folded into the q_norm weight, exact in bf16),
    k = rope(rms(k) w_k), v passes through (Qwen3 has no v_norm); BF8 K / V;
  - the cb_k / cb_v rings KR / VR = 192 / 144 pages (GRES_DK2A_KR / VR; L1: see l1_bytes and the note at KR);
  - a program of layers [layer_offset, layer_offset + layers) reads / writes those layers of the ring (ct 14)."""
import os

import math

import torch

import ttnn
from models.demos.gemma4_31b_qb2.resident_m7.common import f32_bits, rect_cover, seam_split
from models.demos.gemma4_31b_qb2.resident_m7.d_k2a_sched import kmax, pack_s, root_cap
from models.demos.gemma4_31b_qb2.resident_m7.drafter_common import D_HEAD_DIM, D_LAYERS, RING_PAGES, TILE

HEADS, UNITS, PER_HEAD = 8, 60, 15
DTS, DTF = 8, 16
BF8, BF16, F32 = 1088, 2048, 4096
# CB indices (d_k2a_common.hpp)
(CB_Q, CB_K, CB_V, CB_MASK, CB_ONE, CB_S, CB_P, CB_M, CB_L, CB_OWN, CB_PARTS, CB_OUT, CB_SC, CB_DIAG, CB_CP, CB_RA,
 CB_RB, CB_SCHED) = range(18)
CB_NEWK, CB_NEWV = 24, 25
(HCB_RX, HCB_W, HCB_CS, HCB_ONE, HCB_SQ, HCB_RINV, HCB_XN, HCB_QO, HCB_KR, HCB_VN, HCB_KV8, HCB_OG) = range(12)
HCB_FEED = 24
MAX_Q, NV_S, PT_S, PT_F = 4, 9, DTS + 2, DTF + 2
L1_BASE, L1_TOP = 112_576, 1_458_880  # the CB tops' convention: the L1 floor - 32 KB (K2A_DESIGN_v2 addendum 2 (3a))
# the plan's rings: unit top 1,295,296. K2a's 240 / 216 (1,425,856) clash in the D-program's process: L1 buffers are
# allocated at one address in every bank, and the hub's top-16 records (v3, 128 KB) + candidates + the ingest's put
# the L1 floor near 1,397,568 on every core (GI1 mock)
KR = int(os.environ.get("GRES_DK2A_KR", "192"))  # cb_k pages (multiple of 16)
VR = int(os.environ.get("GRES_DK2A_VR", "144"))  # cb_v pages (multiple of 72)
assert KR % 16 == 0 and VR % 72 == 0, (KR, VR)
U_SEMS, H_SEMS = 14, 6
assert RING_PAGES == 72  # d_k2a_common.hpp kRingTiles


def draft_norm_tiles(vectors, layers=D_LAYERS, offset=0):
    """[layers * 2 * 32, 512] bf16 (resident_m5_3 attention.norm_tiles' layout): rows 32 (2 l + k) .. + 32 = layer
    l's q_norm weight / 16 (k = 0: the 1 / sqrt(head_dim) score scale folded in) or k_norm weight (k = 1), replicated
    over the tile's 32 rows"""
    out = torch.zeros(layers * 2 * TILE, 512, dtype=torch.bfloat16)
    scale = 1.0 / (D_HEAD_DIM**0.5)
    assert scale == 2.0**-4  # a power of two: w * scale is exact in bf16
    for l in range(layers):
        L = vectors["layers"][offset + l]
        for k, w in enumerate((L["q_norm"].float() * scale, L["k_norm"].float())):
            out[(2 * l + k) * TILE : (2 * l + k + 1) * TILE, : w.numel()] = w.bfloat16()
    return out


def rope_tables(theta, positions, head_dim=D_HEAD_DIM):
    """tt/dflash.py rope_tables: HF rotate_half cos / sin [positions, head_dim] (float32, then bf16 on device)"""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    f = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    emb = torch.cat([f, f], dim=-1)
    return emb.cos(), emb.sin()


class DK2aAttention:
    def __init__(self, mesh, M, layers, capacity, slots, num_streamers, used, window, eps, theta, vectors,
                 layer_offset=0, rope=None):
        """M rows per user block (2, 4 or 8); capacity: positions (the RoPE tables' rows); slots: the ring's user
        slots; used: the D-program's cores (logical (x, y)); window: the drafter's sliding_window; (v2) rope: an existing
        (cos, sin) pair of row-major [rows, 256] bf16 DRAM tables (one 512 B page per position, the same address on
        every chip) used instead of building this program's own, with capacity = rows (the F_POSITION bound)"""
        assert M in (2, 4, 8), M
        self.mesh, self.M, self.U, self.layers, self.layer_offset = mesh, M, 32 // M, layers, layer_offset
        assert layer_offset + layers <= D_LAYERS
        self.capacity, self.slots, self.window, self.eps = capacity, slots, window, eps
        self.num_streamers = num_streamers
        grid = mesh.compute_with_storage_grid_size()
        free = sorted({(x, y) for x in range(grid.x) for y in range(grid.y)} - set(used), key=lambda c: (c[1], c[0]))
        assert len(free) >= HEADS + UNITS, f"d_k2a needs {HEADS + UNITS} cores, has {len(free)}"
        C = ttnn.CoreCoord
        self.heads = [C(*c) for c in free[:HEADS]]
        self.units = [C(*c) for c in free[HEADS : HEADS + UNITS]]  # kv head g: units [15 g, 15 g + 15)
        self.spare = [C(*c) for c in free[HEADS + UNITS :]]
        self.flat = self.units  # (the hub's error words: the unit cores)
        CR = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])  # noqa: E731
        self.grid_u, self.grid_h = CR(self.units), CR(self.heads)
        self.phys = lambda c: mesh.worker_core_from_logical_core(c)  # noqa: E731
        self.KMAX, self.NP = kmax(window), root_cap(window) * kmax(window) * PT_S
        self.KSC = max(self.KMAX, 8)
        self.CQ = MAX_Q * DTS
        rep = ttnn.ReplicateTensorToMesh(mesh)
        dram = ttnn.DRAM_MEMORY_CONFIG
        # DRAM staging: Q (bf16 tiles: (g * tiles + T) * Dt + d), K and V (BF8: g * Dt + d)
        q_pages = 4 * pack_s(M)[4] * DTS
        self.q_stage = ttnn.from_torch(torch.zeros(1, 1, 32, 32 * q_pages), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=mesh, mesh_mapper=rep, memory_config=dram)
        self.k_stage, self.v_stage = (
            ttnn.from_torch(torch.zeros(1, 1, 32, 32 * 4 * DTS), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                            device=mesh, mesh_mapper=rep, memory_config=dram)
            for _ in range(2)
        )
        # the head cores' norm weights and RoPE tables (row-major [capacity, 256] bf16: one 512 B page per position)
        self.w_t = ttnn.from_torch(draft_norm_tiles(vectors, layers, layer_offset), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=dram)
        if rope is not None:  # (v2) the target engine's sliding tables: theta and head dim are the drafter's
            cos_t, sin_t = rope
            for t in (cos_t, sin_t):
                shape = tuple(t.shape)
                assert t.layout == ttnn.ROW_MAJOR_LAYOUT and t.dtype == ttnn.bfloat16 and shape[-1] == D_HEAD_DIM, \
                    (t.layout, t.dtype, shape)
                assert t.memory_config().buffer_type == ttnn.BufferType.DRAM and not t.memory_config().is_sharded()
                assert math.prod(shape[:-1]) == capacity, (shape, capacity)
            self.cos_t, self.sin_t = cos_t, sin_t
        else:
            cos, sin = rope_tables(theta, capacity)
            self.cos_t, self.sin_t = (
                ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                mesh_mapper=rep, memory_config=dram)
                for t in (cos, sin)
            )
        ttnn.synchronize_device(mesh)
        self.io = [self.q_stage, self.k_stage, self.v_stage, self.w_t, self.cos_t, self.sin_t]

    def xy(self, c):
        p = self.phys(c)
        return p.x | (p.y << 16)

    # ---- descriptors
    def ct(self):
        # 6: group kinds (tables only: unused in drafter mode); 13: the drafter window; 14: the ring layer offset
        return [self.layers, self.M, self.num_streamers, f32_bits(self.eps), f32_bits(1 / 256), f32_bits(1 / 512), 1,
                KR, VR, self.NP, UNITS, PER_HEAD, self.CQ, self.window, self.layer_offset]

    def unit_cbs(self):
        bf16, bf8, u32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint32
        return [  # (index, pages, dtype, page bytes)
            (CB_Q, self.CQ, bf16, BF16), (CB_K, KR, bf8, BF8), (CB_V, VR, bf8, BF8), (CB_MASK, 4, bf16, BF16),
            (CB_ONE, 2, bf16, BF16), (CB_S, NV_S, bf16, BF16), (CB_P, NV_S, bf16, BF16), (CB_M, 2, bf16, BF16),
            (CB_L, 2, bf16, BF16), (CB_OWN, 2 * PT_F, bf16, BF16), (CB_PARTS, self.NP, bf16, BF16),
            (CB_OUT, 32, bf16, BF16), (CB_SC, self.KSC, bf16, BF16), (CB_DIAG, self.KSC, bf16, BF16),
            (CB_CP, PT_F, bf16, BF16), (CB_RA, PT_F, bf16, BF16), (CB_RB, PT_F, bf16, BF16), (CB_SCHED, 1, u32, 4096),
            (CB_NEWK, DTF, bf8, BF8), (CB_NEWV, DTF, bf8, BF8),
        ]

    def head_cbs(self):
        bf16, bf8, f32, u32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint32
        return [
            (HCB_RX, 32, bf16, BF16), (HCB_W, 32, bf16, BF16), (HCB_CS, 48, bf16, BF16), (HCB_ONE, 1, bf16, BF16),
            (HCB_SQ, 16, f32, F32), (HCB_RINV, 1, f32, F32), (HCB_XN, 16, bf16, BF16), (HCB_QO, 16, bf16, BF16),
            (HCB_KR, 16, bf16, BF16), (HCB_VN, 16, bf16, BF16), (HCB_KV8, 16, bf8, BF8), (HCB_OG, 16, bf16, BF16),
            (HCB_FEED, 1, u32, 512),
        ]

    @staticmethod
    def _tot(cbs):
        return sum(-(-p * b // 16) * 16 for _, p, _, b in cbs)

    def l1_bytes(self):
        return {"unit": self._tot(self.unit_cbs()), "unit_top": L1_BASE + self._tot(self.unit_cbs()),
                "head": self._tot(self.head_cbs()), "head_top": L1_BASE + self._tot(self.head_cbs()), "limit": L1_TOP}

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
        return [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.grid_u, initial_value=0) for i in range(U_SEMS)] + [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=self.grid_h, initial_value=0) for i in range(H_SEMS)]

    def kernels(self, kernel, dm, compute_cfg, streamer_cores, positions_t, slots_t, ring, sem_o, sem_rxaddr,
                fid_head=ttnn.MathFidelity.HiFi4, fid_unit=ttnn.MathFidelity.HiFi2):
        """reader / writer / compute descriptors of one chip (the same on every chip); positions_t / slots_t: the
        launch rows (DRAM [1, 32] int32 rows, 128 B pages); ring: d_ingest.DrafterRing"""
        NCRISC, BRISC = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
        NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default
        assert ring.slots == self.slots
        kv = [a for _ in range(self.layers) for a in (ring.k.buffer_address(), ring.v.buffer_address())]
        hxy = [self.xy(c) for c in self.heads]
        uxy = [self.xy(c) for c in self.units]
        # unit cores: common args (d_k2a_unit_reader.cpp; tables, scratch, P, the group words and pools unused)
        head = 15
        ucom = [positions_t.buffer_address(), slots_t.buffer_address(), self.capacity, self.slots, 0, 0, 0]
        o_gmap = head
        o_pool = o_gmap + 8
        o_kv = o_pool + 6
        o_hxy = o_kv + len(kv)
        o_uxy = o_hxy + HEADS
        ucom += [o_gmap, o_pool, o_kv, self.q_stage.buffer_address(), self.k_stage.buffer_address(),
                 self.v_stage.buffer_address(), o_hxy, o_uxy]
        assert len(ucom) == head
        ucom += [0] * 8 + [0] * 6 + kv + hxy + uxy
        # head cores (no full layers: the full RoPE tables 0)
        seam = seam_split(32)
        srects = rect_cover(self.mesh, streamer_cores, seam)
        sxy = [self.xy(c) for c in streamer_cores]
        hhead = 14
        o_rect = hhead
        o_sxy = o_rect + len(srects)
        o_huxy = o_sxy + len(sxy)
        hcom = [positions_t.buffer_address(), slots_t.buffer_address(), self.capacity, self.slots,
                self.cos_t.buffer_address(), self.sin_t.buffer_address(), 0, 0,
                self.w_t.buffer_address(), sem_o, sem_rxaddr, o_rect, o_sxy, o_huxy]
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
            ("d_k2a_unit_reader.cpp", self.grid_u, ur, ucom, dm(NCRISC, NOC1)),
            ("d_k2a_unit_writer.cpp", self.grid_u, uw, ucom, dm(BRISC, NOC0)),
            ("d_k2a_unit_compute.cpp", self.grid_u, uc, [], compute_cfg(fid_unit, False, True)),
            ("d_k2a_head_reader_v2.cpp", self.grid_h, hr, hcom, dm(NCRISC, NOC1)),  # (v3) R4 unicasts
            ("d_k2a_head_writer.cpp", self.grid_h, hw, hwcom, dm(BRISC, NOC0)),
            ("d_k2a_head_compute.cpp", self.grid_h, hc, [], compute_cfg(fid_head, True, True)),
        ):
            kd = kernel(src, grid, ct, rt, cfg)
            if com:
                kd.common_runtime_args = com
            out.append(kd)
        return out
