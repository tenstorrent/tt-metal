# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Attention cores of the resident Gemma engine (step 2: real attention for one user, M <= 8 rows at positions
p .. p + M - 1), see kernels/attn_common.hpp for the schedule.

The engine decodes from the ttnn QB2 path's paged BF8 KV cache ([pages, local kv heads, 128, head_dim] per layer, K
and V apart; sliding layers a bounded ring of 9 pages, positions modulo 1152), its page tables and its decode RoPE
tables, so the ttnn prefill fills the cache and the resident engine continues from it (AttnInputs.from_generator).
AttnInputs.synthetic allocates the same layout without the ttnn model (performance runs).
"""
import math
import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4_31b_qb2.resident_m3_2.common import (
    EPS, LAYERS, N_CHIPS, TILE, f32_bits, is_full, rect_cover, seam_split,
)

A_RX, A_Q, A_PARTS, A_PADDR_L, A_PADDR_G, A_OADDR = range(6)
GROUPS, GROUP_CORES = 4, 8
FULL_WORKERS = 28
DTS, DTF = 8, 16
PAGE, RING_PAGES = 128, 9
BF8, BF16, F32 = 1088, 2048, 4096
KINDS = ("sliding_attention", "full_attention")

# CB indices (attn_common.hpp)
(CB_Q, CB_K, CB_V, CB_WMASK, CB_ONE, CB_S, CB_P, CB_M, CB_L, CB_OWN, CB_PARTS, CB_POUT, CB_OUT, CB_SC, CB_SC2, CB_DIAG,
 CB_POS, CB_Q32, CB_KV32, CB_W, CB_CS, CB_SQ, CB_RINV, CB_XN, CB_KR, CB_VN, CB_K8, CB_V8, CB_TK, CB_TV, CB_TMASK,
 CB_QREADY, CB_RX, CB_OSTAGE, CB_PTAB, CB_CP, CB_RA, CB_RB, CB_K2) = range(39)
# M1L: a full-layer worker walks its position tiles in chunks of this many (L1 constant in context)
FULL_CHUNK = int(os.environ.get("GRES_FULL_CHUNK", "10"))
# M1L: full-only worker groups (a merger + 6 workers each) under 2 super-mergers, on cores the engine leaves free
EXTRA_GROUPS = int(os.environ.get("GRES_FULL_EXTRA", "5"))
# M1L narrow / wide full layers (attn_common.hpp kWideFrom): from this many DRAM position tiles on (32K positions),
# all full-layer workers with K on the writer; below it, step 2's 28 workers with K and V on the reader, which costs
# less at short and mid context (R = 8 at 1K / 8K: -0.3 / -0.5 ms, bench/resident/m1l/ab_*.json)
WIDE_FROM = int(os.environ.get("GRES_FULL_WIDE_FROM", "1000"))
# cb_k / cb_k2 pages (K windows of 16 per position tile): how far K runs ahead of the compute
K_PAGES = int(os.environ.get("GRES_K_PAGES", "64"))
# (L1 knobs, M1L) the mergers' cb_v and cb_k / cb_k2 pages: mergers top R = 8 L1 at 262K. K pages 32 frees 69,632 B
# per merger at no measured cost (bench/resident/m1l/l1knob); a merger-only cb_v of 20 frees 65,280 B but costs up
# to 0.8 ms at 262K (off by default)
MERGER_CV = int(os.environ.get("GRES_MERGER_CV", "0"))
MERGER_K_PAGES = int(os.environ.get("GRES_MERGER_K_PAGES", "32"))
# (M3) cb_k / cb_k2 hold whole K windows of 16 pages: the writer's pre-Q prefetch count is pages // 16 (attn_common.hpp
# kPreK) and no reservation may straddle the CB end
assert K_PAGES % 16 == 0 and K_PAGES >= 16 and MERGER_K_PAGES % 16 == 0 and MERGER_K_PAGES >= 16, (K_PAGES, MERGER_K_PAGES)
# the non-leader attention compute's fidelity (default: the engine's attn_fidelity) and fp32 dest accumulation. M1L:
# bf16 dest by default (as ttnn's decode SDPA): -0.2 .. -0.5 ms at 128K / 262K, and against the unquantized HF layer
# at 128K the same as fp32 dest (layer 59: 0.98579 vs 0.98645; ttnn 0.97434), bench/resident/m1l/tf_hf_*.json
OTHER_FID = os.environ.get("GRES_ATTN_FID", "")
OTHER_FP32 = int(os.environ.get("GRES_ATTN_FP32", "0"))


@dataclass
class AttnInputs:
    """the KV cache and tables the engine decodes from (one user)"""

    kv: list  # per layer (K, V) ttnn tensors [pages, H, 128, D] bf8
    tables: dict  # kind -> page table row of the user (torch int, logical page -> physical page)
    rope: dict  # kind -> (cos, sin) ttnn row-major [positions, D] bf16
    capacity: int  # positions the tables cover

    @staticmethod
    def from_generator(gen, state):
        """the ttnn generator's cache state (models/demos/gemma4_31b_qb2/tt/generator.py CacheState) of user 0"""
        shared = gen.model.shared_setup
        return AttnInputs(
            kv=list(state.kv),
            tables={kind: state.page_tables[kind][0].clone() for kind in state.page_tables},
            rope={kind: (shared[("rope", kind)]["decode_cos"], shared[("rope", kind)]["decode_sin"]) for kind in KINDS},
            capacity=state.capacity,
        )

    @staticmethod
    def synthetic(mesh, capacity, layers=LAYERS, checkpoint=None, random=True):
        """caches in the ttnn layout (random contents, else zeros), identity page tables, the HF RoPE tables"""
        from transformers import AutoConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        from models.demos.gemma4_31b_qb2.resident_m3_2.weights import Checkpoint

        cfg = AutoConfig.from_pretrained(Checkpoint(checkpoint).path, local_files_only=True).text_config
        pages = -(-capacity // PAGE)
        rope, kv = {}, []
        rep = ttnn.ReplicateTensorToMesh(mesh)
        positions = min(cfg.max_position_embeddings, pages * PAGE + 64)
        for kind in KINDS:
            cos, sin = Gemma4TextRotaryEmbedding(cfg)(
                torch.empty(1, dtype=torch.bfloat16), torch.arange(positions).reshape(1, -1), layer_type=kind
            )
            rope[kind] = tuple(
                ttnn.from_torch(t[0].bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for t in (cos, sin)
            )
        for l in range(layers):
            full = is_full(l)
            shape = (pages + 2 if full else RING_PAGES + 2, 1 if full else 4, PAGE, 512 if full else 256)
            kv.append(tuple(
                ttnn.from_torch(torch.randn(shape).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                device=mesh, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if random else
                ttnn.zeros(shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for _ in range(2)
            ))
        tables = {"sliding_attention": torch.arange(pages) % RING_PAGES, "full_attention": torch.arange(pages)}
        return AttnInputs(kv, tables, rope, pages * PAGE)


def norm_tiles(layers, checkpoint=None):
    """[layers * 2 * 32, 512] bf16: rows 32 (2 l + k) .. + 32 = the q (k = 0) / k (k = 1) norm weight of layer l,
    replicated over the tile's 32 rows (tile (2 l + k, d) = weight columns of tile d)"""
    from models.demos.gemma4_31b_qb2.resident_m3_2.weights import Checkpoint

    ck = Checkpoint(checkpoint)
    out = torch.zeros(layers * 2 * TILE, 512, dtype=torch.bfloat16)
    for l in range(layers):
        for k, name in enumerate(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
            w = ck.layer(l, name).bfloat16()
            out[(2 * l + k) * TILE : (2 * l + k + 1) * TILE, : w.numel()] = w
    return out


class Attention:
    """the 32 attention cores of each chip: 4 groups of 8, each a free 4 x 2 (else 2 x 4) block of the worker grid
    (row-major in the block: leader L_g first, merger G_g second)"""

    def __init__(self, mesh, R, M, layers, capacity, used, num_streamers):
        self.mesh, self.R, self.M, self.layers = mesh, R, M, layers
        self.num_streamers = num_streamers  # (VERIFY_STEP2 V5: from the streamer grid, not a constant)
        grid = mesh.compute_with_storage_grid_size()
        free = {(x, y) for x in range(grid.x) for y in range(grid.y)} - set(used)
        groups = []
        for w, h in ((4, 2), (2, 4)):
            for y in range(grid.y - h + 1):
                for x in range(grid.x - w + 1):
                    cells = [(x + dx, y + dy) for dy in range(h) for dx in range(w)]
                    if len(groups) < GROUPS and all(c in free for c in cells):
                        groups.append(cells)
                        free -= set(cells)
        assert len(groups) == GROUPS, f"no room for {GROUPS} attention groups of {GROUP_CORES}"
        self.cores = [[ttnn.CoreCoord(x, y) for x, y in grp] for grp in groups]
        # M1L full-only cores: super-mergers S_0 / S_1, then EXTRA_GROUPS groups (merger G'_h + 6 workers)
        spare = sorted(free, key=lambda c: (c[1], c[0]))
        ng = max(0, min(EXTRA_GROUPS, (len(spare) - 2) // 7))
        take = [ttnn.CoreCoord(x, y) for x, y in spare[: 2 + 7 * ng]] if ng else []
        self.supers = take[:2]
        self.xgroups = [take[2 + 7 * h : 2 + 7 * h + 7] for h in range(ng)]
        self.extra = take
        flat = [c for grp in self.cores for c in grp]
        self.flat = flat + self.extra  # every attention core (the 32 + the full-only ones)
        self.leaders = [grp[0] for grp in self.cores]
        self.mergers = [grp[1] for grp in self.cores] + self.supers + [grp[0] for grp in self.xgroups]
        self.workers = [c for grp in self.cores for c in grp[2:]] + [c for grp in self.xgroups for c in grp[1:]]
        assert R <= 8, "the attention cores take 1 .. 8 rows"
        self.QTs, self.QTf = -(-2 * R // TILE), -(-8 * R // TILE)
        self.CQ = max(self.QTs * DTS, self.QTf * DTF)
        self.PTs, self.PTf = self.QTs * DTS + 2 * self.QTs, self.QTf * DTF + 2 * self.QTf
        # position tiles per chunk: sliding <= ceil(33 / 7) (window 1024 + M rows: one chunk), full FULL_CHUNK (M1L:
        # a full worker merges a running partial over its chunks, so no CB grows with the context)
        self.NVs = 5
        self.NVf = FULL_CHUNK
        # cb_v pages, a multiple of NVs and NVf: also how far the reader runs ahead of the compute (step 2 had 80 at
        # capacity 32K: two sliding layers of V; 2 NVf = 20 throttled the reader and cost R = 1 up to 0.8 ms)
        cv = int(os.environ.get("GRES_CV", "0")) or max(80, 2 * self.NVf)
        unit = math.lcm(self.NVs, self.NVf)
        self.CV = max(unit, cv - cv % unit)
        self.CV_M = max(unit, MERGER_CV - MERGER_CV % unit) if MERGER_CV else self.CV  # (the mergers')
        self.CS_W = max(self.QTs * self.NVs, self.QTf * self.NVf)
        self.CS_L = self.QTf * 2
        # M1L tree: the super-mergers S_0 / S_1 report to L_0 (slots 4, 5) when its parts fit (QTf = 1, R <= 4); at
        # R = 8 a 6-partial L_0 overflows L1, so S_i hangs under G_{2 + i} (slot 6) instead, one merge level deeper
        sup_mode = os.environ.get("GRES_SUPER_PARENT", "auto")
        self.sup_l0 = sup_mode == "l0" or (sup_mode == "auto" and self.QTf == 1)
        ns = len(self.supers)
        self.NP_L = max(7 * self.PTs, (4 + (ns if self.sup_l0 else 0)) * self.PTf)
        self.NP_G = (6 if self.sup_l0 or not ns else 7) * self.PTf
        self.pmax = -(-capacity // PAGE)
        # cb_ptab: positions [0, 32), slots [32, 64), scratch pages [64, 80), the slot's 6 group tables from 80
        self.ptab_words = 80 + 6 * self.pmax
        C = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])  # noqa: E731
        self.grid_all, self.grid_l, self.grid_g = C(self.flat), C(self.leaders), C(self.mergers)
        self.grid_w = C(self.workers)
        self.grid_nl = C(self.mergers + self.workers)
        self.grid_lg = C(self.leaders + self.mergers)
        self.phys = lambda c: mesh.worker_core_from_logical_core(c)  # noqa: E731
        self._tree()

    def _tree(self):
        """the full-layer worker tree: wf order (subtree roots first), per core (parent, slot, box, merger, children)"""
        order, node = [], {}
        key = lambda c: (c.x, c.y)  # noqa: E731
        l0 = self.cores[0][0]
        for grp in self.cores:  # G_g (wf 7g) and its 6 workers
            order += grp[1:]
        half = (len(self.xgroups) + 1) // 2
        sup_children = {0: self.xgroups[:half], 1: self.xgroups[half:]}
        for i, sc in enumerate(self.supers):
            order.append(sc)
            for grp in sup_children[i]:
                order += grp
        wf = {key(c): i for i, c in enumerate(order)}
        A_L, A_G = 3, 4  # A_PADDR_L, A_PADDR_G
        # L_0's children: the 4 group mergers, and S_0 / S_1 (sup_l0) or S_i as G_{2 + i}'s 7th child (slot 6); every
        # subtree root still precedes its subtree in wf order, so the active children are a prefix of the slots
        under_g = {} if self.sup_l0 else {2 + i: sc for i, sc in enumerate(self.supers)}
        root_kids = [grp[1] for grp in self.cores] + (list(self.supers) if self.sup_l0 else [])
        node[key(l0)] = (0, 0, 0, 0, [wf[key(c)] for c in root_kids])
        for g, grp in enumerate(self.cores):
            kids = [wf[key(c)] for c in grp[2:]] + ([wf[key(under_g[g])]] if g in under_g else [])
            node[key(grp[1])] = (self.xy(l0), g, A_L, 1, kids)
            for i, c in enumerate(grp[2:]):
                node[key(c)] = (self.xy(grp[1]), i, A_G, 0, [])
        for i, sc in enumerate(self.supers):
            where = (self.xy(l0), 4 + i, A_L) if self.sup_l0 else (self.xy(self.cores[2 + i][1]), 6, A_G)
            node[key(sc)] = (*where, 1, [wf[key(grp[0])] for grp in sup_children[i]])
            for h, grp in enumerate(sup_children[i]):
                node[key(grp[0])] = (self.xy(sc), h, A_G, 1, [wf[key(c)] for c in grp[1:]])
                for k, c in enumerate(grp[1:]):
                    node[key(c)] = (self.xy(grp[0]), k, A_G, 0, [])
        self.wf, self.node, self.full_workers = wf, node, len(order)
        # the skew: a full worker's partial passes `lev` merges on its way out (its own if it has children, then each
        # ancestor's, L_0's included); with s tiles per level (GRES_FULL_SKEW "s" or "s_R<=4,s_R=8") a core reads
        # s * (mean - lev) tiles more, cumulative over wf order (it sums to 0: the ranges still tile [0, f_n))
        par = {}
        for k_, (_, _, _, _, kids) in node.items():
            for w in kids:
                par[key(order[w])] = k_
        lev = []
        for c in order:
            n_, k_ = int(bool(node[key(c)][4])), key(c)
            while k_ in par:
                n_, k_ = n_ + 1, par[k_]
            lev.append(n_)  # (the last hop is L_0's merge)
        sk = [float(v) for v in os.environ.get("GRES_FULL_SKEW", "6").split(",")]  # (M1L sweep: 6 at R = 1 and 8)
        s = sk[0] if self.QTf == 1 or len(sk) == 1 else sk[1]
        mean = sum(lev) / len(lev)
        cum = [round(s * sum(mean - v for v in lev[:w])) for w in range(len(lev) + 1)]
        assert cum[-1] == 0
        dmax = max(abs(cum[w + 1] - cum[w]) for w in range(len(lev)))
        self.skew = {key(c): (cum[w] & 0xFFFFFFFF, cum[w + 1] & 0xFFFFFFFF) for w, c in enumerate(order)}
        self.skew_min = len(order) * (dmax + 2) if s else 0xFFFFFFFF
        self.levels = lev

    def tree_args(self, c):
        p = self.node.get((c.x, c.y))
        if p is None:
            return [0, 0, 0, 0, 0] + [0] * 7 + [0, 0, 0xFFFFFFFF]
        parent, slot, box, merger, kids = p
        assert len(kids) <= 7
        lo, hi = self.skew.get((c.x, c.y), (0, 0))
        return [parent, slot, box, merger, len(kids)] + kids + [0] * (7 - len(kids)) + [lo, hi, self.skew_min]

    def xy(self, c):
        p = self.phys(c)
        return p.x | (p.y << 16)

    def cbs(self):
        """per role (leaders, mergers, workers) its own descriptors, cb_q first: a CB spanning roles would make the
        allocator stack every role's CBs on every core; cb_q (the Q multicast target) then has the same address on
        every attention core. Addresses other cores need (rx, parts) are the same within a role and published."""
        R = self.R
        tb = 64 * R

        def cb(idx, grid, pages, dt, page):
            fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid, format_descriptors=[fd])

        bf16, bf8, f32, u32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint32
        ptmax = max(self.PTs, self.PTf)
        q32 = self.QTf * DTF
        out = []
        for role, grid in (("leader", self.grid_l), ("merger", self.grid_g), ("worker", self.grid_w)):
            cs = self.CS_L if role == "leader" else self.CS_W
            out += [
                cb(CB_Q, grid, self.CQ, bf16, BF16),
                cb(CB_ONE, grid, 2, bf16, BF16),
                cb(CB_WMASK, grid, 2, bf16, BF16),
                cb(CB_POS, grid, 1, u32, 128),
                cb(CB_M, grid, 2, bf16, BF16),
                cb(CB_L, grid, 2, bf16, BF16),
                cb(CB_OWN, grid, ptmax, bf16, BF16),
                cb(CB_PTAB, grid, 1, u32, self.ptab_words * 4),
                cb(CB_S, grid, cs, bf16, BF16),
                cb(CB_P, grid, cs, bf16, BF16),
            ]
            if role != "leader":
                kp, cv = (MERGER_K_PAGES, self.CV_M) if role == "merger" else (K_PAGES, self.CV)
                out += [cb(CB_K, grid, kp, bf8, BF8), cb(CB_V, grid, cv, bf8, BF8), cb(CB_K2, grid, kp, bf8, BF8)]
                out += [cb(CB_CP, grid, ptmax, bf16, BF16), cb(CB_RA, grid, ptmax, bf16, BF16),
                        cb(CB_RB, grid, ptmax, bf16, BF16)]
                if role == "worker":  # (the chunk merge's scale and diag CBs; leaders and mergers have them below)
                    out += [cb(CB_SC, grid, 8, bf16, BF16), cb(CB_DIAG, grid, 8, bf16, BF16)]
            if role != "worker":
                out += [
                    cb(CB_SC, grid, 8, bf16, BF16),
                    cb(CB_DIAG, grid, 8, bf16, BF16),
                    cb(CB_PARTS, grid, self.NP_L if role == "leader" else self.NP_G, bf16, BF16),
                ]
            if role == "leader":
                out.append(cb(CB_SC2, grid, 8, bf16, BF16))
            if role == "merger":
                out.append(cb(CB_POUT, grid, self.PTf, bf16, BF16))
            if role == "leader":
                out += [
                    cb(CB_OUT, grid, q32, bf16, BF16),
                    cb(CB_Q32, grid, q32, bf16, BF16),
                    cb(CB_KV32, grid, 16, bf16, BF16),
                    cb(CB_W, grid, 32, bf16, BF16),
                    cb(CB_CS, grid, 48, bf16, BF16),
                    cb(CB_SQ, grid, 16, f32, F32),
                    cb(CB_RINV, grid, 2, f32, F32),
                    cb(CB_XN, grid, 16, bf16, BF16),
                    cb(CB_KR, grid, 16, bf16, BF16),
                    cb(CB_VN, grid, 16, bf16, BF16),
                    cb(CB_K8, grid, 16, bf8, BF8),
                    cb(CB_V8, grid, 16, bf8, BF8),
                    cb(CB_TK, grid, 32, bf8, BF8),
                    cb(CB_TV, grid, 32, bf8, BF8),
                    cb(CB_TMASK, grid, 2, bf16, BF16),
                    cb(CB_QREADY, grid, 1, u32, 16),
                    cb(CB_RX, grid, 144, u32, tb),
                    cb(CB_OSTAGE, grid, 128, u32, tb),
                ]
        return out

    def l1_bytes(self):
        """per role: the CB bytes"""
        roles = {"leader": self.leaders[0], "merger": self.mergers[0], "worker": self.workers[0]}
        res = {}
        for name, c in roles.items():
            total = 0
            for d in self.cbs():
                if d.core_ranges.contains(c):
                    total += d.total_size
            res[name] = total
        return res

    def semaphores(self):
        # 0 .. 5 the schedule's, 6 / 7 this launch's table errors (A_ERR, A_ERR2: read by the hub)
        # 8: the reader's tables loaded (M1L: the writer streams the workers' K tiles from them)
        return [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.grid_all, initial_value=0) for i in range(9)]

    def ct(self, leader_kernel, role="worker"):
        cv, kp = (self.CV_M, MERGER_K_PAGES) if role == "merger" else (self.CV, K_PAGES)
        return [
            self.layers, self.R, self.M, self.num_streamers, self.QTs, self.QTf, self.CQ, self.NVs, self.NVf, self.PTs,
            self.PTf,
            cv, self.CS_L if leader_kernel else self.CS_W, f32_bits(EPS), f32_bits(1 / 256), f32_bits(1 / 512),
            int(leader_kernel), self.full_workers, WIDE_FROM, 7 * GROUPS, kp // 16,
        ]

    def kernels(self, kernel, dm, compute_cfg, streamer_cores, serve, rope, w_addr, kv_addrs,
                sem_o, sem_rxaddr, odump, pdump, fid_leader, fid_other):
        """reader / writer / compute descriptors of one chip (the same on every chip)"""
        NCRISC, BRISC = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
        NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default
        roles = ("leader", "merger", "worker")
        rr = {r: ttnn.RuntimeArgs() for r in roles}
        wr = {r: ttnn.RuntimeArgs() for r in roles}
        cr = {r: ttnn.RuntimeArgs() for r in roles}
        mesh = self.mesh
        seam = seam_split(self.M)
        streamer_rects = rect_cover(mesh, streamer_cores, seam)
        nonleaders = self.mergers + self.workers  # (with the full-only cores: L_0's full-layer Q targets)
        l0 = self.leaders[0]
        kv = [a for pair in kv_addrs for a in pair]
        gmap = [0] * 8
        for l, grp in enumerate(serve.layer_group[: self.layers]):
            gmap[l // 8] |= grp << (4 * (l % 8))
        pools = [serve.pool_pages.get(g_, 0) for g_ in range(6)]
        feed = [serve.positions.buffer_address(), serve.slots.buffer_address()]
        rd_head = feed + [serve.tables.buffer_address(), serve.scratch_t.buffer_address(), self.pmax, serve.capacity,
                          serve.max_users]
        members = [(g, j, c) for g, grp in enumerate(self.cores) for j, c in enumerate(grp)]
        # full-only cores: g >= GROUPS (they skip sliding layers), j >= 1 (never a leader); mergers j = 1
        mergers = {(c.x, c.y) for c in self.mergers}
        members += [(GROUPS, 1 if (c.x, c.y) in mergers else 2, c) for c in self.extra]
        for a, (g, j, c) in enumerate(members):
            grp = self.cores[g] if g < GROUPS else self.cores[0]
            if True:
                role = "leader" if j == 0 else ("merger" if (c.x, c.y) in mergers else "worker")
                wf = self.wf.get((c.x, c.y), 0xFFFFFFFF)
                parts = self.NP_L if j == 0 else (self.NP_G if (c.x, c.y) in mergers else 0)
                tree = self.tree_args(c)
                rr[role][c.x][c.y] = [a, g, j, wf] + rd_head + list(rope) + [w_addr] + gmap + pools + kv + tree
                q_s = rect_cover(mesh, grp[1:], seam) if (j == 0 and g < GROUPS) else [0]
                q_f = rect_cover(mesh, nonleaders, seam) if a == 0 else [0]
                q_fn = rect_cover(mesh, [x for grp in self.cores for x in grp[1:]], seam) if a == 0 else [0]  # narrow
                others = rect_cover(mesh, [x for x in self.flat if x != c], seam) if (a == 0 or (g == 0 and j == 1)) else [0]
                srects = streamer_rects if j == 0 else [0]
                k_addrs = [kv[2 * l] for l in range(self.layers)]
                wr[role][c.x][c.y] = [a, g, j, wf] + feed + [serve.capacity, serve.max_users, self.xy(grp[0]), self.xy(grp[1]),
                                self.xy(l0), parts, sem_o, sem_rxaddr, odump, pdump, self.pmax] + gmap + pools + k_addrs \
                    + tree + q_s + q_f + q_fn + others + srects
                cr[role][c.x][c.y] = [a, g, j, wf, parts] + tree
        # per role (leaders, mergers, workers) its own descriptors: the mergers' CV / K pages may differ (L1 knobs)
        grids = {"leader": self.grid_l, "merger": self.grid_g, "worker": self.grid_w}
        other_cfg = compute_cfg(getattr(ttnn.MathFidelity, OTHER_FID) if OTHER_FID else fid_other, bool(OTHER_FP32), True)
        out = []
        for r in roles:
            ct = self.ct(False, r)
            out += [kernel("attn_reader.cpp", grids[r], ct, rr[r], dm(NCRISC, NOC1)),
                    kernel("attn_writer.cpp", grids[r], ct, wr[r], dm(BRISC, NOC0))]
        out.append(kernel("attn_compute.cpp", self.grid_l, self.ct(True), cr["leader"], compute_cfg(fid_leader, True, True)))
        out += [kernel("attn_compute.cpp", grids[r], self.ct(False, r), cr[r], other_cfg) for r in ("merger", "worker")]
        return out
