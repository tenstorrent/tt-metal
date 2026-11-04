# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 rung 3 (M2 + M4-lite): the DFlash2 drafter's layers and context as MkLayer-style megakernel programs
(doc/ttnn/17-mtp-spec-decode/dflash2_intrace.md §8) -- `GLM53_DF2_LAYER=mk` of tt/dflash2_kind.py.

One generic_op per program on every chip (a MeshProgramDescriptor: the programs differ per chip only in the fabric
lanes' arguments and the ctx fc K offset):
  owners   64 cores, owner k = the hidden column tiles {2k, 2k+1}: the glue phases of kernels/spec/df2_mk_{dm0,dm1,c}.cpp
           (RMS with a 64-way partial-sum gather, the rung-2 fused conv, the residual fold, the head norms + RoPE + the
           K / V cache rows, swiglu)
  readers  the DS bank readers (3 per bank, 24 cores; kernels/mk/reader_{w,a,c}.cpp): the layer's matmuls as
           ReaderPhases on the DSLinear weights of tt/dflash2_mm.py (M1), the o / down partials handed to the fabric
  lanes    2 cores: tt/fabric_ar.FabricPhase ("cb" mode, 8 valid rows): the in-program all-reduce (and, with per-chip
           K slices of fc, the all-gather of the ctx fc output)
The block step: A_0 = [RMS -> kp_attn -> CONV -> qkv -> HEADS], the stock paged SDPA (one launch per kv head), then
per layer the merged [B_i + A_{i+1}] = [o -> AR -> FIN -> RMS -> kp_mlp -> CONV -> ug -> SWI -> down -> AR -> FIN ->
RMS -> kp_attn -> CONV -> qkv -> HEADS]; the last program ends with the final norm (hid). ctx = ONE program
[fc (K-split) -> AR -> RMS_AR (hidden_norm) -> 8 x qkv -> 8 x HEADS (k / v at this step's positions)].

The attention layout trick: the paged SDPA decode takes Q as [1, B, heads, hd] with the SAME attention bound for
every head of a batch row; the drafter's block attends with every row to the same positions (<= c' + K), so Q is
handed over as [1, nh_l / kv group (batch = head), 32 (the rows as "heads"), hd]: its input tiles are the owners'
token-row tiles of each head and its output tiles are the o matmul's input column tiles -- no create_heads /
concat_heads, no sharded chain. The K / V caches are per kv head ([blocks, 1, 32, hd]) so that at TP2 (4 kv heads
per chip) each kv group is one SDPA launch (4 + one concat per layer); at TP8 (1 kv head) one launch per layer.

Rows: the block's 32 rows hold K + 1 <= 8 valid rows (0..K); rows 8..31 are garbage that every phase keeps finite and
row-local (the conv's row shift reads the previous row only; the K / V rows they write sit past the attention bound
and are overwritten before any read). The fabric all-reduce therefore carries rows 0..7 only (b = 8: 512 B per
tile); its consumers zero rows 8..31. Rounding points vs the rung-2 op chain: the o / down partials are bf16 BEFORE
the all-reduce (the lanes sum fp32, pack bf16 per round) instead of one fp32 all-reduce rounded once -> PCC-level,
not bitwise (the host emulation `Df2MkTorch` mirrors the device order, incl. the packer's round-half-away)."""

from __future__ import annotations

import os
import struct

import torch

from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_ref import rotate_half

TILE = 32
BF16, FP32 = torch.bfloat16, torch.float32
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")
DF2_MK_REV = 4  # kernels/spec/df2_mk_*.cpp revision (a define: a long-lived process caches binaries by path + defines); 2: the per-phase CB rewinds; 3: CB_KOUT; 4: one SDPA per layer (merged caches, the per-q-head page table)
BROWS = 8  # valid rows carried by the all-reduce (K + 1 <= 8)
FOLD_MODES = ("spec", "off")  # GLM53_DF2_FOLD: the SDPA's Q form (Df2Mk.__init__)
SDPA_MAX_CORES_PER_HEAD = 64  # sdpa_decode's tree reduction: MAX_TREE_REDUCTION_ROUNDS = 6


def sdpa_cores(nkvh: int, cores: int, override=None) -> int:
    """cores per kv head for the folded SDPA: the grid split over the nkvh groups, capped by the op's tree reduction
    (64) -- TP2 (nkvh 4, 130 cores) 32, TP8 (nkvh 1) 64; GLM53_DF2_SDPA_CORES overrides (the A/B)"""
    if override:
        return max(1, int(override))
    return max(1, min(SDPA_MAX_CORES_PER_HEAD, cores // nkvh))


def fold_view(q_heads: torch.Tensor, G: int) -> torch.Tensor:
    """[1, nqh, 32, hd] (one tile row-block per q head) -> the folded [1, nkvh, G 32, hd]: the SAME bytes (tile
    (g, j, c) of the fold = tile (g G + j, c) of the per-head form), so the owner kernels write q tiles at page
    h HT + c in both forms and the o matmul reads the SDPA output by address in both"""
    _, nqh, t, hd = q_heads.shape
    return q_heads.reshape(1, nqh // G, G * t, hd)


N_OWNERS = 64
# mirrors kernels/spec/df2_mk_common.hpp
PH_RMS, PH_RMS_AR, PH_CONV, PH_FIN, PH_HEADS, PH_SWI = range(6)
SEM_G, SEM_CF, SEM_QKV, SEM_AR, SEM_AR2, SEM_UG, SEM_R = range(7)
CB = dict(X=0, W=1, P=2, G=3, OUT=4, LAND=5, REC=6, CF=7, E=8, J=9, B=10, SH=11, ZERO=12, RES=13, QKV=14, HW=15,
          COS=16, SIN=17, INV=18, HN=19, V=20, SCR=21, GU=22, KOUT=23)  # fmt: skip
RT_W, PB_Q = 24, 8
RT_TAP, RT_TAP2, RT_TAP3 = 20, 21, 22  # the per-phase tap fields (df2_mk_common.hpp)
RMS_SIGNAL = 0x10  # RMS flag: signal the readers after the output
HEADS_LAST, HEADS_NOQ = 1 << 8, 1 << 9  # HEADS flag bits: the program's last q|k|v hand-off; skip the q heads (ctx)


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def bf16_rha(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 values (as fp32) with round half away from zero (the Tensix packer's rounding; tt/fabric_ar.py)"""
    b = t.float().contiguous().view(torch.int32)
    return ((b + 0x8000) & -65536).view(torch.float32)


def mesh_shape(tp: int) -> tuple:
    """the program's mesh shape by chip count: the pair 1x2, the 2x4 mesh, one chip (the host test)"""
    return {1: (1, 1), 2: (1, 2), 8: (2, 4)}[tp]


# ====================================================================== the device programs
class Df2Mk:
    """the rung-3 programs on a mesh (TtnnOps): placement (readers / owners / lanes), the fabric phase, the
    persistent DRAM intermediates, the norm weight row tiles, the per-program descriptor builder"""

    def __init__(self, ops, d, S, *, nblk: int, log=print):
        import ttnn
        from models.autoports.zai_org_glm_5_3_flash.tt import fabric_ar as FAR
        from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import rect_placement
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import reader_assignments

        self.ttnn, self.ops, self.mesh, self.log = ttnn, ops, ops.mesh, log
        self.tp = ops.tp
        assert self.tp > 1, "the rung-3 programs all-reduce over a mesh (pair / 2x4)"
        self.d, self.S = d, S
        self.H, self.ncol, self.hd, self.ht = d.H, d.H // TILE, d.hd, d.hd // TILE
        self.nqh, self.nkvh = S.nh_l, S.nkv_l
        self.G = self.nqh // self.nkvh
        self.it = S.I_l // TILE
        self.nts = max(1, -(-self.it // N_OWNERS))  # m tiles per owner (owner k: tiles k nts .. while < it)
        self.nblk = nblk
        self.rpb = 3
        # placement: the DS bank readers (ttnn's bank-adjacent placement), 64 owners + a root around them
        # (mk_dense.rect_placement; the root is unused), 2 lane cores among the free ones
        self.assign = reader_assignments(self.mesh, self.rpb, "MESH_ROW_PLUS_X")
        taken = {(int(c.x), int(c.y)) for c, _, _ in self.assign}
        self.place = rect_placement(self.mesh, taken)
        self.owners = self.place["k"]
        self.owner_crs = merged_ranges([(int(c.x), int(c.y)) for c in self.owners])
        noc = lambda c: self.mesh.worker_core_from_logical_core(c)  # noqa: E731
        self.owner_noc = [(int(w.x), int(w.y)) for w in (noc(c) for c in self.owners)]
        self.reader_noc = [(int(w.x), int(w.y)) for w in (noc(c) for c, _, _ in self.assign)]
        grid = self.mesh.compute_with_storage_grid_size()
        used = taken | {(int(c.x), int(c.y)) for c in self.owners}
        used.add((int(self.place["root"].x), int(self.place["root"].y)))
        free = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x)) if (x, y) not in used]
        assert len(free) >= 2, "no free cores for the fabric lanes"
        self.lanes = free[:2]
        self.fab = FAR.FabricPhase(self.mesh, BROWS, lanes=self.lanes, mode="cb")
        self.fab.probe()
        self.R, self.C = int(self.mesh.shape[0]), int(self.mesh.shape[1])
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        self._rep = rep
        tab = torch.zeros(1, 128, dtype=torch.int32)
        for k, (x, y) in enumerate(self.owner_noc):
            tab[0, 2 * k], tab[0, 2 * k + 1] = x, y
        self.owner_tab = ttnn.from_torch(tab, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)  # fmt: skip
        # persistent DRAM intermediates (bf16 tiles, page = column tile): never allocated inside a capture
        mk = lambda w: ttnn.empty(
            (1, 1, TILE, w), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )  # noqa: E731
        H = d.H
        self.res, self.x, self.cv, self.hid, self.xn = mk(H), mk(H), mk(H), mk(H), mk(H)
        self.cf_a, self.cf_m = mk(4 * d.ng), mk(4 * d.ng)
        self.qkv = mk((self.nqh + 2 * self.nkvh) * d.hd)
        self.qkv_ctx = [mk((self.nqh + 2 * self.nkvh) * d.hd) for _ in range(d.L)]
        self.gu, self.m = mk(2 * S.I_l), mk(S.I_l)
        self.y_fab = mk(H)  # the fabric phases' unused output tensor (the readers hand their rows to the lanes)
        # ONE Q tensor (batch = q head): the SDPA runs every q head of the layer in one launch over the layer's merged
        # per-kv-head cache. Two views of the same tile order (tile h * HT + j = q head h, hd tile j):
        #   GLM53_DF2_FOLD=off   [1, nqh, 32, hd]: one SDPA user per q head through the per-q-head page table pt_g
        #                        (row h = blocks of kv head h // G): every user scans its kv head's cache (G users
        #                        per cache), and the stock paged default is ONE core per user with 32-token chunks
        #                        (sdpa_decode.cpp kDefaultMaxCoresPerHeadBatch = 1): 16 ms per layer at 131K
        #   GLM53_DF2_FOLD=spec  [1, nkvh, G * 32, hd] (the default): the q heads of kv head g folded as the op's
        #                        speculative multi-position tiles (spec_multi_pos_tiles = G: row-tile j of batch g =
        #                        q head g G + j, cur_pos [nqh] = the per-tile bounds, all c' + K) through the
        #                        per-kv-head page table pt_kv -> the cache is read ONCE per kv head, by
        #                        min(64, cores // nkvh) cores per head (the tree reduction), dynamic 128-token chunks
        self.fold = os.environ.get("GLM53_DF2_FOLD", "spec")
        assert self.fold in FOLD_MODES, self.fold
        self.q_shape = (1, self.nkvh, self.G * TILE, d.hd) if self.fold == "spec" else (1, self.nqh, TILE, d.hd)
        self.q_out = ttnn.empty(self.q_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        pt = torch.stack([(h // self.G) * nblk + torch.arange(nblk, dtype=torch.int32) for h in range(self.nqh)])
        self.pt_g = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)  # fmt: skip
        pt_kv = torch.stack([g * nblk + torch.arange(nblk, dtype=torch.int32) for g in range(self.nkvh)])
        self.pt_kv = ttnn.from_torch(pt_kv, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)  # fmt: skip
        self.pt_bytes = nblk * 4
        grid = ops.grid
        self.sdpa_cores = sdpa_cores(self.nkvh, int(grid.x) * int(grid.y), os.environ.get("GLM53_DF2_SDPA_CORES"))
        self.sdpa_cfg = None
        if self.fold == "spec":
            self.sdpa_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=TILE, k_chunk_size=0,
                                                   max_cores_per_head_batch=self.sdpa_cores)  # fmt: skip
        # the norm weights as fp32 row tiles (row 0 = w)
        self.hidden_norm_row, self.norm_row = self.wrow(S.hidden_norm), self.wrow(S.norm)
        self.rows = [dict(in_ln=self.wrow(Ly["in_ln"]), post_ln=self.wrow(Ly["post_ln"]), q_norm=self.wrow(Ly["q_norm"]),
                          k_norm=self.wrow(Ly["k_norm"])) for Ly in S.layers]  # fmt: skip
        self.scr_bytes = 128 + ((self.pt_bytes + 63) & ~63) + 2048 + 512 + 64
        # GLM53_DF2_TAPS=1: every phase's output also lands in a persistent tap tensor (the owner kernels' RT_TAP*
        # fields), the readers' outputs get per-layer buffers, the SDPA output is copied: a unit reads them after a
        # replay and compares them with the host emulation's stages (tests/s17/jobs/dflash2_mk_taps.py)
        self.taps_on = os.environ.get("GLM53_DF2_TAPS", "0") == "1"
        self.taps = {}
        if self.taps_on:
            for i in range(d.L):
                for nm, w in (("x_ln", H), ("cf", 4 * d.ng), ("conv_a", H), ("qkv", (self.nqh + 2 * self.nkvh) * d.hd),
                              ("q", self.nqh * d.hd), ("k_rows", self.nkvh * d.hd), ("v_rows", self.nkvh * d.hd),
                              ("o", H), ("res", H), ("x_post", H), ("cf2", 4 * d.ng), ("conv_m", H), ("gu", 2 * S.I_l),
                              ("m", S.I_l), ("y", H), ("res2", H)):  # fmt: skip
                    self.taps[f"L{i}.{nm}"] = mk(w)
                self.taps[f"L{i}.att"] = ttnn.empty(self.q_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                                    ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
                self.taps[f"ctx.L{i}.qkv"] = self.qkv_ctx[i]
                self.taps[f"ctx.L{i}.k_rows"] = mk(self.nkvh * d.hd)
                self.taps[f"ctx.L{i}.v_rows"] = mk(self.nkvh * d.hd)
            self.taps["ctx.fc"] = mk(H)
            self.taps["ctx.xn"] = self.xn
            self.taps["hid"] = self.hid
        self._tap = lambda name: (self.taps[name].buffer_address() if self.taps_on else 0)
        self.rev_defs = [("DF2_MK_REV", str(DF2_MK_REV))]
        if getattr(ops, "kv_dtype", "bf16") == "bfp8":  # the owners' cache rows re-encoded as bfp8_b (df2_bfp8.hpp)
            self.rev_defs.append(("DF2_KV_BFP8", "1"))
        self.calls = 0
        self.l1 = {}  # program name -> the readers' per-core L1 bytes (the mock rows)
        log(f"dflash2 mk: readers {len(self.assign)} owners {len(self.owners)} lanes {self.lanes} nts {self.nts} "
            f"fab v {self.fab.v} ch {self.fab.ch} pt {self.pt_g.buffer_address():#x}")  # fmt: skip

    # ------------------------------------------------------------ tensors
    def wrow(self, w: torch.Tensor):
        """a norm weight [W] -> the fp32 [1, 1, 32, W] tile row whose row 0 holds w (page c = column tile c)"""
        ttnn = self.ttnn
        t = torch.zeros(TILE, int(w.numel()))
        t[0] = w.float().reshape(-1)
        return ttnn.from_torch(t.reshape(1, 1, TILE, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                               device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=self._rep)  # fmt: skip

    # ------------------------------------------------------------ the owner kernels
    def _owner_parts(self, phases: list, E, J, SH):
        """phases = [(type, flag, [block fields])] -> (kernels, cbs, semaphores) on the owner cores"""
        ttnn, d = self.ttnn, self.d
        nph, nrd = len(phases), len(self.assign)
        ct = [nph, nrd, N_OWNERS, self.nqh, self.nkvh, self.ht, TILE // d.group, d.ng, self.it, self.nts,
              f32_bits(d.eps), f32_bits(1.0 / d.H), f32_bits(1.0 / d.hd), self.nblk, self.fab.v, self.fab.ch]  # fmt: skip
        assert len(ct) == 16
        for t, f, _ in phases:
            ct += [int(t), int(f)]
        blocks = []
        for _, _, fields in phases:
            assert len(fields) <= RT_W, (len(fields), RT_W)
            blocks += [int(v) for v in fields] + [0] * (RT_W - len(fields))
        lv = self.fab.lane_noc()
        head0 = [E.buffer_address(), J.buffer_address(), SH.buffer_address(), lv[0].x, lv[0].y, lv[1].x, lv[1].y,
                 self.fab.addr["res"]]  # fmt: skip
        rd_flat = [v for xy in self.reader_noc for v in xy]
        rt0, rt1, rtc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k, c in enumerate(self.owners):
            rt0[c.x][c.y] = [k, 2 * k] + head0 + rd_flat + blocks
            rt1[c.x][c.y] = [k, 2 * k, self.owner_tab.buffer_address()] + blocks
            rtc[c.x][c.y] = [k]
        src = ttnn.KernelDescriptor.SourceType.FILE_PATH
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (CB["W"], CB["G"], CB["HW"], CB["INV"]):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        kd = lambda name, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=os.path.join(KDIR, "spec", f"df2_mk_{name}.cpp"), source_type=src, core_ranges=self.owner_crs,
            compile_time_args=ct, runtime_args=rt, defines=list(self.rev_defs), config=cfg)  # fmt: skip
        kernels = [
            kd("dm0", rt0, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            kd("dm1", rt1, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
            kd("c", rtc, cc),
        ]
        bf, f32, u32 = ttnn.bfloat16, ttnn.float32, ttnn.uint32
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(total_size=page * n, core_ranges=self.owner_crs, format_descriptors=[  # noqa: E731
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])  # fmt: skip
        ht, nts = self.ht, self.nts
        cbs = [cb(CB["X"], bf, 2048, 2), cb(CB["W"], f32, 4096, 2), cb(CB["P"], f32, 4096, 1), cb(CB["G"], f32, 4096, 2),
               cb(CB["OUT"], bf, 2048, max(4, ht, nts)), cb(CB["LAND"], u32, N_OWNERS * 128, 1), cb(CB["REC"], u32, 128, 1),
               cb(CB["CF"], bf, 2048, 4), cb(CB["E"], bf, 2048, 4), cb(CB["J"], bf, 2048, 2), cb(CB["B"], bf, 2048, 2),
               cb(CB["SH"], bf, 2048, 1), cb(CB["ZERO"], bf, 2048, 1), cb(CB["RES"], bf, 2048, 2), cb(CB["QKV"], bf, 2048, ht),
               cb(CB["HW"], f32, 4096, ht), cb(CB["COS"], bf, 2048, ht), cb(CB["SIN"], bf, 2048, ht), cb(CB["INV"], f32, 4096, 1),
               cb(CB["HN"], bf, 2048, ht), cb(CB["V"], bf, 2048, ht), cb(CB["SCR"], u32, self.scr_bytes, 1),
               cb(CB["GU"], bf, 2048, 2 * nts), cb(CB["KOUT"], bf, 2048, ht)]  # fmt: skip
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.owner_crs, initial_value=0) for i in range(SEM_UG + 1)]
        return kernels, cbs, sems

    # ------------------------------------------------------------ the programs
    def _program(self, name: str, phases: list, mm: list, fab_iters: int, io: list, E, J, SH, koff_tiles: int = 0):
        """phases: the owner phases; mm: [(unit, x, y, wait, signal, fab)] in reader order (unit = a DsMM; fab = the
        partial goes to the lanes); fab_iters: the all-reduces of the program; koff_tiles: the per-chip K offset of
        phase 0's activation (the ctx fc K-split: chip r reads K tiles [r koff, ..)); io: the tensors touched"""
        ttnn = self.ttnn
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MMPhase, ReaderPhases

        phs = [MMPhase(u.ds, kb=u.kb, wait=w, signal=s) for u, _, _, w, s, _ in mm]
        rd = ReaderPhases(self.mesh, phs, readers_per_bank=self.rpb)
        self.l1[name] = rd.l1_bytes()
        xs, ys = [x for _, x, _, _, _, _ in mm], [y for _, _, y, _, _, _ in mm]
        raw, rcbs = rd.parts_raw(xs, ys, consumers=self.owner_noc)
        fab_mask = sum(1 << i for i, (_, _, _, _, _, f) in enumerate(mm) if f)
        assert bin(fab_mask).count("1") == fab_iters, (fab_mask, fab_iters)
        ok, oc, osem = self._owner_parts(phases, E, J, SH)
        w_name, w_ct, w_rt, w_defs, w_cfg = raw["w"]
        a_name, a_ct, a_rt, a_defs, a_cfg = raw["a"]
        c_name, c_ct, c_rt, c_defs, c_cfg = raw["c"]
        a_defs = list(a_defs) + self.rev_defs
        if fab_mask:
            n_rt = len(next(iter(a_rt.values())))
            a_defs += [("MK_FAB_MASK", str(fab_mask)), ("MK_FAB_BROWS", str(self.fab.brows)), ("MK_FAB_RT", str(n_rt))]
            a_rt = {core: list(v) + self.fab.producer_args() for core, v in a_rt.items()}
        if koff_tiles:
            a_defs.append(("MK_KSPLIT", "1"))
        rsems = [ttnn.SemaphoreDescriptor(id=SEM_R, core_ranges=rd.cores, initial_value=0)]
        sems = osem + rsems
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                chip = r * self.C + c
                art = a_rt
                if koff_tiles:  # phase 0's n0 -> (koff << 16) | n0 on this chip (kernels/mk/reader_a.cpp MK_KSPLIT)
                    art = {}
                    for core, v in a_rt.items():
                        v = list(v)
                        if v[2] != 0xFFFFFFFF:
                            v[2] = ((chip * koff_tiles) << 16) | v[2]
                        art[core] = v
                kernels = ok + [rd.kernel(w_name, w_ct, w_rt, list(w_defs) + self.rev_defs, w_cfg),
                                rd.kernel(a_name, a_ct, art, a_defs, a_cfg),
                                rd.kernel(c_name, c_ct, c_rt, list(c_defs) + self.rev_defs, c_cfg)]  # fmt: skip
                cbs = oc + rcbs
                if fab_iters:
                    lk, lcbs = self.fab.lane_parts(r, c, fab_iters, len(self.assign), self.owner_noc, cons_sem=SEM_AR,
                                                   cons_sem2=SEM_AR2)  # fmt: skip
                    prog = ttnn.ProgramDescriptor(kernels=kernels + lk, semaphores=sems, cbs=cbs + lcbs)
                    self.fab.add_connections(prog, r, c, send_kernel_index=len(kernels))
                else:
                    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog  # (rt_guard: the lane args differ per chip by design)
        io = list(io) + [u.ds.weight for u, _, _, _, _, _ in mm] + xs + ys + [self.owner_tab, E, J, SH]
        if fab_iters:
            io += self.fab.io()
        seen, uniq = set(), []
        for t in io:
            if id(t) not in seen:
                seen.add(id(t))
                uniq.append(t)
        self.ops._tick(1, f"mk_{name}")
        ttnn.generic_op(uniq, mpd)
        self.calls += 1

    # phase builders
    def _ph_rms(self, src, w, out, signal: bool, ar: int | None = None, tap: str = "", tap2: str = ""):
        a = lambda t: t.buffer_address()  # noqa: E731
        flag = (RMS_SIGNAL if signal else 0) | (0 if ar is None else ar)
        f = [a(src), a(w), a(out)] + [0] * (RT_TAP - 3) + [self._tap(tap) if tap else 0, self._tap(tap2) if tap2 else 0]
        return (PH_RMS if ar is None else PH_RMS_AR, flag, f)

    def _ph_conv(self, x, cf, out, base4, side: int, tap: str = ""):
        a = lambda t: t.buffer_address()  # noqa: E731
        return (PH_CONV, side, [a(x), a(cf), a(out), a(base4)] + [0] * (RT_TAP - 4) + [self._tap(tap) if tap else 0])

    def _ph_fin(self, res_in, cf, res_out, base4, ordinal: int, tap: str = "", tap2: str = ""):
        a = lambda t: t.buffer_address()  # noqa: E731
        f = [a(res_in), a(cf), a(res_out), a(base4)] + [0] * (RT_TAP - 4)
        f += [self._tap(tap) if tap else 0, self._tap(tap2) if tap2 else 0]
        return (PH_FIN, (1 << 4) | ordinal, f)

    def _ph_heads(self, qkv, rows, cos, sin, pos, Kc, Vc, i: int, last: bool, noq: bool, tap: str = ""):
        a = lambda t: t.buffer_address()  # noqa: E731
        flag = i | (HEADS_LAST if last else 0) | (HEADS_NOQ if noq else 0)
        fields = [a(qkv), a(rows["q_norm"]), a(rows["k_norm"]), a(cos), a(sin), a(pos), a(self.pt_g), self.pt_bytes]
        fields += [a(self.q_out)] * self.nkvh + [a(Kc[0])] * self.nkvh + [a(Vc[0])] * self.nkvh  # (one tensor each)
        fields += [0] * (RT_TAP - len(fields))
        if tap and self.taps_on:  # L{i} / ctx.L{i}: q (none in the ctx), k_rows, v_rows
            fields += [self._tap(tap + ".q") if not noq else 0, self._tap(tap + ".k_rows"), self._tap(tap + ".v_rows")]
        return (PH_HEADS, flag, fields)

    def _ph_swi(self, gu, m):
        return (PH_SWI, 0, [gu.buffer_address(), m.buffer_address()])

    def _rd(self, i: int, nm: str):
        """a reader phase's output buffer: the shared one, or the layer's tap in taps mode (cf, cf2, qkv, gu, m)"""
        if self.taps_on:
            return self.taps[f"L{i}.{nm}"]
        return dict(cf=self.cf_a, cf2=self.cf_m, qkv=self.qkv, gu=self.gu, m=self.m)[nm]

    def _mm(self, unit, x, y, wait=None, signal=None, fab=False):
        return (unit, x, y, wait, signal, fab)

    def layer_a0(self, L, res_src, cos, sin, pos, KC):
        """A_0: [RMS(in_ln) -> kp_attn -> CONV -> qkv -> HEADS] on res_src (the embedding rows)"""
        E, J, SH = KC["E_t"], KC["J_t"], KC["Sh_t"]
        rows = self.rows[0]
        cf, qkv = self._rd(0, "cf"), self._rd(0, "qkv")
        phases = [self._ph_rms(res_src, rows["in_ln"], self.x, True, tap="L0.x_ln"),
                  self._ph_conv(self.x, cf, self.cv, L["base4_attn"], 0, tap="L0.conv_a"),
                  self._ph_heads(qkv, rows, cos, sin, pos, L["Kc"], L["Vc"], 0, True, False, tap="L0")]  # fmt: skip
        mm = [self._mm(L["kp_attn"], self.x, cf, (SEM_R, N_OWNERS), SEM_CF),
              self._mm(L["qkv"], self.cv, qkv, (SEM_R, N_OWNERS), SEM_QKV)]  # fmt: skip
        io = [res_src, rows["in_ln"], rows["q_norm"], rows["k_norm"], cos, sin, pos, self.pt_g, L["base4_attn"]]
        io += [self.q_out] + list(L["Kc"]) + list(L["Vc"]) + [self.x, cf, self.cv, qkv] + list(self.taps.values())
        self._program("a0", phases, mm, 0, io, E, J, SH)

    def layer_ba(self, i: int, L, Ln, att, res_in, cos, sin, pos, KC):
        """[B_i + A_{i+1}] (Ln = the next layer, or None: the last layer + the final norm into hid)"""
        E, J, SH = KC["E_t"], KC["J_t"], KC["Sh_t"]
        rows = self.rows[i]
        cf, cf2, gu, m = self._rd(i, "cf"), self._rd(i, "cf2"), self._rd(i, "gu"), self._rd(i, "m")
        if self.taps_on:  # the SDPA output (transient) kept
            self.ttnn.copy(att, self.taps[f"L{i}.att"])
            self.ops._tick(1, "tap_copy")
        phases = [self._ph_fin(res_in, cf, self.res, L["base4_attn"], 0, tap=f"L{i}.res", tap2=f"L{i}.o"),
                  self._ph_rms(self.res, rows["post_ln"], self.x, True, tap=f"L{i}.x_post"),
                  self._ph_conv(self.x, cf2, self.cv, L["base4_mlp"], 0, tap=f"L{i}.conv_m"),
                  self._ph_swi(gu, m),
                  self._ph_fin(self.res, cf2, self.res, L["base4_mlp"], 1, tap=f"L{i}.res2", tap2=f"L{i}.y")]  # fmt: skip
        mm = [self._mm(L["o"], att, self.y_fab, None, None, True),
              self._mm(L["kp_mlp"], self.x, cf2, (SEM_R, N_OWNERS), SEM_CF),
              self._mm(L["ug"], self.cv, gu, (SEM_R, N_OWNERS), SEM_UG),
              self._mm(L["down"], m, self.y_fab, (SEM_R, N_OWNERS), None, True)]  # fmt: skip
        io = [res_in, att, rows["post_ln"], L["base4_attn"], L["base4_mlp"], self.res, self.x, cf, cf2, self.cv, gu, m,
              self.y_fab] + list(self.taps.values())  # fmt: skip
        if Ln is not None:
            rn = self.rows[i + 1]
            cfn, qkvn = self._rd(i + 1, "cf"), self._rd(i + 1, "qkv")
            phases += [self._ph_rms(self.res, rn["in_ln"], self.x, True, tap=f"L{i + 1}.x_ln"),
                       self._ph_conv(self.x, cfn, self.cv, Ln["base4_attn"], 0, tap=f"L{i + 1}.conv_a"),
                       self._ph_heads(qkvn, rn, cos, sin, pos, Ln["Kc"], Ln["Vc"], 0, True, False, tap=f"L{i + 1}")]  # fmt: skip
            mm += [self._mm(Ln["kp_attn"], self.x, cfn, (SEM_R, N_OWNERS), SEM_CF),
                   self._mm(Ln["qkv"], self.cv, qkvn, (SEM_R, N_OWNERS), SEM_QKV)]  # fmt: skip
            io += [rn["in_ln"], rn["q_norm"], rn["k_norm"], cos, sin, pos, self.pt_g, Ln["base4_attn"], qkvn, cfn]
            io += [self.q_out] + list(Ln["Kc"]) + list(Ln["Vc"])
            name = "ba"
        else:
            phases.append(self._ph_rms(self.res, self.norm_row, self.hid, False))
            io += [self.norm_row, self.hid]
            name = "b_last"
        self._program(name, phases, mm, 2, io, E, J, SH)

    def ctx(self, x_taps, fc, Ls, cos, sin, pos, KC, koff_tiles: int):
        """[fc (K-split, -> AR) -> RMS_AR (hidden_norm) -> 8 x qkv -> 8 x HEADS (k / v at this step's rows)]"""
        E, J, SH = KC["E_t"], KC["J_t"], KC["Sh_t"]
        nL = len(Ls)
        phases = [self._ph_rms(self.y_fab, self.hidden_norm_row, self.xn, True, ar=0, tap2="ctx.fc")]
        phases += [self._ph_heads(self.qkv_ctx[i], self.rows[i], cos, sin, pos, L["Kc"], L["Vc"], i, i == nL - 1, True,
                                  tap=f"ctx.L{i}") for i, L in enumerate(Ls)]  # fmt: skip
        mm = [self._mm(fc, x_taps, self.y_fab, None, None, True)]
        mm += [self._mm(L["qkv"], self.xn, self.qkv_ctx[i], (SEM_R, N_OWNERS) if i == 0 else None, SEM_QKV)
               for i, L in enumerate(Ls)]  # fmt: skip
        io = [x_taps, self.hidden_norm_row, self.xn, cos, sin, pos, self.pt_g, self.y_fab] + self.qkv_ctx + [self.q_out]
        io += list(self.taps.values())
        for L in Ls:
            io += list(L["Kc"]) + list(L["Vc"])
        for r in self.rows:
            io += [r["q_norm"], r["k_norm"]]
        self._program("ctx", phases, mm, 1, io, E, J, SH, koff_tiles=koff_tiles)

    def attention(self, L, cur, scale: float):
        """ONE stock paged SDPA decode launch per layer over the layer's merged [nkvh blocks, 1, 32, hd] cache, the bound
        page cur [1, nqh] (every entry c' + K) -> the o matmul's input (the q-head tile order; no concat; TP2 and TP8
        the same form). fold=spec: Q [1, nkvh, G 32, hd] as G multi-position tiles per kv head (the cache read once
        per head, sdpa_cores cores per head); fold=off: Q [1, nqh, 32, hd], one user per q head (see __init__)"""
        ttnn, ops = self.ttnn, self.ops
        ops._tick(1, "sdpa")
        if self.fold == "spec":
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                self.q_out, L["Kc"][0], L["Vc"][0], page_table_tensor=self.pt_kv, cur_pos_tensor=cur, scale=scale,
                compute_kernel_config=ops.hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=self.sdpa_cfg,
                spec_multi_pos_tiles=self.G)  # fmt: skip
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            self.q_out, L["Kc"][0], L["Vc"][0], page_table_tensor=self.pt_g, cur_pos_tensor=cur, scale=scale,
            compute_kernel_config=ops.hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip


# ====================================================================== the host emulation (TorchOps)
class Df2MkTorch:
    """the same programs on the torch backend: per-chip torch tensors, the phases' arithmetic in the device order
    (one bf16 rounding per RMS / conv / RoPE output, bf16 partials before the all-reduce, the fabric's per-round
    round-half-away packing, rows >= 8 of an all-reduced tile zero)"""

    def __init__(self, ops, d, S, *, nblk: int, log=print):
        self.ops, self.d, self.S, self.nblk = ops, d, S, nblk
        self.tp = ops.tp
        self.R, self.C = mesh_shape(self.tp)
        self.H, self.hd = d.H, d.hd
        self.nqh, self.nkvh = S.nh_l, S.nkv_l
        self.G = self.nqh // self.nkvh
        z = lambda w: ops.zeros((1, 1, TILE, w))  # noqa: E731
        self.res, self.x, self.cv, self.hid, self.xn = z(d.H), z(d.H), z(d.H), z(d.H), z(d.H)
        self.q_out = [ops.zeros((1, self.G, TILE, d.hd)) for _ in range(self.nkvh)]
        self.hidden_norm_row, self.norm_row = S.hidden_norm.float(), S.norm.float()
        self.rows = [dict(in_ln=Ly["in_ln"].float(), post_ln=Ly["post_ln"].float(), q_norm=Ly["q_norm"].float(),
                          k_norm=Ly["k_norm"].float()) for Ly in S.layers]  # fmt: skip
        self.calls = 0
        self.l1 = {}
        self.probe = None  # a dict: stage name -> chip 0's tensor (clone), the kind's stage names where they coincide

    def _p(self, name, t):
        if self.probe is not None:
            self.probe[name] = t.detach().clone()
        return t

    # the phases (per chip tensors: [32, W] float views)
    @staticmethod
    def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.reshape(1, -1)).to(BF16)

    def _conv(self, x, cf, base4, side):
        from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_kind import conv_fused_host

        return conv_fused_host(x.reshape(TILE, -1), cf.reshape(TILE, -1), base4.reshape(TILE, -1), side, self.KC)

    def _allreduce(self, parts: list) -> list:
        """bf16 partials per chip (row-major chip order over the R x C mesh) -> the reduced bf16 tiles on every chip
        (rows >= BROWS zero): round 1 over the pair axis, round 2 over the row axis, the packer's rounding per round"""
        R, C = self.R, self.C
        xs = [p.float().reshape(TILE, -1) for p in parts]
        xs = [x.clone() for x in xs]
        for x in xs:
            x[BROWS:] = 0
        grid = [[xs[r * C + c] for c in range(C)] for r in range(R)]
        r1 = []
        for c in range(C):
            s = grid[0][c]
            for r in range(1, R):
                s = s + grid[r][c]
            r1.append(bf16_rha(s) if R > 1 else s)
        acc = None
        for q in range(0, C, 2):
            s = r1[q] + r1[q + 1] if C > 1 else r1[q]
            acc = s if acc is None else acc + s
        out = bf16_rha(acc).to(BF16) if C > 1 or R > 1 else acc.to(BF16)
        return [out.clone() for _ in range(self.tp)]

    def _mm(self, unit, x):
        return unit.raw([t.reshape(1, 1, TILE, -1) for t in x])

    def _heads(self, qkv, rows, cos, sin, pos, Kc, Vc, noq: bool, prefix: str = ""):
        """per chip: the q / k / v heads of qkv [32, (nqh + 2 nkvh) hd]: q -> q_out (norm + RoPE), k -> Kc (norm +
        RoPE) rows pos[t], v -> Vc rows"""
        d, hd = self.d, self.hd
        qt = torch.zeros(TILE, self.nqh * hd, dtype=BF16)
        kt = torch.zeros(TILE, self.nkvh * hd, dtype=BF16)
        vt = torch.zeros(TILE, self.nkvh * hd, dtype=BF16)
        for r in range(self.tp):
            X = qkv[r].reshape(TILE, -1)
            p = pos[r].reshape(-1).long()
            cs, sn = cos[r].reshape(TILE, hd).float(), sin[r].reshape(TILE, hd).float()
            for h in range(self.nqh + 2 * self.nkvh):
                x = X[:, h * hd : (h + 1) * hd]
                if h < self.nqh:
                    if noq:
                        continue
                    y = self._rms(x, rows["q_norm"], d.eps).float()
                    y = (y * cs + rotate_half(y) * sn).to(BF16)
                    self.q_out[h // self.G][r][0, h % self.G] = y
                    if r == 0:
                        qt[:, h * hd : (h + 1) * hd] = y
                    continue
                isk = h < self.nqh + self.nkvh
                g = h - self.nqh if isk else h - self.nqh - self.nkvh
                if isk:
                    y = self._rms(x, rows["k_norm"], d.eps).float()
                    y = (y * cs + rotate_half(y) * sn).to(BF16)
                    cache = Kc[g][r]
                else:
                    y, cache = x, Vc[g][r]
                if r == 0:
                    (kt if isk else vt)[:, g * hd : (g + 1) * hd] = y
                for t in range(TILE):
                    pt = int(p[t])
                    if 0 <= pt < cache.shape[0] * TILE:
                        cache[pt // TILE, 0, pt % TILE] = y[t]
        if prefix:
            if not noq:
                self._p(prefix + ".q", qt)
            self._p(prefix + ".k_rows", kt)
            self._p(prefix + ".v_rows", vt)

    def layer_a0(self, L, res_src, cos, sin, pos, KC):
        self.KC = KC
        rows = self.rows[0]
        for r in range(self.tp):
            self.x[r].copy_(self._rms(res_src[r].reshape(TILE, -1), rows["in_ln"], self.d.eps).reshape(self.x[r].shape))
        self._p("L0.x_ln", self.x[0])
        cf = self._mm(L["kp_attn"], self.x)
        self._p("L0.cf", cf[0])
        for r in range(self.tp):
            self.cv[r].copy_(self._conv(self.x[r], cf[r], L["base4_attn"][r], 0).reshape(self.cv[r].shape))
        self._p("L0.conv_a", self.cv[0])
        qkv = self._mm(L["qkv"], self.cv)
        self._p("L0.qkv", qkv[0])
        self._heads(qkv, rows, cos, sin, pos, L["Kc"], L["Vc"], False, prefix="L0")
        self.cf_a_t = cf  # the attention conv's cf: the FIN of [B_0 + A_1] reads it (the device keeps it in cf_a)
        self.ops._tick(1, "mk_a0")
        self.calls += 1

    def layer_ba(self, i, L, Ln, att, res_in, cos, sin, pos, KC):
        d = self.d
        rows = self.rows[i]
        # att [1, nqh, 32, hd] per chip -> the o input [32, nqh hd]: on the device the SDPA output's tile pages (h, k)
        # ARE the column tiles of the [32, nqh hd] activation (the layout trick); in torch that is a permute
        o_in = [t[0].permute(1, 0, 2).reshape(1, 1, TILE, -1).contiguous() for t in att]
        self._p(f"L{i}.att", att[0])
        o = self._allreduce(self._mm(L["o"], o_in))
        self._p(f"L{i}.o", o[0])
        for r in range(self.tp):
            res = (
                res_in[r].float().reshape(TILE, -1) + self._conv(o[r], self.cf_a_t[r], L["base4_attn"][r], 1).float()
            ).to(BF16)
            self.res[r].copy_(res.reshape(self.res[r].shape))
            self.x[r].copy_(self._rms(res, rows["post_ln"], d.eps).reshape(self.x[r].shape))
        self._p(f"L{i}.res", self.res[0])
        self._p(f"L{i}.x_post", self.x[0])
        cf2 = self._mm(L["kp_mlp"], self.x)
        self._p(f"L{i}.cf2", cf2[0])
        for r in range(self.tp):
            self.cv[r].copy_(self._conv(self.x[r], cf2[r], L["base4_mlp"][r], 0).reshape(self.cv[r].shape))
        gu = self._mm(L["ug"], self.cv)
        self._p(f"L{i}.gu", gu[0])
        m = []
        for r in range(self.tp):
            g = gu[r].reshape(TILE, -1)
            half = g.shape[-1] // 2
            m.append(
                (g[:, :half].float() * torch.nn.functional.silu(g[:, half:].float())).to(BF16).reshape(1, 1, TILE, -1)
            )
        self._p(f"L{i}.conv_m", self.cv[0])
        self._p(f"L{i}.m", m[0])
        y = self._allreduce(self._mm(L["down"], m))
        self._p(f"L{i}.y", y[0])
        for r in range(self.tp):
            res = (self.res[r].float().reshape(TILE, -1) + self._conv(y[r], cf2[r], L["base4_mlp"][r], 1).float()).to(
                BF16
            )
            self.res[r].copy_(res.reshape(self.res[r].shape))
        self._p(f"L{i}.res2", self.res[0])
        if Ln is not None:
            rn = self.rows[i + 1]
            for r in range(self.tp):
                self.x[r].copy_(self._rms(self.res[r].reshape(TILE, -1), rn["in_ln"], d.eps).reshape(self.x[r].shape))
            self._p(f"L{i + 1}.x_ln", self.x[0])
            cf = self._mm(Ln["kp_attn"], self.x)
            self._p(f"L{i + 1}.cf", cf[0])
            self.cf_a_t = cf
            for r in range(self.tp):
                self.cv[r].copy_(self._conv(self.x[r], cf[r], Ln["base4_attn"][r], 0).reshape(self.cv[r].shape))
            self._p(f"L{i + 1}.conv_a", self.cv[0])
            qkv = self._mm(Ln["qkv"], self.cv)
            self._p(f"L{i + 1}.qkv", qkv[0])
            self._heads(qkv, rn, cos, sin, pos, Ln["Kc"], Ln["Vc"], False, prefix=f"L{i + 1}")
            self.ops._tick(1, "mk_ba")
        else:
            for r in range(self.tp):
                self.hid[r].copy_(
                    self._rms(self.res[r].reshape(TILE, -1), self.norm_row, d.eps).reshape(self.hid[r].shape)
                )
            self._p("hid", self.hid[0])
            self.ops._tick(1, "mk_b_last")
        self.calls += 1

    def ctx(self, x_taps, fc, Ls, cos, sin, pos, KC, koff_tiles: int):
        self.KC = KC
        d = self.d
        kt = koff_tiles * TILE
        parts = [
            (x_taps[r].reshape(TILE, -1)[:, r * kt : (r + 1) * kt].float() @ fc.W[r]).to(BF16) for r in range(self.tp)
        ]
        y = self._allreduce(parts)
        self._p("ctx.fc", y[0])
        for r in range(self.tp):
            self.xn[r].copy_(self._rms(y[r], self.hidden_norm_row, d.eps).reshape(self.xn[r].shape))
        self._p("ctx.xn", self.xn[0])
        for i, L in enumerate(Ls):
            qkv = self._mm(L["qkv"], self.xn)
            self._p(f"ctx.L{i}.qkv", qkv[0])
            self._heads(qkv, self.rows[i], cos, sin, pos, L["Kc"], L["Vc"], True, prefix=f"ctx.L{i}")
        self.ops._tick(1, "mk_ctx")
        self.calls += 1

    def attention(self, L, cur, scale: float):
        """the SDPA over the per-kv-head caches: att [1, nqh, 32, hd] per chip (rows attend positions <= cur)"""
        out = []
        for r in range(self.tp):
            n = int(cur[r].reshape(-1)[0]) + 1
            a = torch.zeros(1, self.nqh, TILE, self.hd)
            for g in range(self.nkvh):
                K = L["Kc"][g][r].float().reshape(-1, self.hd)[:n]  # [positions, hd]
                V = L["Vc"][g][r].float().reshape(-1, self.hd)[:n]
                q = self.q_out[g][r].float()  # [1, G, 32, hd]
                s = torch.einsum("gtd,sd->gts", q[0], K) * scale
                a[0, g * self.G : (g + 1) * self.G] = torch.einsum("gts,sd->gtd", torch.softmax(s, -1), V)
            out.append(a.to(BF16))
        self.ops._tick(1, "sdpa")
        return out


# ====================================================================== the owner CBs' host model (the 2026-10-23 01:02Z watcher assert)
CB_PAGES_FIXED = dict(X=2, W=2, P=1, G=2, LAND=1, REC=1, CF=4, E=4, J=2, B=2, SH=1, ZERO=1, RES=2, INV=1, SCR=1)


def cb_pages(ht: int, nts: int) -> dict:
    """the owner CBs' page counts as Df2Mk._owner_parts allocates them"""
    return dict(CB_PAGES_FIXED, OUT=max(4, ht, nts), QKV=ht, HW=ht, COS=ht, SIN=ht, HN=ht, V=ht, GU=2 * nts, KOUT=ht)


def cb_ops(phases: list, k: int, *, nqh: int, nkvh: int, ht: int, nts: int, it: int) -> dict:
    """the owner kernels' CB traffic per RISC per phase for owner k: {risc: [[(cb, op, pages), ...] per phase]}, op =
    "push" (reserve + push by a producer) / "pop" (wait + pop by a consumer), in the kernels' order (kernels/spec/
    df2_mk_{dm0,dm1,c}.cpp); phases = [(type, flag)]"""
    nh_all = nqh + 2 * nkvh
    b, n, c = [], [], []
    for t, flag in phases:
        pb, pn, pc = [], [], []
        if t in (PH_RMS, PH_RMS_AR):
            pb += [("X", "push", 2), ("W", "push", 2), ("OUT", "pop", 2)]
            pn += [("P", "pop", 1), ("G", "push", 2)]
            pc += [
                ("X", "pop", 0),
                ("P", "push", 1),
                ("G", "pop", 2),
                ("OUT", "push", 2),
                ("X", "pop", 2),
                ("W", "pop", 2),
            ]
        elif t in (PH_CONV, PH_FIN):
            fin = t == PH_FIN
            pb += (
                [
                    ("J", "push", 2),
                    ("SH", "push", 1),
                    ("ZERO", "push", 1),
                    ("X", "push", 2),
                    ("CF", "push", 4),
                    ("E", "push", 4),
                    ("B", "push", 2),
                ]
                + ([("RES", "push", 2)] if fin else [])
                + [("OUT", "pop", 2)]
            )
            for _ in range(2):
                pc += [("OUT", "push", 1), ("X", "pop", 1), ("CF", "pop", 2), ("E", "pop", 2), ("B", "pop", 1)]
                if fin:
                    pc += [("RES", "pop", 1)]
            pc += [("J", "pop", 2), ("SH", "pop", 1), ("ZERO", "pop", 1)]
        elif t == PH_HEADS:
            noq = bool(flag & HEADS_NOQ)
            if k < nh_all:
                isq, isk = k < nqh, nqh <= k < nqh + nkvh
                if isq and noq:
                    pass
                elif isq or isk:
                    pb += [("QKV", "push", ht), ("HW", "push", ht), ("COS", "push", ht), ("SIN", "push", ht)]
                    ocb = "OUT" if isq else "KOUT"  # one consumer RISC per CB: the BRISC takes q, the NCRISC k
                    pc += [
                        ("INV", "push", 1),
                        ("HN", "push", ht),
                        ("QKV", "pop", ht),
                        ("HW", "pop", ht),
                        ("INV", "pop", 1),
                        (ocb, "push", ht),
                        ("HN", "pop", ht),
                        ("COS", "pop", ht),
                        ("SIN", "pop", ht),
                    ]
                    if isq:
                        pb += [("OUT", "pop", ht)]
                    else:
                        pn += [("KOUT", "pop", ht)]
                else:
                    pb += [("V", "push", ht)]
                    pn += [("V", "pop", ht)]
        else:  # SWI
            if k * nts < it:
                pb += [("GU", "push", 2 * nts), ("OUT", "pop", nts)]
                pc += [("OUT", "push", nts), ("GU", "pop", 2 * nts)]
        b.append(pb)
        n.append(pn)
        c.append(pc)
    return dict(brisc=b, ncrisc=n, trisc=c)


def cb_check(
    phases: list,
    *,
    nqh: int,
    nkvh: int,
    ht: int,
    nts: int,
    it: int,
    owners: int = N_OWNERS,
    rewind: bool = True,
    single_consumer: bool = True,
) -> dict:
    """asserts, for every owner k, (a) the balance: per phase and CB, the pages pushed == the pages popped and exactly
    one producer / one consumer RISC; (b) the no-straddle rule: with the local pointers of every RISC simulated
    (wrap at the CB end; rewind = every RISC rewinds every CB at the end of each phase, as the kernels do), no chunk
    crosses the CB's end (the device's fifo pointer assert / a pack past the CB). Returns the max chunk position
    reached per CB (rewind=False reproduces the watcher assert of 2026-10-23: CB_OUT at the SWI / HEADS chunks)"""
    pages = cb_pages(ht, nts)
    worst = {}
    for k in range(owners):
        ops = cb_ops(phases, k, nqh=nqh, nkvh=nkvh, ht=ht, nts=nts, it=it)
        if single_consumer:
            # the CB protocol's hidden assumption (the 2026-10-23 01:52Z silicon taps, df2_mk_common.hpp): over the
            # WHOLE program every CB has one consumer RISC and one producer RISC -- cb_wait_front snapshots the shared
            # pages_acked once, so a second consumer arriving early is released by the other consumer's phases' pushes
            roles = {}
            for r, seq in ops.items():
                for ph in seq:
                    for cb, op, n in ph:
                        if n:
                            roles.setdefault((cb, op), set()).add(r)
            for (cb, op), rs in roles.items():
                assert (
                    len(rs) == 1
                ), f"owner {k}: {cb} has {len(rs)} {op} RISCs {sorted(rs)} over the program (one per CB)"
        pos = {r: {cb: [0, 0] for cb in pages} for r in ops}  # per RISC: [write pos, read pos] in pages
        for p in range(len(phases)):
            pushed, popped, prod, cons = {}, {}, {}, {}
            for r, seq in ops.items():
                for cb, op, n in seq[p]:
                    if n == 0:
                        continue
                    size = pages[cb]
                    assert n <= size, f"phase {p} {r} {cb}: chunk {n} > CB {size} pages"
                    i = 0 if op == "push" else 1
                    at = pos[r][cb][i]
                    assert (
                        at + n <= size
                    ), f"owner {k} phase {p} {phases[p]} {r}: {op} {n} pages of {cb} at page {at} of {size} straddles the CB end"
                    worst[cb] = max(worst.get(cb, 0), at + n)
                    pos[r][cb][i] = (at + n) % size
                    (pushed if op == "push" else popped)[cb] = (pushed if op == "push" else popped).get(cb, 0) + n
                    (prod if op == "push" else cons).setdefault(cb, set()).add(r)
            for cb in set(pushed) | set(popped):
                assert pushed.get(cb, 0) == popped.get(
                    cb, 0
                ), f"owner {k} phase {p} {phases[p]}: {cb} pushed {pushed.get(cb, 0)} popped {popped.get(cb, 0)}"
                assert (
                    len(prod[cb]) == 1 and len(cons[cb]) == 1
                ), f"owner {k} phase {p}: {cb} producers {prod[cb]} consumers {cons[cb]}"
            if rewind:
                for r in pos:
                    for cb in pos[r]:
                        pos[r][cb] = [0, 0]
    return worst


def program_phase_types(kind: str, n_layers: int = 8) -> list:
    """the (type, flag) lists of the four program kinds, as Df2Mk builds them"""
    if kind == "a0":
        return [(PH_RMS, RMS_SIGNAL), (PH_CONV, 0), (PH_HEADS, HEADS_LAST)]
    if kind == "ba":
        return [
            (PH_FIN, 1 << 4),
            (PH_RMS, RMS_SIGNAL),
            (PH_CONV, 0),
            (PH_SWI, 0),
            (PH_FIN, (1 << 4) | 1),
            (PH_RMS, RMS_SIGNAL),
            (PH_CONV, 0),
            (PH_HEADS, HEADS_LAST),
        ]
    if kind == "b_last":
        return [(PH_FIN, 1 << 4), (PH_RMS, RMS_SIGNAL), (PH_CONV, 0), (PH_SWI, 0), (PH_FIN, (1 << 4) | 1), (PH_RMS, 0)]
    if kind == "ctx":
        return [(PH_RMS_AR, RMS_SIGNAL)] + [
            (PH_HEADS, i | HEADS_NOQ | (HEADS_LAST if i == n_layers - 1 else 0)) for i in range(n_layers)
        ]
    raise ValueError(kind)
