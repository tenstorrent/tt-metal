# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill megakernel (PMK): the mHC PRE and POST phases of the persistent prefill layer program (Stage 16's contract
doc/ttnn/16-prefill-production/prefill_megakernel.md §2 roles, §3 P1 / P10 / P16, §4b CB ranges and counters;
Stage 7a's draft doc/ttnn/07a-fused-streaming-layer/prefill_mk_collectives.md §5).

Math (tt/mhc_pre.py reference_step / tt/mhc_post.py reference_step are the oracles), per token of the 4-stream fp32
residual x [T, 16384]:
    pre : m = (x @ fn^T) / rms(x); z = m * scale + bias; pre = sigmoid(z[0:4]) + eps; post = 2 sigmoid(z[4:8]);
          comb = Sinkhorn(softmax(z[8:24] as 4 x 4)); h = sum_s pre_s x_s; hn = h / rms(h) * w      -> x-hat (bf16)
    post: x'_s = post_s * y + sum_i comb[i][s] x_i                                                  -> the new residual

Geometry (contract §2; the Stage 16 note NOTE_16_GEOMETRY.txt: every piece builds BOTH):
    TP8  SP = 256 rows per chip:   8 row blocks (32 rows) x 8 column slabs = 64 members, 8 roots; member (b, j) owns the
         column tiles [16 j, 16 j + 16) of all 4 streams of block b = 64 fp32 tiles = 256 KB, RESIDENT in CB_X for the
         whole layer (loaded from the DRAM residual at the program boundary, written back at the end: F_X_LOAD /
         F_X_STORE; inside a stack program it never leaves L1). NRB = 1 block per group.
    TP2  SP = 1,024 rows per chip: the same 64 + 8 cores; group g handles blocks g, g + 8, g + 16, g + 24 (NRB = 4) and
         the 64 MB residual stays in DRAM: each block streams through CB_X for the pre and again for the post, the
         post's output tiles go back to the DRAM residual pages in place.
This is Stage 7a's MhcPrePf (G = 8, P = 8, CPC = 16, PF_FPU arithmetic) + mhc_post_prefill_7a as phases: the kernel
bodies are 7a's (kernels/pmk/mhc_*.cpp document the derivation per function), with
    * the residual resident / in place instead of DRAM tensors per call;
    * the post's coefficient tiles packed by the ROOT (K_s | L_s, 7a's mhc_post_pf7a reader layout) and unicast to its
      members' CB_PCOEF (barriered before its round-2 message): no post / comb DRAM round trip, no per-core relayout;
    * program-wide cumulative hand-off counters (noc_semaphore_wait_min; `round0` = layer x nsub x nrb folded into
      every expected value, never reset within the program: integrator 10:02Z, contract §4b);
    * H1 / H7 (x-hat): the member's 16 bf16 tiles pushed into the AG store (XS CB 35 on MM core rank * blocks + b at
      column 512 j', FS_ARRIVE 15 raised on the holder) or, standalone, written to a DRAM hn tensor;
    * H6 / H10 (y): the RS lanes land the member's 32 x 512 fp32 rows in CB_Y and raise FS_REDUCED (14); standalone,
      the member reads them from a DRAM y tensor;
    * phase args at PH_CT0 / ph_rt0 (Stage 14's composition convention) so the integrator can include the kernels in its
      owner composites.
CB map (contract §4b): mHC pre 0..20 (7a's 0..19 + TOKEN 20), mHC post 21..28 (Y 21, PCOEF 22, PSTAGE 23; 24..28 free).
Semaphores: 0..3 the pre's rounds; HN 4 (x-hat landed, on the MM cores), FS_REDUCED 14 (y landed, on the members),
SEM_C 12 (coefficient tiles landed, on the members; new).

`Plan` is pure python (no ttnn): geometry, CB bytes per role, placement, the counter protocol check and the time
estimate — tests/s16/pmk_mhc_test.py runs it on a CPU. `PmkMhc` (ttnn) builds the kernels / CBs / semaphores as
Stage 14-style role parts and a standalone ProgramDescriptor (the mock job and the device bench).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

TILE = 32
N_STREAMS = 4
HIDDEN = 4096
COL_TILES = HIDDEN // TILE  # 128
N_MIX = 24
P = 8  # members per group (column slabs)
CPC = COL_TILES // P  # 16 column tiles per member and stream
X_TILES = N_STREAMS * CPC  # 64 residual tiles per member block
N_GROUPS = 8
FP32_T, BF16_T = 4096, 2048
KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "pmk")
PMK_REV = "6"  # kernel revision define: bump on every kernel source edit (the JIT cache keys file-path kernels)

# mirrors kernels/pmk/mhc_common.hpp
CB = dict(GATHER=0, MSG=1, X=2, X32=3, FN=4, W=5, PART=6, PRE=7, H=8, S2=9, INV=10, OUT=11, H16=12, RED=14, PLANE=15,
          COEF=16, SB=17, INVR=18, SCR=19, TOKEN=20, Y=21, PCOEF=22, PSTAGE=23, DBG=24)  # fmt: skip
CB_PRE_RANGE, CB_POST_RANGE = range(0, 21), range(21, 29)
SEM_G1, SEM_G2, SEM_M1, SEM_M2 = 0, 1, 2, 3
SEM_XH_DEFAULT = 15  # integrator 10:02Z: FS_ARRIVE, the H1 / H2 holder's word on the AG-store MM cores (coll polls it)
SEM_Y_DEFAULT = 14  # contract: FS_REDUCED (the member's reduced 32 x 512 fp32 rows landed in CB_Y)
F_X_LOAD, F_X_STORE, F_XH_DRAM, F_Y_DRAM, F_PC_DRAM, F_INIT, F_FN_PREFETCH, F_PROBE = 1, 2, 4, 8, 16, 32, 64, 128
F_SB_PRELUDE = 256  # root P10: the integrator's prelude pushes sublayer 1's SB tiles (else the stream part loads them)
PARTS = (1, 10, 16)  # PMK_MHC_PART: P1 = pre(sub 0), P10 = post(sub 0) + pre(sub 1), P16 = post(sub 1); 0 = all in one
N_TAPS = 4  # PMK_PROBE taps per round: z planes, the pass, the pass with ITERS = 1, the sync's diag record (rev 6)
PLANE_MARKER, PLANE_DIAG, PACK_MAX_ATTEMPTS = 25, 26, 2  # mirrors mhc_common.hpp (the root's verified pack, rev 6)
TAP_DIAG, DIAG_MAGIC, DIAG_WORDS = 3, 0xD1A60006, 12  # tap 3 = the sync RISC's diagnostics record (rev 6)


def marker_bits(r: int) -> int:
    """mirrors mhc_common.hpp marker_bits: the fp32 bit pattern the root's pass stores in plane PLANE_MARKER as its
    last SFPU store of round r (512 + 4 (r + 1): finite, nonzero, distinct per round)"""
    return (0x44000000 + (r + 1) * 0x10000) & 0xFFFFFFFF


def marker_final(r: int) -> int:
    """mirrors mhc_common.hpp marker_final: the pushed COEF page's marker (cycle 3 = the sign-flipped marker_bits)"""
    return marker_bits(r) | 0x80000000


NO_GATE = 0xFFFFFFFF
# fp32 CBs read by copy_tile: unpacked straight to DST (exact fp32 on the SFPU); CB_X (matmul operand) and CB_PRE
# (diagonal matmul operand) and CB_H16 (matmul operand) keep the default unpack
FP32_COPY_CBS = ("GATHER", "X32", "H", "INV", "PLANE", "SB", "Y", "PCOEF")  # (W is bf16: default unpack)


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


# ============================================================================================ pure-python plan
@dataclass
class Plan:
    """Geometry, CB bytes, placement and the counter protocol of the mHC phases for `sp_rows` SP rows per chip."""

    sp_rows: int = 256
    nsub: int = 2  # sublayers per program (attention, FFN)
    y_fp32: bool = True  # H6 / H10 y is fp32 (the RS lanes reduce in fp32); False = bf16 (TP8 op-graph y)
    hn_fp32: bool = False  # x-hat bf16 (contract H1); fp32 for a standalone fp32 consumer
    y_lane: bool = True  # y landed by the RS lanes (CB_Y holds the sublayer's NRB blocks); False = DRAM-read ring of 1
    sem_xh: int = SEM_XH_DEFAULT
    sem_y: int = SEM_Y_DEFAULT
    probe: bool = False  # root: CB_DBG + the taps of every coefficient pass (PMK_PROBE)
    grid: tuple = (11, 10)  # p300 worker grid (doc/hardware.md: 11 x 10 = 110 cores)

    def __post_init__(self):
        assert self.sp_rows % (TILE * N_GROUPS) == 0, self.sp_rows
        self.n_blocks = self.sp_rows // TILE
        self.nrb = self.n_blocks // N_GROUPS
        self.resident = self.nrb == 1  # TP8: one block per group, in L1 across the layer
        self.y_page = FP32_T if self.y_fp32 else BF16_T
        self.hn_page = FP32_T if self.hn_fp32 else BF16_T
        self.step = N_GROUPS

    # ------------------------------------------------------------------------------------ placement
    def placement(self) -> dict:
        """member (b, j) at logical (j, b), root b at (P, b) (7a's PF_MCAST column layout: a column of members shares
        a fn^T slab, so a later multicast of fn^T / w goes down a NoC rectangle). Any placement works for the kernels
        (they only see NoC coordinates), the integrator may remap."""
        assert self.grid[0] > P and self.grid[1] >= N_GROUPS, self.grid
        return dict(members={(b, j): (j, b) for b in range(N_GROUPS) for j in range(P)},
                    roots={b: (P, b) for b in range(N_GROUPS)})  # fmt: skip

    # ------------------------------------------------------------------------------------ CBs
    def cbs(self) -> list:
        """[(name, index, page bytes, pages, roles)] with roles in {"member", "root", "all"} (all = uniform address:
        remote cores write into it)."""
        nrb, cpc = self.nrb, CPC
        return [
            ("GATHER", CB["GATHER"], FP32_T, 2 * P, "all"),
            ("MSG", CB["MSG"], 512, 1, "all"),
            ("PCOEF", CB["PCOEF"], FP32_T, nrb * 2 * N_STREAMS, "all"),
            ("X", CB["X"], FP32_T, X_TILES, "member"),  # + alias X32 (same allocation)
            ("FN", CB["FN"], BF16_T, X_TILES, "member"),
            ("W", CB["W"], BF16_T, cpc, "member"),  # bf16: the checkpoint's norm weight IS bf16 (exact)
            ("PART", CB["PART"], FP32_T, 2, "member"),
            ("PRE", CB["PRE"], FP32_T, N_STREAMS, "member"),
            ("H", CB["H"], FP32_T, cpc, "member"),  # + alias H16
            ("S2", CB["S2"], FP32_T, 1, "member"),
            ("INV", CB["INV"], FP32_T, 1, "member"),
            ("OUT", CB["OUT"], self.hn_page, cpc, "member"),
            ("TOKEN", CB["TOKEN"], 64, 1, "member"),
            ("Y", CB["Y"], self.y_page, (nrb if self.y_lane else 1) * cpc, "member"),
            ("PSTAGE", CB["PSTAGE"], FP32_T, 2 * N_STREAMS, "member"),
            ("RED", CB["RED"], FP32_T, 1, "root"),
            ("PLANE", CB["PLANE"], FP32_T, 1, "root"),
            ("COEF", CB["COEF"], FP32_T, 1, "root"),
            ("SB", CB["SB"], FP32_T, 3, "root"),
            ("INVR", CB["INVR"], FP32_T, 1, "root"),
            ("SCR", CB["SCR"], FP32_T, 2, "root"),
        ] + ([("DBG", CB["DBG"], FP32_T, N_TAPS, "root")] if self.probe else [])

    def l1_bytes(self) -> dict:
        """CB bytes per role (the L1 the integrator must give these phases; the residual block is the X entry)."""
        out = dict(member=0, root=0)
        for name, _, page, pages, roles in self.cbs():
            for r in ("member", "root"):
                if roles in (r, "all"):
                    out[r] += page * pages
        out["member_resident_only"] = (FP32_T * X_TILES) if self.resident else 0
        # dead during the MM phases P3 / P7 (between P1 and P10): everything but X (resident), PCOEF (LIVE P1 -> P10,
        # root-written; uniform address), Y (lane landed), TOKEN, and FN / W only when the sublayer-1 prefetch is gated
        # after P7 (F_FN_PREFETCH + the family's IN0-free gate): the integrator may alias its MM CBs onto these
        dead = sum(page * pages for name, _, page, pages, roles in self.cbs()
                   if roles == "member" and name not in ("X", "Y", "TOKEN"))  # fmt: skip
        out["member_dead_during_mm"] = dead
        return out

    # ------------------------------------------------------------------------------------ counters
    def counter_protocol(self, layers: int = 2) -> dict:
        """Simulate the program-wide cumulative counters over `layers` layers of nsub sublayers x nrb blocks (one
        member, its root, one holder): every wait_min resolves with the expected value round0 + idx (members) /
        (round0 + idx) * P (root), nothing is ever reset, and the holder's FS_ARRIVE word reaches 8 x (layer + 1)
        x nsub x nrb (coll's cumulative AG1 >= 8, AG2 >= 16 per layer). Returns the final counts."""
        m = dict(M1=0, M2=0, Y=0)  # (no coefficient counter: M2 >= idx implies the root's barriered K/L unicast landed)
        r = dict(G1=0, G2=0)
        holder = 0  # FS_ARRIVE on one AG-store core: 8 members per block raise it
        per_layer = self.nsub * self.nrb
        for layer in range(layers):
            round0 = layer * per_layer
            for sub in range(self.nsub):
                for k in range(self.nrb):
                    idx = round0 + sub * self.nrb + k + 1
                    r["G1"] += P
                    assert r["G1"] >= idx * P
                    m["M1"] += 1
                    assert m["M1"] >= idx
                    r["G2"] += P
                    assert r["G2"] >= idx * P
                    m["M2"] += 1
                    assert m["M2"] >= idx
                    holder += P  # this block's 8 members' x-hat slices
                    assert holder >= 8 * idx
                for k in range(self.nrb):
                    idx = round0 + sub * self.nrb + k + 1
                    m["Y"] += 1
                    assert m["M2"] >= idx and m["Y"] >= idx
            assert m["M1"] == m["M2"] == m["Y"] == (layer + 1) * per_layer
        return dict(final=dict(member=m, root=r, holder=holder), waits_member=layers * per_layer,
                    waits_root=layers * per_layer * P, holder_per_layer=8 * per_layer)  # fmt: skip

    # ------------------------------------------------------------------------------------ estimate
    def estimate_us(self) -> dict:
        """us per SUBLAYER (pre + post) from Stage 7a's d0 timing bisect at T 256 (work_log 05:59-06:18Z, PF_FPU,
        G 8 x P 8 = the same per-member work as one block here) with the per-call DRAM / launch / sync floor removed:
          phase 1 (64 fn + 64 x x^T matmul_tiles, FPU)  37 us   (measured: removing them saved 37)
          root: round-1 sums + z, Sinkhorn pass, msgs    ~11 us   (Sinkhorn 5 measured; sums / messages ~6)
          collapse 64 + S2 16 matmul_tiles (FPU)         ~15 us   (hidden in the bisect; counted here)
          finish 16 x (h * inv * w) SFPU                  15 us   (measured)
          post 16 units x ~5.5 us SFPU (7a: copies 21 + math 30 us at 9.3 units per core)   88 us
        The post's FPU is idle (exact fp32 SFPU FMAs: the residual must not go through tf32). Per block; at TP2 the
        4 blocks of a group run back to back (+ the DRAM reads of x / y per block: 320 KB per member per phase, hidden
        under the previous block's compute except the first)."""
        pre_fpu, pre_root, pre_coll, pre_fin = 37.0, 11.0, 15.0, 15.0
        post = 16 * 5.5
        per_block = dict(pre=pre_fpu + pre_root + pre_coll + pre_fin, post=post)
        per_block["fpu_busy_us"] = pre_fpu + pre_coll
        per_block["fpu_busy_share"] = per_block["fpu_busy_us"] / (per_block["pre"] + per_block["post"])
        dram_first = 0.0 if self.resident else 2 * 320e3 / 480e9 * 1e6 * 64  # all members' first-block reads
        sub = dict(pre=per_block["pre"] * self.nrb, post=per_block["post"] * self.nrb + dram_first)
        sub["total"] = sub["pre"] + sub["post"]
        sub["fpu_busy_share"] = (
            per_block["fpu_busy_share"] * (per_block["pre"] + per_block["post"]) * self.nrb / sub["total"]
        )
        return dict(per_block=per_block, per_sublayer=sub, sp_rows=self.sp_rows, nrb=self.nrb, resident=self.resident)


# ============================================================================================ ttnn program
def _ttnn():
    import ttnn  # noqa: PLC0415

    return ttnn


@dataclass
class SubWeights:
    """One sublayer's device tensors (the layouts of tt/mhc_pre_pf.MhcPrePf): fn [16384, 32] bf16 (page = K tile
    s * 128 + c), sb [96, 32] fp32 (scale row, bias row, identity), w [32, 4096] bf16 (row 0 = the norm weight, exact).
    """

    fn: object
    sb: object
    w: object
    host: dict = field(default_factory=dict)

    @classmethod
    def build(cls, mesh, fn_t, scale, bias, norm_w):
        import torch  # noqa: PLC0415

        ttnn = _ttnn()
        put = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        fn_pad = torch.zeros(N_STREAMS * HIDDEN, TILE)
        fn_pad[:, :N_MIX] = fn_t.float()
        sb = torch.zeros(3 * TILE, TILE)
        sb[0, :N_MIX] = scale.float()
        sb[TILE, :N_MIX] = bias.float()
        sb[2 * TILE :] = torch.eye(TILE)
        wt = torch.zeros(TILE, HIDDEN)
        wt[0] = norm_w.float()
        # w as bf16 tiles (the checkpoint's norm weights are bf16: the tile holds them exactly; copy_tile of a bf16 CB
        # into the fp32 DST is the same fp32 value 7a's fp32 copy gave) -- 32 KB less per member than 7a's fp32 tiles
        return cls(fn=put(fn_pad.to(torch.bfloat16), ttnn.bfloat16), sb=put(sb, ttnn.float32),
                   w=put(wt.to(torch.bfloat16), ttnn.bfloat16),
                   host=dict(fn_t=fn_t.float(), scale=scale.float(), bias=bias.float(), w=norm_w.float()))  # fmt: skip

    @classmethod
    def from_source(cls, mesh, source, cfg, L: int, which: str, norm_name: str):
        """as tt/mhc_pre_pf.MhcPrePf.from_source"""
        import torch  # noqa: PLC0415

        n = cfg.hc_mult
        scale = source.host(L, f"hc_{which}_scale").float().reshape(-1)
        per_mix = torch.cat([scale[0].repeat(n), scale[1].repeat(n), scale[2].repeat(n * n)])
        return cls.build(mesh, source.host(L, f"hc_{which}_fn").reshape(n * cfg.hidden_size, (2 + n) * n), per_mix,
                         source.host(L, f"hc_{which}_base").float().reshape(-1),
                         source.host(L, norm_name).float().reshape(-1))  # fmt: skip

    @classmethod
    def random(cls, mesh, seed: int = 0):
        """synthetic weights with the checkpoint's magnitudes (mock / smoke)"""
        import torch  # noqa: PLC0415

        g = torch.Generator().manual_seed(seed)
        fn_t = torch.randn(N_STREAMS * HIDDEN, N_MIX, generator=g) * 0.01
        scale = torch.cat([torch.full((4,), 0.5), torch.full((4,), 0.5), torch.full((16,), 0.5)])
        bias = torch.zeros(N_MIX)
        bias[0:4] = 0.5
        bias[8:24] = torch.eye(4).reshape(-1) * 2.0
        return cls.build(mesh, fn_t, scale, bias, 1.0 + 0.1 * torch.randn(HIDDEN, generator=g))


class PmkMhc:
    """The mHC phases of one layer program on one device (replicated on a mesh): weights of `nsub` sublayers, the
    role parts (Stage 14 composition) and a standalone descriptor (bench / mock)."""

    def __init__(self, mesh, subs: list, plan: Plan, *, hc_eps: float, x_eps: float, h_eps: float, iters: int,
                 placement: dict | None = None):  # fmt: skip
        assert len(subs) == plan.nsub, (len(subs), plan.nsub)
        self.mesh, self.subs, self.plan = mesh, subs, plan
        self.iters, self.hc_eps, self.x_eps, self.h_eps = iters, hc_eps, x_eps, h_eps
        ttnn = _ttnn()
        g = mesh.compute_with_storage_grid_size()
        plan.grid = (int(g.x), int(g.y))
        self.place = placement or plan.placement()
        self.members = [self.place["members"][(b, j)] for b in range(N_GROUPS) for j in range(P)]
        self.roots = [self.place["roots"][b] for b in range(N_GROUPS)]
        crs = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))  # noqa: E731
                                               for x, y in cores])  # fmt: skip
        self.m_crs, self.r_crs, self.all_crs = crs(self.members), crs(self.roots), crs(self.members + self.roots)

    def _noc(self, xy):
        ttnn = _ttnn()
        c = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
        return int(c.x), int(c.y)

    @staticmethod
    def _acc(t) -> list:
        return list(_ttnn().TensorAccessorArgs(t).get_compile_time_args())

    def cb_descriptors(self) -> list:
        ttnn = _ttnn()
        fmt = {FP32_T: ttnn.float32, BF16_T: ttnn.bfloat16, 512: ttnn.float32, 64: ttnn.float32}
        role_crs = dict(member=self.m_crs, root=self.r_crs, all=self.all_crs)
        out = []
        for name, idx, page, pages, roles in self.plan.cbs():
            dt = fmt[page] if name not in ("OUT", "Y") else (ttnn.float32 if page == FP32_T else ttnn.bfloat16)
            fmts = [ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)]
            alias = {"X": CB["X32"], "H": CB["H16"]}.get(name)
            if alias is not None:
                fmts.append(ttnn.CBFormatDescriptor(buffer_index=alias, data_format=dt, page_size=page))
            out.append(ttnn.CBDescriptor(total_size=page * pages, core_ranges=role_crs[roles], format_descriptors=fmts))
        return out

    def semaphore_descriptors(self, handoff: bool = True) -> list:
        """0..3 on members + roots; FS_REDUCED (14) on the members in the standalone program (the member's own DRAM
        read replaces the lanes; declared so the id is valid). No coefficient semaphore (7a review: 12 = XS_FREE). The
        integrator declares FS_ARRIVE (15) on the holders and FS_REDUCED itself: handoff=False leaves it out."""
        ttnn = _ttnn()
        p = self.plan
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all_crs, initial_value=0) for i in range(4)]
        if handoff:
            sems.append(ttnn.SemaphoreDescriptor(id=p.sem_y, core_ranges=self.m_crs, initial_value=0))
        return sems

    def _compute_config(self, full_sync: bool, role: str | None = None):
        """members: full-sync fp32 DST (the post's 8-tile pass; the pre's <= 3-tile bodies run under it too, as
        Stage 14's owner composites do); root: HALF-sync as 7a's standalone root and Stage 14's root composite
        (silicon 10:08Z: the root's plane-layout coefficient pass under full sync gave exact pre / post planes (DST face
        0) but wrong comb planes (faces 1-2); the pass was only ever validated half-sync)."""
        ttnn = _ttnn()
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True, dst_full_sync_en=full_sync)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        # role-scoped (M-A, 2026-10-24 05:05Z): a role's config marks only ITS OWN fp32 copy CBs. One table for both
        # roles marked the ROOT-only PLANE 15 / SB 17 on the MEMBER kernel too, harmless in the mhc-only programs but
        # merged into a fused composite (merge_compute: per-CB union) it flipped the kda items' CBs 15 / 17 to
        # UnpackToDestFp32 -> wrong item terms in every fused program with an mhc phase (A15 / P1+A2F / A2F+P10)
        own = None
        if role is not None:
            own = {name for name, _, _, _, roles in self.plan.cbs() if roles in (role, "all")}
            if role == "member":
                own |= {"X32", "H16"}  # aliases of the member's X / H allocations
        for name in FP32_COPY_CBS:
            if name == "Y" and not self.plan.y_fp32:
                continue
            if own is not None and name not in own:
                continue
            modes[CB[name]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        return cc

    def parts(self, x, hns: list, ys: list, posts: list, combs: list, *, flags: int, part: int = 0,
              token_wait0: int = 0, xh_targets: dict | None = None, layer: int = 0, fn_gate=(NO_GATE, 0),
              taps=None) -> tuple:  # fmt: skip
        """Role parts of ONE part (PMK_MHC_PART = part: 1 / 10 / 16, or 0 = all three in one body, the standalone
        unit): ([{name, cores, ct, rt {(x, y): args}, defines, config}], cbs, semaphores). The ct / rt layout is the
        same for every part (a part ignores what it does not need), so the integrator includes each source once per
        part with the same arg bases.
        x: the DRAM residual [1, 1, SP, 16384] fp32 (resident: read at F_X_LOAD / written at F_X_STORE; streamed: read
        and written per block). hns / ys / posts / combs: per-sublayer DRAM tensors (hn [1,1,SP,4096], y [1,1,SP,4096],
        post [1,1,SP,4], comb [1,1,SP,16]); unused ones may be any tensor of the right kind (same memory config / page
        size). flags: F_* bits (F_INIT: the first layer's P1 only; F_FN_PREFETCH: sublayer 1's fn^T / w read at P1's
        end after the family's IN0-free gate fn_gate = (semaphore id, cumulative value), NO_GATE = none).
        xh_targets[(sub, b, j)] = (noc x, noc y, L1 address) of the member's 32 x 512 bf16 slice in the AG store (push
        mode, flags without F_XH_DRAM; FS_ARRIVE is raised on that core). layer: this layer's index in the program
        (round0 = layer x nsub x nrb is folded into every counter wait: cumulative, never reset). taps: the DRAM
        taps tensor [1, 1, n_blocks * nsub * N_TAPS * 32, 32] fp32 (plan.probe + F_PROBE), else any DRAM tensor."""
        ttnn = _ttnn()
        p = self.plan
        assert part in (0,) + PARTS, part
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        dm0 = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)
        dm1 = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)
        cc, cc_root = self._compute_config(True, "member"), self._compute_config(False, "root")
        defines = [("PMK_REV", PMK_REV), ("PMK_NSUB", str(p.nsub)), ("PMK_NRB", str(p.nrb)),
                   ("PMK_RESIDENT", "1" if p.resident else "0"), ("PMK_SEM_Y", str(p.sem_y)),
                   ("PMK_SEM_XH", str(p.sem_xh)), ("PMK_MHC_PART", str(part))]  # fmt: skip
        if p.probe:
            defines.append(("PMK_PROBE", "1"))
        sub0 = self.subs[0]
        taps_t = taps if taps is not None else posts[0]
        m_stream_ct = (
            [p.y_page] + self._acc(x) + self._acc(sub0.fn) + self._acc(sub0.w) + self._acc(hns[0]) + self._acc(ys[0])
        )
        r_stream_ct = self._acc(sub0.sb)
        r_sync_ct = self._acc(posts[0]) + self._acc(combs[0]) + self._acc(taps_t)
        compute_ct = [self.iters, f32_bits(self.hc_eps), f32_bits(1.0 / (N_STREAMS * HIDDEN)), f32_bits(self.x_eps),
                      f32_bits(1.0 / HIDDEN), f32_bits(self.h_eps)]  # fmt: skip
        n_rounds = p.nsub * p.nrb if part == 0 else (p.nrb if part in (1, 10) else 0)
        ms_rt, my_rt, mc_rt, rs_rt, ry_rt, rc_rt = {}, {}, {}, {}, {}, {}
        round0 = layer * p.nsub * p.nrb
        for b in range(N_GROUPS):
            root = self.place["roots"][b]
            rnoc = self._noc(root)
            mem = [self.place["members"][(b, j)] for j in range(P)]
            for j, core in enumerate(mem):
                ms_rt[core] = [x.buffer_address(), b, p.step, j * CPC, flags, int(fn_gate[0]), int(fn_gate[1])] + [
                    v for s in range(p.nsub) for v in (self.subs[s].fn.buffer_address(), self.subs[s].w.buffer_address(),
                                                        hns[s].buffer_address(), ys[s].buffer_address())]  # fmt: skip
                push = []
                for s in range(p.nsub):
                    for k in range(p.nrb):
                        tgt = (xh_targets or {}).get((s, b + k * p.step, j), (0, 0, 0))
                        push += list(tgt)
                my_rt[core] = [rnoc[0], rnoc[1], j, flags, round0] + push
                mc_rt[core] = [token_wait0]
            # rev 6: CB_SB is per sublayer: sb(0) (P1, F_INIT), flags, sb(1) (P10 unless F_SB_PRELUDE)
            rs_rt[root] = [sub0.sb.buffer_address(), flags, self.subs[min(1, p.nsub - 1)].sb.buffer_address()]
            ry_rt[root] = [flags, b, p.step, round0, taps_t.buffer_address()] + [
                v for s in range(p.nsub) for v in (posts[s].buffer_address(), combs[s].buffer_address())] + [
                v for c in mem for v in self._noc(c)]  # fmt: skip
            rc_rt[root] = [n_rounds]
        src = lambda n: os.path.join(KERNEL_DIR, n)  # noqa: E731
        roles = [
            dict(name=src("mhc_stream.cpp"), cores=self.m_crs, ct=m_stream_ct, rt=ms_rt, defines=[("PMK_MEMBER", "1")] + defines, config=dm0),
            dict(name=src("mhc_sync.cpp"), cores=self.m_crs, ct=[], rt=my_rt, defines=[("PMK_MEMBER", "1")] + defines, config=dm1),
            dict(name=src("mhc_compute.cpp"), cores=self.m_crs, ct=compute_ct, rt=mc_rt, defines=[("PMK_MEMBER", "1")] + defines, config=cc),
            dict(name=src("mhc_stream.cpp"), cores=self.r_crs, ct=r_stream_ct, rt=rs_rt, defines=[("PMK_ROOT", "1")] + defines, config=dm0),
            dict(name=src("mhc_sync.cpp"), cores=self.r_crs, ct=r_sync_ct, rt=ry_rt, defines=[("PMK_ROOT", "1")] + defines, config=dm1),
            dict(name=src("mhc_compute.cpp"), cores=self.r_crs, ct=compute_ct, rt=rc_rt, defines=[("PMK_ROOT", "1")] + defines, config=cc_root),
        ]  # fmt: skip
        return roles, self.cb_descriptors(), self.semaphore_descriptors(handoff=bool(flags & F_Y_DRAM))

    @staticmethod
    def _rt(per_core: dict):
        ttnn = _ttnn()
        rt = ttnn.RuntimeArgs()
        for (x, y), args in per_core.items():
            rt[x][y] = [int(a) for a in args]
        return rt

    def descriptor(self, *args, **kwargs):  # (part=..., probe via plan.probe)
        """standalone ProgramDescriptor (the roles as kernels; the integrator composes parts() instead)"""
        ttnn = _ttnn()
        roles, cbs, sems = self.parts(*args, **kwargs)
        kernels = [ttnn.KernelDescriptor(kernel_source=r["name"], source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                         core_ranges=r["cores"], compile_time_args=r["ct"], defines=r["defines"],
                                         runtime_args=self._rt(r["rt"]), config=r["config"]) for r in roles]  # fmt: skip
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    def __call__(self, x, ys: list, *, write_pc: bool = True, part: int = 0, probe=None):
        """Standalone (bench / mock): x [1, 1, SP, 16384] fp32 DRAM residual, UPDATED IN PLACE through the nsub
        sublayers; ys[s] [1, 1, SP, 4096] (fp32 / bf16 per plan.y_fp32) the sublayer outputs -> (hns, posts, combs
        [, taps]): hn [1,1,SP,4096] bf16 per sublayer, post [1,1,SP,4] / comb [1,1,SP,16] fp32 per sublayer (7a's
        layouts); taps [1,1,n_blocks*nsub*N_TAPS*32,32] fp32 when plan.probe (decode with decode_taps). part != 0
        builds that part's program alone (mock compile only: CB state does not persist across programs)."""
        ttnn = _ttnn()
        p = self.plan
        t = int(x.shape[-2])
        assert t == p.sp_rows and x.dtype == ttnn.float32 and int(x.shape[-1]) == N_STREAMS * HIDDEN, (t, x.dtype)
        dev = x.device()
        mk = lambda shape, dt: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
        hns = [mk((1, 1, t, HIDDEN), ttnn.float32 if p.hn_fp32 else ttnn.bfloat16) for _ in range(p.nsub)]
        posts = [mk((1, 1, t, N_STREAMS), ttnn.float32) for _ in range(p.nsub)]
        combs = [mk((1, 1, t, N_STREAMS * N_STREAMS), ttnn.float32) for _ in range(p.nsub)]
        taps = mk((1, 1, p.n_blocks * p.nsub * N_TAPS * TILE, TILE), ttnn.float32) if p.probe else None
        flags = F_XH_DRAM | F_Y_DRAM | F_INIT | F_FN_PREFETCH | (F_PC_DRAM if write_pc else 0)
        flags |= (F_X_LOAD | F_X_STORE) if p.resident else 0
        flags |= F_PROBE if p.probe else 0
        desc = self.descriptor(x, hns, ys, posts, combs, flags=flags, part=part, taps=taps)
        io = [x] + [t_ for s in self.subs for t_ in (s.fn, s.sb, s.w)] + list(ys) + hns + posts + combs
        io += [taps] if taps is not None else []
        ttnn.generic_op(io, desc)
        return (hns, posts, combs) + ((taps,) if taps is not None else ())


# ============================================================================================ taps
def decode_taps(taps_host, plan: Plan, rb: int, sub: int):
    """taps [n_blocks * nsub * N_TAPS * 32, 32] host fp32 -> dict(z, pass, iters1): each [24, 32] (plane, token) of
    row block rb / sublayer sub, re-laid from the plane layout (tt/mhc_pre.from_planes); tap TAP_DIAG is the sync's
    record (decode_pack_diag)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import from_planes  # noqa: PLC0415

    tiles = taps_host.float().reshape(-1, TILE, TILE)
    base = (rb * plan.nsub + sub) * N_TAPS
    names = ("z", "pass", "iters1")
    return {n: from_planes(tiles[base + i], N_MIX, TILE) for i, n in enumerate(names)}


def decode_pack_diag(taps_host, plan: Plan, rb: int, sub: int) -> dict:
    """the root's verified-pack diagnostics of (rb, sub): tap TAP_DIAG's first DIAG_WORDS uint32 words, written by the
    sync RISC from its mailbox scratch (never a packer-written page): magic (valid iff == DIAG_MAGIC), r, sub, k, the
    expected marker, plane-25 lanes 0 / 31 as read from the pushed COEF page, marker_ok (all 32 lanes), attempts1 /
    attempts2 (the pack thread's CB_RED words: unverified channel), rev, idx. pack_lost = the small retry bound ran
    out (the marker was not in the packed page)."""
    import torch  # noqa: PLC0415

    tiles = taps_host.float().reshape(-1, TILE, TILE)
    w = tiles[(rb * plan.nsub + sub) * N_TAPS + TAP_DIAG].contiguous().reshape(-1).view(torch.int32)
    u = lambda i: int(w[i].item()) & 0xFFFFFFFF  # noqa: E731
    d = dict(valid=u(0) == DIAG_MAGIC, r=u(1), sub=u(2), k=u(3), marker=hex(u(4)), seen0=hex(u(5)), seen31=hex(u(6)),
             marker_ok=u(7), attempts1=u(8), attempts2=u(9), rev=u(10), idx=u(11))  # fmt: skip
    d["marker_expected"] = u(4) == marker_final(d["r"]) if d["valid"] else False
    d["pack_lost"] = d["valid"] and d["marker_ok"] == 0
    return d


def host_planes(z24: "torch.Tensor", iters: int, hc_eps: float):
    """the pass's arithmetic on host (float64): z24 [T, 24] -> [24, T] planes of pre | post | comb after `iters`
    Sinkhorn iterations (iters = 1: softmax + the first column pass only)"""
    import torch  # noqa: PLC0415

    z = z24.double()
    pre = torch.sigmoid(z[:, :4]) + hc_eps
    post = 2 * torch.sigmoid(z[:, 4:8])
    comb = torch.softmax(z[:, 8:].reshape(-1, 4, 4), dim=-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return torch.cat([pre, post, comb.reshape(-1, 16)], dim=1).T
