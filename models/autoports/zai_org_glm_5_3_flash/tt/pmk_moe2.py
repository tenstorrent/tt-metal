# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PMK MoE phase set v2 (piece 'moe2', doc/ttnn/16-prefill-production/pmk/moe2/README.md): the levers that lift v1's
DRAM floor (tt/pmk_moe.py keeps the v1 arithmetic; this module extends it, every v1 hand-off word stays valid):

  lever 2  XFFN_X_TILED  the routed x rows gathered over NoC1 from the family's L1 x-hat_f store (XS 35, H2 / H8: block
           G on holder core G, bf16 TILE) straight into the TILE CB (no DRAM x_ext gather, no tilize on the compute);
  lever 3  PMK_MCAST     an expert with >= 2 M groups is streamed ONCE from DRAM by the first core of a rectangle of G
           cores and multicast to the others (7a's PfStream credit / VALID protocol, kernels/pfs lineage): the weight
           re-read factor becomes ceil(ng / G) / ng;
  lever 1  the per-row combine in L1: NOT buildable at any bucket on this chip with a fixed-order fp32 sum -- the
           capacity arithmetic is `combine_l1_capacity` (the README's proof); the v2 combine keeps v1's DRAM out_rm
           transposition (bf16 rows written once, read once) and adds the shared partial as the last term;
  lever 4  the packed 8,192-row bucket: `Geometry(c=8192)` plans within the 1,026 KB budget (W=8,192 = rows);
  lever 5  PMK_SHARED (the P13 partial = the (K + 1)-th combine term, weight 1.0, summed last), PMK_IDS_U16 /
           PMK_WT_BF16 (the router tail's [C, K] uint16 ids / bf16 weights, router_tail/writer.cpp RTAIL_WEIGHTS);
           P13 itself = two PfStreamPhases (7a's PfStreamMM: gate_up, then down) on the members -> y_s (`SharedP13`).

Pure-python: Geometry / cb_plan2 / streaming_model2 / lever_table / combine_l1_capacity / rectangles (the budget test and
the README tables). Device: PmkMoe2 (a PmkMoe whose parts() adds the v2 args, defines, CBs and semaphores)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as V1
from models.autoports.zai_org_glm_5_3_flash.tt.pmk_moe import (  # noqa: F401  (re-exported for the tests)
    CB_BUDGET,
    PAGE,
    TILE,
    WTB_BFP4,
    Geometry,
    lcm,
    plan_words,
    r16,
)

PMK_MOE2_REV = 5
XS_BYTES_PER_HOLDER = 32 * 4096 * 2  # one 32-row block of x-hat_f, bf16 TILE (the family's XS 35)
MEMBER_SPARE_B = 22_528  # family README: the TP8 member's spare below the CB end
# the family member's CBs IDLE during P12-P14 (family README, member table; XS 4/35/40, the residual 2, PRE 7, COEF 24,
# CROW 25 and PCOEF 23's sub 1 stay LIVE): GATHER 0 (64 KB) + MSG 1 (0.5) + W/PART 5/41 (64) + PART 6 (8) + H/OUT 8/42
# (64) + S2 9 (4) + INV 10 (4) + OUT 11 (32) + POST Y 21 (64) + Y2 22 (8) + YS 26 (8) + post OUT 27 (8) + PARK 29 (102:
# the v2 form streams its own rings, PARK is free) + ARGS 30 (1) = 431.5 KB
MEMBER_ALIASABLE_B = sum(kb * 1024 for kb in (64, 0.5, 64, 8, 64, 4, 4, 32, 64, 8, 8, 8, 102, 1))
HIFI_CYC = {"LoFi": 16, "HiFi2": 32, "HiFi3": 48, "HiFi4": 64}
FFN_FID_VALUE = {
    "LoFi": 0,
    "HiFi2": 2,
    "HiFi3": 3,
    "HiFi4": 4,
}  # ckernel::MathFidelity (PMK_FFN_FID, as pmk_layer.FID_VALUE)


@dataclass(frozen=True)
class Levers:
    x_l1: bool = True  # lever 2
    mcast: int = 1  # lever 3: rectangle size G (1 = off)
    shared: bool = False  # lever 5: the P13 partial as the (K + 1)-th combine term
    ids_u16: bool = False  # lever 5: the router tail's layouts
    comb_l1: bool = False  # lever 1 (infeasible: the ledger shows why; kept as a switch for the arithmetic)
    hist_sliced: bool = False  # open item: plan_dest / plan_runs read hist [P, EP] in 8-row slices (-(P - 8) EP words)
    out_bfp8: bool = False  # PMK_OUT_BFP8: expert partial rows as row-bfp8 (64 + 32 NC bytes per N chunk), the
    #                         combine rebuilds bfp8 tiles and COL-broadcasts the weights (README sec 9); bf16 = reference
    yw_db: bool = False  # PMK_YW_DB (PERF.md sec 9): the writer re-lays a whole 32-row block into one of two block
    #                         scratches and pops CB_YRM before the DRAM writes land (the down-proj credit return path)
    yw_coal: int = 1  # PMK_YW_COAL (PERF.md sec 10, with yw_db): N chunks coalesced per row write (C x RB bytes)
    ring_slots: int = 2  # PMK_RING_SLOTS (PERF.md sec 11, perf5): chunks per WGU / WD ring on both sides (2 | 4)
    gu_split: bool = (
        False  # PMK_GU_SPLIT (PERF.md sec 12, perf5): gate|up columns split across the rectangle's members,
    )
    #                         act slices sent to the row owners (CB_ACTS, SEM_ACT), down on the owners (x_l1 0 only)
    comb_split: bool = (
        False  # PMK_COMB_SPLIT (PERF.md sec 15): the combine's odd slots rebuilt by the BRISC (CIN2 / CW2),
    )
    #                           the y write on the NCRISC; + the combine probe
    yw_brisc: bool = (
        False  # PMK_YW_BRISC (PERF.md sec 16): the row-bfp8 re-lay on the reader BRISC (inside its waits), the
    )
    #                         writer only issues the row writes (with yw_db)
    # Stage 14 B2.3 (w8/progb_b23_design.md; the self-stream form, G = 1):
    valid: bool = False  # XFFN_VALID_ROWS at x_l1 0: the writer gathers only a group's valid rows (no padding rows)
    snake: bool = False  # XFFN_SNAKE: units dealt by cost class, snaking over the cores (the op path's served dealing)
    xs_rows: bool = False  # PMK_XS_ROWS: x-hat_f as ROW_MAJOR rows in the holders' L1 (32 rows x H bf16 = 256 KB per
    #                         holder), gathered by the writer's KC-tile row reads into CB_XRM instead of from x_ext (x_l1 0)
    comb_group: int = 1  # PMK_COMB_GROUP: tokens per read barrier in the bf16 combine gather (1 / 2 / 4)
    comb_slot: int = 0  # PMK_COMB_SLOT: CIN pages per token slot (0 = KK x PT; 16 pads the K + 1 = 9 terms so that the
    #                         slots tile the idle XRM alias group at x_l1 0)
    comb_bufs: int = 2  # CIN / CW capacity in token groups (2 = the reader fills one group while the compute runs one)
    out_tm: bool = (
        False  # PMK_OUT_TM: token-major out_rm -- page (part, token) = the token's K row-parts contiguous (the
    )
    #                         plan's disp carries the pair index; the writer writes each 512-B segment at slot j; the
    #                         combine reads ONE K x PART block per token-part); self-stream bf16 form with VALID


BFP8_TILE_B, ROWB_EXP_PAD = 1088, 64
LPT_NG_MAX = 1024  # moe_common.hpp LPT_NG_MAX


def lpt_scratch_bytes(ep: int) -> int:
    """moe_common.hpp lpt_scratch_bytes: gtab[32] u32 | ng[EP] u16 | order[EP] u16 | hist[NG_MAX + 2] u16 | load[32] u16 |
    owner[EP] u8"""
    return 256 + 2 * ep + 2 * ep + 2 * (LPT_NG_MAX + 2) + 128 + ep


SPLIT_RMAX = 2  # PMK_GU_SPLIT_RMAX: row tiles per stream at most (XT = RMAX x KC tiles; 2 keeps the pair's XT at +8 KB)


def split_rows(g: int, gu_t: int, suba: int, gupt: int, rmax: int = SPLIT_RMAX) -> int:
    """moe_common.hpp split_rows: row tiles per stream of a g-member rectangle = min(g, RMAX, GUP tiles / the widest
    column slice); split_cols gives member pos its slice (c0, cw) in SUBA-tile units"""
    units = gu_t // suba
    cwmax = -(-units // g) * suba
    return max(1, min(g, rmax, gupt // cwmax))


def split_cols(pos: int, g: int, gu_t: int, suba: int) -> tuple:
    units = gu_t // suba
    return (pos * units // g) * suba, ((pos + 1) * units // g - pos * units // g) * suba


def stream_assignment(cnt, g_sizes, mg: int = 1, mode: int = 2, weights=None) -> tuple:
    """the host reference of moe_common.hpp stream_lpt (PERF.md sec 11): expert -> streamer from the live counts.
    cnt: rows per expert; g_sizes: the 24 rectangles' member counts; mode 1 = the expert's bank's 3 streamers
    (s & 7 == e & 7), 2 = any streamer; 0 = the static map (e mod 8) + 8 ((e / 8) mod 3). -> (owner [E] (None =
    unserved), load [NS] in streams)"""
    ns = len(g_sizes)
    w = (
        [256] * ns if weights is None else [int(round(float(v) * 256)) for v in weights]
    )  # uint16 fixed point (1.0 = 256)
    ng = [(((m + 31) // 32) + mg - 1) // mg if m else 0 for m in cnt]
    ng = [min(v, LPT_NG_MAX) for v in ng]
    owner, load = [None] * len(cnt), [0] * ns
    if mode == 0:
        for e, v in enumerate(ng):
            if v:
                s = (e % 8) + 8 * ((e // 8) % 3)
                owner[e] = s
                load[s] += -(-v // g_sizes[s]) * w[s]
        return owner, load
    order = sorted((e for e in range(len(cnt)) if ng[e]), key=lambda e: (-ng[e], e))
    for e in order:
        best, bs = None, 0
        for s in range(ns):
            if mode == 1 and (s & 7) != (e & 7):
                continue
            c = load[s] + -(-ng[e] // g_sizes[s]) * w[s]
            if best is None or c < best:
                best, bs = c, s
        owner[e] = bs
        load[bs] = best
    return owner, load


def rowb_bytes(h_t: int, nc: int) -> int:
    """one row-bfp8 partial row: (H_T / NC) chunks of [64 B exponent pairs][32 NC B mantissas]"""
    return (h_t // nc) * (ROWB_EXP_PAD + 32 * nc)


# ---------------------------------------------------------------------------------------------- CB ledger (v2)
def cb_plan2(g: Geometry, mg: int, kc: int, nc: int, lv: Levers = Levers(), standalone: bool = True,
             plan_p: int = 0, plan_nb: int = 1) -> dict:  # fmt: skip
    """the v2 CB allocations with the aliases (one allocation, several indices), as cb_plan: with lever 2 CB_XRM is
    gone and CB_XT is double-buffered (the writer fills chunk k + 1 while the compute runs k); CB_TAB carries the 64 B
    of multicast sync words; the combine's CIN / CW hold K + 1 terms with the shared partial; PMK_IDS_U16 needs 4 CHUNK
    + 16 words of plan scratch. `standalone`: the XS store (256 KB on the holders) is this program's own CB (the mock /
    bench); in the family it is XS 35 (already counted in the member census)."""
    ffn = V1.ffn_cb_bytes(g, mg, kc, nc)
    ffn["TAB"] += 64 + 64 + lpt_scratch_bytes(g.ep)  # the sync words; perf5: the PMK_STREAM_LPT scratch (64-B aligned)
    S = max(2, int(lv.ring_slots))  # perf5: S-chunk multicast rings (v1 sizes them at 2)
    ffn["WGU"], ffn["WD"] = S * kc * 2 * g.w_t * WTB_BFP4, S * g.w_t * nc * WTB_BFP4
    gupt = mg * 2 * g.w_t  # the gate|up partial tiles (GUP); PMK_GU_SPLIT: R x cw <= gupt by split_rows
    if lv.gu_split:
        assert not lv.x_l1, "PMK_GU_SPLIT: the DRAM x gather path only"
        ffn["ACTS"] = (gupt // 2) * 2048  # this member's act slice, ONE block (aliased on COUT: idle during the FFN)
        ffn["XT"] = SPLIT_RMAX * kc * 2048  # the stream's R row tiles of one K chunk, tilized (one block)
        # round 2 (PERF.md sec 14): with R rows per chunk a 2-row XRM holds ONE chunk's rows, so the DRAM x gather
        # serialised with every gate|up chunk (c14: 15.8 us per chunk instead of ~9); two chunks of rows (+16 KB)
        ffn["XRM"] = 2 * SPLIT_RMAX * kc * 2048
    if lv.x_l1:
        ffn["XT"] = 2 * mg * kc * 2048
        ffn.pop("XRM")
    kk = g.k + (1 if lv.shared else 0)
    part = g.h * 2 // g.h_split
    pt = part // 2048
    cg, cslot, cbufs = int(lv.comb_group), int(lv.comb_slot) or kk * pt, int(lv.comb_bufs)  # B2.3 PMK_COMB_GROUP
    assert 32 % cg == 0 and cslot >= kk * pt and cbufs >= 1, (cg, cslot, cbufs)
    comb = dict(CIN=cbufs * cg * cslot * 2048, CW=cbufs * cg * kk * 4096, CIDX=V1.SYNC_WORDS_B + g.k * 32 * 4 + 4096,
                COUT=32 * pt * 2048, CRM=32 * pt * 4096)  # fmt: skip
    if lv.out_bfp8:
        nt = g.h_t // g.h_split  # the part's column tiles (the combine's unit output)
        assert nt % nc == 0 and nt % 4 == 0, (nt, nc)
        part_b = (nt // nc) * (ROWB_EXP_PAD + 32 * nc)
        comb = dict(CIN=2 * nt * BFP8_TILE_B, CW=2 * 4096, CIDX=V1.SYNC_WORDS_B + g.k * 32 * 4 + 4096 + 64 + 32 * part_b,
                    COUT=nt * 2048, CRM=nt * 4096)  # fmt: skip
        if lv.shared:
            comb["CSH"] = nt * 2048  # the P13 partial's tiles (bf16) for one unit
        if lv.comb_split:  # two single-block gather CBs (one per RISC) instead of CIN's two blocks; CW2 one weight page
            comb["CIN"], comb["CIN2"], comb["CW2"] = nt * BFP8_TILE_B, nt * BFP8_TILE_B, 4096
        cw = max(1, int(lv.yw_coal)) if lv.yw_db else 1
        assert (g.h_t // nc) % cw == 0, (g.h_t // nc, cw)
        ffn["YRM"] = 2 * cw * mg * nc * BFP8_TILE_B  # two groups of cw N chunks x mg rows (yw_db) / the v1 2 slots
        # ffn_writer.hpp's row-bfp8 assembly: 2 rows of RB past the WTAB words (64-B aligned) -- inside WTAB, not GUP;
        # yw_db: two group scratches of mg x 32 rows x cw RB
        ffn["WTAB"] += (
            64 + (2 * mg * 32 * cw if lv.yw_db else 2) * (ROWB_EXP_PAD + 32 * nc) + (64 if lv.yw_brisc else 0)
        )  # + the 2 descriptors
    pw = plan_words(replace(g, p=plan_p) if plan_p else g, plan_nb)  # PMK_PLAN_P / _RUNS_NB: the plan's words
    plan_w = max(pw["count"], pw["dest"], pw["runs"]) + (4 * pw["chunk"] + 16 if lv.ids_u16 else 0)
    if lv.hist_sliced:
        plan_w -= (g.p - 8) * g.ep  # the resident hist copy (127 KB at P = 110) becomes an 8-row window
    sizes = {**ffn, **comb, "PLAN": plan_w * 4}
    aliases = {"WGU": ("COUT", "PLAN"), "GUP": ("CRM",), "WD": ("CW",), "TAB": ("CIDX",)}
    aliases["XT" if lv.x_l1 else "XRM"] = ("CIN",)
    if lv.comb_split:
        # CW2 is allocated ALONE: an alias group shares its bytes, and CW (the NCRISC's even-slot weight tile) and CW2 (the
        # BRISC's odd-slot one) are both live in the combine -- round 3's c17 / c18 (PCC 0.985) had CW2 in the WD group
        # on top of CW: the two weight tiles overwrote each other and the BRISC's capacity zeroing wiped CW's pages
        aliases["YRM"] = (
            "CIN2",
        )  # YRM (FFN only; 2 x coal x NC bfp8 tiles = the same 1,088-B pages) carries the second block
    if "CSH" in comb and (not lv.gu_split or comb["CSH"] <= ffn["ACT"]):
        aliases["ACT"] = ("CSH",)  # ACT (the FFN's activation tiles) is idle in the combine; PMK_GU_SPLIT: CB_ACT must
        #                             stay exactly ONE block (its base never moves), so only when CSH fits inside it
    if lv.gu_split:
        aliases["ACTS"] = ("COUT",)  # the act slice block (FFN) and the combine's output block (same 2 KB pages)
    page_of = dict(
        PAGE,
        CSH=2048,
        ACTS=2048,
        CW2=4096,
        **({"YRM": BFP8_TILE_B, "CIN": BFP8_TILE_B, "CIN2": BFP8_TILE_B} if lv.out_bfp8 else {}),
    )
    nt = g.h_t // g.h_split
    blocks = V1.push_blocks(g, mg, kc, nc, kk=kk, out_bfp8=lv.out_bfp8, nt=nt)
    if not lv.out_bfp8 and (cg > 1 or cslot != kk * pt):  # B2.3: one push = a token group's slots / weights
        assert not lv.comb_split, "PMK_COMB_GROUP: the bf16-row gather only"
        blocks.update(CIN=cg * cslot * 2048, CW=cg * kk * 4096)
    if "CSH" in comb:
        blocks["CSH"] = nt * 2048
    if lv.comb_split:
        blocks["CIN2"], blocks["CW2"] = nt * BFP8_TILE_B, 4096
    if lv.gu_split:  # the writer pushes KC pages per row tile; XT / GUP / ACTS move as whole blocks
        blocks.update(XRM=kc * 2048, XT=SPLIT_RMAX * kc * 2048, GUP=gupt * 4096, ACTS=(gupt // 2) * 2048)
    groups = V1.alias_groups(sizes, aliases, page_of, blocks, list(ffn))
    if lv.mcast > 1:
        # lever 3's multicast rings hold EXACTLY two pushes: a receiver's slot is the streamer's get_write_ptr
        # (ffn_reader.hpp: every unit pushes an even chunk count on a 2-slot ring) or (n - 1) mod 2 x the chunk past the
        # published base (program B's exp_streamer.hpp). PLAN over WGU rounded TP8's ring to 3 slots (221,184 = 3 x
        # 73,728): misaligned slots on the receivers. Then PLAN (dead after the plan barriers, as over WGU) goes over
        # GUP; a ring still rounded past two pushes leaves its group (allocated alone). (TP2's WGU is already 2 pushes.)
        cap_of = lambda gs, m: next(gr[1] for gr in gs if m in gr[0])  # noqa: E731
        if cap_of(groups, "WGU") != S * blocks["WGU"]:
            aliases.update(WGU=("COUT",), GUP=("CRM", "PLAN"))
            groups = V1.alias_groups(sizes, aliases, page_of, blocks, list(ffn))
        for ring in ("WGU", "WD"):
            if cap_of(groups, ring) != S * blocks[ring]:
                for name, al in list(aliases.items()):
                    aliases[name] = tuple(m for m in al if m != ring)
                aliases.pop(ring, None)
                groups = V1.alias_groups(sizes, aliases, page_of, blocks, list(ffn))
        for ring in ("WGU", "WD"):
            assert cap_of(groups, ring) == S * blocks[ring], (ring, cap_of(groups, ring), blocks[ring])
    bad = V1.check_pushes(groups, blocks)
    assert not bad, bad
    moe = sum(t for _, t, _ in groups)
    xs = XS_BYTES_PER_HOLDER if ((lv.x_l1 or lv.xs_rows) and standalone) else 0
    return dict(groups=groups, total=moe + xs, moe=moe, xs=xs, ffn=ffn, comb=comb, plan=pw, blocks=blocks,
                member_fits=moe <= MEMBER_SPARE_B + MEMBER_ALIASABLE_B)  # fmt: skip


BLOCK_CANDIDATES = ((4, 8, 8), (2, 8, 8), (2, 4, 8), (1, 4, 8), (1, 4, 4), (1, 2, 4), (1, 1, 4))


def fit_levers(g: Geometry, lv: Levers, budget: int = CB_BUDGET, standalone: bool = True) -> Levers:
    """the lever set a STANDALONE program (the mock / bench: the XS store on every core) can afford: lever 2 is
    dropped (DRAM x gather) when no block candidate fits the budget with the 256 KB store -- the TP2 pair's 32-tile
    slices (GUP 262 KB at MG = 1) are that case; the family's member placement counts XS separately"""
    if not lv.x_l1:
        return lv
    blk = choose_blocks2(g, lv, budget, standalone=standalone)
    if cb_plan2(g, *blk, lv, standalone)["total"] <= budget:
        return lv
    return replace(lv, x_l1=False)


def choose_blocks2(g: Geometry, lv: Levers = Levers(), budget: int = CB_BUDGET, mg_max: int = 4, member: bool = False,
                   standalone: bool = True) -> tuple:  # fmt: skip
    """(MG, KC, NC): the widest candidate whose v2 CBs fit `budget` (member=True: the family member's spare +
    aliasable bytes with XS live, i.e. the production placement on the 64 members)"""
    for m_, k_, n_ in BLOCK_CANDIDATES:
        if m_ > mg_max or g.h_t % k_ or g.h_t % n_ or (g.h_t // k_) % lv.ring_slots or (g.h_t // n_) % lv.ring_slots:
            continue
        p = cb_plan2(g, m_, k_, n_, lv, standalone)
        if p["member_fits"] if member else p["total"] <= budget:
            return m_, k_, n_
    return 1, 1, 4


# ---------------------------------------------------------------------------------------------- streaming (v2)
def streaming_model2(g: Geometry, mg: int, lv: Levers = Levers(), fid: str = "HiFi2", dram_gbps: float = 480.0,
                     mhz: float = 1350.0, rows_per_expert=None, noc_gbps_core: float = 60.0) -> dict:  # fmt: skip
    """per chip, per layer: DRAM bytes split into the WEIGHT term (streamed once per unit, / G with the multicast) and
    the PER-ROW terms (x gather 8 x 8 KB, the out_rm round trip 8 x 16 KB, y 8 KB), the FPU floor, the MATH ceiling
    = FPU useful / max(DRAM, FPU, NoC gather) and the TRUE crossover (rows where FPU = DRAM; None when the per-row
    DRAM alone exceeds the per-row FPU: v1's formula assumed every byte fixed). rows_per_expert: real routing or None.
    """
    per_expert = g.h_t * 2 * g.w_t * WTB_BFP4 + g.w_t * g.h_t * WTB_BFP4
    if rows_per_expert is None:
        rows_per_expert = [g.n / g.e] * g.e
    ngs = [-(-(-(-r // 32)) // mg) if r else 0 for r in rows_per_expert]
    G = max(1, lv.mcast)
    streams = sum((ng if ng < 2 or G < 2 else -(-ng // G)) for ng in ngs)  # DRAM reads of a slice
    units = sum(ngs)
    tile_rows = sum(-(-r // 32) for r in rows_per_expert)
    rows = sum(rows_per_expert)
    weight_b = streams * per_expert
    x_dram_b = 0 if lv.x_l1 else rows * g.h * 2
    x_noc_b = rows * g.h * 2 if lv.x_l1 else 0  # L1 -> L1 over NoC1, spread over the P cores
    out_row_b = rowb_bytes(g.h_t, 8) if lv.out_bfp8 else g.h * 2  # bf16 8 KB; row-bfp8 5,120 B at NC = 8
    out_b = 0 if lv.comb_l1 else rows * out_row_b * 2
    ys_b = g.c * g.h * 2 if lv.shared else 0  # the P13 partial read by the combine
    y_b = g.c * g.h * 2
    dram_b = weight_b + x_dram_b + out_b + y_b + ys_b
    cyc = HIFI_CYC[fid]
    tmm_per_row = g.h_t * 2 * g.w_t + g.w_t * g.h_t
    useful_us = (g.n / 32) * tmm_per_row * cyc / g.p / mhz
    padded_us = tile_rows * tmm_per_row * cyc / g.p / mhz
    dram_us = dram_b / dram_gbps / 1e3
    noc_us = x_noc_b / g.p / noc_gbps_core / 1e3  # per-core ingress at the gather rate (32-B pieces: ~60 GB/s)
    span = max(dram_us, padded_us, noc_us)
    per_row_dram_us = (
        ((0 if lv.x_l1 else 8 * g.h * 2) + (0 if lv.comb_l1 else 8 * 2 * out_row_b) + g.h * 2) / dram_gbps / 1e3
    )
    per_row_fpu_us = (g.k / 32) * tmm_per_row * cyc / g.p / mhz
    fixed_us = weight_b / dram_gbps / 1e3
    cross = fixed_us / (per_row_fpu_us - per_row_dram_us) if per_row_fpu_us > per_row_dram_us else None
    return dict(per_expert_mb=per_expert / 1e6, units=units, streams=streams, reread_factor=streams / g.e,
                weight_mb=weight_b / 1e6, x_dram_mb=x_dram_b / 1e6, x_noc_mb=x_noc_b / 1e6, out_mb=out_b / 1e6,
                y_mb=(y_b + ys_b) / 1e6, dram_mb=dram_b / 1e6, dram_us=dram_us, noc_us=noc_us, fpu_useful_us=useful_us,
                fpu_padded_us=padded_us, span_floor_us=span, math_ceiling=useful_us / span,
                per_row_dram_us=per_row_dram_us, per_row_fpu_us=per_row_fpu_us, crossover_rows=cross,
                gflop=(g.n / 32) * tmm_per_row * 2 * 32**3 / 1e9)  # fmt: skip


def lever_table(c: int = 2048, w: int = 256, p: int = 110, fid: str = "HiFi2") -> list:
    """the README's before / after rows: (label, blocks, model) for v1, +lever 2, +lever 3, +lever 1 (hypothetical),
    at the given bucket"""
    g = Geometry(c=c, w=w, p=p, h_split=4)  # v2's combine units (sec 13)
    l1 = c <= 2048  # the x-hat_f store fits L1 (16 MB over 64 holders) only up to the 2,048 bucket
    rows = []
    for label, lv in (("v1 (DRAM x, DRAM out_rm)", Levers(x_l1=False)),
                      ("+ lever 2 (x from L1 XS)" if l1 else "  lever 2 n/a (x store 64 MB: DRAM)", Levers(x_l1=l1)),
                      ("+ lever 3 (mcast G=4)", Levers(x_l1=l1, mcast=4)),
                      ("+ bfp8 partials (PMK_OUT_BFP8)", Levers(x_l1=l1, mcast=4, out_bfp8=True)),
                      ("+ lever 1 (L1 combine: infeasible)", Levers(x_l1=l1, mcast=4, comb_l1=True)),
                      ("+ lever 1 + 2 (both infeasible at > 2,048)", Levers(x_l1=True, mcast=4, comb_l1=True))):  # fmt: skip
        blk = choose_blocks2(g, replace(lv, x_l1=l1))
        rows.append((label, blk, streaming_model2(g, blk[0], lv, fid)))
    return rows


def combine_l1_capacity(g: Geometry, cores: int = 110, l1_budget: int = CB_BUDGET) -> dict:
    """lever 1's capacity proof: a fixed-order (deterministic, batch-invariant) combine needs, per chip, the fp32 y
    accumulator of every row in flight (the expert-major sweep keeps EVERY row in flight for the whole phase: its 8
    pairs are spread over the 288 experts) plus one reserved landing row per token (the next expected slot, the
    deadlock-free minimum) -- against the chip's CB budget"""
    y_fp32 = g.c * g.h * 4
    landing = g.c * g.h * 2
    per_core = (y_fp32 + landing) / cores
    ffn_set = 350 * 1024  # the (2, 4, 8) v2 FFN set
    return dict(
        y_fp32_mb=y_fp32 / 1e6,
        landing_mb=landing / 1e6,
        per_core_kb=per_core / 1024,
        budget_kb=l1_budget / 1024,
        fits=per_core <= l1_budget,
        fits_with_ffn=per_core + ffn_set <= l1_budget,  # a flat 110-core program (the standalone bench)
        fits_member=per_core + ffn_set <= MEMBER_SPARE_B + MEMBER_ALIASABLE_B,
    )  # the family's members


# ---------------------------------------------------------------------------------------------- rectangles (lever 3)
def rectangles(cores: list, noc: dict, g_size: int, grid_x: int, grid_y: int, orient: str = "") -> list:
    """partition the logical cores into rectangles of g_size cores along the orientation that wastes fewer cores
    (vertical runs of a column, or horizontal runs of a row; orient = "v" / "h" forces one -- ProgramB tries the
    other when the first yields < 24 rectangles, e.g. G = 3 with the streamers in rows 8-9); each rectangle's NoC
    bounding box must hold exactly its worker cores (7a's PfStreamMM._rectangle rule). -> [[(x, y), ...], ...];
    cores left over stream their own units"""
    if g_size <= 1:
        return []
    cs = set(cores)
    waste_v = (grid_y % g_size) * grid_x
    waste_h = (grid_x % g_size) * grid_y
    out = []
    if orient == "v" or (not orient and waste_v <= waste_h):
        for x in range(grid_x):
            for y0 in range(0, grid_y - g_size + 1, g_size):
                out.append([(x, y) for y in range(y0, y0 + g_size)])
    else:
        for y in range(grid_y):
            for x0 in range(0, grid_x - g_size + 1, g_size):
                out.append([(x, y) for x in range(x0, x0 + g_size)])
    good = []
    for r in out:
        if not all(c in cs for c in r):
            continue
        v = [noc[c] for c in r]
        lx, ly, hx, hy = min(a for a, _ in v), min(b for _, b in v), max(a for a, _ in v), max(b for _, b in v)
        inside = {c for c in cores if lx <= noc[c][0] <= hx and ly <= noc[c][1] <= hy}
        if inside == set(r):
            good.append(r)
    return good


def cover_rectangles(cores: list, noc: dict, n: int, gmax: int, grid_x: int, grid_y: int, hard_max: int = 8) -> list:
    """PERF.md sec 7: partition the compute cores (every non-streamer core) into exactly n exact NoC boxes -- runs of
    consecutive cores along one axis -- so that as many as possible are math members: pieces of <= gmax cores
    (widened up to hard_max when n boxes cannot hold every core at gmax), the piece count raised from the minimum
    (ceil(L / g) per run) to n by splitting the largest pieces first; when the cores fragment into more than n runs
    the smallest runs stay idle (as in the box placement). Both orientations are tried; the one covering the most
    cores, then the narrowest size spread, wins. -> [[(x, y), ...], ...] (n boxes, sizes 1..hard_max)"""
    cs = set(cores)
    best = None
    for orient in ("v", "h"):
        runs = []
        lines = range(grid_x) if orient == "v" else range(grid_y)
        for a in lines:
            cells = [(a, b) if orient == "v" else (b, a) for b in (range(grid_y) if orient == "v" else range(grid_x))]
            run = []
            for c in cells + [None]:
                if c is not None and c in cs:
                    run.append(c)
                elif run:
                    runs.append(run)
                    run = []
        runs.sort(key=len, reverse=True)
        runs = runs[:n]  # more runs than boxes: the smallest stay idle
        for g in range(gmax, hard_max + 1):
            pieces = [-(-len(r) // g) for r in runs]
            if sum(pieces) > n:
                continue
            while sum(pieces) < n:  # split the run with the largest current piece size
                cand = [j for j in range(len(runs)) if pieces[j] < len(runs[j])]
                if not cand:
                    break
                i = max(cand, key=lambda j: (len(runs[j]) / pieces[j], len(runs[j])))
                pieces[i] += 1
            if sum(pieces) != n:
                continue
            boxes = []
            for r, k in zip(runs, pieces):
                q, rem = divmod(len(r), k)
                at = 0
                for j in range(k):
                    sz = q + (1 if j < rem else 0)
                    boxes.append(r[at : at + sz])
                    at += sz
            ok = True  # exactness: the NoC bounding box of each piece holds exactly its cores
            for b in boxes:
                v = [noc[c] for c in b]
                lx, ly, hx, hy = (
                    min(a for a, _ in v),
                    min(b_ for _, b_ in v),
                    max(a for a, _ in v),
                    max(b_ for _, b_ in v),
                )
                if {c for c in cores if lx <= noc[c][0] <= hx and ly <= noc[c][1] <= hy} != set(b):
                    ok = False
                    break
            if not ok:
                continue
            covered = sum(len(b) for b in boxes)
            spread = max(len(b) for b in boxes) - min(len(b) for b in boxes)
            key = (-covered, spread, g)
            if best is None or key < best[0]:
                best = (key, boxes)
            break  # the smallest g that fits this orientation
    return best[1] if best else []


def tp8_expectation(n_math: int, c_tile: float, rows: int = 2048, mg_reuse: int = 1, mhz: float = 1350.0,
                    weight_mb: float = 510.0, dram_gbps: float = 493.5) -> dict:  # fmt: skip
    """PERF.md sec 8: ms per layer per chip at TP8 for N math cores at c_tile cycles per (weight tile x row tile) with
    the unpacked weight tile reused across mg_reuse row tiles: products = row-tile slots (rows x 8 / 32) x tiles per
    expert; span = max(DRAM floor, products x c_tile / (N x mhz)). The 173-cycle budget = 493.5 GB/s / 110 cores at
    576 B per tile for ONE row tile per streamed tile."""
    tiles_per_expert = 128 * 16 + 8 * 128  # TP8: H_T x GU_T + W_T x H_T (bfp4, 576 B)
    products = (rows * 8 // 32) * tiles_per_expert
    dram_ms = weight_mb * 1e6 / (dram_gbps * 1e9) * 1e3
    fpu_ms = products * c_tile / (n_math * mhz * 1e6) * 1e3
    return dict(products=products, dram_ms=round(dram_ms, 3), fpu_ms=round(fpu_ms, 3), span_ms=round(max(dram_ms, fpu_ms), 3),
                c_tile_for_dram_floor=round(dram_ms * n_math * mhz * 1e3 / products, 1))  # fmt: skip


# ---------------------------------------------------------------------------------------------- the device program
class PmkMoe2(V1.PmkMoe):
    """v1's PmkMoe with the v2 levers: `levers`, the holders' XS store (standalone: filled from x [C, H] TILE by the
    NCRISC, barrier SEM_XS; in the family: the lanes' H8), the rectangles, the shared partial y_s, the tail layouts."""

    def __init__(self, mesh, *, levers: Levers = Levers(), cb_xs: int = 35, holders=None, ffn_fid: str = "", bw_stage: int = -1,
                 plan_early=None, **kw):  # fmt: skip
        kw.setdefault("h_split", 4)  # v2 default: 32-tile combine units (CRM 128 KB, COUT 64 KB)
        super().__init__(mesh, **kw)
        self.lv, self.cb_xs = levers, int(cb_xs)
        # Stage 14 B2.3 PMK_FFN_FID: the FFN matmuls at their own fidelity under the kernel's `fid` (the combine's): LoFi
        # is bitwise HiFi2 for bfp4 x bf16, but the combine's bf16 x weight multiplies at LoFi lose ~2 % of y's norm
        self.ffn_fid = str(ffn_fid or self.fid)
        assert self.ffn_fid in FFN_FID_VALUE, self.ffn_fid
        # the M-B1 bisect standalone (PMK_BW_STAGE: 0 = the launch alone, 1 = + the plan, 2 = + the FFN, -1 = off = + the
        # combine): traced replay differences split program B's time by phase (no output below the full program)
        self.bw_stage = int(bw_stage)
        # Stage 14 B2.3 step (2) PMK_PLAN_EARLY (fused window only, kernels/pmk/b_wrap_w.cpp): the plan runs on the first
        # plan_cores cores BEFORE B's hand-over (idle MM-only BRISCs under P13), after `sig_n` +1s on semaphore `sig` (the
        # members' router-tail writers: ids rows landed), barriers on semaphores `sem_a` / `sem_b` (ids A / P12 / P13 never
        # touch on those cores); B itself skips the plan (PMK_PLAN_SKIP). dict(sig, sig_n, sem_a, sem_b)
        self.plan_early = dict(plan_early) if plan_early else None
        if self.plan_early:
            assert self.plan_p < self.p, "PMK_PLAN_EARLY: the plan cores are a prefix of B's cores (plan_cores)"
        self.holders = [tuple(h) for h in holders] if holders else None  # the 64 x-hat_f holders (block G at [G])
        import ttnn

        grid = mesh.compute_with_storage_grid_size()
        self.grid = (int(grid.x), int(grid.y))
        self.noc_of = dict(zip(self.cores, self.noc))
        self.rects = rectangles(self.cores, self.noc_of, self.lv.mcast, *self.grid) if self.lv.mcast > 1 else []
        if self.rects:  # the lever-3 credit protocol, simulated before any program is built (7a's check_ring2)
            from models.autoports.zai_org_glm_5_3_flash.tt import pfs_credit_model

            _, kc_, nc_ = self.blocks(2048)
            pfs_credit_model.check_ring2(self.lv.mcast, self.h_t // kc_ + self.h_t // nc_, slots=self.lv.ring_slots)
        self._ttnn = ttnn

    def blocks(self, c: int) -> tuple:
        if self.blocks_explicit:
            return self.blocks_explicit
        return choose_blocks2(self.geometry(c), self.lv)

    def plan(self, c: int) -> dict:
        return cb_plan2(self.geometry(c), *self.blocks(c), self.lv, plan_p=self.plan_p if self.plan_p < self.p else 0,
                        plan_nb=self.plan_nb)  # fmt: skip

    def cb_descriptors(self, plan: dict) -> list:
        ttnn = self._ttnn
        fmt = dict(
            self.CB_FMT,
            CSH=ttnn.bfloat16,
            ACTS=ttnn.bfloat16,
            CW2=ttnn.float32,
            **({"YRM": ttnn.bfloat8_b, "CIN": ttnn.bfloat8_b, "CIN2": ttnn.bfloat8_b} if self.lv.out_bfp8 else {}),
        )
        cbs = []
        for members, total, pages in plan["groups"]:
            fds = [ttnn.CBFormatDescriptor(buffer_index=self.cb_index(m), data_format=fmt.get(m, ttnn.uint32),
                                           page_size=pages[m]) for m in members]  # fmt: skip
            cbs.append(ttnn.CBDescriptor(total_size=total, core_ranges=self.crs, format_descriptors=fds))
        return cbs

    def cb_index(self, name: str) -> int:
        if name == "CSH":
            return self.cb0 + 15
        if name == "ACTS":
            return self.cb0 + 16  # moe_common.hpp CB_ACTS (PMK_GU_SPLIT)
        if name == "CIN2":
            return self.cb0 + 17  # moe_common.hpp CB_CIN2 (PMK_COMB_SPLIT)
        if name == "CW2":
            return self.cb0 + 18
        return super().cb_index(name)

    def out_row_bytes(self) -> int:
        return rowb_bytes(self.h_t, self.blocks(2048)[2]) if self.lv.out_bfp8 else self.h * 2

    def l1_bytes(self, c: int) -> int:
        return self.plan(c)["total"]

    def xs_holders(self, c: int) -> list:
        n = c // TILE
        hs = self.holders or self.cores
        assert len(hs) >= n, (len(hs), n)
        return hs[:n]

    def parts(self, x_ext, ids_rm, wt, out_rm, y, c: int, *, x_tile=None, y_s=None, wait=None, h9=None,
              xs_base=None, window: bool = False) -> tuple:  # fmt: skip
        """as PmkMoe.parts, plus: x_tile [1, 1, C, H] bf16 TILE (standalone lever 2: the holders' XS fill source;
        the family passes None and lands XS itself), y_s [C, H] bf16 ROW_MAJOR (lever 5: the P13 partial).
        With PMK_IDS_U16 ids_rm is [1, C x K] uint16 ROW_MAJOR and wt [1, C x K] bf16 ROW_MAJOR, ONE page each (the
        router tail's rows, 16 B per token)."""
        ttnn = self._ttnn
        raw, cbs, sems, io = super().parts(x_ext, ids_rm, wt, out_rm, y, c, wait=wait, h9=h9)
        g = self.geometry(c)
        mg, kc, nc = self.blocks(c)
        plan = self.plan(c)
        lv = self.lv
        w_file, ct_w, rt_w, d_w, cfg_w = raw["w"]
        a_file, ct_a, rt_a, d_a, cfg_a = raw["a"]
        c_file, ct_c, rt_c, d_c, cfg_c = raw["c"]
        common = [("PMK_MOE2_REV", str(PMK_MOE2_REV)), ("PMK_MOE_CB_XS", str(self.cb_xs))]
        cfid = os.environ.get("GLM53_PMK_B_COMB_FID", "")  # the combine's own fidelity (comb_compute.hpp PMK_COMB_FID)
        if cfid:
            common.append(("PMK_COMB_FID", str({"LoFi": 0, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}[cfid])))
        if lv.x_l1:
            common += [("XFFN_X_TILED", "1"), ("XFFN_VALID_ROWS", "1")]
        else:  # B2.3: the x_l1 0 path's knobs (PMK_XS_ROWS gathers the valid rows only: a padding row has no holder)
            assert not (lv.xs_rows and lv.mcast > 1), "PMK_XS_ROWS: the self-stream form (G = 1)"
            if lv.valid or lv.xs_rows:
                common.append(("XFFN_VALID_ROWS", "1"))
            if lv.xs_rows:
                common.append(("PMK_XS_ROWS", "1"))
        if lv.snake:
            assert lv.mcast == 1, "XFFN_SNAKE deals the self-streamed (G = 1) units only"
            common.append(("XFFN_SNAKE", "1"))
        if self.bw_stage >= 0:
            common.append(("PMK_BW_STAGE", str(self.bw_stage)))
        if (
            self.plan_early and window
        ):  # (the fused window only: a standalone B -- e.g. the unit's control -- plans itself)
            pe = self.plan_early
            common += [("PMK_PLAN_EARLY", "1"), ("PMK_PLAN_SKIP", "1"), ("PMK_PLAN_SIG_SEM", str(int(pe["sig"]))),
                       ("PMK_PLAN_SIG_N", str(int(pe["sig_n"]))), ("PMK_PLAN_SEM_A", str(int(pe["sem_a"]))),
                       ("PMK_PLAN_SEM_B", str(int(pe["sem_b"])))]  # fmt: skip
        if lv.out_tm:
            assert (
                not lv.x_l1 and (lv.valid or lv.xs_rows) and lv.mcast == 1 and not lv.out_bfp8 and not lv.comb_split
            ), lv
            part_b = self.h * 2 // self.h_split
            common += [("PMK_OUT_TM", str(int(c))), ("PMK_TM_PART", str(part_b)), ("PMK_TM_K", str(self.k))]
        if lv.comb_group > 1 or lv.comb_slot:
            assert not lv.out_bfp8 and not lv.comb_split, "PMK_COMB_GROUP: the bf16-row gather only"
            common.append(("PMK_COMB_GROUP", str(int(lv.comb_group))))
            if lv.comb_slot:
                common.append(("PMK_COMB_SLOT", str(int(lv.comb_slot))))
        if lv.mcast > 1 and self.rects:
            common.append(("PMK_MCAST", "1"))
        if lv.shared:
            common.append(("PMK_SHARED", "1"))
        if lv.ids_u16:
            common += [("PMK_IDS_U16", "1"), ("PMK_IDS_K", str(self.k)), ("PMK_WT_BF16", "1")]
        if lv.out_bfp8:
            common += [("PMK_OUT_BFP8", "1"), ("PMK_OUT_NC", str(nc)), ("PMK_OUT_NT", str(self.h_t // self.h_split))]
            modes = list(cfg_c.unpack_to_dest_mode)
            modes[self.cb1 + 1] = ttnn.UnpackToDestMode.UnpackToDestFp32  # CB_CRM: the fp32 -> bf16 copy pass is exact
            cfg_c.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        # comb_reader zeroes every page of CB_CW once: its REAL capacity = the CW alias group's pages (16, 2026-10-22)
        cw_pages = next(t // pg["CW"] for mem, t, pg in plan["groups"] if "CW" in mem)
        common = common + [("PMK_CW_PAGES", str(cw_pages))]
        if lv.yw_db:
            assert lv.out_bfp8, "PMK_YW_DB is the row-bfp8 writer's knob"
            common = common + [("PMK_YW_DB", "1"), ("PMK_YW_COAL", str(max(1, int(lv.yw_coal))))]
            if lv.yw_brisc:
                common = common + [("PMK_YW_BRISC", "1")]
        d_w, d_a, d_c = list(d_w) + common, list(d_a) + common, list(d_c) + common
        if self.ffn_fid != self.fid:
            d_c.append(("PMK_FFN_FID", str(FFN_FID_VALUE[self.ffn_fid])))
        # ---- BRISC: the rectangle args (PMK_RT_MC)
        rect_of = {}
        for ri, r in enumerate(self.rects):
            for pos, core in enumerate(r):
                rect_of[core] = (ri, pos)
        nr = len(self.rects)
        for xy in self.cores:
            base = len(rt_w[xy])
            if xy in rect_of:
                ri, pos = rect_of[xy]
                r = self.rects[ri]
                v = [self.noc_of[c_] for c_ in r]
                sx, sy = self.noc_of[r[0]]
                mc = [
                    len(r),
                    ri,
                    pos,
                    nr,
                    sx,
                    sy,
                    min(a for a, _ in v),
                    min(b for _, b in v),
                    max(a for a, _ in v),
                    max(b for _, b in v),
                ]  # fmt: skip  # G = THIS rectangle's size
            else:
                mc = [lv.mcast, nr, 0, nr, 0, 0, 0, 0, 0, 0]  # ri = NR: never a rectangle member
            rt_w[xy] = rt_w[xy] + mc
            d_w_core = base
        d_w.append(("PMK_RT_MC", str(d_w_core)))
        # ---- NCRISC: the XS holders (PMK_RT_XS), the standalone fill (PMK_RT_XSF), the shared partial (PMK_RT_YS)
        holders = self.xs_holders(c) if (lv.x_l1 or lv.xs_rows) else []
        hidx = {h: i for i, h in enumerate(holders)}
        # the standalone fill source: lever 2 = x [C, H] TILE (x_tile); PMK_XS_ROWS = x_ext's rows (one row per page)
        fill_src = x_ext if lv.xs_rows else x_tile
        for xy in self.cores:
            a = list(rt_a[xy])
            b_xs = len(a)
            a += [0, len(holders)] + [v for h in holders for v in self.noc_of[h]]  # [0] = xs base: patched below
            b_xsf = len(a)
            a += [hidx.get(xy, 0xFFFFFFFF), fill_src.buffer_address() if fill_src is not None else 0, self.h_t]
            b_ys = len(a)
            a += [y_s.buffer_address() if y_s is not None else 0]
            rt_a[xy] = a
        d_a += [("PMK_RT_XS", str(b_xs)), ("PMK_RT_XSF", str(b_xsf)), ("PMK_RT_YS", str(b_ys))]
        if lv.x_l1 and x_tile is not None:
            d_a.append(("PMK_XS_FILL", "1"))
            io.append(x_tile)
        if lv.xs_rows and xs_base is None:  # standalone: the holders fill their row blocks from x_ext (already in io)
            d_a.append(("PMK_XS_FILL", "1"))
        if lv.shared:
            assert y_s is not None, "lever 5 needs the P13 partial y_s"
            io.append(y_s)
        # ---- CBs: the v2 plan (+ the XS store on the holders, standalone); semaphores 9..11
        cbs = self.cb_descriptors(plan)
        if lv.x_l1 or lv.xs_rows:
            # the XS store: in the family it is XS 35 on the 64 members (the holders) and the family passes its address
            # (xs_base); standalone every core of the set allocates it (256 KB), so the kernel takes its own CB address
            crs_xs = (
                self.crs
                if xs_base is None
                else ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in holders])
            )
            cbs.append(ttnn.CBDescriptor(total_size=XS_BYTES_PER_HOLDER, core_ranges=crs_xs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=self.cb_xs, data_format=ttnn.bfloat16, page_size=2048)]))  # fmt: skip
        for xy in self.cores:
            rt_a[xy][b_xs] = int(xs_base or 0)
        for sid in (9, 10, 11):
            sems.append(ttnn.SemaphoreDescriptor(id=sid, core_ranges=self.crs, initial_value=0))
        raw = dict(
            w=(w_file, ct_w, rt_w, d_w, cfg_w), a=(a_file, ct_a, rt_a, d_a, cfg_a), c=(c_file, ct_c, rt_c, d_c, cfg_c)
        )
        return raw, cbs, sems, io

    def scratch(self, c: int) -> dict:
        key = ("scratch", c)
        if (
            key not in self.bufs and self.lv.out_tm
        ):  # PMK_OUT_TM: [H_SPLIT x C pages of K x PART bytes] (the same bytes)
            ttnn = self._ttnn
            self.bufs[key] = dict(
                out_rm=ttnn.empty((self.h_split * c, self.k * self.h // self.h_split), dtype=V1.BF16,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=V1.DRAM),
                y=ttnn.empty((1, 1, c, self.h), dtype=V1.BF16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=V1.DRAM))  # fmt: skip
        if key not in self.bufs and self.lv.out_bfp8:  # out_rm as row-bfp8 bytes: a bf16-typed [CAP, ROWB / 2] page row
            ttnn = self._ttnn
            g = self.geometry(c)
            self.bufs[key] = dict(
                out_rm=ttnn.empty((g.cap, rowb_bytes(self.h_t, self.blocks(c)[2]) // 2), dtype=V1.BF16,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=V1.DRAM),
                y=ttnn.empty((1, 1, c, self.h), dtype=V1.BF16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=V1.DRAM))  # fmt: skip
        s = super().scratch(c)
        if self.lv.shared and "y_s" not in s:
            ttnn = self._ttnn
            if self.lv.out_bfp8:  # the PfStream pair's native output: bf16 TILE [C, H]
                s["y_s"] = ttnn.empty((1, 1, c, self.h), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                      memory_config=V1.DRAM)  # fmt: skip
            else:
                s["y_s"] = ttnn.empty((c, self.h), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                      memory_config=V1.DRAM)  # fmt: skip
        return s

    def __call__(self, x_ext, ids_rm, wt, c: int, *, x_tile=None, **kw):
        s = self.scratch(c)
        prog, io = self.program(x_ext, ids_rm, wt, s["out_rm"], s["y"], c, x_tile=x_tile, y_s=s.get("y_s"), **kw)
        self._ttnn.generic_op(io, prog)
        self.calls += 1
        return s["y"]


def tail_layout_inputs(idx, sel, scaling: float):
    """lever 5: the ROUTER TAIL's layouts from the op path's idx / sel (the bench feeds the tail-layout plan from the
    same routing): ids [1, C x K] uint16 ROW_MAJOR, w [1, C x K] bf16 ROW_MAJOR (bf16_rne(s_j scaling / (sum s + eps)):
    router_tail/writer.cpp's RTAIL_WEIGHTS formula, so the bench's weights are the tail's to the bit). ONE page each:
    the plan / combine readers address the tensor as page 0 (page_size = C x 2 x K); [C, K] would be C pages (W7 s3w)"""
    import ttnn

    c, k = int(idx.shape[-2]), int(idx.shape[-1])
    # (ttnn.reshape takes no uint16: through uint32 -> one ROW_MAJOR row -> tile -> uint16 -> the one-page row)
    i32 = ttnn.typecast(idx, ttnn.uint32)
    r32 = ttnn.to_layout(ttnn.reshape(i32, (c, k)), ttnn.ROW_MAJOR_LAYOUT, memory_config=V1.DRAM)
    ttnn.deallocate(i32)
    row = ttnn.to_layout(ttnn.reshape(r32, (1, c * k)), ttnn.TILE_LAYOUT, memory_config=V1.DRAM)
    ttnn.deallocate(r32)
    t16 = ttnn.typecast(row, ttnn.uint16)
    ttnn.deallocate(row)
    ids = ttnn.to_layout(t16, ttnn.ROW_MAJOR_LAYOUT, memory_config=V1.DRAM)
    ttnn.deallocate(t16)
    w = ttnn.multiply(sel, ttnn.reciprocal(ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-20)))
    w = ttnn.typecast(ttnn.multiply(w, scaling), ttnn.bfloat16)
    w = ttnn.to_layout(ttnn.reshape(w, (1, c * k)), ttnn.ROW_MAJOR_LAYOUT, memory_config=V1.DRAM)
    return ids, w


# ---------------------------------------------------------------------------------------------- P13 (lever 5)
class SharedP13:
    """the shared expert's 1/TP slice as TWO PfStreamPhases (7a's tt/pf_stream.PfStreamMM, the sec-4c body: members
    activation-stationary, weights streamed once and multicast by the bank streamers): gu = x-hat_f @ W_gu [H, 2 W_s]
    (bfp8) -> SwiGLU -> act [C, W_s]; down = act @ W_d [W_s, H] -> y_s [C, H]. The partial y_s is the combine's last
    term (PMK_SHARED). Host-side plan only in this piece: the SwiGLU between the two streams is a member compute
    epilogue the pfs kernels do not have yet (README sec 6: open item for 7a); the mock builds both stream programs."""

    def __init__(self, mesh, w_gu, w_d, *, members, streamers, nb: int = 4, nw: int = 32, kb: int = 8, park: int = 4):
        import ttnn
        from models.autoports.zai_org_glm_5_3_flash.tt import pf_stream as PFS

        self.gu = PFS.PfStreamMM(mesh, w_gu, members=members, streamers=streamers, nb=nb, nw=nw, kb=kb, park=park,
                                 w_dtype=ttnn.bfloat8_b)  # fmt: skip
        kt_d = int(w_d.shape[0]) // 32  # the down stream's K = W_s (8 tiles at TP8): an even number of K-blocks
        kb_d = min(kb, kt_d // 2)  # per N-block (7a's lane parity), so KB = 4 at TP8
        self.dn = PFS.PfStreamMM(mesh, w_d, members=members, streamers=streamers, nb=nb, nw=nw, kb=kb_d, park=park,
                                 w_dtype=ttnn.bfloat8_b, sync=self.gu.sync, slots=2)  # fmt: skip

    def footprint(self) -> dict:
        return dict(gu=self.gu.footprint(), dn=self.dn.footprint())

    def programs(self, x_gu, x_act):
        """the two standalone PfStream programs (the mock builds them; the layer runs them as phases of the members'
        composite with the SwiGLU epilogue between)"""
        return self.gu.program(x_gu), self.dn.program(x_act)


# ---------------------------------------------------------------------------------------------- the stream-rate probe
def stream_rate(probe_chips: list, mhz: float = 1350.0) -> list:
    """per chip, from PmkMoe.probe_read() (kernels/prefill/moe_xffn_probe.hpp rows: reader slots R_GU_READ 0,
    R_DN_READ 1, R_KB 5, span 13, start 14, end 15): the ACHIEVED expert-weight streaming rate = bytes streamed by
    every core / the stream phase's span across the cores (first start .. last end: the number Stage 16 compares with
    the ~480-512 GB/s GDDR6 peak), the rate while reads are in flight (bytes / sum of read-active cycles per core,
    averaged: the per-core NoC-request-bound rate), and the read-active fraction of the span"""
    out = []
    for ch in probe_chips:
        # program B (ProgramB.probe_extra_rows): the 24 streamer rows hold the DRAM bytes (the receivers read none:
        # their R_KB stays 0 -- the 18:25Z "0 B" probe); otherwise the readers are the streamers (v1 / v2)
        src = "streamer" if "streamer" in ch else "reader"
        r = ch[src].to(dtype=__import__("torch").float64)
        kb, span = r[:, 5], r[:, 13]
        start, end = r[:, 14], r[:, 15]
        active = r[:, 0] + r[:, 1]
        if float(span.max()) <= 0:
            out.append(dict(mock=True))
            continue
        phase_cyc = float((end - start.min()).max())  # low 32 bits of the clock: one launch never wraps 3 s
        total_b = float(kb.sum()) * 1024
        gbps_span = total_b / (phase_cyc / mhz / 1e6) / 1e9
        per_core = (kb * 1024) / (active / mhz / 1e6).clamp(min=1e-9) / 1e9
        per_core_span = (kb * 1024) / (span / mhz / 1e6).clamp(min=1e-9) / 1e9  # each core over its own span
        out.append(dict(mock=False, source=src, bytes_mb=round(total_b / 1e6, 1), span_us=round(phase_cyc / mhz, 1),
                        dram_gbps_span=round(gbps_span, 1), dram_gbps_active_core_mean=round(float(per_core.mean()), 2),
                        active_fraction=round(float((active / span.clamp(min=1)).mean()), 3),
                        core_gbps_min=round(float(per_core_span.min()), 2), core_gbps_max=round(float(per_core_span.max()), 2)))  # fmt: skip
    return out


# the streamer row's slots (kernels/pmk/exp_streamer.hpp exps::S_*)
S_GU_READ, S_DN_READ, S_GU_CREDIT, S_DN_CREDIT, S_STREAMS, S_KB, S_MC_ISSUE, S_MC_FLUSH = 0, 1, 2, 3, 4, 5, 6, 7
S_GU_CHUNKS, S_DN_CHUNKS, S_PLAN_WAIT, S_MC_ACK, S_RD_ISSUE, S_SPAN, S_START, S_END = 8, 9, 10, 11, 12, 13, 14, 15


def progb_scoreboard(ch: dict, mhz: float = 1350.0) -> dict:
    """program B's credit / wait scoreboard of one chip from PmkMoe.probe_read() (PERF.md sec 3): fractions of each
    role's span (mean over its cores) -- streamer: DRAM read-active (issue .. barrier; residual barrier wait with
    PMK_STREAM_PREFETCH), credit wait (READY >= G n: the members have not freed the slot), multicast flush wait (the
    previous multicast still leaving the NIU), plan wait, final ack wait; member reader: data wait (credit sent ..
    VALID: the streamer is late) and slot wait (cb_reserve_back: the compute has not popped), member writer: the x
    gather, the y write, the YRM wait. A fraction near 1 on ONE side names the bottleneck: streamer credit wait ~ 1
    + member slot wait ~ 1 = compute-bound members; member data wait ~ 1 + streamer read ~ 1 = DRAM / streamer-bound;
    streamer flush ~ 1 = multicast-bound (NoC); writer x gather ~ 1 with member slot wait ~ 1 = the gather starves
    the compute."""
    torch = __import__("torch")
    out = {}
    f = lambda num, den: round(float((num / den.clamp(min=1)).mean()), 3)  # noqa: E731
    if "streamer" in ch:
        st = ch["streamer"].to(torch.float64)
        sp = st[:, S_SPAN]
        rows = st[sp > 0]
        if rows.shape[0]:
            sp = rows[:, S_SPAN]
            kb = rows[:, S_KB]
            busy = (rows[:, S_GU_READ] + rows[:, S_DN_READ] + rows[:, S_GU_CREDIT] + rows[:, S_DN_CREDIT] + rows[:, S_MC_ISSUE]
                    + rows[:, S_MC_FLUSH] + rows[:, S_PLAN_WAIT] + rows[:, S_MC_ACK] + rows[:, S_RD_ISSUE])  # fmt: skip
            strm = rows[:, S_STREAMS] % 65536
            out["streamer"] = dict(
                cores=int(rows.shape[0]), prefetch=int(rows[0, S_STREAMS]) >> 16, streams=int(strm.sum()),
                # perf5 (PERF.md sec 11): the per-streamer spread -- the phase ends with the SLOWEST streamer
                span_us_min=round(float(sp.min()) / mhz, 1), span_us_max=round(float(sp.max()) / mhz, 1),
                span_imbalance=round(float(sp.max() / sp.mean()), 3), streams_min=int(strm.min()), streams_max=int(strm.max()),
                mb_min=round(float(kb.min()) * 1024 / 1e6, 1), mb_max=round(float(kb.max()) * 1024 / 1e6, 1),
                mcast_issue=f(rows[:, S_MC_ISSUE], sp), read_issue=f(rows[:, S_RD_ISSUE], sp), unattributed=round(1 - float((busy / sp).mean()), 3),
                chunks_gu=int(rows[:, S_GU_CHUNKS].sum()), chunks_dn=int(rows[:, S_DN_CHUNKS].sum()),
                mb=round(float(kb.sum()) * 1024 / 1e6, 1), span_us_mean=round(float(sp.mean()) / mhz, 1),
                read_active=f(rows[:, S_GU_READ] + rows[:, S_DN_READ], sp), read_gu=f(rows[:, S_GU_READ], sp),
                read_dn=f(rows[:, S_DN_READ], sp), credit_wait=f(rows[:, S_GU_CREDIT] + rows[:, S_DN_CREDIT], sp),
                credit_gu=f(rows[:, S_GU_CREDIT], sp), credit_dn=f(rows[:, S_DN_CREDIT], sp),
                mcast_flush_wait=f(rows[:, S_MC_FLUSH], sp), plan_wait=f(rows[:, S_PLAN_WAIT], sp),
                ack_wait=f(rows[:, S_MC_ACK], sp),
                gbps_per_core_mean=round(float(((kb * 1024) / (sp / mhz / 1e6)).mean()) / 1e9, 2))  # fmt: skip
    rd = ch["reader"].to(torch.float64)
    rd = rd[rd[:, 4] > 0]  # members with units (R_UNITS)
    if rd.shape[0]:
        sp = rd[:, 13]
        out["member_reader"] = dict(cores=int(rd.shape[0]), units=int(rd[:, 4].sum()),
                                    data_wait=f(rd[:, 0] + rd[:, 1], sp), data_wait_gu=f(rd[:, 0], sp), data_wait_dn=f(rd[:, 1], sp),
                                    slot_wait=f(rd[:, 2] + rd[:, 3], sp), slot_wait_gu=f(rd[:, 2], sp), slot_wait_dn=f(rd[:, 3], sp),
                                    relay=f(rd[:, 7], sp),  # PMK_YW_BRISC: the re-lay rows done inside the waits
                                    span_us_mean=round(float(sp.mean()) / mhz, 1))  # fmt: skip
    wr_all = ch["writer"].to(torch.float64)
    cb = wr_all[wr_all[:, 11] > 0]  # the combine probe (comb_reader.hpp XFFN_PROBE slots 8-12)
    if cb.shape[0]:
        csp = cb[:, 11]
        out["combine"] = dict(cores=int(cb.shape[0]), gather=f(cb[:, 8], csp), rebuild=f(cb[:, 9], csp), y=f(cb[:, 10], csp),
                              span_us_mean=round(float(csp.mean()) / mhz, 1), span_us_max=round(float(csp.max()) / mhz, 1),
                              start_after_ffn_start_us=round(float(((cb[:, 12] - cb[:, 14]) % 2**32).mean()) / mhz, 1))  # fmt: skip
        rb = ch["reader"].to(torch.float64)
        rb = rb[rb[:, 11] > 0]
        if rb.shape[0]:
            out["combine"]["brisc_half"] = dict(
                gather=f(rb[:, 8], rb[:, 11]),
                rebuild=f(rb[:, 9], rb[:, 11]),
                span_us_mean=round(float(rb[:, 11].mean()) / mhz, 1),
            )
    wr = wr_all[wr_all[:, 7] > 0]  # writers that wrote rows (W_Y_KB)
    if wr.shape[0]:
        sp = wr[:, 13]
        out["member_writer"] = dict(cores=int(wr.shape[0]), x_gather=f(wr[:, 0], sp), x_slot_wait=f(wr[:, 1], sp),
                                    y_wait=f(wr[:, 2], sp), y_write=f(wr[:, 3], sp), ctl_wait=f(wr[:, 4], sp),
                                    x_kb=int(wr[:, 6].sum()), y_kb=int(wr[:, 7].sum()), span_us_mean=round(float(sp.mean()) / mhz, 1))  # fmt: skip
    return out


def lpt_weights_from(spec):
    """the bench's lpt_w arg: a list of 24 numbers (us per chunk per streamer, any scale), or the path of a previous
    row's .out (its per_streamer us_per_chunk), normalised to the fastest streamer = 1.0; None / "" = unweighted"""
    if not spec:
        return None
    if isinstance(spec, str):
        import json

        o = json.load(open(spec))
        ps = (o.get("result") or o).get("per_streamer")
        vals = [d["us_per_chunk"] for d in sorted(ps, key=lambda d: d["s"])]
    else:
        vals = [float(v) for v in spec]
    lo = min(vals)
    return [v / lo for v in vals]


def progb_per_streamer(pm, ch: dict, mhz: float = 1350.0) -> dict:
    """perf5: the 24 streamer rows joined with their rectangles (size, position) and the members' reader waits per
    rectangle -- the per-rectangle RATE differences behind the span imbalance with balanced bytes (PERF.md sec 12)"""
    torch = __import__("torch")
    st = ch["streamer"].to(torch.float64)
    rd = ch["reader"].to(torch.float64)
    idx = {c: i for i, c in enumerate(pm.cores)}
    out = []
    for s_ in range(min(int(st.shape[0]), len(pm.rects))):
        r = st[s_]
        sp = max(float(r[S_SPAN]), 1.0)
        rect = pm.rects[s_]
        mem = rd[[idx[c] for c in rect]]
        busy = mem[mem[:, 4] > 0]
        out.append(dict(s=s_, g=len(rect), xy=pm.streamers[s_], rect0=rect[0], span_us=round(sp / mhz, 1), mb=round(float(r[S_KB]) * 1024 / 1e6, 1),
                        streams=int(r[S_STREAMS]) % 65536, chunks=int(r[S_GU_CHUNKS] + r[S_DN_CHUNKS]),
                        us_per_chunk=round(sp / mhz / max(1, int(r[S_GU_CHUNKS] + r[S_DN_CHUNKS])), 2),
                        issue=round(float(r[S_MC_ISSUE]) / sp, 3), credit=round(float(r[S_GU_CREDIT] + r[S_DN_CREDIT]) / sp, 3),
                        read=round(float(r[S_GU_READ] + r[S_DN_READ]) / sp, 3), flush=round(float(r[S_MC_FLUSH]) / sp, 3),
                        members_busy=int(busy.shape[0]), units=int(busy[:, 4].sum()) if busy.shape[0] else 0,
                        data_wait=round(float(((busy[:, 0] + busy[:, 1]) / busy[:, 13].clamp(min=1)).mean()), 3) if busy.shape[0] else None,
                        slot_wait=round(float(((busy[:, 2] + busy[:, 3]) / busy[:, 13].clamp(min=1)).mean()), 3) if busy.shape[0] else None))  # fmt: skip
    return out


# ---------------------------------------------------------------------------------------------- the pipeline model
@dataclass(frozen=True)
class PipeCost:
    """the program-B pipeline model's constants (progb_pipeline; PERF.md sec 3). Cycle constants are per-event
    costs of the kernels as written (ffn_compute.hpp / ffn_writer.hpp / exp_streamer.hpp); c_mm and c_pack32 are
    the two the silicon rows calibrate (progb_calibrate)."""

    mhz: float = 1350.0
    dram_gbps_streamer: float = 493.5 / 24  # 20.6: 7a's 493.5 GB/s (DRAM_GBPS_STREAMERS below) over the 24 streamers
    dram_lat_us: float = 1.5  # a DRAM read's round trip
    mcast_gbps: float = 40.0  # one core's multicast write rate (the NIU's send side)
    noc_lat_us: float = 1.0  # a NoC transfer's latency (multicast landing, a gather barrier)
    c_mm: float = 40.0  # cycles per tile-matmul incl. the bfp4 B unpack (HiFi2 ~ 32 cycles of math)
    c_pack32: float = 130.0  # cycles per fp32 tile pack with L1 accumulation (4 KB read + 4 KB write of L1)
    c_pack8: float = 40.0  # cycles per bfp8 tile pack
    c_sync: float = 150.0  # cycles per dest acquire / commit / wait / release round (one subblock)
    c_act: float = 300.0  # cycles per activation tile (2 fp32 copies + silu-glu + pack)
    c_issue: float = 40.0  # cycles per NoC transaction issue on a data-movement RISC
    c_rowb: float = 300.0  # cycles per row-bfp8 row re-lay (NC x 10 volatile L1 ops at NC = 4) + its write
    suba: int = 4
    subw: int = 4


def progb_pipeline(h_t: int, w_t: int, kc: int, nc: int, *, mg: int = 1, g: int = 2, n_streams: int = 12,
                   rows: int = 32, x_l1: bool = True, prefetch: bool = False, cost: PipeCost = PipeCost(),
                   mcast_issue_frac: float = 1.0) -> dict:  # fmt: skip
    """a discrete-event model of ONE streamer rectangle under the lever-3 credit protocol (exp_streamer.hpp sends,
    ffn_reader.hpp PMK_MCAST_EXT receives, ffn_compute.hpp consumes, ffn_writer.hpp gathers x / writes y), with the
    2-slot rings (WGU / WD on the members, XT, YRM) and the streamer's own 2-slot ring: n_streams streams of
    H_T / KC gate|up chunks then H_T / NC down chunks, every member of the rectangle computing one M group of `rows`
    (G identical members: one is simulated; CTL_SKIP members only receive). Returns the span, the chip rate (x 24
    streamers), and the per-phase stall fractions on each side -- the "credit stalls per phase" the host can state
    without a device (the tt-metal mock builds kernels, it does not run them)."""
    wtb = WTB_BFP4
    gu_bb, dn_bb = kc * 2 * w_t * wtb, w_t * nc * wtb
    n_gu, n_dn = h_t // kc, h_t // nc
    mt = -(-rows // 32)
    mhz = cost.mhz
    us = lambda cyc: cyc / mhz  # noqa: E731
    t_read = lambda b: cost.dram_lat_us + b / (cost.dram_gbps_streamer * 1e3)  # noqa: E731
    t_send = lambda b: b / (cost.mcast_gbps * 1e3)  # noqa: E731
    t_mcast = lambda b: cost.noc_lat_us + t_send(b)  # noqa: E731
    nsub_gu = (2 * w_t) // cost.suba
    t_cmp_gu = us(mt * nsub_gu * (cost.c_sync + kc * cost.suba * cost.c_mm + cost.suba * cost.c_pack32))
    nsub_dn = nc // cost.subw
    t_cmp_dn = us(mt * nsub_dn * (cost.c_sync + w_t * cost.subw * cost.c_mm + cost.subw * cost.c_pack8))
    t_act = us(mt * w_t * cost.c_act)
    # the x gather per gate|up chunk: TILE pieces from the holders' L1 (2 x 32 B per row per K tile) or x_ext rows
    t_gather = (
        us(rows * kc * 2 * cost.c_issue) + cost.noc_lat_us if x_l1 else us(rows * cost.c_issue) + cost.dram_lat_us
    )
    t_ywrite = us(rows * cost.c_rowb) + cost.noc_lat_us
    chunks = []
    for _ in range(n_streams):
        chunks += [(0, i, gu_bb) for i in range(n_gu)] + [(1, i, dn_bb) for i in range(n_dn)]
    N = len(chunks)
    t_issue, t_rd, t_cred, t_mc, t_data, t_cs, t_ce = ([0.0] * N for _ in range(7))
    gu_ids, dn_ids = [c for c in range(N) if chunks[c][0] == 0], [c for c in range(N) if chunks[c][0] == 1]
    t_xt, t_yw = {}, {}
    st = dict(s_read=[0.0, 0.0], s_credit=[0.0, 0.0], s_flush=[0.0, 0.0], s_mc_issue=[0.0, 0.0], m_data=[0.0, 0.0], m_xt=[0.0, 0.0],
              m_yslot=[0.0, 0.0], m_cmp=[0.0, 0.0], w_gather=0.0, w_ywrite=0.0)  # fmt: skip
    gather_end, ywrite_end = 0.0, 0.0
    iter_start = 0.0
    gi = {c: k for k, c in enumerate(gu_ids)}
    di = {c: k for k, c in enumerate(dn_ids)}
    for c in range(N):
        down, idx, b = chunks[c]
        # ---- streamer: the chunk's read
        if prefetch:
            if c == 0:
                t_issue[0] = 0.0
                t_rd[0] = t_read(b)
                iter_start = t_rd[0]
                st["s_read"][0] += t_rd[0]
        else:
            if c == 0:
                t_issue[0] = 0.0
            else:
                flush = t_mc[c - 1]  # (t_mc already includes the transmit: flushed() returns at once)
                t_issue[c] = flush
            t_rd[c] = t_issue[c] + t_read(b)
            st["s_read"][down] += t_rd[c] - t_issue[c]
        # ---- the member's credit for chunk c: after it pushed c - 1 and the compute popped c - 2 (its slot)
        t_cred[c] = max(t_data[c - 1] if c >= 1 else 0.0, t_ce[c - 2] if c >= 2 else 0.0)
        if prefetch and c + 1 < N:
            flush = max(iter_start, (t_mc[c - 1] + t_send(chunks[c - 1][2])) if c >= 1 else 0.0)
            st["s_flush"][down] += flush - iter_start
            t_issue[c + 1] = flush
            t_rd[c + 1] = flush + t_read(chunks[c + 1][2])
            ready = max(t_rd[c], t_issue[c + 1])
        else:
            ready = t_rd[c]
        t_mc[c] = max(ready, t_cred[c])
        st["s_credit"][down] += t_mc[c] - ready
        t_data[c] = t_mc[c] + t_mcast(b)
        # the multicast TRANSMIT sits inside the issue call (the NIU queue holds ~1 packet: exp_streamer S_MC_ISSUE)
        mc_end = t_mc[c] + t_send(b) * mcast_issue_frac
        st["s_mc_issue"][down] += mc_end - t_mc[c]
        if prefetch and c + 1 < N:
            nxt_iter = max(mc_end, t_rd[c + 1])  # the barrier for chunk c + 1
            st["s_read"][chunks[c + 1][0]] += nxt_iter - mc_end
            iter_start = nxt_iter
        else:
            t_mc[c] = mc_end  # the next read is issued after the transmit
        # ---- the member: writer gather / compute / y write
        prev_end = t_ce[c - 1] if c >= 1 else 0.0
        if not down:
            k = gi[c]
            xt_free = t_ce[gu_ids[k - 2]] if k >= 2 else 0.0
            g0 = max(gather_end, xt_free)
            gather_end = g0 + t_gather
            st["w_gather"] += t_gather
            t_xt[c] = gather_end
            t_cs[c] = max(t_data[c], prev_end, t_xt[c])
            st["m_data"][0] += max(0.0, t_data[c] - max(prev_end, t_xt[c]))
            st["m_xt"][0] += max(0.0, t_xt[c] - max(prev_end, t_data[c]))
            t_ce[c] = t_cs[c] + t_cmp_gu
            st["m_cmp"][0] += t_cmp_gu
        else:
            k = di[c]
            y_free = t_yw[dn_ids[k - 2]] if k >= 2 else 0.0
            base = prev_end + (t_act if idx == 0 else 0.0)
            t_cs[c] = max(t_data[c], base, y_free)
            st["m_data"][1] += max(0.0, t_data[c] - max(base, y_free))
            st["m_yslot"][1] += max(0.0, y_free - max(base, t_data[c]))
            t_ce[c] = t_cs[c] + t_cmp_dn
            st["m_cmp"][1] += t_cmp_dn
            y0 = max(ywrite_end, t_ce[c])
            ywrite_end = y0 + t_ywrite
            st["w_ywrite"] += t_ywrite
            t_yw[c] = ywrite_end
    span = max(t_ce[-1], ywrite_end)
    total_b = sum(b for _, _, b in chunks)
    fr = lambda v: round(v / span, 3)  # noqa: E731
    return dict(span_us=round(span, 1), chip_gbps=round(total_b * 24 / (span * 1e3), 1),
                streamer_gbps=round(total_b / (span * 1e3), 2), bytes_mb_streamer=round(total_b / 1e6, 2), chunks=N,
                us_per_chunk=dict(read_gu=round(t_read(gu_bb), 2), read_dn=round(t_read(dn_bb), 2), mcast_gu=round(t_mcast(gu_bb), 2),
                                  mcast_dn=round(t_mcast(dn_bb), 2), cmp_gu=round(t_cmp_gu, 2), cmp_dn=round(t_cmp_dn, 2),
                                  gather_gu=round(t_gather, 2), ywrite_dn=round(t_ywrite, 2), act=round(t_act, 2)),
                streamer=dict(read_gu=fr(st["s_read"][0]), read_dn=fr(st["s_read"][1]), credit_gu=fr(st["s_credit"][0]),
                              credit_dn=fr(st["s_credit"][1]), flush_gu=fr(st["s_flush"][0]), flush_dn=fr(st["s_flush"][1]),
                              mcast_issue=fr(st["s_mc_issue"][0] + st["s_mc_issue"][1])),
                member=dict(compute_gu=fr(st["m_cmp"][0]), compute_dn=fr(st["m_cmp"][1]), data_wait_gu=fr(st["m_data"][0]),
                            data_wait_dn=fr(st["m_data"][1]), xt_wait_gu=fr(st["m_xt"][0]), yslot_wait_dn=fr(st["m_yslot"][1])),
                writer=dict(gather=fr(st["w_gather"]), ywrite=fr(st["w_ywrite"])),
                bound=("members (credits)" if st["s_credit"][0] + st["s_credit"][1] > 0.5 * span else
                       "streamer (DRAM / multicast)" if st["m_data"][0] + st["m_data"][1] > 0.3 * span else "mixed"))  # fmt: skip


def progb_calibrate(measured: dict, h_t: int = 128, w_t: int = 32, *, g: int = 2, n_streams: int = 12,
                    mm_grid=tuple(range(16, 200, 4)), pack_grid=tuple(range(40, 900, 10))) -> dict:  # fmt: skip
    """fit (c_mm, c_pack32) -- the two per-tile compute costs -- to the measured silicon spans
    {(kc, nc, x_l1): us} (the 18:25Z TP2 L4 rows: (1, 4, True): 36,480; (4, 4, False): 30,222); the other constants
    stay at PipeCost's. Returns the best fit, its spans and relative errors, and the stall table at the fit."""
    best = None
    for mm in mm_grid:
        for pk in pack_grid:
            cost = replace(PipeCost(), c_mm=float(mm), c_pack32=float(pk))
            err = 0.0
            spans = {}
            for (kc, nc, x_l1), us_meas in measured.items():
                r = progb_pipeline(h_t, w_t, kc, nc, g=g, n_streams=n_streams, x_l1=x_l1, cost=cost)
                spans[(kc, nc, x_l1)] = r["span_us"]
                err += ((r["span_us"] - us_meas) / us_meas) ** 2
            if best is None or err < best[0]:
                best = (err, mm, pk, spans)
    err, mm, pk, spans = best
    cost = replace(PipeCost(), c_mm=float(mm), c_pack32=float(pk))
    table = {k: progb_pipeline(h_t, w_t, k[0], k[1], g=g, n_streams=n_streams, x_l1=k[2], cost=cost) for k in measured}
    rel = {k: round((spans[k] - v) / v, 3) for k, v in measured.items()}
    return dict(c_mm=mm, c_pack32=pk, spans=spans, rel_err=rel, cost=cost, table=table)


# ---------------------------------------------------------------------------------------------- the NoC combine (eval)
# the family's non-member cells and what their L1 has free during P14 (family README census, v2 form: the FFN set
# streams its own rings, so rings 43..46, kda prep 32..37, PARK 29 and 40..42 are idle on the MM cores; the 4 tail
# cores do not run P14 at all; the 2 lanes are coll's: every CB / semaphore)
NONMEMBER_CELLS = dict(root=8, mm_only=32, tail=4, lane=2)
FREE_B = dict(
    mm_only=(293 + 92 + 102 + 128) * 1024 + 142 * 1024,
    root=(293 + 92 + 102 + 128) * 1024 + 145 * 1024,
    tail=875 * 1024,
    lane=0,
)
FFN_SET_B = 405_312  # the (2, 4, 8) / h_split 4 v2 moe set on a core that runs P14 (roots, MM-only)


def noc_combine_capacity(c: int, owners: str = "nonmember", partial_b: int = 8192, y_fp32: bool = True,
                         pool_rows: int = 2) -> dict:  # fmt: skip
    """PMK_NOC_COMBINE evaluation (README sec 11): every token owned by one core; the expert cores send each (token,
    slot) partial row over the NoC into the owner's landing; the owner adds in FIXED slot order (out-of-order rows
    wait in the landing: one reserved row per token = the deadlock-free minimum, plus a pool), keeps y (fp32 16 KB, or
    bf16 8 KB -- not bitwise vs the fp32 L1-accumulated DRAM combine) and writes y once.
    owners: 'nonmember' = roots + MM-only + tails (44; the lanes are coll's), 'mm_only' = 32, 'all' = 110 (flat)."""
    n = {"nonmember": 44, "mm_only": 32, "all": 110}[owners]
    tok = -(-c // n)
    y_b = tok * (4096 * 4 if y_fp32 else 4096 * 2)
    landing_b = tok * partial_b + pool_rows * partial_b
    need = y_b + landing_b
    avail = {
        "nonmember": FREE_B["mm_only"] - FFN_SET_B,
        "mm_only": FREE_B["mm_only"] - FFN_SET_B,
        "all": CB_BUDGET - FFN_SET_B,
    }[owners]
    noc_mb = c * 8 * partial_b / 1e6
    return dict(owners=n, tokens_per_core=tok, y_kb=y_b / 1024, landing_kb=landing_b / 1024, need_kb=need / 1024,
                avail_kb=avail / 1024, tail_avail_kb=(FREE_B["tail"] - 0) / 1024, fits=need <= avail,
                fits_tail_only=need <= FREE_B["tail"], noc_mb=noc_mb, y_write_mb=c * 8192 / 1e6,
                breaks=("y accumulator" if y_b > avail else "landing") if need > avail else None)  # fmt: skip


# ---------------------------------------------------------------------------------------------- program B (sec 12)
# Stage 16 (13:4xZ): each FFN layer = two persistent programs in one trace, A = P0..P11, B = P12..P16 = the flat
# MoE program. 7a's traced DRAM bench (13:46Z): 24 bank-adjacent streamer cores sum to 493.5 GB/s per chip (96 % of
# peak) vs 304 GB/s for all-core rotating reads -> the expert slices are read by the 24 streamers (contiguous per bank,
# the mk-stream-v1 manner) and MULTICAST to compute rectangles (kernels/pmk/exp_streamer.hpp + the PMK_MCAST_EXT
# receiver path of ffn_reader.hpp); the other 86 cores are expert compute + token owners of the L1 fixed-order combine.
DRAM_GBPS_STREAMERS = 493.5
N_STREAMERS = 24
SEM_ACT = 14  # moe_common.hpp SEM_ACT (PMK_GU_SPLIT)
SYNC_SHARD_WORDS = 64  # the streamer's sync shard: 256 B (exp_streamer.hpp exps::SYNC_B)


def pack_expert_streams(gate_up: list, down: list) -> tuple:
    """host torch: expert e's gate|up [H, 2W] (tile-permuted as served) then down [W, H] as consecutive pages of
    bank e mod 8 of ONE interleaved tile tensor (page p -> bank p mod 8, in-bank index p / 8): -> ([pages * 32, 32]
    tile array, m0 [E] in-bank tile index of each expert, pages)"""
    import torch

    e_n = len(gate_up)
    used = [0] * 8
    m0 = []
    tiles_e = []
    for e in range(e_n):
        gu, dn = gate_up[e].float(), down[e].float()
        h, gw = gu.shape
        w, h2 = dn.shape
        assert h2 == h
        tg = gu.reshape(h // 32, 32, gw // 32, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)  # K-row-major
        td = dn.reshape(w // 32, 32, h // 32, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)
        t = torch.cat([tg, td])
        m0.append(used[e % 8])
        used[e % 8] += int(t.shape[0])
        tiles_e.append(t)
    pages = max(used) * 8
    out = torch.zeros(pages, 32, 32)
    for e, t in enumerate(tiles_e):
        idx = (e % 8) + 8 * (m0[e] + torch.arange(int(t.shape[0])))
        out[idx] = t
    return out.reshape(pages * 32, 32), m0, pages


def program_b_plan(c: int, g_size: int = 2, w: int = 256, p_total: int = 110, landing_bfp8: bool = False,
                   mg: int = 1, kc: int = 4, nc: int = 8, fid: str = "HiFi2", dram_gbps: float = DRAM_GBPS_STREAMERS,
                   mhz: float = 1350.0, tile_owners: bool = False, slots_in_flight: int = 2, n_ffn=None) -> dict:  # fmt: skip
    """the program-B census and arithmetic (README sec 12): roles, per-core L1 by role, the byte / us table at the
    streamers' 493.5 GB/s with the L1 fixed-order combine (out_rm = the y write only), the ceiling. n_ffn: the
    member count when the rectangles are unequal (PERF.md sec 7 cover: sum of the rectangle sizes, <= 86)"""
    g = Geometry(c=c, w=w, p=p_total - N_STREAMERS)
    n_compute = p_total - N_STREAMERS  # 86
    n_ffn = N_STREAMERS * g_size if n_ffn is None else int(n_ffn)  # the rectangles' cores
    n_pure = n_compute - n_ffn  # owners without FFN work
    assert n_pure >= 0, (g_size, n_ffn)
    gu_bb = kc * 2 * g.w_t * WTB_BFP4
    dn_bb = g.w_t * nc * WTB_BFP4
    # the receiver FFN set (no own DRAM reads; lever 2 XT; v1 roles)
    ffn = dict(WGU=2 * gu_bb, WD=2 * dn_bb, XT=2 * mg * kc * 2048, GUP=mg * 2 * g.w_t * 4096, ACT=mg * g.w_t * 2048,
               YRM=2 * nc * 2048, TAB=6 * 1024, CTL=256)  # fmt: skip
    ffn_b = sum(ffn.values())
    # the L1 combine (owners = all 86 compute cores): fp32 y + one reserved landing row per token + a 2-row pool
    tok = -(-c // n_compute)
    row_b = rowb_bytes(g.h_t, nc) if landing_bfp8 else g.h * 2
    y_b = tok * g.h * 4
    landing_b = (tok + 2) * row_b
    if tile_owners:
        # the FPU adds TILE ROWS (32 tokens x a tile column), so an owner owns whole 32-token blocks: y = one tile row
        # fp32 (512 KB) and the landing holds the 32 rows of `slots_in_flight` slots; 64 blocks need 64 owners
        n_owner_blocks = c // 32
        assert n_owner_blocks <= n_compute, (n_owner_blocks, n_compute)
        tok = 32
        y_b = 32 * g.h * 4
        landing_b = slots_in_flight * 32 * row_b
        n_pure = n_owner_blocks  # the 64 block owners carry no FFN set (the ledger below: 'pure_owner')
        n_ffn = n_compute - n_pure  # 22 FFN cores at 2,048 rows
        g_size = max(1, n_ffn // N_STREAMERS)  # rectangles shrink to what is left (0 -> the streamers feed 1 core)
        n_ffn = max(n_ffn, 1)
    # the plan scratch with hist sliced (open item 6.1) on the pure owners + streamers (P_plan = n_compute)
    pw = plan_words(g)
    plan_b = (max(pw["count"], pw["dest"], pw["runs"]) - (g.p - 8) * g.ep) * 4
    # the x-hat_f store: 64 blocks x 2 half-blocks (128 KB) over the 24 streamers (2 each) + the pure owners (2 each)
    half = XS_BYTES_PER_HOLDER // 2
    xs_slots = 2 * N_STREAMERS + 2 * n_pure
    xs_on_ffn = max(0, 128 - xs_slots)  # half-blocks that must go to FFN cores (1 each)
    roles = dict(
        streamer=dict(cores=N_STREAMERS, ring=2 * max(gu_bb, dn_bb), sync=128, tables=3 * g.ep * 4, xs=2 * half, plan=0),
        ffn_owner=dict(cores=n_ffn, ffn=ffn_b, y=y_b, landing=landing_b, xs=half if xs_on_ffn else 0, plan=plan_b),
        pure_owner=dict(cores=n_pure, y=y_b, landing=landing_b, xs=2 * half, plan=plan_b),
    )  # fmt: skip
    for r in roles.values():
        r["total"] = sum(v for k_, v in r.items() if k_ != "cores")
        r["fits"] = r["total"] <= CB_BUDGET
    # bytes / us: weights once (streams = experts x ceil(ng / G)), y write only, x from the L1 store
    g110 = replace(g, p=p_total)
    m = streaming_model2(g110, mg, Levers(x_l1=True, mcast=g_size, comb_l1=True), fid, dram_gbps=dram_gbps, mhz=mhz)
    fpu_ffn_us = m["fpu_useful_us"] * p_total / max(n_ffn, 1)  # the FFN runs on the rectangles' cores only
    span = max(m["dram_us"], fpu_ffn_us, m["noc_us"])
    return dict(roles=roles, tokens_per_owner=tok, xs_half_blocks=dict(slots=xs_slots, on_ffn=xs_on_ffn),
                weight_mb=m["weight_mb"], reread=m["reread_factor"], y_write_mb=m["y_mb"], dram_us=m["dram_us"],
                fpu_useful_us_110=m["fpu_useful_us"], fpu_ffn_cores_us=fpu_ffn_us, span_floor_us=span,
                math_ceiling=m["fpu_useful_us"] / span,  # the contract's MATH: FLOPs / (110-core peak x span)
                combine_fits=all(r["fits"] for r in roles.values()), n_ffn=n_ffn, n_pure=n_pure)  # fmt: skip


MEMBER_RECT = [(x, y) for y in range(8) for x in range(3, 11)]  # the family's 64 members (XS 35 holders), 11 x 10 grid


def adj_pmk_streamers(mesh, members=None, avoid=None) -> list:
    """7a's 'adj_pmk' placement (dram_read_bench.py, 13:46Z: 493.5 GB/s): ttnn's bank-adjacent reader cores
    (MESH_ROW_PLUS_X, 3 per bank) outside the member rectangle, and for each adjacent core inside it the nearest free
    core outside (Manhattan), bank kept. Returned as streamer s = index_in_bank * 8 + bank (exp_streamer.hpp's rule:
    streamer s serves the experts with e mod 8 = bank, (e / 8) mod 3 = index). Mock fallback: the family's pattern.
    `avoid` (Stage 16 mb8: the fused layer program at TP8): cores treated like the rectangle -- the lanes / tails hold
    program A's fabric connections, whose words take the reserved semaphore ids 10.. that B's ids 5-14 would occupy"""
    import ttnn

    g = mesh.compute_with_storage_grid_size()
    allc = [(x, y) for y in range(int(g.y)) for x in range(int(g.x))]
    ms = set(members or MEMBER_RECT) | {tuple(c) for c in (avoid or ())}
    try:
        P = ttnn._ttnn.operations.matmul
        adj = P.dram_bank_reader_assignments(mesh, num_workers_per_dram_bank=3,
                                             dram_reader_placement=P.DramReaderPlacement.MESH_ROW_PLUS_X)  # fmt: skip
        adj = [((int(c.x), int(c.y)), int(b)) for c, b, _ in adj]
    except Exception:  # noqa: BLE001
        adj = [
            ((x, y), b) for b, (x, y) in enumerate([(x, y) for y in (0, 1, 4, 6, 9) for x in (0, 1, 2, 6, 7, 8)][:24])
        ]
        adj = [(c, b % 8) for c, b in adj]
    taken, placed = set(), []  # (core, bank, index in bank)
    per_bank = {}
    for c, b in adj:
        i = per_bank.get(b, 0)
        per_bank[b] = i + 1
        if c not in ms:
            placed.append((c, b, i))
            taken.add(c)
    per_bank2 = {}
    for c, b in adj:
        i = per_bank2.get(b, 0)
        per_bank2[b] = i + 1
        if c in ms:
            free = [f for f in allc if f not in ms and f not in taken]
            best = min(free, key=lambda f: abs(f[0] - c[0]) + abs(f[1] - c[1]))
            placed.append((best, b, i))
            taken.add(best)
    order = sorted(placed, key=lambda t: (t[2], t[1]))
    return [c for c, _, _ in order]


class SelfStreamB(PmkMoe2):
    """program B's TP8 FORM OF RECORD (PERF.md sec 17-18, the 7f row: 4,553 us vs the op path's 4,088 at TP8, PCC
    0.99999): the v1 self-streaming phase set -- every compute core streams its own (expert, M group) units from pl.xsl
    by the address table (no streamer cores, no cross-core credit chain, no stream copy: in place by construction) and
    the bf16 partials + the bf16 combine (no row-bfp8 rebuild). Built through ProgramB(..., selfstream=True) so the stack
    driver's kwargs select it (GLM53_PMK_B_SELFSTREAM=1); the program-B-only knobs (lpt, dual, coal, splits, inplace,
    lpt_w, pack_rne) are inert here and the attributes the driver reads report so. y_s (PMK_SHARED) is ROW_MAJOR
    [C, H] bf16 on this path: the caller lays its shared partial out as rows before the copy."""

    inplace, lpt, gu_split, prefetch, mcast_linked, mcast_unicast, mcast_dual = (
        True,
        0,
        False,
        False,
        False,
        False,
        False,
    )
    rect_mode, gmax, lpt_w, probe_extra_rows = "self", 0, None, 0

    def __init__(self, mesh, **kw):
        super().__init__(mesh, **kw)
        self.streamers, self.rect_sizes, self.g_size = [], [], 1

    def load_streams(self, *a, **k):
        return 0  # nothing to pack: the units read the resident expert tensors by the address table

    @staticmethod
    def kwargs_from_program_b(kw: dict) -> dict:
        """the ProgramB kwargs -> this form's: mcast 1, bf16 partials, no writer / combine knobs; blocks default to the
        op path's TP8 geometry (4, 8, 8) with the DRAM x gather (x_l1 0: PERF.md sec 18), (2, 8, 8) with the XS store"""
        lv = kw.get("levers") or Levers(x_l1=False, mcast=1)
        lv = replace(lv, mcast=1, out_bfp8=False, yw_db=False, yw_coal=1, comb_split=False, yw_brisc=False, gu_split=False,
                     ring_slots=2)  # fmt: skip
        # (PMK_XS_ROWS: the 256-KB row store leaves room for (2, 8, 8) only -- (4, 8, 8) + the store = 1,255,588 B)
        blocks = kw.get("blocks") or ((2, 8, 8) if (lv.x_l1 or lv.xs_rows) else (4, 8, 8))
        keep = ("num_experts", "top_k", "hidden", "limit", "gate_up", "down", "fid", "probe", "full_sync", "gu_rt", "gu_ct",
                "h_split", "cb0", "cb1", "cores", "cb_xs", "holders", "ffn_fid", "bw_stage", "plan_cores", "plan_runs_nb",
                "plan_early")  # fmt: skip
        out = {k: kw[k] for k in keep if k in kw}
        out.update(levers=lv, blocks=tuple(blocks))
        return out


CB_B_ARG = (
    30  # GLM53_PMK_ARGS=table on standalone program B: the arg-row scratch (2 slots: BRISC / NCRISC), free in B's map
)


class ProgramB(PmkMoe2):
    """program B's FFN half on silicon geometry: the 24 bank-adjacent STREAMER cores run kernels/pmk/exp_streamer.hpp
    (expert slices contiguous per bank, one interleaved tensor from pack_expert_streams) and multicast each chunk to
    their rectangle of G compute cores (PMK_MCAST_EXT receivers); the compute cores = the other P - 24 cores run the
    plan (P_compute barriers; they also raise the streamers' SEM_PLAN3), the FFN as receivers, and -- in this mock /
    bench form -- v1's DRAM combine (the L1 fixed-order combine of sec 12 is planned by program_b_plan; its owner
    kernels are the next step). Rectangles: G-runs of compute cores (exact NoC boxes), rectangle i <-> streamer i."""

    def __new__(cls, mesh, *a, selfstream: bool = False, **kw):
        if selfstream:  # the TP8 form of record: a SelfStreamB (PERF.md sec 17-18); ProgramB.__init__ is then skipped
            return SelfStreamB(mesh, **SelfStreamB.kwargs_from_program_b(kw))
        return super().__new__(cls)

    def __init__(self, mesh, *, streamers=None, g_size: int = 2, prefetch: bool = False, rect_mode: str = "cover",
                 gmax: int = 4, mcast_linked: bool = False, mcast_unicast: bool = False, mcast_dual: bool = False,
                 lpt: int = 0, gu_split: bool = False, inplace: bool = True, lpt_w=None, comb_split: bool = False,
                 yw_brisc: bool = False, selfstream: bool = False, pack_rne: bool = False, compute_only=None,
                 **kw):  # fmt: skip
        """rect_mode "cover" (PERF.md sec 7, default): every compute core is a member -- 24 exact boxes of 1..gmax
        cores cover all 86 (per-rectangle G at runtime on both sides); "box": the 18:25Z placement (24 boxes of
        g_size). prefetch: PMK_STREAM_PREFETCH on the streamers (PERF.md knob 2); probe=True (PmkMoe) adds the 24
        streamer rows to the probe tensor (exp_streamer.hpp S_* slots: progb_scoreboard). lpt (PERF.md sec 11,
        PMK_STREAM_LPT): 0 = the static expert -> streamer map, 1 = live LPT within the expert's bank (3 streamers),
        2 = live LPT over all 24 streamers (stream_assignment is the host reference)"""
        import ttnn

        self.prefetch = bool(prefetch)
        self.rect_mode, self.gmax, self.mcast_linked = str(rect_mode), int(gmax), bool(mcast_linked)
        self.mcast_unicast, self.mcast_dual = bool(mcast_unicast), bool(mcast_dual)
        self.lpt = int(lpt)
        assert self.lpt in (0, 1, 2), lpt
        self.gu_split = bool(gu_split)
        # PERF.md sec 15: per-streamer LPT cost weights (us per chunk of each rectangle measured on the previous row:
        # the same bytes run up to 1.4x slower on some boxes) -- 24 floats, 1.0 = the fastest; None = unweighted
        self.lpt_w = [float(v) for v in lpt_w] if lpt_w else None
        # PMK_B_INPLACE (PERF.md sec 13, Stage 12's ruling): the streamers read pl.xsl's resident ND-sharded expert pages
        # in place (per-pair 1,152-B reads through the [E, 2] address table + TensorAccessor shard bases) -- no stream
        # copy (-2,038 MB per layer at TP2, -510 MB at TP8), per-chip correct by construction (fixes the chip-0 replica)
        self.inplace = bool(inplace)
        self.probe_extra_rows = N_STREAMERS
        grid = mesh.compute_with_storage_grid_size()
        allc = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x))]
        if streamers is None:
            streamers = adj_pmk_streamers(mesh)  # 7a's adj_pmk: bank-adjacent outside the member rectangle
        self.streamers = [tuple(c) for c in streamers][:N_STREAMERS]
        assert len(self.streamers) == N_STREAMERS, len(self.streamers)
        compute = [c for c in allc if c not in set(self.streamers)]
        if compute_only is not None:  # Stage 16 M-B: B as a window of the fused layer program runs its receivers on the
            # family's 64 members only (the MM-only role cannot also hold the receiver CBs beside the streamer rings)
            keep = {tuple(c) for c in compute_only}
            compute = [c for c in compute if c in keep]
        kw.setdefault("levers", Levers(x_l1=True, mcast=g_size))
        if self.gu_split:
            kw["levers"] = replace(kw["levers"], gu_split=True)
        if comb_split:
            kw["levers"] = replace(kw["levers"], comb_split=True)
        if yw_brisc:
            kw["levers"] = replace(kw["levers"], yw_brisc=True)
        kw.setdefault("holders", [c for c in MEMBER_RECT if c in set(compute)])  # the XS 35 holders = the members
        super().__init__(mesh, cores=compute, **kw)
        if self.rect_mode == "cover":
            rects = cover_rectangles(self.cores, self.noc_of, N_STREAMERS, self.gmax, *self.grid)
            assert len(rects) == N_STREAMERS, (len(rects), self.gmax)
        else:
            rects = rectangles(self.cores, self.noc_of, g_size, *self.grid)
            if len(rects) < N_STREAMERS:  # the other orientation (G = 3: vertical runs lose the streamer rows 8-9)
                alt = rectangles(self.cores, self.noc_of, g_size, *self.grid, orient="h")
                alt_v = rectangles(self.cores, self.noc_of, g_size, *self.grid, orient="v")
                rects = max((alt, alt_v), key=len)
            assert len(rects) >= N_STREAMERS, (len(rects), g_size)
        self.rects = rects[:N_STREAMERS]  # rectangle i is fed by streamer i
        self.g_size = max(len(r) for r in self.rects)  # the largest rectangle (the streamer kernel's CT G)
        self.rect_sizes = sorted((len(r) for r in self.rects), reverse=True)
        self.s_noc = {c: (lambda w: (int(w.x), int(w.y)))(mesh.worker_core_from_logical_core(ttnn.CoreCoord(*c)))
                      for c in self.streamers}  # fmt: skip
        self.s_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in self.streamers])
        # trace-light levers (Stage 14, 2026-10-23 16:59:28Z): GLM53_PMK_RECT_MERGE=1 = the compute / streamer core sets
        # as RECTANGLES (one CoreRange per core paid the CB configs / semaphores / launch message / binary relays per
        # core per launch), GLM53_PMK_ARGS=table = the DM kernels' runtime args in a DRAM table (2-word headers)
        self.set_trace_modes(
            rect=os.environ.get("GLM53_PMK_RECT_MERGE", "0") == "1",
            table=os.environ.get("GLM53_PMK_ARGS", "trace") == "table",
        )
        # the streamers' sync words (READY_GU, READY_D, PARK_GU, PARK_D, VAL_GU, VAL_D at 16-B stride, the DUAL GO /
        # DONE words at 96.., the NCRISC's 16-B-aligned VALID sources at 128 / 144: exp_streamer.hpp SYNC_B = 256 B):
        # one L1 shard each
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(self.s_crs, [1, SYNC_SHARD_WORDS], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        import torch

        self.sync = ttnn.from_torch(torch.zeros(N_STREAMERS, SYNC_SHARD_WORDS, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mc, mesh_mapper=self.repl)  # fmt: skip
        self.stream_t, self.m0_t = None, None

    def load_streams(self, gate_up_host: list, down_host: list):
        """the expert slices (host torch, as served: gate|up tile-permuted) -> the per-bank contiguous stream tensor"""
        import torch

        import ttnn

        if self.inplace:  # nothing to pack: the streamers read the resident expert tensors
            return 0
        tiles, m0, pages = pack_expert_streams(gate_up_host, down_host)
        self.stream_t = ttnn.from_torch(tiles.reshape(1, 1, -1, 32), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                                        device=self.mesh, memory_config=V1.DRAM, mesh_mapper=self.repl)  # fmt: skip
        g = self.geometry(2048)
        m0v = torch.zeros(1, g.ep, dtype=torch.int32)
        m0v[0, : len(m0)] = torch.tensor(m0, dtype=torch.int32)
        self.m0_t = ttnn.from_torch(m0v, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                    memory_config=V1.DRAM, mesh_mapper=self.repl)  # fmt: skip
        return pages

    def parts(self, *a, **kw):
        ttnn = self._ttnn
        raw, cbs, sems, io = super().parts(*a, **kw)
        c = a[5]
        g = self.geometry(c)
        mg, kc, nc = self.blocks(c)
        sync = self.sync.buffer_address()
        w_file, ct_w, rt_w, d_w, cfg_w = raw["w"]
        d_w = [d for d in d_w if d[0] != "PMK_MCAST"] + [("PMK_MCAST", "1"), ("PMK_MCAST_EXT", "1")]
        if self.mcast_dual:
            d_w.append(("PMK_MCAST_DUAL", "1"))  # the receivers wait the NCRISC half's VALID too (semaphores 12 / 13)
        # the 4 extra MC words (ready_gu, ready_d, park_gu, park_d addresses on the streamer) + the streamers' NoC list
        # + (perf5) the streamers' g table (PMK_RT_GS: n, then the member count of rectangle s) for PMK_STREAM_LPT
        s_flat = [v for cc in self.streamers for v in self.s_noc[cc]]
        suba = 8 if self.full_sync and (2 * self.w_t) % 8 == 0 else 4  # PmkMoe.parts' SUBA (the slice unit)
        gupt = mg * 2 * self.w_t
        # the LPT cost = ceil(ng / rows per stream): the rectangle size, or split_rows(g) under PMK_GU_SPLIT (both kernels
        # derive the same rows per stream from g, SUBA, GUP tiles and RMAX)
        g_tab = [split_rows(len(r), 2 * self.w_t, suba, gupt) if self.gu_split else len(r) for r in self.rects]
        w_tab = [int(round(v * 256)) for v in self.lpt_w] if self.lpt_w else []
        assert not w_tab or len(w_tab) == N_STREAMERS, len(w_tab)
        split_defs = [("PMK_GU_SPLIT", "1"), ("PMK_GU_SPLIT_RMAX", str(SPLIT_RMAX)), ("PMK_GU_SPLIT_GUPT", str(gupt)),
                      ("PMK_GU_SPLIT_SUBA", str(suba))] if self.gu_split else []  # fmt: skip
        for xy in self.cores:
            args = rt_w[xy]
            args += [sync + 0, sync + 16, sync + 32, sync + 48]
            base = len(args)
            args += [N_STREAMERS] + s_flat
            base_gs = len(args)
            args += [N_STREAMERS] + g_tab + w_tab
        d_w.append(("PMK_RT_STREAMERS", str(base)))
        d_w.append(("PMK_RT_GS", str(base_gs)))
        if self.lpt:
            d_w.append(("PMK_STREAM_LPT", str(self.lpt)))
            if w_tab:
                d_w.append(("PMK_LPT_W", "1"))
        if self.lv.ring_slots != 2:
            d_w.append(("PMK_RING_SLOTS", str(int(self.lv.ring_slots))))
        d_w = d_w + split_defs
        # PMK_MCAST_EXT: a receiver's READY credits / PARK bases go to ITS STREAMER (rectangle ri <-> streamer ri), not to
        # the rectangle's first receiver that PmkMoe2.parts put at MC + 4 / 5 (16, 2026-10-22 review: a certain hang)
        mc0 = int(dict(d_w)["PMK_RT_MC"])
        for xy in self.cores:
            ri = rt_w[xy][mc0 + 1]
            if ri < len(self.rects):
                rt_w[xy][mc0 + 4], rt_w[xy][mc0 + 5] = self.s_noc[self.streamers[ri]]
        if self.lv.comb_split:
            # PMK_COMB_SPLIT: the BRISC gets the combine reader's CT args (PMK_CT_COMBR = the NCRISC's ct_a[4:10]) and
            # RT args (PMK_RT_COMBR = [out_rm, dest, wt, u0, nu], y_s at PMK_RT_YS); the NCRISC gets the y address
            # (PMK_RT_Y) for the y write; PMK_CW2_PAGES = CW2's real capacity
            a_file, ct_a, rt_a, d_a, cfg_a = raw["a"]
            c_file, ct_c, rt_c, d_c, cfg_c = raw["c"]
            y_t, y_s = a[4], kw.get("y_s")
            ct_combr = len(ct_w)
            ct_w = list(ct_w) + list(ct_a[4:10])
            d_a_dict = dict(d_a)
            b_comb = int(d_a_dict["PMK_RT_COMB"])
            for xy in self.cores:
                args = rt_w[xy]
                base_cr = len(args)
                args += list(rt_a[xy][b_comb : b_comb + 5])
                base_ys = len(args)
                args += [int(y_s.buffer_address()) if y_s is not None else 0]
                rt_a[xy] = list(rt_a[xy]) + [int(y_t.buffer_address())]
            base_y = len(rt_a[self.cores[0]]) - 1
            cw2_pages = next(t // pg["CW2"] for mem, t, pg in self.plan(c)["groups"] if "CW2" in mem)
            d_w = d_w + [("PMK_COMB_SPLIT", "1"), ("PMK_CT_COMBR", str(ct_combr)), ("PMK_RT_COMBR", str(base_cr)),
                         ("PMK_RT_YS", str(base_ys)), ("PMK_CW2_PAGES", str(cw2_pages))]  # fmt: skip
            raw["a"] = (
                a_file,
                ct_a,
                rt_a,
                list(d_a) + [("PMK_COMB_SPLIT", "1"), ("PMK_RT_Y", str(base_y)), ("PMK_CW2_PAGES", str(cw2_pages))],
                cfg_a,
            )
            raw["c"] = (c_file, ct_c, rt_c, list(d_c) + [("PMK_COMB_SPLIT", "1")], cfg_c)
        raw["w"] = (w_file, ct_w, rt_w, d_w, cfg_w)
        if self.gu_split:
            # the NCRISC (act sends) and the compute: the defines; the NCRISC: PMK_RT_MEMBERS = g, then the NoC (x, y) of
            # the rectangle's members by position (row r's owner is member pos r); SEM_ACT 14 on the receivers
            a_file, ct_a, rt_a, d_a, cfg_a = raw["a"]
            c_file, ct_c, rt_c, d_c, cfg_c = raw["c"]
            rect_of = {}
            for r in self.rects:
                for core in r:
                    rect_of[core] = r
            for xy in self.cores:
                r = rect_of.get(xy, [xy])
                base_m = len(rt_a[xy])
                rt_a[xy] = (
                    list(rt_a[xy]) + [len(r)] + [v for c_ in r for v in self.noc_of[c_]] + [0] * (2 * (8 - len(r)))
                )
            d_a = list(d_a) + split_defs + [("PMK_RT_MEMBERS", str(base_m))]
            raw["a"] = (a_file, ct_a, rt_a, d_a, cfg_a)
            raw["c"] = (c_file, ct_c, rt_c, list(d_c) + split_defs, cfg_c)
            sems.append(ttnn.SemaphoreDescriptor(id=SEM_ACT, core_ranges=self.crs, initial_value=0))
            assert max(len(r) for r in self.rects) <= 8, self.rect_sizes
        # the streamer kernel
        gu_bb = kc * 2 * self.w_t * WTB_BFP4
        slot_bb = max(gu_bb, self.w_t * nc * WTB_BFP4)  # exp_streamer.hpp SLOT_BB: a slot holds either chunk
        ct_s = [self.h_t, self.w_t, kc, nc, mg, self.e, g.ep, WTB_BFP4, self.g_size]
        if self.inplace:  # CT0 + 9: the expert tensors' TensorAccessorArgs (every expert shares the ND-shard layout)
            acc = lambda t_: list(ttnn.TensorAccessorArgs(t_).get_compile_time_args())  # noqa: E731
            ct_s = ct_s + acc(self.gate_up[0]) + acc(self.down[0])
            assert (
                self.shard_bases()
            ), "PMK_B_INPLACE: the expert tensors must be ND-sharded [rows, 64] (fused_decoder._nd_dram)"
        b = self.tables(c)
        pa = self.bufs["probe"].buffer_address() if self.probe else 0
        rt_s = {}
        for si, cc in enumerate(self.streamers):
            r = self.rects[si]
            v = [self.noc_of[c_] for c_ in r]
            rt_s[cc] = (
                [
                    b["cnt"].buffer_address(),
                    b["start"].buffer_address(),
                    self.tbl.buffer_address()
                    if self.inplace
                    else self.m0_t.buffer_address(),  # RT0 + 2: [E, 2] addresses | m0
                    si,
                    sync,
                    min(x for x, _ in v),
                    min(y for _, y in v),
                    max(x for x, _ in v),
                    max(y for _, y in v),
                    self.p,
                    0 if self.inplace else self.stream_t.buffer_address(),  # RT0 + 10: the stream tensor's base
                    pa,
                    2 * self.p + si,  # RT0 + 11 / 12: the probe tensor, this streamer's row
                    len(r),  # RT0 + 13: this rectangle's member count g
                    N_STREAMERS,
                ]
                + g_tab
                + w_tab
            )  # fmt: skip  # RT0 + 14: n, then every rectangle's g, then the weights
        d_s = [("PMK_CT_STREAM", "0"), ("PMK_RT_STREAM", "0"), ("PMK_MOE_CB0", str(self.cb0)), ("PMK_MOE_CB1", str(self.cb1)),
               ("PMK_MOE_CB_XS", str(self.cb_xs)), ("PMK_P", str(self.p)), ("PMK_MOE_REV", str(V1.PMK_MOE_REV))]  # fmt: skip
        if self.probe:
            d_s.append(("XFFN_PROBE", "1"))
        if self.lpt:
            d_s.append(("PMK_STREAM_LPT", str(self.lpt)))
            if w_tab:
                d_s.append(("PMK_LPT_W", "1"))
        if self.lv.ring_slots != 2:
            d_s.append(("PMK_RING_SLOTS", str(int(self.lv.ring_slots))))
        d_s += split_defs
        if self.inplace:
            d_s.append(("PMK_B_INPLACE", "1"))
        if self.prefetch:
            d_s.append(("PMK_STREAM_PREFETCH", "1"))
        if self.mcast_linked:
            d_s.append(("PMK_MCAST_LINKED", "1"))
        if self.mcast_unicast:
            d_s.append(("PMK_MCAST_UNICAST", "1"))
        if self.mcast_dual:
            d_s.append(("PMK_MCAST_DUAL", "1"))
        dm = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        )
        raw["s"] = ("moe_stream_brisc.cpp", ct_s, rt_s, d_s, dm)
        if self.mcast_dual:  # the NCRISC half-streamer on the same cores, NoC1, the same arg lists
            dm1 = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
            )
            raw["s1"] = ("moe_stream_ncrisc.cpp", ct_s, rt_s, d_s, dm1)
        cbs.append(ttnn.CBDescriptor(total_size=self.lv.ring_slots * slot_bb, core_ranges=self.s_crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=self.cb_index("WGU"), data_format=ttnn.bfloat4_b, page_size=WTB_BFP4)]))  # fmt: skip
        # exp_streamer reads its tables at TAB + SYNC_WORDS_B; perf5: the PMK_STREAM_LPT scratch past them (64-B aligned)
        tab_b = -(-(V1.SYNC_WORDS_B + 3 * g.ep * 4) // 64) * 64 + 64 + lpt_scratch_bytes(g.ep)
        if self.inplace:  # + the [E, 2] address table and the current expert's shard bases (W_T + H_T / 2 uint64)
            tab_b += 64 + 2 * self.e * 4 + 64 + 8 * (self.w_t + self.h_t // 2) + 64
        cbs.append(ttnn.CBDescriptor(total_size=tab_b, core_ranges=self.s_crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=self.cb_index("TAB"), data_format=ttnn.uint32, page_size=tab_b)]))  # fmt: skip
        for sid in (7, 10, 11):
            sems.append(ttnn.SemaphoreDescriptor(id=sid, core_ranges=self.s_crs, initial_value=0))
        if self.mcast_dual:  # the second VALID per ring on the receivers (and allocated on the streamers: same address)
            for sid in (12, 13):
                sems.append(ttnn.SemaphoreDescriptor(id=sid, core_ranges=self.crs, initial_value=0))
                sems.append(ttnn.SemaphoreDescriptor(id=sid, core_ranges=self.s_crs, initial_value=0))
        io += [self.sync] if self.inplace else [self.stream_t, self.m0_t, self.sync]
        return raw, cbs, sems, io

    def kernel(self, name, ct, rt, defs, cfg):
        if name in ("moe_stream_brisc.cpp", "moe_stream_ncrisc.cpp"):
            ttnn = self._ttnn
            rta = ttnn.RuntimeArgs()
            for (x, y), args in rt.items():
                rta[x][y] = [int(v) for v in args]
            return ttnn.KernelDescriptor(kernel_source=os.path.join(V1.KDIR, "pmk", name),
                                         source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.s_crs,
                                         compile_time_args=ct, runtime_args=rta, config=cfg, defines=list(defs))  # fmt: skip
        return super().kernel(name, ct, rt, defs, cfg)

    def set_trace_modes(self, rect: bool, table: bool):
        """the trace-light levers for this instance's next program() builds: rect = the core sets as disjoint
        rectangles (pmk_layer.core_rects; the same cores, so every address / arg is unchanged), table = the DM kernels'
        args from the DRAM table (_arg_table). Switchable per build (the A/B unit compiles every mode before a capture).
        """
        ttnn = self._ttnn
        if not hasattr(self, "_crs_single"):
            self._crs_single, self._s_crs_single = self.crs, self.s_crs
        if rect:
            from models.autoports.zai_org_glm_5_3_flash.tt.pmk_layer import core_rects

            rs = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))  # noqa: E731
                                                  for x0, y0, x1, y1 in core_rects(cores)])  # fmt: skip
            self.crs, self.s_crs = rs(self.cores), rs(self.streamers)
        else:
            self.crs, self.s_crs = self._crs_single, self._s_crs_single
        self.rect_merge, self.table_mode = bool(rect), bool(table)

    def _arg_table(self, raw: dict, cbs: list, io: list, c: int, dyn_vals: frozenset = frozenset()):
        """GLM53_PMK_ARGS=table: the DM kernels' (compute cores' moe_brisc / moe_ncrisc, the streamers' BRISC / NCRISC)
        runtime-arg vectors as rows of ONE DRAM table per (bucket, kernel set), written at the FIRST build (the compile
        launch: no host write inside a capture); the TRISC keeps its (small) args in the launch.
        THE DYNAMIC BLOCK (coordinator 2026-10-23 18:43:37Z; 16's served 45-layer mock 18:37Z: the served chain hands
        program B fresh x_ext / ids_rm / wt / hb tensors per call, so rows changed between the compile launch and the
        capture build): a word whose value at the table fill is the buffer address of a CALLER tensor (dyn_vals:
        program()'s inputs, never B's own scratch) is DYNAMIC for its row. Each launch carries [table address, row,
        n_dyn] + n_dyn (position, current value) pairs -- n_dyn fixed per row from the fill, so the arg COUNT never
        changes between builds (generic_op's cache) -- and PMK_B_PROLOGUE patches them into the row it copied. Every
        later build compares the STATIC words only and raises naming the row / position / old / new value: a caller
        word that is not a plain address (an offset into one) fails loudly here, never as a stale replay."""
        import torch

        ttnn = self._ttnn
        spec = [("w", self.cores, 0), ("a", self.cores, 1), ("s", self.streamers, 0)] + (
            [("s1", self.streamers, 1)] if "s1" in raw else []
        )
        base, n = {}, 0
        for k, cores, _ in spec:
            base[k] = n
            n += len(cores)
        host = {base[k] + i: [int(v) for v in raw[k][2][xy]] for k, cores, _ in spec for i, xy in enumerate(cores)}
        width = max(len(v) for v in host.values())
        page = max(64, -(-4 * width // 64) * 64)
        key = (int(c), tuple(sorted(raw)))
        tabs = self.__dict__.setdefault("_arg_tabs", {})
        tb = tabs.get(key)
        if tb is None:
            t = ttnn.from_torch(torch.zeros(n, page // 4, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=self.repl)  # fmt: skip
            full = torch.zeros(n, page // 4, dtype=torch.int64)
            for r_, v in host.items():
                full[r_, : len(v)] = torch.tensor(v, dtype=torch.int64)
            full = torch.where(full >= 2**31, full - 2**32, full).to(torch.int32)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(full, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                            mesh_mapper=self.repl), t)  # fmt: skip
            dyn = {r_: [j for j, v in enumerate(vs) if v in dyn_vals] for r_, vs in host.items()}
            tb = tabs[key] = dict(t=t, rows=host, page=page, n=n, width=width, dyn=dyn)
        else:
            dyn = tb["dyn"]
            bad = []
            for r_ in sorted(set(host) | set(tb["rows"])):
                old, new = tb["rows"].get(r_), host.get(r_)
                if old is None or new is None or len(old) != len(new):
                    bad.append((r_, "len", None if old is None else len(old), None if new is None else len(new)))
                    continue
                ds = set(dyn.get(r_, ()))
                bad += [(r_, j, old[j], new[j]) for j in range(len(new)) if j not in ds and old[j] != new[j]]
            if bad:
                raise ValueError(f"ProgramB GLM53_PMK_ARGS=table: {len(bad)} STATIC arg word(s) changed after the table fill "
                                 f"(row, position, old, new): {bad[:8]} -- not a caller-tensor address: extend dyn_vals")  # fmt: skip
        addr = int(tb["t"].buffer_address())
        for k, cores, slot in spec:
            f, ct, rt, defs, cfg = raw[k]
            raw[k] = (f, ct, {xy: [addr, base[k] + i, len(dyn[base[k] + i])]
                              + [w for j in dyn[base[k] + i] for w in (j, host[base[k] + i][j])] for i, xy in enumerate(cores)},
                      list(defs) + [("PMK_B_ARG_TABLE", "1"), ("PMK_B_ARG_CB", str(CB_B_ARG)), ("PMK_B_ARG_PAGE", str(page)),
                                    ("PMK_B_ARG_SLOT", str(slot))], cfg)  # fmt: skip
        allc = self.crs.merge(self.s_crs)
        cbs.append(ttnn.CBDescriptor(total_size=2 * page + 64, core_ranges=allc, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_B_ARG, data_format=ttnn.uint32, page_size=2 * page + 64)]))  # fmt: skip
        nd = [len(v) for v in dyn.values()]
        self.arg_table_report = dict(rows=n, page=page, width=width, bytes=n * page, dyn_words_max=max(nd, default=0),
                                     dyn_words_total=sum(nd))  # fmt: skip
        return raw, cbs, list(io) + [tb["t"]]

    @staticmethod
    def _caller_addresses(*ts) -> frozenset:
        """the buffer addresses of program()'s CALLER tensors (the dynamic block's provenance rule; B's own scratch
        out_rm / y / y_s is persistent per bucket and stays static)"""
        out = set()
        for t in ts:
            if t is not None and hasattr(t, "buffer_address"):
                out.add(int(t.buffer_address()))
        return frozenset(out)

    def program(self, x_ext, ids_rm, wt, out_rm, y, c: int, **kw):
        ttnn = self._ttnn
        raw, cbs, sems, io = self.parts(x_ext, ids_rm, wt, out_rm, y, c, **kw)
        from models.autoports.zai_org_glm_5_3_flash.tt.cb_guard import check

        if getattr(self, "table_mode", False):
            caller = [x_ext, ids_rm, wt] + [v for k_, v in kw.items() if k_ != "y_s"]
            raw, cbs, io = self._arg_table(raw, cbs, io, c, self._caller_addresses(*caller))
        check(cbs, "pmk_moe2_progB")
        kernels = [self.kernel(*raw[r]) for r in ("w", "a", "c", "s") + (("s1",) if self.mcast_dual else ())]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs), io

    # ------------------------------------------------------------------------------------------ family registration
    PHASES = ("P12", "P13", "P14", "P15")

    def phase_parts(self, c: int, *, x_tile=None, y_s=None, ids_rm=None, wt=None, x_ext=None) -> dict:
        """program B's phases for the family descriptor (prefill_megakernel.md sec 4b ids; tt/pmk_layer.PhaseBody's
        fields): {phase: {role: {"dm0"/"dm1"/"c": (kernel file, ct, rt, defines, config)}, "cbs": {role: [(index,
        format, page, pages, push)]}, "sems": [...], "waits"/"signals", "io"}}. The three composites run the phases
        back to back on their cores (one kernel per RISC per role); the family splices their sources as bodies and
        keeps the CB indices (44..59, 27 / 28 on the compute cores; 45 / 50 rings on the streamers). Push sizes come
        from push_blocks and are asserted (check_pushes) by cb_plan2."""
        s = self.scratch(c)
        g = self.geometry(c)
        mg, kc, nc = self.blocks(c)
        plan = self.plan(c)
        raw, cbs, sems, io = self.parts(x_ext if x_ext is not None else s["y"], ids_rm, wt, s["out_rm"], s["y"], c,
                                        x_tile=x_tile, y_s=y_s if y_s is not None else s.get("y_s"))  # fmt: skip
        cb_uses = {"receiver": [], "streamer": []}
        for members, total, pages in plan["groups"]:
            for m in members:
                cb_uses["receiver"].append((self.cb_index(m), str(self.CB_FMT.get(m, "uint32")), pages[m],
                                            total // pages[m], plan["blocks"].get(m)))  # fmt: skip
        gu_bb = kc * 2 * self.w_t * WTB_BFP4
        slot_bb = max(gu_bb, self.w_t * nc * WTB_BFP4)
        cb_uses["streamer"] = [(self.cb_index("WGU"), "bfloat4_b", WTB_BFP4, self.lv.ring_slots * slot_bb // WTB_BFP4, slot_bb),
                               (self.cb_index("TAB"), "uint32", -(-(V1.SYNC_WORDS_B + 3 * g.ep * 4) // 64) * 64, 1, None)]  # fmt: skip
        if self.lv.x_l1:
            cb_uses["receiver"].append((self.cb_xs, "bfloat16", 2048, self.h_t, None))  # XS 35: the family's H8 store
        roles = {"receiver": {"dm0": raw["w"], "dm1": raw["a"], "c": raw["c"]}, "streamer": {"dm0": raw["s"]}}
        table = {
            "P12": dict(
                what="plan steps 1-3 on the receivers (BRISC), ids / weights from the tail layouts (PMK_IDS_U16 / PMK_WT_BF16)",
                cores="receiver",
                waits="H8: FS_ARRIVE 15 >= 16 (wait=) + TAIL 13",
                signals="barriers 5, 6, 7 (+ the streamers' 7)",
            ),
            "P13": dict(
                what="shared expert: SharedP13 (two PfStreamMM phases on the members) -> y_s bf16 TILE",
                cores="family members",
                waits="H8",
                signals="y_s complete (DRAM) before P15",
            ),
            "P14": dict(
                what="expert FFN: streamers (exp_streamer.hpp, VALID 10 / 11) -> receiver rectangles (xffn phases, XFFN_X_TILED gather from XS 35, row-bfp8 partials to out_rm)",
                cores="streamer + receiver",
                waits="7 on the streamers",
                signals="barrier 8 (every core's partials landed)",
            ),
            "P15": dict(
                what="combine (row-bfp8 tiles rebuilt, COL-broadcast weights, fp32 slot order, + y_s last) -> y bf16 TILE",
                cores="receiver",
                waits="8",
                signals="H9: +1 per unit per 32-row block (PMK_H9) -> the RS lanes",
            ),
        }
        return dict(phases=table, roles=roles, cbs=cb_uses, sems=[(d.id, "receiver") for d in sems if d.core_ranges == self.crs]
                    + [(sid, "streamer") for sid in (7, 10, 11)], io=io, blocks=(mg, kc, nc), levers=self.lv)  # fmt: skip
