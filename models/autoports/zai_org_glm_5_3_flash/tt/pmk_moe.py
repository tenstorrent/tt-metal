# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill megakernel (PMK) -- the MoE PHASE SET (prefill_megakernel.md P12 tail / P14 / combine; piece 'moe'):
the served XSL expert FFN of a sparse_kda layer as phases of ONE persistent program on the MM cores, built from the
EXISTING Stage 16 prefill MoE kernels (no rewrites, sec 0): the multi-core routing plan (kernels/prefill/moe_plan_mc_*:
counts -> region layout + dest + staging -> disp runs), the expert-per-core FFN (moe_xffn_*: one expert M group per core
at a time, the expert's slice weights streamed by the BRISC into double-buffered CBs under the compute) and the rm
combine (moe_combine_rm_*: y[t] = sum_j w[t, j] out[dest[j C + t]], fp32 in slot order) -- see kernels/pmk/moe_common.hpp
for the composition (phase namespaces, barriers on semaphores 5..8, H8 wait / H9 per-block signals).

Geometry (sec 1b): parameterized by the slice width W (tiles) of the served expert tensors (TP8: 8, TP2: 32), the
rows C of the bucket and the MM grid; blocks (MG, KC, NC) = the widest that fit the CB budget (xffn's rule: TP8
(4, 8, 8), TP2 (1, 4, 4) -- the pair's 32-tile slices re-read weights per tile row: README 'what limits it').

Standalone: `PmkMoe(...)(x_ext, ids_rm, wt, c)` -> y [1, 1, C, H] bf16 TILE (the routed partial of this chip's slice;
the shared expert and the reduce-scatter are outside: README). `parts()` gives the integrator the raw kernel pieces
(per-RISC source + compile-time / runtime args, CBs, semaphores, io) to splice into tt/pmk_layer.py."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

import ttnn

KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")
TILE = 32
BF16, FP32 = ttnn.bfloat16, ttnn.float32
DRAM = ttnn.DRAM_MEMORY_CONFIG
WTB_BFP4 = 576
CB_BUDGET = 1026 * 1024  # prefill_megakernel.md sec 4: ~1,026 KB of CBs per core (CB base 308,352, end <= 1,334,720)
SYNC_WORDS_B = (
    128  # moe_common.hpp: CB_TAB's multicast sync + push-probe prefix (CB_CIDX aliases TAB: its words start past it)
)
PMK_MOE_REV = 5  # kernel revision (defines: a long-lived process rebuilds after a kernel edit)


def r16(v: int) -> int:
    return -(-v // 16) * 16


def lcm(a: int, b: int) -> int:
    from math import gcd

    return a * b // gcd(a, b)


# ---------------------------------------------------------------------------------------------- pure-python model


def _h9_args(h9) -> list:
    """the H9 runtime args: list [(x, y, base)] = every consumer's per-32-row-block counter words (base + 16 block); dict
    (TABLE mode, non-B lever (2)) dict(targets=[(x, y, sync base)], table=[(target index, word)] per row block): the unit
    of block b raises only targets[t].base + 16 w, (t, w) = table[b]"""
    if not h9:
        return [0]
    if isinstance(h9, dict):
        tg, tb = list(h9["targets"]), list(h9["table"])
        return [len(tg)] + [int(v) for xyb in tg for v in xyb] + [int(v) for tw in tb for v in tw]
    return [len(h9)] + [int(v) for xyb in h9 for v in xyb]


@dataclass(frozen=True)
class Geometry:
    """the per-chip MoE geometry (ttnn-free: the budget test and the README arithmetic use it)"""

    c: int  # rows of the bucket (every chip runs all of them on its slice: XSL)
    h: int = 4096
    w: int = 256  # the slice width (I / TP): TP8 256, TP2 1024
    e: int = 288
    k: int = 8
    p: int = 110  # MM cores
    h_split: int = 2  # combine column parts per 32-token chunk

    @property
    def h_t(self):
        return self.h // TILE

    @property
    def w_t(self):
        return self.w // TILE

    @property
    def n(self):
        return self.c * self.k

    @property
    def ep(self):
        return -(-self.e // 32) * 32

    @property
    def cap(self):
        return -(-(self.n + 32 * self.e) // 32) * 32


def ffn_cb_bytes(g: Geometry, mg: int, kc: int, nc: int) -> dict:
    """bytes of the FFN phase's CBs (moe_xffn_common.hpp roles; EpMoe._xffn's l1())"""
    w_t, wtb = g.w_t, WTB_BFP4
    tab_words = 2 * g.ep + (2 * g.e + 15) // 16 * 16 + mg * 32 + 16 + 2 * (w_t + g.h_t // 2)
    return dict(CTL=64, WGU=2 * kc * 2 * w_t * wtb, XRM=2 * mg * kc * 2048, XT=mg * kc * 2048, WD=2 * w_t * nc * wtb,
                WCTL=64, TAB=tab_words * 4, WTAB=mg * 32 * 4, GUP=mg * 2 * w_t * 4096, ACT=mg * w_t * 2048,
                YRM=2 * nc * 2048)  # fmt: skip


def plan_words(g: Geometry, nb: int = 1) -> dict:
    """scratch words of the 3 plan steps (moe_plan_mc_*.cpp) on P cores; nb = plan_runs' expert batches
    (PMK_PLAN_RUNS_NB: RMAX = one batch's worst case)"""
    chunk = r16(-(-g.n // g.p))
    epc = -(-g.e // g.p)
    rmax = r16(-(-epc // max(1, int(nb))) * g.c)
    # GLM53_PMK_PLAN_WIN (words; Stage 14 W1, the whole-prompt MoE group): runs assembles the owned regions in windows
    # of max(win, chunk) words (moe_plan_mc_runs.cpp's RMAX < EPC C form, nb 1 only) -- the plan's L1 then stops growing with c
    win = int(os.environ.get("GLM53_PMK_PLAN_WIN", "0") or 0)
    if win > 0 and rmax > r16(win):
        assert int(nb) <= 1, (nb, win, "GLM53_PMK_PLAN_WIN: the windowed plan runs one expert batch")
        rmax = max(r16(win), chunk)
    return dict(chunk=chunk, epc=epc, rmax=rmax, count=chunk + g.ep + 32,
                dest=g.p * g.ep + 3 * chunk + 4 * g.ep + 32,
                runs=g.p * g.ep + 2 * g.ep + 32 + 2 * rmax + 32 * g.p + 3 * g.p)  # fmt: skip


def combine_cb_bytes(g: Geometry) -> dict:
    part = g.h * 2 // g.h_split
    pt = part // 2048
    assert pt * 2048 == part and 1 <= pt <= 4, (g.h, g.h_split)
    return dict(CIN=2 * g.k * pt * 2048, CW=2 * g.k * 4096, CIDX=SYNC_WORDS_B + g.k * 32 * 4 + 4096, COUT=32 * pt * 2048,
                CRM=32 * pt * 4096)  # fmt: skip


# the combine / plan CBs alias FFN CBs (the phases never overlap in time): alias group -> members
ALIASES = {"WGU": ("COUT", "PLAN"), "GUP": ("CRM",), "XRM": ("CIN",), "WD": ("CW",), "TAB": ("CIDX",)}
PAGE = dict(
    CTL=32,
    WGU=WTB_BFP4,
    XRM=2048,
    XT=2048,
    WD=WTB_BFP4,
    WCTL=32,
    GUP=4096,
    ACT=2048,
    YRM=2048,
    CIN=2048,
    CW=4096,
    COUT=2048,
    CRM=4096,
    PLAN=64,
)  # TAB / WTAB / CIDX: one page = the whole CB


def push_blocks(g: Geometry, mg: int, kc: int, nc: int, kk: int | None = None, out_bfp8: bool = False,
                nt: int | None = None) -> dict:  # fmt: skip
    """bytes of ONE push (cb_push_back) per CB, as the kernels issue them (constant per launch: the units always push
    whole MG-row blocks; mt < MG only leaves rows unused). Members that are never pushed (scratch) are absent."""
    w_t, wtb = g.w_t, WTB_BFP4
    kk = g.k if kk is None else kk
    part = g.h * 2 // g.h_split
    pt = part // 2048
    out = dict(CTL=32, WCTL=32, WGU=kc * 2 * w_t * wtb, WD=w_t * nc * wtb, XRM=mg * kc * 2048, XT=mg * kc * 2048,
               GUP=mg * 2 * w_t * 4096, ACT=mg * w_t * 2048, YRM=nc * (1088 if out_bfp8 else 2048),
               CIN=(nt * 1088) if out_bfp8 else kk * pt * 2048, CW=4096 if out_bfp8 else kk * 4096,
               COUT=(nt * 2048) if out_bfp8 else 32 * pt * 2048, CRM=(nt * 4096) if out_bfp8 else 32 * pt * 4096)  # fmt: skip
    return out


def alias_groups(sizes: dict, aliases: dict, pages: dict, blocks: dict, order: list, slack: int = 64 * 1024) -> list:
    """[(members, total, {member: page})]: one allocation per alias group whose capacity is a multiple of every
    member's page AND push block (CB blocks never straddle the ring end: prefill_megakernel.md sec 10 rule 1; the
    13:44:48Z watcher assert was WGU's 128-page blocks in an allocation sized by PLAN and rounded to the PAGE lcm).
    Single-page scratch members (not in `pages`: PLAN, TAB, ...) take the whole allocation as their page and never
    constrain it. When the rounding to the pushing members' lcm would cost more than `slack`, the member whose removal
    shrinks it most leaves the group and is allocated alone (the planner's budget then decides the blocks)."""
    names = list(order) + [n for n in sizes if n not in order]
    groups, done = [], set()

    def rounded_for(members):
        total = max(sizes[m] for m in members)
        mult = 1
        for m in members:
            if m in pages:
                mult = lcm(mult, pages[m])
                if m in blocks:
                    mult = lcm(mult, blocks[m])
        return -(-total // mult) * mult, total

    for name in names:
        if name in done:
            continue
        members = [name] + [m for m in aliases.get(name, ()) if m not in done]
        while len(members) > 1:
            rounded, total = rounded_for(members)
            if rounded - total <= slack:
                break
            # drop the member whose removal helps most (never the group's owner `name`)
            best = min(members[1:], key=lambda m: rounded_for([x for x in members if x != m])[0])
            members.remove(best)
        rounded, _ = rounded_for(members)
        pg = {m: (pages[m] if m in pages else rounded) for m in members}
        groups.append((tuple(members), rounded, pg))
        done.update(members)
    return groups


def check_pushes(groups: list, blocks: dict) -> list:
    """every member's capacity % push block == 0 (and % page == 0) -> [] or the failing (member, total, block)s"""
    bad = []
    for members, total, pg in groups:
        for m in members:
            if total % pg[m] or (m in blocks and total % blocks[m]):
                bad.append((m, total, blocks.get(m), pg[m]))
    return bad


def cb_plan(g: Geometry, mg: int, kc: int, nc: int) -> dict:
    """{"groups": [(members, total_bytes, {member: page})], "total": bytes}: the program's CB allocations with the
    aliases, every total a multiple of each member's page size AND push block (sec 10 rule 1; the 13:44:48Z assert)"""
    ffn = ffn_cb_bytes(g, mg, kc, nc)
    comb = combine_cb_bytes(g)
    pw = plan_words(g)
    sizes = {**ffn, **comb, "PLAN": max(pw["count"], pw["dest"], pw["runs"]) * 4}
    blocks = push_blocks(g, mg, kc, nc)
    groups = alias_groups(sizes, ALIASES, PAGE, blocks, list(ffn))
    assert not check_pushes(groups, blocks), check_pushes(groups, blocks)
    return dict(groups=groups, total=sum(t for _, t, _ in groups), ffn=ffn, comb=comb, plan=pw, blocks=blocks)


BLOCK_CANDIDATES = ((4, 8, 8), (2, 8, 8), (2, 4, 8), (1, 4, 8), (1, 4, 4), (1, 2, 4), (1, 1, 4))


def choose_blocks(g: Geometry, budget: int = CB_BUDGET, mg_max: int = 4, explicit=None) -> tuple:
    """(MG, KC, NC): xffn's rule -- the widest candidate whose CBs (with the aliases) fit the budget"""
    if explicit:
        mg, kc, nc = explicit
        assert cb_plan(g, mg, kc, nc)["total"] <= budget and nc % 4 == 0, (explicit, cb_plan(g, mg, kc, nc)["total"])
        return mg, kc, nc
    for m_, k_, n_ in BLOCK_CANDIDATES:
        if m_ <= mg_max and g.h_t % k_ == 0 and g.h_t % n_ == 0 and cb_plan(g, m_, k_, n_)["total"] <= budget:
            return m_, k_, n_
    return 1, 1, 4


def streaming_model(
    g: Geometry, mg: int, fid: str = "HiFi2", dram_gbps: float = 480.0, mhz: float = 1350.0, rows_per_expert=None
) -> dict:
    """the README's streaming arithmetic (per chip, per layer): expert bytes, DRAM floor, FPU floor, the re-read
    factor of the M groups, the MATH utilization ceiling. rows_per_expert: a list (real routing) or None (uniform)."""
    wtb = WTB_BFP4
    gu_bytes = g.h_t * 2 * g.w_t * wtb  # one expert slice's gate|up
    dn_bytes = g.w_t * g.h_t * wtb
    per_expert = gu_bytes + dn_bytes
    if rows_per_expert is None:
        rows_per_expert = [g.n / g.e] * g.e
    units = sum(-(-(-(-r // 32)) // mg) if r else 0 for r in rows_per_expert)  # groups of <= MG tile rows
    tile_rows = sum(-(-r // 32) for r in rows_per_expert)
    weight_bytes = units * per_expert
    x_bytes = sum(rows_per_expert) * g.h * 2  # every pair's row gathered once (8 KB bf16)
    out_bytes = sum(rows_per_expert) * g.h * 2 * 2  # out_rm written, read back by the combine
    y_bytes = g.c * g.h * 2
    dram_bytes = weight_bytes + x_bytes + out_bytes + y_bytes
    # FPU: tile-matmuls per tile row = gate|up (H_T x 2 W_T) + down (W_T x H_T); cycles per tile-matmul by fidelity
    cyc = {"LoFi": 16, "HiFi2": 32, "HiFi3": 48, "HiFi4": 64}[fid]
    tmm_per_row = g.h_t * 2 * g.w_t + g.w_t * g.h_t
    useful_tmm = (g.n / 32) * tmm_per_row  # the valid rows only
    padded_tmm = tile_rows * tmm_per_row
    fpu_us = padded_tmm * cyc / g.p / mhz
    useful_us = useful_tmm * cyc / g.p / mhz
    dram_us = dram_bytes / dram_gbps / 1e3
    weights_us = weight_bytes / dram_gbps / 1e3
    flop = useful_tmm * 2 * 32**3
    return dict(per_expert_mb=per_expert / 1e6, units=units, reread_factor=weight_bytes / (g.e * per_expert),
                weight_mb=weight_bytes / 1e6, x_mb=x_bytes / 1e6, out_roundtrip_mb=out_bytes / 1e6,
                dram_mb=dram_bytes / 1e6, weights_us=weights_us, dram_floor_us=dram_us, fpu_padded_us=fpu_us,
                fpu_useful_us=useful_us, gflop=flop / 1e9, math_ceiling=useful_us / max(dram_us, fpu_us),
                crossover_rows=g.c * dram_us / useful_us if useful_us else None)  # fmt: skip


# ---------------------------------------------------------------------------------------------- the phase set
class PmkMoe:
    def __init__(self, mesh, *, num_experts: int, top_k: int, hidden: int, limit: float, gate_up: list, down: list,
                 cores=None, cb0: int = 44, cb1: int = 27, blocks=None, h_split: int = 2, fid: str = "", full_sync: bool = False,
                 probe: bool = False, gu_rt: bool = False, gu_ct: int = 0, plan_cores: int = 0,
                 plan_runs_nb: int = 1):  # fmt: skip
        """gate_up / down: this chip's slice tensors of every expert (fused_decoder.load_xsl_experts: bfloat4_b, ND-sharded
        [rows, 64]); cores: the MM cores (logical (x, y) list; None = the whole grid); cb0 / cb1: the CB bases (sec 4b map; cb1 < 31:
        the combine's tilize CBs);
        fid: math fidelity of the composite compute kernel (FFN and combine; "" = GLM53_PF_MOE_FID or HiFi2)."""
        self.mesh = mesh
        self.e, self.k, self.h, self.limit = int(num_experts), int(top_k), int(hidden), float(limit)
        self.gate_up, self.down = list(gate_up), list(down)
        assert len(self.gate_up) == self.e and len(self.down) == self.e, (len(self.gate_up), self.e)
        self.w_t = int(self.down[0].shape[-2]) // TILE
        self.h_t = self.h // TILE
        assert int(self.gate_up[0].shape[-1]) == 2 * self.w_t * TILE, (self.gate_up[0].shape, self.w_t)
        grid = mesh.compute_with_storage_grid_size()
        if cores is None:
            cores = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x))]
        self.cores = [(int(x), int(y)) for x, y in cores]
        self.p = len(self.cores)
        # Stage 14 B2.3 PMK_PLAN_P: the routing plan on the FIRST plan_cores cores only (0 = all P): its cost is linear in
        # its core count (~6 us per plan core: every core reads all P hist rows, a 288 x P prefix, P-wide barriers) and its
        # tables are partition-invariant (the single-core fill order), so any plan core count gives the same tables
        self.plan_p = int(plan_cores) or self.p
        assert 1 <= self.plan_p <= self.p, (self.plan_p, self.p)
        self.plan_nb = max(1, int(plan_runs_nb))  # PMK_PLAN_RUNS_NB: plan_runs' expert batches (L1 at few plan cores)
        self.crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in self.cores]
        )
        self.noc = [(int(w.x), int(w.y)) for w in (mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                                                   for x, y in self.cores)]  # fmt: skip
        self.cb0, self.cb1, self.h_split = int(cb0), int(cb1), int(h_split)
        self.blocks_explicit = tuple(blocks) if blocks else None
        self.fid = fid or os.environ.get("GLM53_PF_MOE_FID", "") or "HiFi2"
        self.full_sync, self.probe = bool(full_sync), bool(probe)
        # PMK_GU_RT (moe2/PERF.md sec 7): the gate|up subblock holds all MG rows (rt = mt) x gu_ct columns; gu_ct 0 =
        # auto: 1 at MG >= 2 (ct < rt: the LLK reuses every unpacked weight tile across the rows), SUBA at MG = 1
        self.gu_rt, self.gu_ct = bool(gu_rt), int(gu_ct)
        self.repl = ttnn.ReplicateTensorToMesh(mesh) if int(mesh.get_num_devices()) > 1 else None
        self.bufs = {}
        self.tbl = None
        self.calls = 0

    # ------------------------------------------------------------------------------------------ geometry / budget
    def geometry(self, c: int) -> Geometry:
        return Geometry(c=c, h=self.h, w=self.w_t * TILE, e=self.e, k=self.k, p=self.p, h_split=self.h_split)

    def plan_geometry(self, c: int) -> Geometry:
        """the routing plan's geometry: P = the plan cores (PMK_PLAN_P)"""
        return Geometry(c=c, h=self.h, w=self.w_t * TILE, e=self.e, k=self.k, p=self.plan_p, h_split=self.h_split)

    def blocks(self, c: int) -> tuple:
        return choose_blocks(self.geometry(c), explicit=self.blocks_explicit)

    def l1_bytes(self, c: int) -> int:
        return cb_plan(self.geometry(c), *self.blocks(c))["total"]

    def shard_bases(self) -> bool:
        """every expert weight ND-sharded as [full height, 2 tiles] shards (the served layout): per-shard base
        addresses instead of TensorAccessor page math per read (EpMoe._xffn_full_height_shards)"""
        try:
            for t_, rows in (
                (self.gate_up[0], int(self.gate_up[0].shape[-2])),
                (self.down[0], int(self.down[0].shape[-2])),
            ):
                spec = t_.memory_config().nd_shard_spec
                if spec is None or list(spec.shard_shape)[-2:] != [rows, 2 * TILE]:
                    return False
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------------------------------ device tables
    def _mk(self, shape):
        return ttnn.from_torch(torch.zeros(*shape, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.mesh, memory_config=DRAM, mesh_mapper=self.repl)  # fmt: skip

    def tables(self, c: int) -> dict:
        """the plan's device tables for a bucket (allocated once per bucket: the trace reads / writes them in place)"""
        if c not in self.bufs:
            g = self.geometry(c)
            pw = plan_words(
                self.plan_geometry(c), self.plan_nb
            )  # PMK_PLAN_P: chunk / epc / rmax / hist rows on plan_p cores
            lo = torch.zeros(1, 32, dtype=torch.int32)  # XSL: every expert is local (lo = 0)
            self.bufs[c] = dict(n=g.n, cap=g.cap, ep=g.ep, chunk=pw["chunk"], epc=pw["epc"], rmax=pw["rmax"],
                                disp=self._mk((1, g.cap)), dest=self._mk((1, g.n)), cnt=self._mk((1, g.ep)),
                                start=self._mk((1, g.ep)), meta=self._mk((1, 32)), hist=self._mk((self.plan_p, g.ep)),
                                stage=self._mk((self.plan_p, pw["chunk"])),
                                lo=ttnn.from_torch(lo, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                                   memory_config=DRAM, mesh_mapper=self.repl))  # fmt: skip
        if self.tbl is None:  # the experts' weight buffers are static: one address table [E, 2] uint32
            addrs = [
                int(a_) for g_, d_ in zip(self.gate_up, self.down) for a_ in (g_.buffer_address(), d_.buffer_address())
            ]
            assert max(addrs) < 2**32, max(addrs)
            tbl = torch.tensor([a_ - 2**32 if a_ >= 2**31 else a_ for a_ in addrs], dtype=torch.int32).reshape(
                1, -1
            )
            self.tbl = ttnn.from_torch(tbl, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                       memory_config=DRAM, mesh_mapper=self.repl)  # fmt: skip
        if self.probe and self.bufs.get("probe") is None:
            # [2 P (+ extra rows: ProgramB's 24 streamers), 16] uint32: reader rows, writer rows, then the extra
            self.bufs["probe"] = self._mk((2 * self.p + int(getattr(self, "probe_extra_rows", 0)), 16))
        return self.bufs[c]

    # ------------------------------------------------------------------------------------------ the program pieces
    def parts(self, x_ext, ids_rm, wt, out_rm, y, c: int, *, wait=None, h9=None) -> tuple:
        """({"w" / "a" / "c": (kernel file, compile-time args, {(x, y): runtime args}, defines, config)}, cbs,
        semaphores, io) of the MoE phase set on the MM cores.
        x_ext [C + 32, H] bf16 ROW_MAJOR (row C = zeros: the padding rows' source); ids_rm [1, k C] uint32 ROW_MAJOR
        (pair p = j C + t: the top-k expert ids, slot-major); wt [1, 1, C, k] fp32 TILE (the pairs' routing weights);
        out_rm [CAP, H] bf16 ROW_MAJOR scratch; y [1, 1, C, H] bf16 TILE (this chip's routed partial).
        wait = (semaphore id, count): H8 -- the BRISC waits >= count on that id before the plan; h9 = [(x, y, base)]:
        consumer NoC cores + the L1 base of their per-32-row-block counter words (16-B stride), +1 per combine unit
        (a block is complete at >= h_split)."""
        g = self.geometry(c)
        b = self.tables(c)
        mg, kc, nc = self.blocks(c)
        plan = cb_plan(g, mg, kc, nc)
        P, cores = self.p, self.cores
        acc = lambda t_: list(ttnn.TensorAccessorArgs(t_).get_compile_time_args())  # noqa: E731
        ids, lo, hist = ids_rm.buffer_address(), b["lo"].buffer_address(), b["hist"].buffer_address()
        dest, cnt, start, meta = (b[k_].buffer_address() for k_ in ("dest", "cnt", "start", "meta"))
        stage, disp, tbl = b["stage"].buffer_address(), b["disp"].buffer_address(), self.tbl.buffer_address()
        xa, oa, ya, wa = x_ext.buffer_address(), out_rm.buffer_address(), y.buffer_address(), wt.buffer_address()
        pa = self.bufs["probe"].buffer_address() if self.probe else 0  # (tables() keeps it at the top level)
        # combine units: (32-token chunk, column part)
        hs = self.h_split
        part = self.h * 2 // hs
        pt = part // 2048
        units = (c // TILE) * hs
        per = -(-units // P)
        cunits = lambda i: (min(i * per, units), min(per, units - min(i * per, units)))  # noqa: E731
        # PMK_COMB_STRIDE=1 (non-B lever (2)'s dealing, 2026-10-26): core i takes units i, i + P, i + 2 P, .. -- round r covers
        # the blocks [r P / h_split, (r + 1) P / h_split): the first blocks complete mid-combine (a streamed P15 can start);
        # contiguous (u0, nu) put both column parts of a block on one core, completing every block at the combine's end
        self.comb_stride = int(os.environ.get("PMK_COMB_STRIDE", "0") or 0) == 1
        self.comb_units = units  # (register_b's per-rank start offset wraps modulo this)
        if self.comb_stride:
            cunits = lambda i: (i, len(range(i, units, P)) if i < units else 0)  # noqa: E731
        subw = 4
        suba = 8 if self.full_sync and (2 * self.w_t) % 8 == 0 else subw
        # ---- compile-time args per RISC (phase bases as defines)
        ct_count = [g.n, self.e, g.ep, b["chunk"]]
        PP = self.plan_p  # PMK_PLAN_P: the plan steps' P (= P unless plan_cores)
        ct_dest = [g.n, self.e, g.ep, b["chunk"], PP, c]
        ct_runs = [self.e, g.ep, b["chunk"], PP, c, g.cap, b["epc"], b["rmax"]]
        ct_ffn_r = (
            [self.h_t, self.w_t, kc, nc, mg, self.e, g.ep, P, WTB_BFP4] + acc(self.gate_up[0]) + acc(self.down[0])
        )
        ct_comb_w = [pt, self.h_t, hs]
        ct_w, bases_w = [], {}
        for name, ct_ in (
            ("COUNT", ct_count),
            ("DEST", ct_dest),
            ("RUNS", ct_runs),
            ("FFN", ct_ffn_r),
            ("COMB", ct_comb_w),
        ):
            bases_w[name] = len(ct_w)
            ct_w += ct_
        ct_ffn_a, ct_comb_r = [self.h_t, nc, kc, mg], [c, self.k, self.h * 2, part, pt, hs]
        ct_a = ct_ffn_a + ct_comb_r
        bases_a = dict(FFN=0, COMB=len(ct_ffn_a))
        ct_ffn_c, ct_comb_c = [self.h_t, self.w_t, kc, nc, mg, subw, suba], [self.k, pt]
        ct_c = ct_ffn_c + ct_comb_c
        bases_c = dict(FFN=0, COMB=len(ct_ffn_c))
        # ---- runtime args per core
        flat_noc = [v for xy in self.noc for v in xy]
        rt_w, rt_a, rt_c, rbases_w, rbases_a = {}, {}, {}, {}, {}
        for i, xy in enumerate(cores):
            u0, nu = cunits(i)
            segs = [("COUNT", [ids, lo, hist, i]), ("DEST", [ids, lo, hist, dest, cnt, start, meta, i, stage]),
                    ("RUNS", [hist, cnt, start, meta, stage, disp, i]),
                    ("FFN", [xa, disp, cnt, start, tbl, i, g.cap, pa]), ("COMB", [ya, u0, nu])]  # fmt: skip
            args = []
            for name, a_ in segs:
                rbases_w[name] = len(args)
                args += a_
            rbases_w["CORES"] = len(args)
            args += flat_noc
            rbases_w["WAIT"] = len(args)
            args += [int(wait[1])] if wait else [0]
            rbases_w["H9"] = len(args)
            args += _h9_args(h9)
            rt_w[xy] = args
            a_args = [oa, xa, disp, g.cap, pa, P, i]
            rbases_a["FFN"] = 0
            rbases_a["COMB"] = len(a_args)
            a_args += [oa, dest, wa, u0, nu]
            rbases_a["CORES"] = len(a_args)
            rt_a[xy] = a_args + flat_noc
            rt_c[xy] = [nu]
        # ---- defines
        common = [("PMK_MOE_CB0", str(self.cb0)), ("PMK_MOE_CB1", str(self.cb1)), ("PMK_P", str(P)), ("PMK_MOE_REV", str(PMK_MOE_REV)),
                  ("XFFN_X_ON_NOC1", "1"), ("PLAN_SORTED", "1")]  # fmt: skip
        if self.shard_bases():
            common.append(("XFFN_SHARD_BASES", "1"))
        if self.plan_p < P:
            common.append(("PMK_PLAN_P", str(self.plan_p)))
        if self.plan_nb > 1:
            common.append(("PMK_PLAN_RUNS_NB", str(self.plan_nb)))
        if self.probe:
            common.append(("XFFN_PROBE", "1"))
        d_w = common + [(f"PMK_CT_{k_}", str(v)) for k_, v in bases_w.items()] + \
            [(f"PMK_RT_{k_}", str(v)) for k_, v in rbases_w.items()]  # fmt: skip
        if wait:
            d_w.append(("PMK_MOE_WAIT_SEM", str(int(wait[0]))))
        if self.comb_stride:
            d_w += [("PMK_COMB_STRIDE", str(P)), ("PMK_COMB_UNITS", str(units))]
        if h9:
            d_w.append(("PMK_H9", "1"))
            if isinstance(h9, dict):  # the TABLE mode (P15 streamed behind the combine): one word per row block
                d_w.append(("PMK_H9_TABLE", "1"))
        d_a = common + [(f"PMK_CT_{k_}", str(v)) for k_, v in bases_a.items()] + \
            [(f"PMK_RT_{k_}", str(v)) for k_, v in rbases_a.items()]  # fmt: skip
        if self.comb_stride:  # the combine reader's unit loop (comb_reader.hpp) strides with the writer's
            d_a += [("PMK_COMB_STRIDE", str(P)), ("PMK_COMB_UNITS", str(units))]
        d_c = (
            common
            + [(f"PMK_CT_{k_}", str(v)) for k_, v in bases_c.items()]
            + [("PMK_RT_FFN", "0"), ("PMK_RT_COMB", "0")]
        )
        dest_tiles = 8 if self.full_sync else 4  # fp32 dest: half-sync 4 tiles, full-sync 8
        if self.gu_rt:
            ct = self.gu_ct or (1 if mg >= 2 else suba)
            assert ct >= 1 and (2 * self.w_t) % ct == 0 and mg * ct <= dest_tiles, (mg, ct, dest_tiles)
            d_c = d_c + [("PMK_GU_RT", "1"), ("PMK_GU_CT", str(ct)), ("PMK_DEST_TILES", str(dest_tiles))]
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, self.fid), math_approx_mode=False,
                                          fp32_dest_acc_en=True, dst_full_sync_en=self.full_sync)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[self.cb0 + 8] = ttnn.UnpackToDestMode.UnpackToDestFp32  # CB_GUP: the fp32 partials copied exactly
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        raw = dict(
            w=("moe_brisc.cpp", ct_w, rt_w, d_w, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            a=("moe_ncrisc.cpp", ct_a, rt_a, d_a, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
            c=("moe_trisc.cpp", ct_c, rt_c, d_c, cc),
        )
        cbs = self.cb_descriptors(plan)
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.crs, initial_value=0) for i in (5, 6, 7, 8)]
        io = [x_ext, ids_rm, wt, b["lo"], b["hist"], b["dest"], b["cnt"], b["start"], b["meta"], b["stage"], b["disp"],
              self.tbl, out_rm, y] + list(self.gate_up) + list(self.down)  # fmt: skip
        if self.probe:
            io.append(self.bufs["probe"])
        return raw, cbs, sems, io

    CB_INDEX = dict(
        CTL=0, WGU=1, XRM=2, XT=3, WD=4, WCTL=5, TAB=6, WTAB=7, GUP=8, ACT=9, YRM=10, PLAN=11, CIN=12, CW=13, CIDX=14
    )  # moe_common.hpp: PMK_MOE_CB0 + k
    CB_INDEX_LOW = dict(COUT=0, CRM=1)  # PMK_MOE_CB1 + k (< 32: the combine's tilize helper)

    def cb_index(self, name: str) -> int:
        return self.cb0 + self.CB_INDEX[name] if name in self.CB_INDEX else self.cb1 + self.CB_INDEX_LOW[name]

    CB_FMT = dict(
        WGU=ttnn.bfloat4_b,
        WD=ttnn.bfloat4_b,
        XRM=BF16,
        XT=BF16,
        ACT=BF16,
        YRM=BF16,
        GUP=FP32,
        CIN=BF16,
        CW=FP32,
        COUT=BF16,
        CRM=FP32,
    )  # others uint32

    def cb_descriptors(self, plan: dict) -> list:
        cbs = []
        for members, total, pages in plan["groups"]:
            fds = [ttnn.CBFormatDescriptor(buffer_index=self.cb_index(m), data_format=self.CB_FMT.get(m, ttnn.uint32),
                                           page_size=pages[m]) for m in members]  # fmt: skip
            cbs.append(ttnn.CBDescriptor(total_size=total, core_ranges=self.crs, format_descriptors=fds))
        return cbs

    def kernel(self, name, ct, rt: dict, defs, cfg):
        rta = ttnn.RuntimeArgs()
        for (x, y), args in rt.items():
            rta[x][y] = [int(a) for a in args]
        return ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pmk", name),
                                     source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.crs,
                                     compile_time_args=ct, runtime_args=rta, config=cfg, defines=list(defs))  # fmt: skip

    def program(self, x_ext, ids_rm, wt, out_rm, y, c: int, **kw):
        raw, cbs, sems, io = self.parts(x_ext, ids_rm, wt, out_rm, y, c, **kw)
        from models.autoports.zai_org_glm_5_3_flash.tt.cb_guard import check

        check(cbs, "pmk_moe")
        kernels = [self.kernel(*raw[r]) for r in ("w", "a", "c")]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs), io

    def scratch(self, c: int) -> dict:
        """the per-bucket scratch tensors (allocated once: a trace must not allocate): out_rm, y"""
        g = self.geometry(c)
        key = ("scratch", c)
        if key not in self.bufs:
            self.bufs[key] = dict(
                out_rm=ttnn.empty((g.cap, self.h), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM),
                y=ttnn.empty((1, 1, c, self.h), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM))  # fmt: skip
        return self.bufs[key]

    def __call__(self, x_ext, ids_rm, wt, c: int, **kw):
        """standalone: ONE program (plan + FFN + combine) -> y [1, 1, C, H] bf16 TILE (a persistent scratch: copy it
        out before the next call)"""
        s = self.scratch(c)
        prog, io = self.program(x_ext, ids_rm, wt, s["out_rm"], s["y"], c, **kw)
        ttnn.generic_op(io, prog)
        self.calls += 1
        return s["y"]

    def probe_read(self) -> list:
        """XFFN_PROBE counters of the last run per chip (kernels/prefill/moe_xffn_probe.hpp): [{"reader", "writer"}]"""
        t = self.bufs["probe"]
        v = ttnn.to_torch(t, mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh, 0)) if self.repl is not None \
            else ttnn.to_torch(t)  # fmt: skip
        v = v.to(torch.int64) & 0xFFFFFFFF
        extra = int(getattr(self, "probe_extra_rows", 0))
        p2 = 2 * self.p + extra
        out = []
        for i in range(v.shape[0] // p2):
            ch = dict(reader=v[i * p2 : i * p2 + self.p], writer=v[i * p2 + self.p : i * p2 + 2 * self.p])
            if extra:  # ProgramB: the streamers' rows (kernels/pmk/exp_streamer.hpp S_* slots)
                ch["streamer"] = v[i * p2 + 2 * self.p : (i + 1) * p2]
            out.append(ch)
        return out


def inputs_from_op_path(x, idx, sel, scaling: float):
    """the phase set's inputs from the op path's (x [1,1,C,H] bf16 TILE, idx [1,1,C,k] uint16, sel [1,1,C,k] fp32 --
    EpMoe.__call__'s inputs) with EpMoe's own ops, so the two paths see IDENTICAL x_ext / ids / weights:
    -> (x_ext [C + 32, H] bf16 ROW_MAJOR, ids_rm [1, k C] uint32, wt [1, 1, C, k] fp32 TILE)"""
    c, h, k = int(x.shape[-2]), int(x.shape[-1]), int(idx.shape[-1])
    ids_t = ttnn.transpose(ttnn.typecast(idx, ttnn.uint32), -2, -1)
    ids_rm = ttnn.to_layout(ttnn.reshape(ids_t, (1, c * k)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
    ttnn.deallocate(ids_t)
    w = ttnn.multiply(sel, ttnn.reciprocal(ttnn.add(ttnn.sum(sel, dim=-1, keepdim=True), 1e-20)))
    wt = ttnn.multiply(w, scaling)
    ttnn.deallocate(w)
    x_ext = ttnn.to_layout(ttnn.pad(ttnn.reshape(x, (c, h)), ((0, TILE), (0, 0)), 0.0), ttnn.ROW_MAJOR_LAYOUT,
                           memory_config=DRAM)  # fmt: skip
    return x_ext, ids_rm, wt
