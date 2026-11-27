# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16 M-B (operator / coordinator 2026-10-24 05:12:04Z): PROGRAM B AS THE LAST WINDOW OF THE FUSED LAYER PROGRAM.

Program B (tt/pmk_moe2.ProgramB: plan, routed experts, combine) joins program A's fused program (A15 = P1 .. P10) as
phase P14 of the family engine. Its own kernels run unchanged, each wrapped (kernels/pmk/b_wrap_*.cpp):
  - receivers = B's compute cores (ProgramB(compute_only=members): the family's 64 members), streamers = B's 24
    bank-adjacent cores (on the root / MM-only BRISCs + NCRISCs); every other core of a hosting composite skips B
    (guard word 0);
  - L1: B's receiver CBs (1,107,556 B at TP2) exceed every window region, so B's window is a RECLAIM window
    (pmk_layer.win_reclaim): laid out from the CB base over every dead allocation; it takes FRESH indices first and
    reuses earlier windows' indices of the same data format / unpack mode for the rest (the member union is otherwise
    47 + 17 + 4 > 64); pmk_window.hpp enter_reclaim balances and re-syncs their page counters;
  - the CB map reaches the kernels as PMK_MOE_CBX_<NAME> (moe_common.hpp), the unpack-to-dest modes are remapped;
  - B's arg bases (PMK_CT_* / PMK_RT_*) become phase-relative (PH_CT0 + k, ph_rt0 + guard + k);
  - the TRISC part is a flex part (dst_any, fid_mode "template": PMK_PART_FID = B's HiFi2 under the items' HiFi4;
    ffn_compute.hpp / comb_compute.hpp select the templated matmul / broadcast multiply);
  - semaphores 5 .. 13 are handed over A -> B by the two-round rendezvous on id 14 (b_wrap.hpp).
M-B1 stages B's inputs (x_tile / x_ext / ids_rm / wt / y_s) from the chain's op path (the router is not in-program
yet): the unit labels its row pre-staged."""

from __future__ import annotations

import dataclasses
import os
import re

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as V1
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "pmk")
FMT = {ttnn.bfloat16: "bf16", ttnn.float32: "fp32", ttnn.uint32: "u32", ttnn.bfloat4_b: "bf4", ttnn.bfloat8_b: "bf8"}
RDV = 14  # b_wrap.hpp PMK_BW_RDV
GW_DM, GW_C, GW_S = 6, 1, 7  # the guard words (b_wrap.hpp): receivers DM / TRISC, streamers DM (+ rel_off)
LOW32 = ("COUT", "CRM")  # the combine's tilize helper takes CB indices < 32 (moe_common.hpp)
# RAW CBs: touched only as words (the data-movement RISCs; CTL's [mt] word on the TRISC through its read pointer), never
# unpacked / packed -> their index's data format / unpack mode is irrelevant: any dead index serves
RAW = ("CTL", "WCTL", "TAB", "WTAB", "PLAN", "CIDX")
NAMES = ("CTL", "WGU", "XRM", "XT", "WD", "WCTL", "TAB", "WTAB", "GUP", "ACT", "YRM", "PLAN", "CIN", "CW", "CIDX",
         "COUT", "CRM", "CSH", "ACTS", "CIN2", "CW2")  # fmt: skip


def rebase_defines(defs, guard: int) -> list:
    """B's arg-base defines -> phase-relative: PMK_CT_X k -> (PH_CT0 + k), PMK_RT_X k -> (ph_rt0 + guard + k)"""
    out = []
    for k, v in defs:
        k, sv = str(k), str(v)
        if re.fullmatch(r"PMK_CT_[A-Z0-9_]+", k) and sv.isdigit():
            out.append((k, f"(PH_CT0 + {int(sv)})"))
        elif re.fullmatch(r"PMK_RT_[A-Z0-9_]+", k) and sv.isdigit():
            out.append((k, f"(ph_rt0 + {guard + int(sv)})"))
        else:
            out.append((k, sv))
    return out


def _b_formats(b) -> dict:
    """B's CB name -> ttnn data format (PmkMoe2.cb_descriptors' map; uint32 default)"""
    f = dict(b.CB_FMT, CSH=ttnn.bfloat16, ACTS=ttnn.bfloat16, CW2=ttnn.float32)
    if b.lv.out_bfp8:
        f.update(YRM=ttnn.bfloat8_b, CIN=ttnn.bfloat8_b, CIN2=ttnn.bfloat8_b)
    return {n: f.get(n, ttnn.uint32) for n in NAMES}


# DataFormat codes (CBFormatDescriptor.data_format_as_uint8) -> engine format names
FMT_CODE = {0: "fp32", 5: "bf16", 6: "bf8", 7: "bf4", 24: "u32"}


def _streamer_cbs_desc(cbs, b) -> list:
    """the streamers' two CBs FROM ProgramB.parts' own descriptors: [(name, total bytes, page, engine fmt)] -- the ring's
    format follows B (coordinator 19:56:43Z: program B's stream formats / fidelity may change for the accuracy gap)"""
    s_crs, out = str(b.s_crs), {}
    want = {b.cb_index("WGU"): "WGU", b.cb_index("TAB"): "TAB"}
    for d in cbs:
        if str(d.core_ranges) != s_crs:
            continue
        for f in d.format_descriptors:
            n = want.get(int(f.buffer_index))
            if n is not None:
                out[n] = (n, int(d.total_size), int(f.page_size), FMT_CODE[int(f.data_format_as_uint8)])
    assert set(out) == {"WGU", "TAB"}, ("program B's streamer CB descriptors", sorted(out))
    return [out["WGU"], out["TAB"]]


def _streamer_cbs(b, c: int) -> list:
    """the streamers' two CBs as ProgramB.parts declares them (the arithmetic mirror; register_b cross-checks it against
    the descriptors while B's ring is bfp4): [(name, total bytes, page)]"""
    g = b.geometry(c)
    _, kc, nc = b.blocks(c)
    gu_bb = kc * 2 * b.w_t * M2.WTB_BFP4
    slot_bb = max(gu_bb, b.w_t * nc * M2.WTB_BFP4)
    tab_b = -(-(V1.SYNC_WORDS_B + 3 * g.ep * 4) // 64) * 64 + 64 + M2.lpt_scratch_bytes(g.ep)
    if b.inplace:
        tab_b += 64 + 2 * b.e * 4 + 64 + 8 * (b.w_t + b.h_t // 2) + 64
    return [("WGU", b.lv.ring_slots * slot_bb, M2.WTB_BFP4), ("TAB", tab_b, tab_b)]


def _a_unpack_modes(e, role: str, pid: str) -> dict:
    """the merged unpack-to-dest modes of the role's TRISC parts in this program (B excluded): {index: mode}"""
    out = {}
    for p in e.phases:
        if p.pid == pid or p.body.status != "body" or not e.in_scope(p.pid) or not e._hosts(role, "c", p):
            continue
        for sub in p.body.subparts(role, "c"):
            cc = sub.get("compute")
            um = (
                None
                if cc is None
                else (
                    cc.get("unpack_to_dest_mode") if isinstance(cc, dict) else getattr(cc, "unpack_to_dest_mode", None)
                )
            )
            if um is None:
                continue
            for i, m in um.items() if isinstance(um, dict) else enumerate(list(um)):
                if m != ttnn.UnpackToDestMode.Default:
                    out[int(i)] = m
    return out


def _used(e, role: str) -> set:
    """the CB indices this program declares on `role` at launch: the allocations of its ACTIVE phases + CB_ARG (the
    arg-table rows, declared at build). An index registered only by an inactive phase (CB_PROBE 31: P0, probe builds) is
    NOT declared -- B taking it must declare it (register_b's fresh list), else the TRISC compiles its formats Invalid
    """
    return {i for idx, _, _, _ in e._slots(role, e.active_phases()) for i in idx} | {PMK.CB_ARG}


def _assign(e, role: str, want: list, windowed: bool, pid: str) -> dict:
    """B's CB name -> index on `role`. want = [(name, engine fmt, page, mode, low32)]. Fresh indices first (outside every
    allocation of this program and the 4 the window barriers / pad take: pmk_layer._win_layout's free[:4]); on a
    windowed role, then the program's own indices of the same (fmt, page)-compatible format and unpack mode."""
    used = _used(e, role)
    free = [i for i in range(63, -1, -1) if i not in used]
    reserved = set(free[:4]) if windowed else set()
    # (W7 (I1)) the allocations the reclaim keeps in place (XS live through B): never one of B's indices, nor an alias
    prot = set((getattr(e, "reclaim_protect", None) or {}).get(role, ()))
    for grp in getattr(e, "ALIASES", ()) or ():
        if prot & set(grp):
            prot |= set(grp)
    reserved |= prot
    pool = sorted(i for i in free if i not in reserved)
    d = dict(e.cbs.get(role, {}))
    if role in ("root", "mm_only"):
        d.update(e.cbs.get("streamer", {}))
    a_modes = _a_unpack_modes(e, role, pid)
    out, taken = {}, set()
    # the < 32 names first (their candidates are scarce), then the format-bound (unpacked / packed) names, the raw last:
    # fresh indices go where the format matters
    for name, fmt, page, mode, low, raw in sorted(want, key=lambda w: (not w[4], w[5])):
        cand = [i for i in pool if i not in taken and (i < 32 or not low)]
        if cand:
            out[name] = cand[0]
            taken.add(cand[0])
            continue
        if not windowed:
            raise ValueError(
                f"PMK B window on {role}: no free CB index for {name} (in place, no reuse): used {sorted(used)}"
            )
        reuse = [i for i in sorted(used) if i not in taken and i not in reserved and (i < 32 or not low) and i in d
                 and (raw or (d[i].fmt == fmt and a_modes.get(i, ttnn.UnpackToDestMode.Default) == mode))]  # fmt: skip
        if not reuse:
            raise ValueError(f"PMK B window on {role}: no free or reusable ({fmt}, mode {mode}) CB index for {name}; "
                             f"pool {pool}, taken {sorted(taken)}, A formats {sorted((i, d[i].fmt) for i in used if i in d)}")  # fmt: skip
        out[name] = reuse[0]
        taken.add(reuse[0])
    return out


def _h9_rank_patch(d_w, rt_w, cores, guard, h9_by_rank: dict) -> dict:
    """{rank: {core: guard + B's BRISC args with the H9 TABLE segment of that rank}} (the segment has the same length on
    every rank: n targets, 3 n words, 2 words per row block; only the values differ)"""
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as V1  # noqa: PLC0415

    off = next(int(v) for k, v in d_w if str(k) == "PMK_RT_H9")
    seg0 = V1._h9_args(h9_by_rank[0])
    out = {}
    for rank, h9 in h9_by_rank.items():
        seg = V1._h9_args(h9)
        assert len(seg) == len(seg0), (rank, len(seg), len(seg0))
        out[rank] = {}
        for x in cores:
            v = list(rt_w[x])
            assert v[off : off + len(seg0)] == seg0, (x, "the H9 segment at PMK_RT_H9")
            v[off : off + len(seg)] = seg
            out[rank][x] = guard(x) + v
    return out


def _comb_uoff_patch(b, d_w, d_a, rt_a, cores, guard, h9_rk, h9_by_rank) -> dict | None:
    """PMK_COMB_STRIDE with the P15 stream: each chip's combine starts at the half it SENDS (h9_by_rank[rank]["uoff"]):
    the writer's u0 (PMK_RT_COMB + 1, in h9_rk's guarded rows, patched in place) and the reader's u0 (PMK_RT_COMB + 3)
    become u0 + uoff mod units. Returns {rank: {core: guard + reader args}} for the reader RISC, or None"""
    if not (h9_rk and getattr(b, "comb_stride", False)):
        return None
    units = int(b.comb_units)
    off_w = GW_DM + next(int(v) for k, v in d_w if str(k) == "PMK_RT_COMB") + 1  # writer COMB: [y, u0, nu]
    off_a = next(int(v) for k, v in d_a if str(k) == "PMK_RT_COMB") + 3  # reader COMB: [out, dest, wt, u0, nu]
    out = {}
    for rank, d_ in h9_rk.items():
        uo = int(h9_by_rank[rank].get("uoff", 0))
        for x in d_:
            d_[x][off_w] = (d_[x][off_w] + uo) % units
        out[rank] = {}
        for x in cores:
            v = list(rt_a[x])
            v[off_a] = (v[off_a] + uo) % units
            out[rank][x] = guard(x) + v
    return out


def register_b(
    a, b, c: int, *, x_ext, ids_rm, wt, x_tile, y_s=None, pid: str = "P14", xs_base=None, h9_by_rank: dict | None = None
) -> dict:
    """program B as phase `pid` (the reclaim window) of program A's engine `a.engine` (PmkProgramA). B's inputs are
    tensors the caller staged; B writes its own scratch (b.scratch(c): out_rm, y). Returns a report."""
    e = a.engine
    rc = e.place.role_cores()
    s = b.scratch(c)
    # h9_by_rank (non-B lever (2), P15 streamed behind the combine): {rank: dict(targets, table)} -- B's combine raises the
    # P15 lane word of each finished row block (pmk_moe._h9_args TABLE mode); the table differs per chip (own / peer rows)
    raw, _cbs, b_sems, io = b.parts(x_ext, ids_rm, wt, s["out_rm"], s["y"], c, x_tile=x_tile,
                                   y_s=y_s if y_s is not None else s.get("y_s"), xs_base=xs_base,
                                   **({"h9": h9_by_rank[0]} if h9_by_rank else {}))  # fmt: skip
    recv, strm = [tuple(x) for x in b.cores], [tuple(x) for x in b.streamers]
    role_of = {tuple(x): r for r in ("member", "root", "mm_only", "lane", "tail") for x in rc[r]}
    recv_roles = sorted({role_of[x] for x in recv})
    strm_roles = sorted({role_of[x] for x in strm})
    assert recv_roles == ["member"], ("B's receivers must be the members (ProgramB(compute_only=members))", recv_roles)
    # the adj streamers land on root / MM-only / tail cores (never a member or a lane: those composites are A's / coll's)
    assert set(strm_roles) <= {"root", "mm_only", "tail"}, ("B's streamers on root / MM-only / tail cores", strm_roles)
    # ---- the CB maps
    plan = b.plan(c)
    fmts = _b_formats(b)
    f_modes = {}  # B's default index -> mode (its compute config)
    cc_b = raw["c"][4]
    for i, m in enumerate(list(cc_b.unpack_to_dest_mode)):
        if m != ttnn.UnpackToDestMode.Default:
            f_modes[i] = m
    mode_of = lambda name: f_modes.get(b.cb_index(name), ttnn.UnpackToDestMode.Default)  # noqa: E731
    groups = [
        (list(members), int(total), {m: int(pages[m]) for m in members}) for members, total, pages in plan["groups"]
    ]
    want_r = [(n, FMT[fmts[n]], pg[n], mode_of(n), n in LOW32, n in RAW) for mem, _, pg in groups for n in mem]
    scb_d = _streamer_cbs_desc(_cbs, b)
    if scb_d[0][3] == "bf4":  # the mirror and the descriptors agree while the ring is bfp4
        assert [t[:3] for t in scb_d] == [tuple(t) for t in _streamer_cbs(b, c)], (scb_d, _streamer_cbs(b, c))
    scb = [t[:3] for t in scb_d]
    s_fmt = {n: f for n, _, _, f in scb_d}
    # the streamer roles' TRISC runs no B code: both streamer CBs are raw there
    want_s = [(n, s_fmt[n], page, ttnn.UnpackToDestMode.Default, False, True)
              for n, _, page in scb]  # fmt: skip
    rep = dict(receivers=len(recv), streamers=len(strm), roles=dict(receiver=recv_roles, streamer=strm_roles), maps={},
               need={}, fresh={})  # fmt: skip
    al = lambda x: -(-int(x) // e.WIN_ALIGN) * e.WIN_ALIGN  # noqa: E731
    e.win_reclaim = dict(e.win_reclaim or {})
    cbx = {}
    for role in recv_roles + strm_roles:
        windowed = role in e.window_roles
        want = want_r if role in recv_roles else want_s
        m = _assign(e, role, want, windowed, pid)
        cbx[role] = m
        rep["maps"][role] = dict(m)
        d = dict(e.cbs.get(role, {}))
        if role in ("root", "mm_only"):
            d.update(e.cbs.get("streamer", {}))
        if role in recv_roles:
            allocs = [(mem, total, pg) for mem, total, pg in groups]
        else:
            allocs = [([n], total, {n: page}) for n, total, page in scb]
        if (
            windowed
        ):  # the reclaim allocations (placed from the CB base by the engine); fresh indices declared for formats
            ra, fresh = [], []
            # fresh = not declared by this program (_used), NOT "never registered": mbw2 (2026-10-24) put CRM on CB_PROBE
            # 31 (registered by the inactive P0, so `i not in d` skipped its declaration) -> the member TRISC compiled
            # index 31 with format Invalid (255) and B's L1-accumulate pack / tilize ran through it: the mailbox overwrite
            used_r = _used(e, role)
            for mem, total, pg in allocs:
                ra.append(([(m[n], pg[n]) for n in mem], total))
                for n in mem:
                    i = m[n]
                    if i not in used_r:
                        fresh.append(PMK.CBUse(index=i, fmt=FMT[fmts[n]] if role in recv_roles else
                                               s_fmt[n], pages=1, live=(pid,),
                                               page=pg[n], note=f"B {n}", users=(pid,)))  # fmt: skip
            e.win_reclaim[role] = dict(pid=pid, allocs=ra, fresh=fresh)
            rep["need"][role], rep["fresh"][role] = sum(al(t) for _, t in ra), [u.index for u in fresh]
        else:  # in place (a role the program does not window: the roots at TP2 / TP8)
            for mem, total, pg in allocs:
                for n in mem:
                    e._use(role, m[n], s_fmt[n], total // pg[n], (pid,), page=pg[n],
                           note=f"B streamer {n}", users=(pid,), solo=True)  # fmt: skip
            rep["need"][role] = sum(al(t) for _, t, _ in allocs)
    # ---- the parts
    # the bisect knob GLM53_PMK_B_HUB=<i>: the rendezvous hub = receiver i (default 0; -1 = the last) -- does the
    # mbw2 mailbox overwrite on core (3, 0) follow the hub or stay with B's first receiver / rectangle 0?
    n_b = len(recv) + len(strm)
    h6 = e.handoffs.get("H6")
    h6_live = h6 is not None and h6.counter == RDV and e._has_live_producer("H6")
    p10_roles = e.wait_roles(next(p for p in e.phases if p.pid == "P10")) if e.in_scope("P10") else set()

    def base(core):  # PMK_BW_RDV's value on the core after A: H6's wait value where P10 waited it, else 0
        return int(h6.wait_value) if (h6_live and role_of[core] in p10_roles) else 0

    # mb8 (TP8): H6 is an aggregator SET on the members' / roots' id 14 (pmk_assembly._h6_agg) landing at a time no B core
    # knows; a round-1 +1 arriving before it would be erased -> the hub is the first streamer whose id 14 only counts (a
    # role P10 does not wait, base 0). TP2 (lane increments, order-free) keeps receiver 0.
    set_h6 = h6_live and bool(getattr(a, "f_p8p10", False)) and int(getattr(a, "n", 2)) != 2
    hub_env = os.environ.get("GLM53_PMK_B_HUB", "")
    if hub_env:
        hub = recv[int(hub_env)]
    elif set_h6:
        # prefer a table-mode role (2 KB arg rows: the hub's list adds 2 n words; the tail BRISC composite is not in table
        # mode and ends over the 256-arg floor with it: mb8 2x4 mock 19:33Z)
        off = [x for x in strm if role_of[x] not in p10_roles]
        hub = next((x for x in off if role_of[x] in PMK.ARG_TABLE_ROLES), off[0] if off else None)
        assert hub is not None, ("mb8: no streamer off P10's wait roles to host the rendezvous hub", strm_roles)
    else:
        hub = recv[0]
    assert base(hub) == 0 or not set_h6, ("mb8: the rendezvous hub's id 14 must only count", hub, base(hub))
    noc_all = dict(b.noc_of)
    noc_all.update(b.s_noc)
    hx, hy = noc_all[hub]
    hub_list = [v for x in recv + strm for v in noc_all[x]]  # the streamer hub's release list (every B core)

    def guard(core):
        return [1, hx, hy, 1 if core == hub else 0, base(core), n_b]

    def guard_s(core, n_args):  # streamers: + rel_off (the hub's list follows its own B args)
        return guard(core) + [GW_S + n_args if core == hub else 0]

    nobody = os.environ.get("GLM53_PMK_B_NOBODY", "0") == "1"  # the bisect knob: B's wrappers skip its body

    def kd(role):
        stage = os.environ.get(
            "GLM53_PMK_B_STAGE", ""
        )  # the bisect: 1 = plan only, 2 = plan + FFN, 3 = + the combine reader
        # (CIN / CW drained), 4 = + the multiply / L1-acc pack (CRM drained), 5 = + the tilize (COUT popped, no y), 6 / "" = all
        # GLM53_PMK_B_FASTTILIZE (default 1): the combine's tilize on B's own (fast) path in the full-sync fused composite
        fast = os.environ.get("GLM53_PMK_B_FASTTILIZE", "1") == "1" and role in recv_roles
        return ([(f"PMK_MOE_CBX_{n}", str(i)) for n, i in sorted(cbx[role].items())] + ([("PMK_BW_NOBODY", "1")] if nobody else [])
                + ([("PMK_BW_STAGE", str(int(stage)))] if stage else []) + ([("PMK_COMB_FAST_TILIZE", "1")] if fast else []))  # fmt: skip

    w_file, ct_w, rt_w, d_w, cfg_w = raw["w"]
    a_file, ct_a, rt_a, d_a, cfg_a = raw["a"]
    c_file, ct_c, rt_c, d_c, cc_c = raw["c"]
    s_file, ct_s, rt_s, d_s, cfg_s = raw["s"]
    for f_, want_f in (
        (w_file, "moe_brisc.cpp"),
        (a_file, "moe_ncrisc.cpp"),
        (c_file, "moe_trisc.cpp"),
        (s_file, "moe_stream_brisc.cpp"),
    ):
        assert os.path.basename(f_) == want_f, (f_, want_f)
    # the receivers' compute config: B's own, its unpack-to-dest modes on the remapped indices
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i, mo in f_modes.items():
        name = next(n for n in NAMES if b.cb_index(n) == i)
        modes[cbx["member"][name]] = mo
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=cc_c.math_fidelity, math_approx_mode=cc_c.math_approx_mode,
                                      fp32_dest_acc_en=cc_c.fp32_dest_acc_en, dst_full_sync_en=cc_c.dst_full_sync_en)  # fmt: skip
    cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
    parts = {}
    mem_cores = [tuple(x) for x in rc["member"]]
    rt_dm0 = {x: guard(x) + list(rt_w[x]) for x in recv}
    h9_rk = _h9_rank_patch(d_w, rt_w, recv, guard, h9_by_rank) if h9_by_rank else None
    rt_dm1_rk = _comb_uoff_patch(b, d_w, d_a, rt_a, recv, guard, h9_rk, h9_by_rank)
    parts[("member", "dm0")] = dict(
        name="bw",
        src=os.path.join(KDIR, "b_wrap_w.cpp"),
        ct=list(ct_w),
        **({"rt_by_rank": h9_rk} if h9_rk else {"rt": rt_dm0}),
        defines=rebase_defines(d_w, GW_DM),
        kdefines=kd("member"),
        config=cfg_w,
        idle_ok=True,
        noc_xy={x: (1,) for x in recv},
        probe_marks=("B.handover", "B.plan", "B.sweep", "B.comb"),
    )  # fmt: skip  # PMK_PP_MARK in b_wrap_w / moe_brisc
    parts[("member", "dm1")] = dict(
        name="bw",
        src=os.path.join(KDIR, "b_wrap_a.cpp"),
        ct=list(ct_a),
        **({"rt_by_rank": rt_dm1_rk} if rt_dm1_rk else {"rt": {x: guard(x) + list(rt_a[x]) for x in recv}}),
        defines=rebase_defines(d_a, GW_DM),
        kdefines=kd("member"),
        config=cfg_a,
        idle_ok=True,
        probe_marks=("B.handover", "B.ffnw", "B.ffn_barrier", "B.combr"),
    )  # fmt: skip  # b_wrap_a / moe_ncrisc
    parts[("member", "c")] = dict(name="bw", src=os.path.join(KDIR, "b_wrap_c.cpp"), ct=list(ct_c),
                                  rt={x: [1] + list(rt_c[x]) for x in recv}, defines=rebase_defines(d_c, GW_C),
                                  kdefines=kd("member"), compute=cc, dst_any=True, fid_mode="template", idle_ok=True)  # fmt: skip
    assert set(recv) <= set(mem_cores)
    for role in strm_roles:
        mine = [x for x in strm if role_of[x] == role]
        rt0 = {x: guard_s(x, len(rt_s[x])) + list(rt_s[x]) + (hub_list if x == hub else []) for x in mine}
        nxy = {x: (1,) + (tuple(GW_S + len(rt_s[x]) + 2 * i for i in range(n_b)) if x == hub else ()) for x in mine}
        parts[(role, "dm0")] = dict(name="bw", src=os.path.join(KDIR, "b_wrap_s.cpp"), ct=list(ct_s), rt=rt0,
                                    defines=rebase_defines(d_s, GW_S), kdefines=kd(role), config=cfg_s, idle_ok=True,
                                    noc_xy=nxy)  # fmt: skip
        if "s1" in raw:
            s1_file, ct_s1, rt_s1, d_s1, cfg_s1 = raw["s1"]
            parts[(role, "dm1")] = dict(name="bw", src=os.path.join(KDIR, "b_wrap_s1.cpp"), ct=list(ct_s1),
                                        rt={x: guard(x) + [0] + list(rt_s1[x]) for x in mine},
                                        defines=rebase_defines(d_s1, GW_S), kdefines=kd(role), config=cfg_s1,
                                        idle_ok=True)  # fmt: skip
    # ---- semaphores: B's ids + the rendezvous on every hosting role (initial values are B's 0: the wrappers reset)
    for role in recv_roles + strm_roles:
        for sid in range(5, RDV + 1):
            e._sem(sid, role)
    # ---- the engine's PLANNED P14 uses (the family's xffn form: IN0 row K-blocks, its rings, ...) are not B's: B's CBs
    # are the reclaim tab / its in-place streamer uses above, so P14 leaves every other CB use's users / live
    stripped = 0
    for role_, d_ in e.cbs.items():
        for i_, u_ in list(d_.items()):
            if pid in tuple(u_.users or ()) + tuple(u_.live or ()) and not str(u_.note).startswith("B "):
                d_[i_] = dataclasses.replace(u_, live=tuple(q for q in u_.live if q != pid),
                                             users=tuple(q for q in (u_.users or ()) if q != pid))  # fmt: skip
                stripped += 1
    rep["stripped_p14_uses"] = stripped
    # ---- the phase: B's roles, no family waits / signals (B's inputs are staged; its own barriers order it)
    p = next(q for q in e.phases if q.pid == pid)
    p.roles = tuple(recv_roles + strm_roles)
    p.waits, p.signals = (), ()
    e.register_phase(pid, PMK.PhaseBody(parts=parts, io=list(io)))
    rep.update(hub=hub, hub_role=role_of[hub], set_h6=set_h6, n=n_b, base={r: base(next(x for x in recv + strm if role_of[x] == r)) for r in recv_roles + strm_roles},
               dual="s1" in raw, cbx={r: dict(v) for r, v in cbx.items()})  # fmt: skip
    return rep


def register_b_self(
    a, b, c: int, *, x_ext, ids_rm, wt, x_tile, y_s=None, pid: str = "P14", xs_base=None, h9_by_rank: dict | None = None
) -> dict:
    """program B in its SELF-STREAM form (pmk_moe2.SelfStreamB, 14's 7g: every compute core streams its own (expert, M
    group) units from pl.xsl in place, no streamer cores) as phase `pid` (the reclaim window) of the fused layer program
    (b_inside_fused_plan.md I-1 / I-2): the compute cores are b.cores over the WINDOWED member / MM-only / root roles
    (the lanes / tails keep P11's / P15's fabric CBs); every hosting composite runs b_wrap_{w,a,c}.cpp (the guard words
    and the two-round A -> B hand-over on id 14 over b.cores; non-B cores of a role skip on their zero guard word); B's
    CBs are reclaim allocations per role, laid by each role's reclaim window (the c1ec7cfb8b9 guard refuses an unlaid
    one). register_b's model for the streamer form; register_g's for the multi-role receivers. Returns a report."""
    e = a.engine
    rc = e.place.role_cores()
    s = b.scratch(c)
    raw, _cbs, _sems, io = b.parts(x_ext, ids_rm, wt, s["out_rm"], s["y"], c, x_tile=x_tile,
                                   y_s=y_s if y_s is not None else s.get("y_s"), xs_base=xs_base,
                                   **({"h9": h9_by_rank[0]} if h9_by_rank else {}),
                                   **({"window": True} if getattr(b, "plan_early", None) else {}))  # fmt: skip
    assert not list(getattr(b, "streamers", []) or []), "register_b_self: the self-stream form has no streamer cores"
    recv = [tuple(x) for x in b.cores]
    role_of = {tuple(x): r for r in ("member", "root", "mm_only", "lane", "tail") for x in rc[r]}
    recv_roles = sorted({role_of[x] for x in recv})
    assert set(recv_roles) <= {"member", "mm_only", "root"}, (
        "B7g on the member / MM-only / root roles only",
        recv_roles,
    )
    assert all(r in e.window_roles for r in recv_roles), ("B7g's roles must be windowed (the reclaim window)", recv_roles,
                                                          e.window_roles)  # fmt: skip
    plan = b.plan(c)
    fmts = _b_formats(b)
    cc_b = raw["c"][4]
    f_modes = {i: m for i, m in enumerate(list(cc_b.unpack_to_dest_mode)) if m != ttnn.UnpackToDestMode.Default}
    mode_of = lambda name: f_modes.get(b.cb_index(name), ttnn.UnpackToDestMode.Default)  # noqa: E731
    groups = [(list(m_), int(t_), {n: int(pg[n]) for n in m_}) for m_, t_, pg in plan["groups"]]
    want = [(n, FMT[fmts[n]], pg[n], mode_of(n), n in LOW32, n in RAW) for mem, _, pg in groups for n in mem]
    al = lambda x: -(-int(x) // e.WIN_ALIGN) * e.WIN_ALIGN  # noqa: E731
    e.win_reclaim = dict(e.win_reclaim or {})
    cbx, rep = {}, dict(cores=len(recv), roles=recv_roles, maps={}, need={}, fresh={})
    for role in recv_roles:
        m = _assign(e, role, want, True, pid)
        cbx[role] = m
        rep["maps"][role] = dict(m)
        used_r = _used(e, role)
        ra, fresh = [], []
        for mem, total, pg in groups:
            ra.append(([(m[n], pg[n]) for n in mem], total))
            for n in mem:
                if m[n] not in used_r:
                    fresh.append(PMK.CBUse(index=m[n], fmt=FMT[fmts[n]], pages=1, live=(pid,), page=pg[n], note=f"B {n}",
                                           users=(pid,)))  # fmt: skip
        e.win_reclaim[role] = dict(pid=pid, allocs=ra, fresh=fresh)
        rep["need"][role], rep["fresh"][role] = sum(al(t) for _, t in ra), [u.index for u in fresh]
    # ---- the hand-over: at TP8 H6's aggregator SETs the members' / roots' id 14 (P10's wait roles) -> the hub is an
    # MM-only B core (its id 14 only counts, register_g's rule); else B's first core
    n_b = len(recv)
    h6 = e.handoffs.get("H6")
    h6_live = h6 is not None and h6.counter == RDV and e._has_live_producer("H6")
    p10_roles = e.wait_roles(next(p for p in e.phases if p.pid == "P10")) if e.in_scope("P10") else set()
    base = lambda core: int(h6.wait_value) if (h6_live and role_of[core] in p10_roles) else 0  # noqa: E731
    hub = next((x for x in recv if role_of[x] == "mm_only" and base(x) == 0), None)
    hub = hub if hub is not None else next((x for x in recv if base(x) == 0), recv[0])
    assert base(hub) == 0 or not (h6_live and int(getattr(a, "n", 2)) != 2), ("B7g's hub: id 14 must only count", hub)
    hx, hy = b.noc_of[hub]
    guard = lambda core: [1, hx, hy, 1 if core == hub else 0, base(core), n_b]  # noqa: E731
    fast = os.environ.get("GLM53_PMK_B_FASTTILIZE", "1") == "1"

    def kd(role):
        return [(f"PMK_MOE_CBX_{n}", str(i)) for n, i in sorted(cbx[role].items())] + (
            [("PMK_COMB_FAST_TILIZE", "1")] if fast else [])  # fmt: skip

    w_file, ct_w, rt_w, d_w, cfg_w = raw["w"]
    a_file, ct_a, rt_a, d_a, cfg_a = raw["a"]
    c_file, ct_c, rt_c, d_c, cc_c = raw["c"]
    for f_, want_f in ((w_file, "moe_brisc.cpp"), (a_file, "moe_ncrisc.cpp"), (c_file, "moe_trisc.cpp")):
        assert os.path.basename(f_) == want_f, (f_, want_f)
    if not any(str(k_) == "PMK_RT_STREAMERS" for k_, _v in d_w):  # b_wrap's hand-over release list: no streamers (as G)
        base_s = max(len(v) for v in rt_w.values())
        rt_w = {x: list(v) + [0] * (base_s - len(v)) + [0] for x, v in rt_w.items()}
        d_w = list(d_w) + [("PMK_RT_STREAMERS", str(base_s))]
    # PMK_P15_STREAM at TP8 (non-B lever (2)): the H9 TABLE segment per chip on every combine writer, the combine's start
    # per chip on both combine RISCs (PMK_COMB_STRIDE)
    h9_rk = _h9_rank_patch(d_w, rt_w, recv, guard, h9_by_rank) if h9_by_rank else None
    a_rk = _comb_uoff_patch(b, d_w, d_a, rt_a, recv, guard, h9_rk, h9_by_rank)
    sub = lambda rk, mine: {r_: {x: v[x] for x in mine} for r_, v in rk.items()}  # noqa: E731
    parts = {}
    for role in recv_roles:
        mine = [x for x in recv if role_of[x] == role]
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i, mo in f_modes.items():
            name = next(n for n in NAMES if b.cb_index(n) == i)
            modes[cbx[role][name]] = mo
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=cc_c.math_fidelity, math_approx_mode=cc_c.math_approx_mode,
                                          fp32_dest_acc_en=cc_c.fp32_dest_acc_en, dst_full_sync_en=cc_c.dst_full_sync_en)  # fmt: skip
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        rt0 = {"rt_by_rank": sub(h9_rk, mine)} if h9_rk else {"rt": {x: guard(x) + list(rt_w[x]) for x in mine}}
        rt1 = {"rt_by_rank": sub(a_rk, mine)} if a_rk else {"rt": {x: guard(x) + list(rt_a[x]) for x in mine}}
        parts[(role, "dm0")] = dict(name="bw", src=os.path.join(KDIR, "b_wrap_w.cpp"), ct=list(ct_w),
                                    **rt0, defines=rebase_defines(d_w, GW_DM),
                                    kdefines=kd(role), config=cfg_w, idle_ok=True, noc_xy={x: (1,) for x in mine},
                                    probe_marks=(("B.eplan",) if getattr(b, "plan_early", None) else ())
                                    + ("B.handover", "B.plan", "B.sweep", "B.comb"))  # fmt: skip
        parts[(role, "dm1")] = dict(name="bw", src=os.path.join(KDIR, "b_wrap_a.cpp"), ct=list(ct_a),
                                    **rt1, defines=rebase_defines(d_a, GW_DM),
                                    kdefines=kd(role), config=cfg_a, idle_ok=True,
                                    probe_marks=("B.handover", "B.ffnw", "B.ffn_barrier", "B.combr"))  # fmt: skip
        parts[(role, "c")] = dict(name="bw", src=os.path.join(KDIR, "b_wrap_c.cpp"), ct=list(ct_c),
                                  rt={x: [1] + list(rt_c[x]) for x in mine}, defines=rebase_defines(d_c, GW_C),
                                  kdefines=kd(role), compute=cc, dst_any=True, fid_mode="template", idle_ok=True)  # fmt: skip
    for role in recv_roles:
        for sid in range(5, RDV + 1):
            e._sem(sid, role)
    pe = getattr(b, "plan_early", None)
    if (
        pe
    ):  # Stage 14 B2.3 step (2): the early plan's signal + barrier ids on its cores' roles (zero at launch, used once)
        plan_roles = sorted({role_of[tuple(x)] for x in b.cores[: b.plan_p]})
        assert set(plan_roles) <= {"mm_only", "root"}, (
            "the early plan runs on idle MM-only / root BRISCs only",
            plan_roles,
        )
        for sid in (int(pe["sig"]), int(pe["sem_a"]), int(pe["sem_b"])):
            assert sid < 5 or sid > RDV, (
                "the early plan's ids must lie outside B's 5 .. 14 (reset at the hand-over)",
                sid,
            )
            for role in plan_roles:
                e._sem(sid, role)
        rep["plan_early"] = dict(cores=b.plan_p, roles=plan_roles, **{k_: int(v_) for k_, v_ in pe.items()})
    stripped = 0
    for role_, d_ in e.cbs.items():
        for i_, u_ in list(d_.items()):
            if pid in tuple(u_.users or ()) + tuple(u_.live or ()) and not str(u_.note).startswith("B "):
                d_[i_] = dataclasses.replace(u_, live=tuple(q for q in u_.live if q != pid),
                                             users=tuple(q for q in (u_.users or ()) if q != pid))  # fmt: skip
                stripped += 1
    rep["stripped_p14_uses"] = stripped
    p = next(q for q in e.phases if q.pid == pid)
    p.roles = tuple(recv_roles)
    p.waits, p.signals = (), ()
    e.register_phase(pid, PMK.PhaseBody(parts=parts, io=list(io)))
    rep.update(hub=hub, hub_role=role_of[hub], n=n_b, selfstream=True, cbx={r: dict(v) for r, v in cbx.items()})
    print(f"PMK B7g: {n_b} cores on {recv_roles}, reclaim need per role {rep['need']} B, blocks {tuple(b.blocks(c))}, "
          f"groups {[(m_, t_) for m_, t_, _ in groups]}", flush=True)  # fmt: skip
    return rep


# ------------------------------------------------------------------------------------------------- program G (W7 step 5)
G_NAMES = ("CTL", "WGU", "XRM", "XT", "WD", "WCTL", "TAB", "WTAB", "GUP", "ACT", "YRM", "PLAN", "CIN", "CW", "CIDX",
           "COUT", "CRM", "RIN", "CZERO")  # fmt: skip
G_RAW = RAW + ("CZERO",)  # touched as words / NoC copies only


def w7_gang_sets(place) -> list:
    """the gangs of program G inside the fused program (G v2): EVERY GANG INSIDE ONE ROLE -- all cores of a role share
    the role's CB layout, so CB_RIN (and the ring semaphores) sit at the same L1 address on every core of a gang (14's
    contract 1, w8/w7_gang_interface.md) whatever the member / MM-only windows look like. 8 member row gangs (x in
    ring fold order) + the MM-only cores in 4 gangs (columns 0 / 1 of rows 0..7, then the low rows' MM-only cores);
    roots, lanes and tails run no G (their composites skip it)."""
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_gang as PG  # noqa: PLC0415

    rc = place.role_cores()
    gangs = []
    mem = sorted(tuple(c_) for c_ in rc["member"])
    for y in sorted({c_[1] for c_ in mem}):
        row = sorted(c_ for c_ in mem if c_[1] == y)
        assert len(row) == PG.G, ("a member row = one gang", y, row)
        gangs.append([row[PG.GANG_FOLD[s_]] for s_ in range(PG.G)])
    mm = [tuple(c_) for c_ in rc["mm_only"]]
    hi = sorted((c_ for c_ in mm if c_[1] < 8), key=lambda c_: (c_[0], c_[1]))  # column-major: x 0 then x 1
    lo = sorted((c_ for c_ in mm if c_[1] >= 8), key=lambda c_: (c_[1], c_[0]))  # the low rows, row-major
    for grp in (hi, lo):
        assert len(grp) % PG.G == 0, ("MM-only gang cores", grp)
        for k in range(0, len(grp), PG.G):
            seg = grp[k : k + PG.G]
            gangs.append([seg[PG.GANG_FOLD[s_]] for s_ in range(PG.G)])
    PG.gang_ring(gangs)  # the ring rules
    return gangs


def register_g(a, g, c: int, *, x_ext, ids_rm, wt, y_s=None, pid: str = "P14", h9_by_rank: dict | None = None) -> dict:
    """program G (tt/pmk_gang.PmkGang, built on w7_gang_sets) as phase `pid` (the reclaim window) of the fused program,
    in program B's place: the member and MM-only composites run kernels/pmk/g_wrap_{w,a,c}.cpp (b_wrap.hpp's guard words
    and two-round A -> G hand-over on id 14 over the 96 gang cores; it resets ids 5 .. 13 = G's plan / FFN ids and the
    ring's 12 / 13). G's CBs are reclaim allocations on both (windowed) roles; G's inputs are the program's (x_ext:
    P11x, ids / weights: P12's tail -> g built with wt_bf16 and the PMK_IDS_U16 layout by the caller, y_s: P13's rows).
    Returns a report."""
    e = a.engine
    rc = e.place.role_cores()
    s = g.scratch(c)
    raw, _cbs, _sems, io = g.parts(
        x_ext, ids_rm, wt, s["out_rm"], s["y"], c, y_s=y_s, **({"h9": h9_by_rank[0]} if h9_by_rank else {})
    )  # (G v2b: P15 streamed, the H9 TABLE)
    gcores = [tuple(x) for x in g.cores]
    role_of = {tuple(x): r for r in ("member", "root", "mm_only", "lane", "tail") for x in rc[r]}
    g_roles = sorted({role_of[x] for x in gcores})
    assert set(g_roles) <= {"member", "mm_only"}, ("G's gangs on the windowed member / MM-only roles only", g_roles)
    for gang in g.gangs:
        assert len({role_of[tuple(x)] for x in gang}) == 1, ("a gang inside ONE role (CB_RIN's address)", gang)
    assert all(r in e.window_roles for r in g_roles), (
        "G's roles must be windowed (the reclaim window)",
        e.window_roles,
    )
    plan = g.cb_plan(c)
    fmts = {n: FMT[g.CB_FMT.get(n, ttnn.uint32)] for n in G_NAMES}
    cc_g = raw["c"][4]
    f_modes = {i: m for i, m in enumerate(list(cc_g.unpack_to_dest_mode)) if m != ttnn.UnpackToDestMode.Default}
    mode_of = lambda name: f_modes.get(g.cb_index(name), ttnn.UnpackToDestMode.Default)  # noqa: E731
    groups = [(list(m_), int(t_), {n: int(pg[n]) for n in m_}) for m_, t_, pg in plan["groups"]]
    want = [(n, fmts[n], pg[n], mode_of(n), n in LOW32, n in G_RAW) for mem, _, pg in groups for n in mem]
    al = lambda x: -(-int(x) // e.WIN_ALIGN) * e.WIN_ALIGN  # noqa: E731
    e.win_reclaim = dict(e.win_reclaim or {})
    cbx, rep = {}, dict(cores=len(gcores), gangs=len(g.gangs), roles=g_roles, maps={}, need={}, fresh={})
    for role in g_roles:
        m = _assign(e, role, want, True, pid)
        cbx[role] = m
        rep["maps"][role] = dict(m)
        used_r = _used(e, role)
        ra, fresh = [], []
        for mem, total, pg in groups:
            ra.append(([(m[n], pg[n]) for n in mem], total))
            for n in mem:
                if m[n] not in used_r:
                    fresh.append(PMK.CBUse(index=m[n], fmt=fmts[n], pages=1, live=(pid,), page=pg[n], note=f"G {n}",
                                           users=(pid,)))  # fmt: skip
        e.win_reclaim[role] = dict(pid=pid, allocs=ra, fresh=fresh)
        rep["need"][role], rep["fresh"][role] = sum(al(t) for _, t in ra), [u.index for u in fresh]
    # ---- the hand-over: hub = an MM-only gang core (its id 14 only counts: at TP8 H6's aggregator SETs the members' id 14)
    n_g = len(gcores)
    h6 = e.handoffs.get("H6")
    h6_live = h6 is not None and h6.counter == RDV and e._has_live_producer("H6")
    p10_roles = e.wait_roles(next(p for p in e.phases if p.pid == "P10")) if e.in_scope("P10") else set()
    base = lambda core: int(h6.wait_value) if (h6_live and role_of[core] in p10_roles) else 0  # noqa: E731
    hub = next((x for x in gcores if role_of[x] == "mm_only" and base(x) == 0), gcores[0])
    assert base(hub) == 0 or not (h6_live and int(getattr(a, "n", 2)) != 2), ("G's hub: id 14 must only count", hub)
    noc = dict(zip(gcores, g.noc))
    hx, hy = noc[hub]
    guard = lambda core: [1, hx, hy, 1 if core == hub else 0, base(core), n_g]  # noqa: E731
    w_file, ct_w, rt_w, d_w, cfg_w = raw["w"]
    a_file, ct_a, rt_a, d_a, cfg_a = raw["a"]
    c_file, ct_c, rt_c, d_c, cc_c = raw["c"]

    def kd(role):
        return [(f"PMK_MOE_CBX_{n}", str(i)) for n, i in sorted(cbx[role].items())] + [("PMK_COMB_FAST_TILIZE", "1")]

    # G v2b (P15 streamed): the H9 TABLE segment per chip on every combine writer + the combine's start per chip
    h9_rk = _h9_rank_patch(d_w, rt_w, gcores, guard, h9_by_rank) if h9_by_rank else None
    a_rk = _comb_uoff_patch(g, d_w, d_a, rt_a, gcores, guard, h9_rk, h9_by_rank)
    sub = lambda rk, mine_: {r_: {x: v[x] for x in mine_} for r_, v in rk.items()}  # noqa: E731
    parts = {}
    for role in g_roles:
        mine = [x for x in gcores if role_of[x] == role]
        rt0 = {"rt_by_rank": sub(h9_rk, mine)} if h9_rk else {"rt": {x: guard(x) + list(rt_w[x]) for x in mine}}
        rt1 = {"rt_by_rank": sub(a_rk, mine)} if a_rk else {"rt": {x: guard(x) + list(rt_a[x]) for x in mine}}
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i, mo in f_modes.items():
            name = next(n for n in G_NAMES if g.cb_index(n) == i)
            modes[cbx[role][name]] = mo
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=cc_c.math_fidelity, math_approx_mode=cc_c.math_approx_mode,
                                          fp32_dest_acc_en=cc_c.fp32_dest_acc_en, dst_full_sync_en=cc_c.dst_full_sync_en)  # fmt: skip
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        parts[(role, "dm0")] = dict(name="gw", src=os.path.join(KDIR, "g_wrap_w.cpp"), ct=list(ct_w),
                                    **rt0, defines=rebase_defines(d_w, GW_DM),
                                    kdefines=kd(role), config=cfg_w, idle_ok=True, noc_xy={x: (1,) for x in mine},
                                    probe_marks=("G.handover", "G.plan", "G.gang", "G.comb"))  # fmt: skip
        parts[(role, "dm1")] = dict(name="gw", src=os.path.join(KDIR, "g_wrap_a.cpp"), ct=list(ct_a),
                                    **rt1, defines=rebase_defines(d_a, GW_DM),
                                    kdefines=kd(role), config=cfg_a, idle_ok=True,
                                    probe_marks=("G.handover", "G.ffnw", "G.ffn_barrier", "G.combr"))  # fmt: skip
        parts[(role, "c")] = dict(name="gw", src=os.path.join(KDIR, "g_wrap_c.cpp"), ct=list(ct_c),
                                  rt={x: [1] + list(rt_c[x]) for x in mine}, defines=rebase_defines(d_c, GW_C),
                                  kdefines=kd(role), compute=cc, dst_any=True, fid_mode="any", idle_ok=True)  # fmt: skip
    for role in g_roles:
        for sid in range(5, RDV + 1):
            e._sem(sid, role)
    stripped = 0
    for role_, d_ in e.cbs.items():
        for i_, u_ in list(d_.items()):
            if pid in tuple(u_.users or ()) + tuple(u_.live or ()) and not str(u_.note).startswith("G "):
                d_[i_] = dataclasses.replace(u_, live=tuple(q for q in u_.live if q != pid),
                                             users=tuple(q for q in (u_.users or ()) if q != pid))  # fmt: skip
                stripped += 1
    rep["stripped_p14_uses"] = stripped
    p = next(q for q in e.phases if q.pid == pid)
    p.roles = tuple(g_roles)
    p.waits, p.signals = (), ()
    e.register_phase(pid, PMK.PhaseBody(parts=parts, io=list(io)))
    rep.update(hub=hub, hub_role=role_of[hub], n=n_g, cbx={r: dict(v) for r, v in cbx.items()})
    return rep
