# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16 integration: the sparse_kda prefill layer program (tt/pmk_layer.py, the family engine) with the pieces'
REAL phase bodies registered (doc/ttnn/16-prefill-production/prefill_megakernel.md, pmk/REVIEW.md).

A piece's parts() returns role dicts {name (kernel source), cores, ct, rt {(x, y): args}, defines, config}; this module
maps them onto the engine's PhaseBody.parts[(role, risc)] and registers them on the layer's phases. The engine then
composes one kernel per (core role, RISC) out of every phase's body (namespaced PH_CT0 / ph_rt0, defines around the
include, hoisted headers), merges the compute configs, and waits only on hand-offs whose producers are registered (the
incremental units: a stub-produced input is staged by the unit, never waited for).

v1 pieces wired here: mhc (P1 / P10 / P16). PfStream (P3 / P7 / P13: 7a), kda (P4 / P5 / P6 / P9), cache, moe and the
FabricStream collectives register the same way as their parts() land."""

from __future__ import annotations

import os

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as MHC

MHC_PARTS = ((1, "P1"), (10, "P10"), (16, "P16"))


def risc_of(cfg) -> str:
    """dm0 / dm1 / c from a piece's kernel config descriptor"""
    import ttnn  # noqa: PLC0415

    if isinstance(cfg, ttnn.ComputeConfigDescriptor):
        return "c"
    proc = getattr(cfg, "processor", None)
    return "dm0" if proc == ttnn.DataMovementProcessor.RISCV_0 else "dm1"


def family_mhc_placement(layer: "PMK.PmkSparseKdaLayer") -> dict:
    """the mhc piece's placement dict on the FAMILY's cores: member (b, j) = family group b's j-th member, root b =
    the family's b-th root (the kernels only see NoC coordinates; the engine composes on these core sets)"""
    pl = layer.engine.place
    assert len(pl.groups) == MHC.N_GROUPS and all(len(g) == MHC.P for g in pl.groups), (len(pl.groups), MHC.P)
    return dict(members={(b, j): tuple(pl.groups[b][j]) for b in range(MHC.N_GROUPS) for j in range(MHC.P)},
                roots={b: tuple(pl.roots[b]) for b in range(MHC.N_GROUPS)})  # fmt: skip


def mhc_arg_decl(mhc: "MHC.PmkMhc", src: str, role: str, flags: int) -> tuple:
    """(noc_xy offsets, non-zero address offsets) of one mhc kernel's runtime args, from the layout pmk_mhc.parts()
    builds (its kernels' header comments): the engine's audit checks them on every core of the composite.
    member mhc_stream: x address, g, step, c0, flags, gate sem, gate value, then per sublayer fn / w / hn / y addresses;
    member mhc_sync: root (x, y), j, flags, round0, then per (sublayer, block) AG-store (x, y, L1) (push mode only);
    root mhc_stream: sb address, flags; root mhc_sync: flags, g, step, round0, taps address, per sublayer post / comb
    addresses, then the P members' (x, y)."""
    p, name = mhc.plan, os.path.basename(src)
    if name == "mhc_stream.cpp":
        return (), ((0,) + tuple(7 + i for i in range(4 * p.nsub)) if role == "member" else (0,))
    if name == "mhc_sync.cpp" and role == "member":
        push = not (flags & MHC.F_XH_DRAM)
        k = range(p.nsub * p.nrb)
        return (0,) + (tuple(5 + 3 * i for i in k) if push else ()), (tuple(7 + 3 * i for i in k) if push else ())
    if name == "mhc_sync.cpp":
        return tuple(5 + 2 * p.nsub + 2 * m for m in range(MHC.P)), tuple(range(4, 5 + 2 * p.nsub))
    return (), ()


def mhc_bodies(mhc: "MHC.PmkMhc", x, hns: list, ys: list, posts: list, combs: list, *, flags: int, layer: int = 0,
               fn_gate=None, xh_targets=None, taps=None) -> dict:  # fmt: skip
    """{pid: PhaseBody} for P1 / P10 / P16 from the mhc piece's per-part parts(). fn_gate = (semaphore, cumulative
    value) of the family's IN0-free counter (H7b) for the sublayer-1 fn^T / w prefetch, or None = no gate (P7 not
    registered: nothing would raise it)."""
    gate = fn_gate if fn_gate is not None else (MHC.NO_GATE, 0)
    io = [x] + list(hns) + list(ys) + list(posts) + list(combs)
    io += [t for s in mhc.subs for t in (s.fn, s.w, s.sb)]
    if taps is not None:
        io.append(taps)
    out = {}
    for part, pid in MHC_PARTS:
        roles, _, _ = mhc.parts(x, hns, ys, posts, combs, flags=flags, part=part, layer=layer, fn_gate=gate,
                                xh_targets=xh_targets, taps=taps)  # fmt: skip
        parts = {}
        for r in roles:
            role = "member" if ("PMK_MEMBER", "1") in r["defines"] else "root"
            risc = risc_of(r["config"])
            assert (role, risc) not in parts, (pid, role, risc)
            noc_xy, nonzero = mhc_arg_decl(mhc, r["name"], role, flags)
            rt = {tuple(k): list(v) for k, v in r["rt"].items()}
            parts[(role, risc)] = dict(src=r["name"], ct=list(r["ct"]), rt=rt,
                                       noc_xy={c: noc_xy for c in rt}, nonzero={c: nonzero for c in rt},
                                       defines=[(str(k), str(v)) for k, v in r["defines"]],
                                       compute=r["config"] if risc == "c" else None,
                                       config=r["config"] if risc != "c" else None)  # fmt: skip
        out[pid] = PMK.PhaseBody(parts=parts, io=io)
    return out


def register_mhc(layer: "PMK.PmkSparseKdaLayer", bodies: dict):
    """register the mhc bodies on the layer program (the census already carries the mhc CBs: pmk_layer derives them
    from pmk_mhc.Plan.cbs())"""
    for pid, body in bodies.items():
        layer.register_phase(pid, body)
    return layer


# ----------------------------------------------------------------------------------------------------- the kda piece
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_kda as KDA  # noqa: E402

# KdaPhase role -> (composite role, RISC, phase): the item kernels (P4 conv / gates + P5 prepare: one kernel set) run
# on the MEMBERS = the H x CPH item cores; the scan kernels (P6 scan + gated norm, P9 state) on the MM-only cores = the
# kda_scan cores (the family's mm_only composite hosts the kda_scan sub-role)
KDA_ROLE = {
    "item_reader": ("member", "dm0", "P4"),
    "item_compute": ("member", "c", "P4"),
    "item_writer": ("member", "dm1", "P4"),
    "scan_reader": ("mm_only", "dm0", "P6"),
    "scan_compute": ("mm_only", "c", "P6"),
    "scan_writer": ("mm_only", "dm1", "P6"),
}


def kda_phase(layer: "PMK.PmkSparseKdaLayer", kda, *, consumers=None, nprod: int = 1, h3_preraised: bool = True):
    """the kda piece's KdaPhase on the FAMILY's cores and CB index maps: item i = member i (head i // CPH, slot i % CPH;
    members in the placement's order), scan core j = kda_scan j; item / scan CB indices = engine.kda_map (the free
    indices of the member / MM-only composites). h3_preraised: the H3 producers (P3) are not registered -> SEM_QKV
    starts at its full count (the unit stages the H3 store)."""
    import ttnn  # noqa: PLC0415

    e = layer.engine
    pl = e.place
    items = [ttnn.CoreCoord(int(x), int(y)) for x, y in pl.members]
    scans = [ttnn.CoreCoord(int(x), int(y)) for x, y in pl.kda_scan]
    assert set(map(tuple, pl.kda_scan)) == set(map(tuple, pl.mm_only)), "the scan body runs on every MM-only core"
    return KDA.KdaPhase.from_kda(kda, e.kda_geo, item_cores=items, scan_cores=scans, consumers=consumers or [],
                                 nprod=nprod, h3_preraised=h3_preraised, item_map=e.kda_map["item"],
                                 scan_map=e.kda_map["scan"])  # fmt: skip


def kda_arg_decl(name: str, phase: "KDA.KdaPhase") -> tuple:
    """(noc_xy offsets, non-zero address offsets) of one kda kernel's runtime args, from KdaPhase.roles' layout:
    item_reader: head, slot, u, lr, f, C, valid_end, 4 taps, dt_bias, rate, probe; item_writer: head, slot, terms,
    valid_end, next item (x, y), probe, then the head's SPREAD scan cores (x, y); scan_reader: head, vblk, S, terms,
    gate_out, w, valid_end, probe, item_cores, spread; scan_writer: head, vblk, on, S, valid_end, C, u, leader, leader
    (x, y), leader, probe, item_cores, then the SPREAD - 1 peers (x, y), then the consumers (x, y)."""
    sp, ncons = phase.geo.spread, len(phase.consumers)
    if name == "item_reader":
        return (), tuple(range(2, 14))
    if name == "item_writer":
        return (4,) + tuple(7 + 2 * b for b in range(sp)), (2, 3, 6)
    if name == "scan_reader":
        return (), tuple(range(2, 8))
    if name == "scan_writer":
        return (8,) + tuple(13 + 2 * k for k in range(sp - 1 + ncons)), (2, 3, 4, 5, 6, 11)
    return (), ()


def kda_bodies(phase: "KDA.KdaPhase", scr: dict, state_s, state_c, valid_end) -> dict:
    """{P4: items, P6: scan} PhaseBodies from KdaPhase.roles (P5 / P9 run inside these kernel sets: their family
    phases stay stubs). Kernel-level defines = the CB index maps (kda_common.hpp is hoisted out of the phase
    namespaces); P4 carries SEM_QKV's initial value when the phase was built h3_preraised."""
    roles = phase.roles(scr, state_s, state_c, valid_end)
    io = phase.io(scr, state_s, state_c, valid_end)
    parts = {"P4": {}, "P6": {}}
    for name, (src, ct, rt, defs, cfg) in roles.items():
        role, risc, pid = KDA_ROLE[name]
        noc_xy, nonzero = kda_arg_decl(name, phase)
        rt_ = {tuple(k): [int(v) for v in a] for k, a in rt.items()}
        parts[pid][(role, risc)] = dict(
            src=os.path.join(PMK.REPO_ROOT, KDA.KDIR, src), ct=[int(v) for v in ct], rt=rt_,
            defines=[(str(k), str(v)) for k, v in defs], compute=cfg if risc == "c" else None,
            config=cfg if risc != "c" else None, kdefines=phase.kdefines["item" if name.startswith("item") else "scan"],
            noc_xy={c: noc_xy for c in rt_}, nonzero={c: nonzero for c in rt_})  # fmt: skip
    init = {KDA.SEM_QKV: phase.nprod * phase.geo.n_chunks} if phase.h3_preraised else {}
    return {"P4": PMK.PhaseBody(parts=parts["P4"], io=io, sem_init=init), "P6": PMK.PhaseBody(parts=parts["P6"], io=io)}


def register_kda(layer: "PMK.PmkSparseKdaLayer", bodies: dict):
    for pid, body in bodies.items():
        layer.register_phase(pid, body)
    return layer
