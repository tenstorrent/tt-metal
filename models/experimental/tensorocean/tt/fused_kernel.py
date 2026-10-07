# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The fused TensorOcean kernel: one TT-Metalium program on an 11 x 10 grid of Tensix cores.

Core (x, y) owns mesh-column strip x and a range of depth levels (core row y). Each core loads its strip's
tracer values once, makes the shifted copies it needs with the SFPU, streams its edges in blocks of 128
(f, mask; per-edge coefficients multicast down the core column by row 0), computes fluxes on the SFPU
in fp32, keeps them in a small ring buffer in L1, and writes each cell's output (sum of 6 fluxes x 1/area).
Kernels: kernels/reader.cpp, kernels/writer.cpp, kernels/compute.cpp (+ common.h, sfpu_shift.h).

prepare(..., natural=, natural_out=) also builds the natural-layout conversions into the same program
(kernels/relayout_fused.h): before the main loop, each core row reads the natural rows of its levels from DRAM,
the compute engine transposes them, and every column goes to the L1 of the core that owns it; after the main
loop, the outputs come back the same way into natural rows in DRAM. Without them, inputs and outputs are in the
per-core DRAM layout of fused_plan.py (tensorocean.py then converts with two separate programs).
"""
import pathlib
import numpy as np
import os as _os_b
import torch
import ttnn
from models.experimental.tensorocean.tt.fused_plan import CH, make_plan, host_arrays

TRACEABLE = True
CB_STAT_STEPS = int(_os_b.environ.get("CB_STAT_STEPS", "4"))
SENDER_LEVELS = int(_os_b.environ.get("SENDER_LEVELS", "6"))  # the statics-sender row (y = 0) gets a smaller share
LP_OVERRIDE = 0
FR = 5
SB = int(_os_b.environ.get("SB", "3"))  # statics steps per sender DRAM batch
OUT_TILES = int(_os_b.environ.get("OUT_TILES", "8"))  # CB_OUT depth (2 tiles per output part)
CB_F_STEPS = 6
NS = int(_os_b.environ.get("NS", "4"))  # statics ring depth (steps the sender may multicast ahead)
DEBUG = False  # True: reader/writer write per-core cycle counters to the DBG tensor (checks/dbg_v3.py)
PAGE = 2048
L1_CB_BUDGET = 1_420_000  # bytes of circular buffers per core the fused versions may use (P150 limit ~1,461,088)
KDIR = pathlib.Path(__file__).resolve().parent / "kernels"


def _arr(name, vals):
    if isinstance(vals[0], (list, tuple)):
        inner = ", ".join("{" + ", ".join(str(int(v)) for v in row) + "}" for row in vals)
        return f"constexpr int32_t {name}[{len(vals)}][{len(vals[0])}] = {{{inner}}};\n"
    return f"constexpr int32_t {name}[{len(vals)}] = {{{', '.join(str(int(v)) for v in vals)}}};\n"


def header(plan, f_alloc, npk, shift_slot, shift_k, lp, nslot=0, pbase=(0, 0)):
    s = "namespace P {\n"
    for k, v in dict(
        CH=CH,
        H=plan.H,
        NBLK=plan.nblk,
        NOBLK=plan.noblk,
        LC=plan.lc,
        LP=lp,
        NBX=plan.nbx,
        CELL_LEN=plan.cell_len,
        F_ALLOC=f_alloc,
        PAGE=PAGE,
        NPK=npk,
        NSHIFT=len(plan.shifts),
        NFS=len(plan.fshift_groups),
        NSLOT=nslot,
    ).items():
        s += f"constexpr uint32_t {k} = {v};\n"
    s += _arr("PBASE", list(pbase))
    s += _arr("SHIFT_P", [p for p, _ in plan.shifts] or [0])
    s += _arr("SHIFT_R", [r for _, r in plan.shifts] or [0])
    s += _arr("SHIFT_K", shift_k or [0])
    s += _arr("SHIFT_SLOT", shift_slot)
    s += _arr("TAP_P", [[p for p, _ in tl] for tl in plan.taps])
    s += _arr("TAP_OFF", [[o for _, o in tl] for tl in plan.taps])
    s += _arr("LOW_A", [-1 if li is None else li[0] for li in plan.lowidx])
    s += _arr("LOW_B", [-1 if li is None else li[1] for li in plan.lowidx])
    s += _arr("TERM_G", [[g for g, _, _ in plan.terms[p]] for p in ("even", "odd")])
    s += _arr("TERM_DR", [[dr for _, dr, _ in plan.terms[p]] for p in ("even", "odd")])
    s += _arr("TERM_DC", [[dc for _, _, dc in plan.terms[p]] for p in ("even", "odd")])
    s += _arr("FS_G", plan.fshift_groups or [0])
    fsi = [0] * 6
    for i, g in enumerate(plan.fshift_groups):
        fsi[g] = i
    s += _arr("FSHIFT_INDEX", fsi)
    s += f"constexpr uint32_t FS_MASK = {sum(1 << g for g in plan.fshift_groups)};\n"
    s += f"constexpr uint32_t L = {plan.L};\n"
    return s + "}\n"


def source(hdr, name):
    common = (KDIR / "common.h").read_text().replace("#pragma once", "")
    body = (KDIR / name).read_text().replace('#include "common.h"', "")
    body = body.replace(
        '#include "relayout_fused.h"', (KDIR / "relayout_fused.h").read_text().replace("#pragma once", "")
    )
    body = body.replace('#include "sfpu_shift.h"', (KDIR / "sfpu_shift.h").read_text().replace("#pragma once", ""))
    # keep the kernel's own #includes first, then constants + common, then the body
    lines = body.splitlines()
    inc = [l for l in lines if l.startswith("#include")]
    rest = [l for l in lines if not l.startswith("#include")]
    return "\n".join(inc) + "\n#include <cstdint>\n" + hdr + common + "\n".join(rest) + "\n"


def _upload(a, device):
    flat = np.ascontiguousarray(a, np.float32).reshape(-1)
    rows = -(-flat.size // (PAGE // 4))
    buf = np.zeros(rows * (PAGE // 4), np.float32)
    buf[: flat.size] = flat
    return ttnn.from_torch(
        torch.from_numpy(buf.reshape(rows, PAGE // 4)),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def prepare(host, n, levels, device, dtype, natural=None, natural_out=None):
    assert dtype == ttnn.float32, "v2 is an fp32 kernel"
    grid = device.compute_with_storage_grid_size()  # 13 x 10 on a P150, 11 x 10 on a QB2 chip
    plan = make_plan(
        n, levels, gx=int(_os_b.environ.get("GRID_X", min(11, grid.x))), gy=min(10, grid.y)
    )  # 13 columns measured no faster overall (2026-10-07)
    A = host_arrays(plan, host)
    # plane slots per level: [p0, p0 shifted copies..., p1, p1 shifted copies...]; copy_rho[i] = plane[i + rho]
    # v22: DRAM holds the 2 unshifted planes per level; compute makes the shifted copies in L1
    import os as _os

    HOSTSHIFT = bool(_os.environ.get("HOSTSHIFT"))  # debug: host-made copies in the v22 slot order, no SFPU shift
    if HOSTSHIFT:
        C = A["CELL"]
        CS = np.zeros((levels, plan.nbx, 2 + len(plan.shifts), plan.cell_len), np.float32)
        CS[:, :, :2] = C
        for k, (p, r) in enumerate(plan.shifts):
            CS[:, :, 2 + k, : plan.cell_len - r] = C[:, :, p, r:]
        A["CELL"] = CS
    t = {k: _upload(v, device) for k, v in A.items()}
    t["OUT"] = _upload(np.zeros((levels, 2, plan.nbx, plan.out_len), np.float32), device)
    t["DBG"] = ttnn.from_torch(
        torch.zeros(1, 32768, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # plane slots: per plane, slot 0 = unshifted, then its shifted copies
    shift_slot = [[0] * 4 for _ in range(2)]
    shift_k, per_plane = [], [0, 0]
    for p, r in plan.shifts:
        per_plane[p] += 1
        shift_slot[p][r] = per_plane[p]
        shift_k.append(per_plane[p])
    npk = 1 + max(per_plane)
    nslot = 2 + len(plan.shifts)
    ntp = -(-plan.cell_len // 1024)  # tiles per plane for the on-chip shift (DEST tiles 0-2 -> 3-5)
    assert ntp <= 3 and plan.cell_len % 64 == 0, plan.cell_len
    lvl_pitch_b = nslot * plan.cell_len * 4 + (ntp * 4096 - plan.cell_len * 4)  # + pad for the last copy's pack overrun
    slot_tab = [[0] * 4 for _ in range(2)]
    for p in (0, 1):
        slot_tab[p][0] = p
    for k, (p, r) in enumerate(plan.shifts):
        slot_tab[p][r] = 2 + k
    pbase = [0, 1 + per_plane[0]]
    f_alloc = -(-max(plan.nblk * CH, plan.noblk * CH + plan.H + 8) // CH) * CH
    # levels per pass: the largest even count whose plane + F buffers fit the L1 budget
    budget = int(_os_b.environ.get("L1BUDGET_KB", "1000")) * 1024
    # level assignment per y: y=0 (the statics sender) gets SENDER_LEVELS, the rest split evenly
    ny_all = min(10, levels)
    sl = min(SENDER_LEVELS, levels)  # never more levels than exist (else OUT is written past its end)
    if ny_all >= 2 and sl:
        rest = levels - sl
        per = -(-rest // (ny_all - 1))
        lv = [(0, sl)] + [(sl + i * per, min(per, rest - i * per)) for i in range(ny_all - 1)]
    else:
        lv = [(y * plan.lc, min(plan.lc, levels - y * plan.lc)) for y in range(plan.nby)]
    lv = [(a, n_) for a, n_ in lv if n_ > 0]
    lcmax = max(n_ for _, n_ in lv)
    # F arrays: groups 0 (ee) and 1 (eo) always appear together with the same offsets -> one array
    fa_of = [0, 0, 1, 2, 3, 4]
    fterms = {}
    for part in ("even", "odd"):
        seen = []
        for g, dr, dc in plan.terms[part]:
            key = (fa_of[g], dr, dc)
            if key not in seen:
                seen.append(key)
        fterms[part] = seen
    nt = len(fterms["even"])
    assert nt == len(fterms["odd"]) == 5
    assert all(fa == 0 for part in fterms.values() for fa, dr, dc in part if dr), "only array 0 needs a shifted copy"
    # output blocks ready after each F block
    ob_ready, c = [], 0
    for b in range(plan.nblk):
        while c < plan.noblk:
            need = 0
            for part in ("even", "odd"):
                for g, dr, dc in fterms[part]:
                    start = c * CH + dc * plan.H + dr
                    need = max(need, start + CH - 1 + (1 if start % 4 else 0))
            if need <= (b + 1) * CH - 1:
                c += 1
            else:
                break
        ob_ready.append(c if b < plan.nblk - 1 else plan.noblk)
    # v25: CB_OI / CB_OI2 hold a whole output batch (operands assembled early, consumed at the block end)
    batches = [ob_ready[b] - (ob_ready[b - 1] if b else 0) for b in range(plan.nblk - 1)]

    def base_cbs(lp):  # (CB id, total bytes, page bytes) of v27's circular buffers for lp levels per pass
        oi_tiles = max(4, (max(batches) if batches else 1) * 2 * ((lp + 1) // 2))
        return [
            (0, CB_STAT_STEPS * 3 * 4096, 4096),
            (1, 2 * 3 * ((lp // 2 + 1) // 2) * 4096, 4096),
            (21, 2 * 3 * (lp // 4) * 4096 if lp >= 4 else 4096, 4096),
            (22, 2 * lp * 2 * CH * 4, lp * 2 * CH * 4),
            (23, 32, 32),
            (2, oi_tiles * 4096, 4096),
            (3, 4 * 32, 32),
            (4, lp * lvl_pitch_b, lvl_pitch_b),
            (5, 6 * lp * (FR + 1) * CH * 4, (FR + 1) * CH * 4),
            (6, NS * 24 * CH * 4, 24 * CH * 4),
            (10, 2 * lp * 2 * CH * 4, lp * 2 * CH * 4),
            (11, 2 * SB * 24 * CH * 4, 24 * CH * 4),
            (7, 32, 32),
            (8, 32, 32),
            (9, 32, 32),
            (12, 512, 512),
            (13, plan.noblk * CH * 4, plan.noblk * CH * 4),
            (14, oi_tiles * 4096, 4096),
            (15, plan.noblk * CH * 4, plan.noblk * CH * 4),
            (18, 4096, 4096),
            (19, 32, 32),
            (20, 32 * 16, 32),
            (24, 32 * 16, 32),
            (16, CB_F_STEPS * 2 * 4096, 4096),
            (17, OUT_TILES * 4096, 4096),
        ]

    lp = min(lcmax + (lcmax & 1), 16)  # DEST holds 16 levels' F chunks (compute.cpp: chunks 48 + level)
    if LP_OVERRIDE:
        lp = LP_OVERRIDE
    while lp > 2 and (lp * lvl_pitch_b + 6 * lp * (FR + 1) * CH * 4 + 4 * (24 + lp * 2) * CH * 4) > budget:
        lp -= 2
    while lp > 2 and sum(b for _, b, _ in base_cbs(lp)) > L1_CB_BUDGET:  # all of v27's buffers must fit too
        lp -= 2
    nfg = 5
    npass = -(-lcmax // lp)  # same pass count on every core: the statics stream is shared per column
    hdr = header(plan, f_alloc, npk, shift_slot, shift_k, lp, nslot, pbase)
    tg = _arr("TERM_G", [[fa for fa, _, _ in fterms[p]] for p in ("even", "odd")])
    tr = _arr("TERM_DR", [[dr for _, dr, _ in fterms[p]] for p in ("even", "odd")])
    tc = _arr("TERM_DC", [[dc for _, _, dc in fterms[p]] for p in ("even", "odd")])
    import re as _re

    hdr = _re.sub(r"constexpr int32_t TERM_G\[.*?;\n", tg, hdr, flags=_re.S)
    hdr = _re.sub(r"constexpr int32_t TERM_DR\[.*?;\n", tr, hdr, flags=_re.S)
    hdr = _re.sub(r"constexpr int32_t TERM_DC\[.*?;\n", tc, hdr, flags=_re.S)
    if DEBUG:
        hdr = "#define KDEBUG 1\n" + hdr
    import os as _os

    hdr = "".join(f"#define {d} 1\n" for d in _os.environ.get("KDEFS", "").split(",") if d) + hdr
    hdr = (
        f"#define NSLOT_DRAM {2 + len(plan.shifts) if HOSTSHIFT else 2}\n"
        + ("#define HOSTSHIFT 1\n" if HOSTSHIFT else "")
        + hdr
    )  # ablation switches
    hdr = (
        hdr[: hdr.rindex("}")]
        + f"constexpr uint32_t NT = {nt};\nconstexpr uint32_t NFG = {nfg};\nconstexpr uint32_t NS = {NS};\nconstexpr uint32_t FR = {FR};\nconstexpr uint32_t SB = {SB};\nconstexpr uint32_t LVL_PITCH_B = {lvl_pitch_b};\nconstexpr uint32_t NTP = {ntp};\n"
        + _arr("SLOT", slot_tab)
        + "}\n"
    )
    # streaming tables
    maxoff = max(o for tl in plan.taps for _, o in tl)
    shift_limit = [min(plan.cell_len, (b + 1) * CH + maxoff + 8) for b in range(plan.nblk)]
    shift_w = [1 if k == len(plan.shifts) - 1 else 0 for k in range(len(plan.shifts))]  # the writer makes the last one
    hdr = hdr.replace("}\n", "", 1) if False else hdr
    extra = _arr("SHIFT_LIMIT", shift_limit) + _arr("OB_READY", ob_ready) + _arr("SHIFT_W", shift_w or [0])
    hdr = hdr[: hdr.rindex("}")] + extra + "}\n"

    active = []
    for x in range(plan.nbx):
        for y, (l0y, nl) in enumerate(lv):
            active.append((x, y, nl))
    # one rectangle when the active cores fill it: 110 one-core ranges cost ~42 us more per launch (per-core dispatch)
    xs, ys = {x for x, _, _ in active}, {y for _, y, _ in active}
    if len(active) == len(xs) * len(ys) and xs == set(range(len(xs))) and ys == set(range(len(ys))):
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(len(xs) - 1, len(ys) - 1))])
    else:
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y, _ in active])
    ra, wa, ca = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    nby_x = {x: max(y for xx, y, _ in active if xx == x) + 1 for x, _, _ in active}
    noc = lambda x, y: device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    for x, y, nl in active:
        ny = nby_x[x]
        s0, a1, a2 = noc(x, 0), noc(x, 0), noc(x, ny - 1)  # the column, sender included (loopback multicast)
        ra[x][y] = [
            t["CELL"].buffer_address(),
            t["DBG"].buffer_address() if DEBUG else 0,
            x * 10 + y,
            t["INV"].buffer_address(),
            x,
            lv[y][0],
            nl,
            t["FMK"].buffer_address(),
            s0.x,
            s0.y,
            y,
            1,
            npass,
        ]
        ca[x][y] = [nl, npass]
        wa[x][y] = [
            t["OUT"].buffer_address(),
            x,
            lv[y][0],
            nl,
            t["FMK"].buffer_address(),
            t["STAT"].buffer_address(),
            (ny - 1) if y == 0 else 0,
            1 if y == 0 else 0,
            a1.x,
            a1.y,
            a2.x,
            a2.y,
            s0.x,
            s0.y,
            t["DBG"].buffer_address() if DEBUG else 0,
            x * 10 + y,
            npass,
            t["INV"].buffer_address(),
            t["CELL"].buffer_address(),
        ]
    r_ct = []
    for k in ("CELL", "FMK", "STAT", "INV"):
        r_ct += ttnn.TensorAccessorArgs(t[k]).get_compile_time_args()
    w_ct = [0, 1]  # semaphore ids: ready (on the sender), valid (on receivers)
    for k in ("OUT", "FMK", "STAT", "INV", "CELL"):
        w_ct += ttnn.TensorAccessorArgs(t[k]).get_compile_time_args()
    extra_cbs = []
    if natural:
        # FUSED_IN: the input rearranging runs inside this program (kernels/relayout_fused.h): each row of cores reads
        # the natural rows of its levels from DRAM, the compute engine transposes them, and the columns are sent to
        # the cores that own them. FUSED_OUT (natural_out): the outputs go the same way back into natural rows.
        assert npass == 1, "fused rearranging needs all of a core's levels in one pass"
        xs_, ys_ = {x for x, _, _ in active}, {y for _, y, _ in active}
        assert len(active) == len(xs_) * len(ys_), "fused rearranging needs a full core rectangle (GO multicast)"
        nat_list = [
            natural[k]
            for k in ("cell", "normalThicknessFlux1", "normalThicknessFlux2", "advMaskHighOrder1", "advMaskHighOrder2")
        ]
        for x_ in nat_list:
            r_ct += ttnn.TensorAccessorArgs(x_).get_compile_time_args()
        for x_ in natural_out or []:
            w_ct += ttnn.TensorAccessorArgs(x_).get_compile_time_args()
        nat_addr = [x_.buffer_address() for x_ in nat_list]
        out_addr = [o.buffer_address() for o in natural_out or []]
        ncores = len(active)
        for x, y, nl in active:
            ra[x][y] = list(ra[x][y]) + nat_addr  # reader args 13..17
            wa[x][y] = list(wa[x][y]) + out_addr  # writer args 19, 20
            ca[x][y] = list(ca[x][y]) + [x]  # compute arg 2
        M = n + 4
        nrin = 2  # CB_RIN depth in plane units
        if natural_out:  # output units a = (level, part) of a row, one assembler core each, in a CB_RIN slot
            assert max(-(-2 * nl_ // plan.nbx) for _, nl_ in lv) <= min(nrin, 2), "output units per assembler > 2"
        cdiv = lambda a, b_: -(-a // b_)
        tr = cdiv(plan.H, 32)  # tiles per transposed row (v27 column)
        wt_cell, wt_f2, wt_f1 = cdiv(M, 32), cdiv(n + 1, 32), cdiv(2 * n + 1, 32)
        wt_max = max(wt_cell, wt_f2, wt_f1)
        assert tr <= 4 and cdiv(n, 32) <= 4, "a transposed row block must fit DEST (4 fp32 tiles)"
        G = plan.groups
        unit_rows = [
            M // 2,
            M // 2,
            G[4].rows,
            G[5].rows,
            G[4].rows,
            G[5].rows,
            G[0].rows,
            G[2].rows,
            G[0].rows,
            G[2].rows,
        ]
        unit_cols = [M, M] + [n + 1] * 4 + [2 * n + 1] * 4
        zb_words = -(-max(plan.cell_len, plan.f_len) // 4) * 4
        lev_y, lev_li = [0] * levels, [0] * levels
        for y, (a, nl_) in enumerate(lv):
            for i in range(nl_):
                lev_y[a + i], lev_li[a + i] = y, i
        ny = len(lv)
        nocs = [noc(i % plan.nbx, i // plan.nbx) for i in range(ncores)]
        c0, c1 = noc(0, 0), noc(plan.nbx - 1, ny - 1)
        u = lambda nm, v: f"constexpr uint32_t {nm}[{len(v)}] = {{{', '.join(str(int(a)) for a in v)}}};\n"
        r = "namespace R {\n"
        for k, v in dict(
            N=n,
            M=M,
            H=plan.H,
            CELL_LEN=plan.cell_len,
            F_LEN=plan.f_len,
            NBX=plan.nbx,
            L=levels,
            LCM=lcmax,
            NCORES=ncores,
            LVL_PITCH=lvl_pitch_b,
            TR=tr,
            WT_MAX=wt_max,
            WT_CELL=wt_cell,
            WT_F2=wt_f2,
            WT_F1=wt_f1,
            UNIT_PAGES=tr * wt_max,
            ROWB=wt_max * 128,
            ZB_WORDS=zb_words,
            TRO=cdiv(n, 32),
            NRIN=nrin,
            MCX0=c0.x,
            MCY0=c0.y,
            MCX1=c1.x,
            MCY1=c1.y,
        ).items():
            r += f"constexpr uint32_t {k} = {v};\n"
        r += u("BAND0", [a for a, _ in plan.bands]) + u("BANDW", [b - a for a, b in plan.bands])
        r += u("G_COLS", [g.cols for g in plan.groups])
        r += u("UNIT_ROWS", unit_rows) + u("UNIT_COLS", unit_cols)
        r += u("LEV_Y", lev_y) + u("LEV_LI", lev_li) + u("NOCX", [c.x for c in nocs]) + u("NOCY", [c.y for c in nocs])
        r += "}\n"
        r += "constexpr uint32_t CB_SFMK = 25, CB_RIN = 26, CB_TIL = 28, CB_TOUT = 29, CB_ZB = 30;\n"
        r += "constexpr uint32_t SEM_ARRIVE = 2, SEM_GO = 3, SEM_OUT0 = 4;\n"
        hdr = "#define FUSED_IN 1\n" + ("#define FUSED_OUT 1\n" if natural_out else "") + hdr + r
        up = tr * wt_max * 4096  # one plane unit
        extra_cbs = [
            (25, 6 * lcmax * 2 * plan.f_len * 4),
            (26, nrin * up, 4096),
            (28, up, 4096),
            (29, 2 * up, 4096),
            (30, zb_words * 4),
        ]

    def cb(idx, nbytes, page=4096):
        return ttnn.CBDescriptor(
            total_size=nbytes,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=page)],
        )

    cbs = [cb(i, b, pg) for i, b, pg in base_cbs(lp)]
    if natural:  # f/mask now come from CB 25: the two DRAM prefetch buffers shrink to one page
        cbs = [c for c in cbs if c.format_descriptors[0].buffer_index not in (10, 22)] + [
            cb(10, 1024, 1024),
            cb(22, 1024, 1024),
        ]
        cbs += [cb(e[0], e[1], e[2] if len(e) > 2 else e[1]) for e in extra_cbs]
        # measured on the P150: CBs may total 1,572,864 - 111,776 B before program creation fails
        assert sum(c.total_size for c in cbs) <= L1_CB_BUDGET, "fused rearranging buffers do not fit L1"
    if _os_b.environ.get("PRINT_L1"):
        print("CB bytes", sum(c.total_size for c in cbs))
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
    )
    m = [ttnn.UnpackToDestMode.Default] * 64
    for i in (0, 1, 2, 14, 18, 21) + ((26, 28) if natural else ()):
        m[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(m)
    SC = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source(hdr, "reader.cpp"),
            source_type=SC,
            core_ranges=cores,
            compile_time_args=r_ct,
            runtime_args=ra,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=source(hdr, "writer.cpp"),
            source_type=SC,
            core_ranges=cores,
            compile_time_args=w_ct,
            runtime_args=wa,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=source(hdr, "compute.cpp"),
            source_type=SC,
            core_ranges=cores,
            compile_time_args=[],
            runtime_args=ca,
            config=cfg,
        ),
    ]
    sem_ids = (0, 1) + ((2, 3) if natural else ()) + ((4, 5) if natural_out else ())
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in sem_ids]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    io = (
        [t["CELL"], t["FMK"], t["STAT"], t["INV"], t["DBG"]]
        + (nat_list if natural else [])
        + list(natural_out or [])
        + [t["OUT"]]
    )
    return dict(plan=plan, prog=prog, io=io, t=t, hdr=hdr, lp=lp)


def run(s):
    return (ttnn.generic_op(s["io"], s["prog"]),)
