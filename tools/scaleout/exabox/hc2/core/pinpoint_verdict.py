#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""pinpoint_verdict.py: the parser and verdict of the numeric pinpoint. Standard library only.

run.sh runs the pinpoint only after a NUMERIC FAIL (lib/pinpoint.sh): numeric_pinpoint.py opens the mesh again and runs
splits that tell the failing chip's parts apart, then this file turns its result JSON into the lines. Never changes a
run's result: exit 0 always.

  pinpoint_verdict.py targets <numeric.txt> [<c7 report.txt>]       one line per failing chip: <bdf> <tray> <c4> <parts>
  pinpoint_verdict.py report <numeric.txt> <c7 report> <result.json> [<why not complete>]
  pinpoint_verdict.py capture <numeric.txt> <c7 report> [<result.json>]   the in-run captures (NUMERIC-CAPTURE
      lines of the chunk reports) merged into result.json ("captures"); prints the pinpoint's re-trigger: skip (every
      failing chip captured, none recomputed wrong), norepeat (all captured, some reproduce: no repeat split), transient
      (the chips not attributed only counter-mapped transients: the dram and rw splits) or nothing

Lines (no 'NUMERIC FAIL' and no UBB number, so numeric_report.py's readers, fl_report and host-numeric-count skip them):
  NUMERIC-PINPOINT <BDF> tray N component=<compute core (X,Y) of the GXxGY grid, element (R,C) ... | dram ch X bank X |
                   bad words concentrated in DRAM ch X bank X | compute side,
                   likely (<evidence>) | noc0 | noc1 | pcie | core x,y (fpu|sfpu) | fpu | sfpu | compute | dram | clock>
                   hits k/N (split <name>; <detail>)
  NUMERIC-PINPOINT VERDICT <BDF> tray N: <component>[, odd|even lanes][ (may follow output columns; not proof of a DRAM
                   fault)] (<split> k/N)[; eth ...]
  NUMERIC-PINPOINT VERDICT <BDF> tray N: not isolated (0/N: <per split>) ...       the fault did not come back
  NUMERIC-PINPOINT REPEAT <BDF> tray N: fault back within L launches (S s)[, <bank>, odd|even lanes] | wrong from
                   launch 0 ... | not back in ...
  NUMERIC-PINPOINT LOAD <BDF> tray N: numeric checks under <op>: L launches, box kW mean/max; chip peak W, pre-fault C;
                   first bad at launch L: compute core (X,Y) of the grid, element (R,C) (a matmul split)
                   or DRAM bank (a DRAM split only), row, col (tile) | load off (...): the checks ran solo
Splits (numeric_pinpoint.py): repeat once, then the others in this order, in passes until every failing chip is isolated
or the budget is spent:
  repeat   the failing shape again first (one chip's fault hit 1 in 194 launches): run.sh's heavy load op, then
           the checked C4 matmul (x, w, y in DRAM), compared on the device, up to 400 launches or 60 s; stops at the
           first 100-launch window a failing chip is bad in. Names banks and lanes but isolates nothing alone.
  dram     DRAM only, no compute: hr host write + read (cells, PCIe), nr cores read DRAM into L1 (NOC0, tt-metal's DRAM
           read NOC), hl host write + read of L1, nw cores write L1 into DRAM (NOC1, the DRAM write NOC), rw DRAM -> DRAM;
           counted per DRAM bank (interleaved page index mod banks) and 16-bit lane
  mm       the same matmul (fixed inputs) with its tensors in L1 (compute) and in DRAM (compute + DRAM); bad output tiles
           mapped to their DRAM bank
  heavy    a known trigger: back-to-back C4n matmul (x 2048x8192, w 8192x16384) on all 32 chips with x, w,
           y in DRAM (slot heavy), plus an x-in-L1 arm (x row chunks in L1, w read and y written in DRAM: heavy_xl;
           the most sensitive probe seen), a read-only arm (x, w from DRAM, y in L1: heavy_read) and a write-only arm (x,
           w in L1, the full y to DRAM: heavy_write); bad words per DRAM bank (tile column mod banks) and lane
  rw       the known failing shape: the C4n-size matmul (2048x8192 @ 8192x16384) with x, w and y all in DRAM, back to back for
           a 60 s burst on every chip, once a run; bad output tiles mapped to their DRAM bank
  core     per core: FPU (multiply) and SFPU (exp) on L1 tensors height-sharded over the whole grid, one shard a core
  clock    the DRAM matmul again with the failing chips' AICLK capped (only when mm hit and nothing isolated it)
Banks: in the matmul splits (mm, rw, heavy, repeat) a bad word's bank is
its output tile column mod banks, so one bank holding most bad words may just be output columns: 'bad words concentrated
in DRAM ch X bank X', a pattern that never ranks as the cause. A run with n_tiles 511 decoupled the two and its bad words
spread over all 8 banks on odd lanes only: the compute side. A slot may carry per_col (bad words per output column group)
and decoupled (true when its output columns do not map to banks). 'dram ch X bank X' (a DRAM cause) is named only when the
bad words stay in one bank with the columns decoupled, or in the dram split (no compute, banks by page address). Bad words
in several banks that still cluster on one lane parity or on output columns name 'compute side, likely' with the numbers.
ETH: the local splits use no ETH link (inside the box). A fault seen in the C4 matmul (no CCL) excludes ETH;
one seen only in the sustained h3 part (CCL) with clean local splits leaves ETH/CCL suspected.
"""
import json
import os
import re
import sys

import numeric_capture as nc   # standard library only too

BUS_TO_TRAY = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}   # PCI bus & 0xF0 -> tray (ub0_repro.py)
BDF = r"(?:[0-9a-f]{4}:)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])"
FAIL_RE = re.compile(r"^NUMERIC FAIL " + BDF + r" \((?:tray (\d|\?)|UBB(\d))\)(?: chip \S+?)?(?: \[([^\]]*)\])?:", re.I)
CONC = 0.8        # one bank or core holding this share of the bad words: the fault is there (banks: see Banks above)
SPREAD_BANKS = 3  # at least this many banks, each with BANK_MIN of the bad words: 'several banks'
BANK_MIN = 0.05
COL_SKEW = 2.0    # the top output column group holding this many times its even share: the bad words follow columns
CONC_PFX = "bad words concentrated in DRAM"
CONC_WHERE = "bank controller/PHY or the NoC path out of it"
CONC_NOTE = CONC_WHERE + "; never a compute or Tensix-core cause"
COMPUTE = "compute side, likely"
LANE = 0.9        # one 16-bit lane holding this share: named in the detail
PHASES = ("hr", "nr", "hl", "nw", "rw")
PHASE_WHAT = dict(hr="host write+read of DRAM", nr="cores read DRAM over NOC0", hl="host write+read of L1",
                  nw="cores write DRAM over NOC1", rw="DRAM to DRAM copy")
CCL_PARTS = {"h3"}  # chunk_sched.py: the only sustained part with CCL (ETH) traffic


def tray_of(bdf):
    try:
        return BUS_TO_TRAY.get(int(bdf.split(":")[-2], 16) & 0xF0)
    except (IndexError, ValueError, AttributeError):
        return None


def read_lines(path):
    try:
        with open(path, errors="replace") as f:
            return f.read().splitlines()
    except (OSError, TypeError):
        return []


def targets(numeric_lines, c7_lines=()):
    """NUMERIC FAIL lines of run.sh's numeric.txt (C4, local matmul) and the sustained report (C7, '[parts]') ->
    {bdf: dict(tray, c4, parts)}. Old UBB<n> lines are read too (tray n+1)."""
    out = {}
    for src, lines in (("c4", numeric_lines), ("c7", c7_lines)):
        for ln in lines:
            m = FAIL_RE.match(ln.strip())
            if not m:
                continue
            bdf = m.group(1).lower()
            t = m.group(2) if m.group(2) and m.group(2) != "?" else (str(int(m.group(3)) + 1) if m.group(3) else None)
            d = out.setdefault(bdf, dict(tray=int(t) if t else tray_of(bdf), c4=False, parts=set()))
            if src == "c4":
                d["c4"] = True
            elif m.group(4):
                d["parts"] |= {p.strip() for p in m.group(4).split(",") if p.strip()}
            else:
                d["parts"].add("?")
    return out


def top(counts):
    """(index or key, share, total) of the largest count; (None, 0, 0) when all are zero."""
    items = list(counts.items()) if isinstance(counts, dict) else list(enumerate(counts or []))
    tot = sum(v for _, v in items)
    if not tot:
        return None, 0.0, 0
    k, v = max(items, key=lambda kv: kv[1])
    return k, v / tot, tot


def kn(s):
    return f"{int((s or {}).get('bad_checks', 0))}/{int((s or {}).get('checks', 0))}"


def lane_note(per_lane):
    k, share, tot = top(per_lane or [])
    return f", {'odd' if k == 1 else 'even'} lane {round(share * tot)}/{tot}" if tot and share >= LANE else ""


def spread_comp(per_bank, per_lane=None, per_col=None, decoupled=False):
    """Bad words in several banks that cluster on one lane parity or on output columns -> (compute side, detail)."""
    per_bank = list(per_bank or [])
    tot = sum(per_bank)
    hit = sum(1 for v in per_bank if tot and v >= BANK_MIN * tot)
    if hit < SPREAD_BANKS:
        return None, None
    b, bs, _ = top(per_bank)
    lk, ls, lt = top(per_lane or [])
    ck, cs, ct = top(per_col or [])
    lane = bool(lt) and ls >= LANE
    col = bool(ct) and len(per_col) > 1 and cs >= COL_SKEW / len(per_col)
    if not (lane or col):
        return None, None
    ev = [f"bad words in {hit}/{len(per_bank)} banks, top bank {b} {round(bs * 100)}%"]
    if lane:
        ev.append(f"{'odd' if lk == 1 else 'even'} lanes {round(ls * lt)}/{lt}")
    if col:
        ev.append(f"output column group {ck} {round(cs * 100)}% of {len(per_col)} groups")
    detail = (f"banks {per_bank}" + (f", lanes even/odd {per_lane[0]}/{per_lane[1]}" if len(per_lane or []) == 2 else "")
              + (f", output column groups {list(per_col)}" if ct else "") + (", columns decoupled from banks" if decoupled else ""))
    return f"{COMPUTE} ({'; '.join(ev)})", detail


def one_bank(per_bank):
    """(bank, 'bank B n/N words') when one bank holds CONC of the bad words, else (None, None)."""
    b, share, tot = top(per_bank or [])
    return (b, f"bank {b} {round(share * tot)}/{tot} words") if tot and share >= CONC else (None, None)


def bank_comp(per_bank, per_lane=None, per_col=None, decoupled=False):
    """A matmul split's bad words per bank (with lanes, output column groups and whether the columns were decoupled
    from the banks) -> (component, detail) or (None, None). One bank: DRAM only when decoupled, else the pattern."""
    b, words = one_bank(per_bank)
    if words:
        if decoupled:
            return f"dram ch {b} bank {b}", words + ", columns decoupled from banks"
        return f"{CONC_PFX} ch {b} bank {b}", words
    return spread_comp(per_bank, per_lane, per_col, decoupled)


TILE_NOTE = "a tile index mod 8, not a DRAM bank: no DRAM readback failed"
DRAM_READS = ("dram", "mm_dram", "rw_dram", "heavy_read", "heavy_write")   # splits that read DRAM back (core_loc_finding)


def dram_seen(ch):
    """A DRAM readback failed on this chip (a DRAM split, or the operand w/x readback found bad words)."""
    r = ch.get("wx_readback")
    return any((ch.get(s) or {}).get("bad_checks") for s in DRAM_READS) or \
        bool(isinstance(r, dict) and r.get("bad_words"))


def label(comp, detail, ch=None):
    """A finding's component as the verdict and repeat lines print it. With the chip's results (ch), a heavy or repeat
    split's one-bank share is a DRAM bank only when a DRAM readback of that chip failed."""
    if comp.startswith(COMPUTE):
        return comp
    if comp.startswith(CONC_PFX) and ch is not None and not dram_seen(ch):
        m = re.match(r"bad words concentrated in DRAM ch (\d+) bank \d+", comp)
        n = m.group(1) if m else "?"
        return f"wrong words concentrated in output tiles {n} mod 8{lanes_of(detail)} ({TILE_NOTE}; compute side, likely)"
    if comp.startswith(CONC_PFX):   # 82/82 bad column tiles on 03:00.0 were w columns stored in bank 3
        m = re.match(r"bad words concentrated in DRAM ch (\d+) bank \d+(.*)$", comp)
        if not m:   # the bank pattern did not match: no 'transient' or 'never compute' claim
            return f"{comp}{lanes_of(detail)} ({CONC_WHERE})"
        path, rest = (" read path", m.group(2)[len(" read path"):]) if m.group(2).startswith(" read path") else ("", m.group(2))
        comp = f"transient DRAM bank {m.group(1)} read fault{rest}" if not path else f"transient DRAM bank {m.group(1)} fault on the read path{rest}"
        return f"{comp}{lanes_of(detail)} ({CONC_NOTE})"
    return f"{comp}{lanes_of(detail)}"


def wx_note(ch):
    """The operand w/x bit-exact readback of a chip (ch["wx_readback"] = dict(bad_words=N), absent when not run) as one
    clause for the one-bank verdict (0 bad words, storage clean)."""
    r = ch.get("wx_readback")
    if not isinstance(r, dict) or r.get("bad_words") is None:
        return "operand w/x readback not run"
    n = int(r["bad_words"])
    return "storage read back clean" if n == 0 else f"operand w/x readback: {n} bad words, storage is suspect"


def dram_finding(d):
    """The dram split -> (component, detail) or None."""
    if not d or not d.get("bad_checks"):
        return None
    pp = d.get("per_phase") or {}
    hit = [p for p in PHASES if (pp.get(p) or [0, 0])[0]]
    ph = ", ".join(f"{p} {pp[p][0]}/{pp[p][1]}" for p in hit)
    b, bd = one_bank(d.get("per_bank"))   # no compute: banks by page address, so one bank is the DRAM channel
    comp = f"dram ch {b} bank {b}" if bd else None
    if comp:
        via = {("nr",): " via NOC0 read", ("nw",): " via NOC1 write"}.get(tuple(hit), "")
        detail = f"phase {ph}{via}, {bd}"
    else:
        host = set(hit) <= {"hr", "hl"}
        comp = ("pcie" if host else "noc0" if hit == ["nr"] else "noc1" if hit == ["nw"] else "dram")
        detail = f"phase {ph}, spread over banks {d.get('per_bank')}"
    if d.get("reread_n"):
        detail += f", reread same {d.get('reread_same', 0)}/{d['reread_n']}"
    if d.get("inf") or d.get("nan"):
        detail += f", Inf {d.get('inf', 0)} NaN {d.get('nan', 0)}"
    return comp, detail + lane_note(d.get("per_lane"))


def heavy_finding(chip):
    """The heavy split -> (component, k/N, detail) or None. A bank holding most of the DRAM arms' bad words
    names the DRAM channel; the read-only and write-only arms name the path when one is bad and the other clean."""
    hv, xl, hr, hw = (chip.get(s) or {} for s in ("heavy", "heavy_xl", "heavy_read", "heavy_write"))
    hd = chip.get("heavy_dec") or {}   # N not a multiple of 8 tile columns: columns do not map to banks
    arms = (hv, xl, hr, hw)
    if not any(s.get("bad_checks") for s in arms + (hd,)):
        return None
    xs = f"x in L1 {kn(xl)}, " if xl.get("checks") else ""
    k = f"{kn(hv)}; {xs}read only {kn(hr)}, write only {kn(hw)}"
    nb = max((len(s.get("per_bank") or []) for s in arms), default=0)
    per_lane = [sum((s.get("per_lane") or [0, 0])[i] for s in arms) for i in range(2)]
    nc = max((len(s.get("per_col") or []) for s in arms), default=0)
    # a bank comes only from the DRAM readback arms (read only, write only); the compute
    # arms' (heavy, heavy_xl, heavy_dec) tile index mod 8 is no DRAM bank, so they never name one
    dram = [s for s in (hr, hw) if s.get("bad_checks")]
    comp, bd = None, None
    per_bank = [sum((s.get("per_bank") or [0] * nb)[i] for s in (hr, hw)) for i in range(nb)]
    if dram:
        comp, bd = bank_comp(per_bank,
                             [sum((s.get("per_lane") or [0, 0])[i] for s in (hr, hw)) for i in range(2)],
                             [sum((s.get("per_col") or [0] * nc)[i] for s in (hr, hw)) for i in range(nc)],
                             all(s.get("decoupled") for s in dram))
    path = ""
    if (hr.get("bad_checks") or xl.get("bad_checks")) and hw.get("checks") and not hw.get("bad_checks"):
        path = " read path"   # xl reads w from DRAM where hw has it in L1; both write y to DRAM
    elif hw.get("bad_checks") and hr.get("checks") and not hr.get("bad_checks"):
        path = " write path"
    detail = (f"x, w, y in DRAM {kn(hv)}, {xs}read only {kn(hr)}, write only {kn(hw)}"
              + (f", decoupled arm {kn(hd)}" if hd.get("checks") else "")
              + (f", {bd}" if comp else f", DRAM arms spread over banks {per_bank}" if dram else ""))
    if hv.get("reread_n"):
        detail += f", reread same {hv.get('reread_same', 0)}/{hv['reread_n']}"
    inf, nan = sum(int(s.get("inf", 0)) for s in arms), sum(int(s.get("nan", 0)) for s in arms)
    if inf or nan:
        detail += f", Inf {inf} NaN {nan}"
    if not comp or comp.startswith(COMPUTE):
        detail += f", bad in the{path} arms" if path else ""
    elif comp:
        comp += path
    return comp or "heavy", k, detail + lane_note(per_lane)


def core_finding(s, op):
    """A per-core split (fpu or sfpu) -> (component, detail) or None."""
    if not s or not s.get("bad_checks"):
        return None
    c, share, tot = top(s.get("per_core") or {})
    if tot and share >= CONC:
        return f"core {c} ({op})", f"core {c} {round(share * tot)}/{tot} bad elements, grid {s.get('grid')}"
    cores = sorted((s.get("per_core") or {}).items(), key=lambda kv: -kv[1])
    return op, f"{len(cores)} cores bad: " + ", ".join(f"{k} {v}" for k, v in cores[:6])


def repeat_finding(chip):
    """The repeat split -> (component or 'repeat', k/N, detail) or None."""
    s = chip.get("repeat") or {}
    if not s.get("bad_checks"):
        return None
    detail = f"failing shape again, {s.get('bad', 0)} bad words"   # a compute split names no DRAM bank
    if s.get("inf") or s.get("nan"):
        detail += f", Inf {s.get('inf', 0)} NaN {s.get('nan', 0)}"
    return "repeat", kn(s), detail + lane_note(s.get("per_lane") or [0, 0])


COMPUTE_SPLITS = ("mm_l1", "heavy", "heavy_xl", "heavy_dec", "repeat")   # matmul checks with no DRAM readback in them


def core_loc_finding(chip):
    """The matmul splits name the compute core that wrote most wrong words (numeric_pinpoint
    count()'s owner map, cs.core_of) -> (component, split, k/N, detail) or None. A tile index mod 8 is no DRAM bank: a
    DRAM class needs a failed DRAM readback (dram, mm_dram, heavy_read/heavy_write, rw)."""
    cnt, first, sp0 = {}, {}, None
    for sp in COMPUTE_SPLITS:
        s = chip.get(sp) or {}
        for k, v in (s.get("per_core_bad") or {}).items():
            cnt[k] = cnt.get(k, 0) + v
            sp0 = sp0 or sp
        for l in s.get("locs") or []:
            if l.get("core"):
                first.setdefault("%d,%d" % tuple(l["core"]), l)
    if not cnt:
        return None
    k, share, tot = top(cnt)
    if share < CONC:   # no one core holds it: the compute side spread over cores, never a DRAM bank
        cores = sorted(cnt.items(), key=lambda kv: -kv[1])
        return ("compute (core spread)", sp0, kn(chip.get(sp0)), f"{len(cores)} cores wrote {tot} wrong words: "
                + ", ".join(f"core ({c}) {v} ({round(100 * v / tot)}%)" for c, v in cores[:6]))
    l = first.get(k) or {}
    g, e = l.get("grid") or ["?", "?"], l.get("element")
    comp = (f"compute core ({k}) of the {g[0]}x{g[1]} grid" + (f", element ({e[0]},{e[1]}) of its 32x32 output tiles"
            f" (16x16 face {l['face'][0]},{l['face'][1]}, element {l['in_face'][0]},{l['in_face'][1]})" if e else "")
            + (f" on ASIC {l['asic']} ({l['bdf']})" if l.get("asic") and l.get("bdf") else f" on {l['bdf']}" if l.get("bdf") else ""))
    return (comp, sp0, kn(chip.get(sp0)), f"core ({k}) {round(share * tot)}/{tot} wrong words; first at row {l.get('row')} "
            f"col {l.get('col')} (tile {(l.get('tile') or ['?', '?'])[0]},{(l.get('tile') or ['?', '?'])[1]})")


def findings(chip):
    """One chip's split results -> [(component, split, k/N, detail)], most specific first."""
    out = []
    f = core_loc_finding(chip)
    if f:
        out.append(f)
    f = dram_finding(chip.get("dram"))
    if f:
        out.append((f[0], "dram", kn(chip.get("dram")), f[1]))
    l1, dr = chip.get("mm_l1") or {}, chip.get("mm_dram") or {}
    if dr.get("bad_checks") and l1.get("checks") and not l1.get("bad_checks"):
        comp, bd = bank_comp(dr.get("per_bank"), dr.get("per_lane"), dr.get("per_col"), dr.get("decoupled"))
        out.append((comp or "dram", "mm", kn(dr), f"matmul in DRAM {kn(dr)}, in L1 {kn(l1)} clean"
                    + (f", output {bd}" if bd else f", output banks {dr.get('per_bank')}") + lane_note(dr.get("per_lane"))))
    rw = chip.get("rw_dram") or {}
    if rw.get("bad_checks"):   # the full-size DRAM matmul under load: the DRAM path when a bank holds it
        comp, bd = bank_comp(rw.get("per_bank"), rw.get("per_lane"), rw.get("per_col"), rw.get("decoupled"))
        out.append((comp or "dram", "rw", kn(rw), f"C4n-size x/w/y-in-DRAM matmul under load {kn(rw)}, small matmul in "
                    f"L1 {kn(l1)}, in DRAM {kn(dr)}" + (f", output {bd}" if bd else f", output banks {rw.get('per_bank')}")
                    + (f", Inf {rw.get('inf', 0)} NaN {rw.get('nan', 0)}" if rw.get("inf") or rw.get("nan") else "")
                    + lane_note(rw.get("per_lane"))))
    f = heavy_finding(chip)
    if f:
        out.append((f[0], "heavy", f[1], f[2]))
    f = repeat_finding(chip)
    if f:
        out.append((f[0], "repeat", f[1], f[2]))
    core = [(op, core_finding(chip.get(op), op)) for op in ("fpu", "sfpu")]
    core = [(op, f) for op, f in core if f]
    for op, f in core:
        out.append((f[0], "core", kn(chip.get(op)), f[1]))
    if l1.get("bad_checks") and not any(f[0].startswith("core ") for _, f in core):
        out.append(("compute", "mm", kn(l1), f"matmul in L1 {kn(l1)} (no DRAM in the loop), in DRAM {kn(dr)}"))
    ck = chip.get("clock") or {}
    if ck.get("checks") and ck.get("base_bad") and not ck.get("bad_checks"):
        out.append(("clock", "clock", f"{ck.get('base_bad')}/{ck.get('base_checks')}",
                    f"DRAM matmul {ck.get('base_bad')}/{ck.get('base_checks')} at the normal clock, 0/{ck['checks']} "
                    f"at {ck.get('mhz')} MHz"))
    rank = lambda c: (0 if c[0].startswith(("dram ch", "core ", "compute core")) else 1 if c[0] in ("noc0", "noc1", "pcie", "clock")
                      or c[0].startswith(COMPUTE) else 2 if c[0] in ("fpu", "sfpu", "compute", "heavy", "repeat")
                      or c[0].startswith(CONC_PFX) else 3)   # a bank's share alone is a pattern, never the cause
    out = sorted(out, key=rank)
    # a core spread goes after the other compute findings, never under a bank-share pattern
    sp = [c for c in out if c[0] == "compute (core spread)"]
    if sp:
        out = [c for c in out if c[0] != "compute (core spread)"]
        i = next((i for i, c in enumerate(out) if c[0].startswith(CONC_PFX) or rank(c) > 2), len(out))
        out[i:i] = sp
    return out


def isolated(chip):
    """A split named a part: the pinpoint can stop for this chip. Hits in the L1 matmul alone, or heavy hits spread over
    the banks, or the compute side named by lanes or columns, go on to the core split; the repeat split alone never stops
    it (the component splits still run)."""
    return any(c[0] not in ("compute", "heavy", "compute (core spread)") and not c[0].startswith(COMPUTE) and c[1] != "repeat"
               for c in findings(chip or {}))


def wants_clock(chip):
    m = (chip or {}).get("mm_dram") or {}
    return bool(m.get("bad_checks")) and not isolated(chip)


def lanes_of(detail):
    """', odd lanes' when a finding's detail names one lane (lane_note), else ''."""
    m = re.search(r"\b(odd|even) lane \d+/\d+", detail or "")
    return f", {m.group(1)} lanes" if m else ""


def load_line(ld, ch):
    """Numeric pinpoint under run.sh's max-load stage load, for one failing chip: the load, its launches and
    box kW, the chip's peak W and pre-fault peak C, and where its first bad result was (split, launch, owning compute core and
    in-tile element, or a DRAM split's bank; row, col)."""
    if not ld:
        return ""
    if ld.get("off") and not ld.get("bursts"):
        return f"load off ({ld['off']}): the checks ran solo"
    p = (ld.get("chips") or {}).get(ch.get("_bdf")) or {}
    out = (f"under {(ld.get('op') or '?').split(';')[0]}: {ld.get('launches', 0)} launches, box "
           f"{ld.get('box_kw_mean')} kW mean, {ld.get('box_kw_max')} kW max (32 chips' W); chip peak {p.get('w')} W, "
           + (f"{p['c']} C at launch {p['c_launch']} ({p['c_split']}, before any fault)" if p.get("c") is not None
              else "no pre-fault temperature"))
    locs = sorted((l for v in ch.values() if isinstance(v, dict) for l in (v.get("locs") or []) if l.get("launch")),
                  key=lambda l: l["launch"])
    if locs:
        l = locs[0]
        out += (f"; first bad at launch {l['launch']} ({l['split']}): "
                + (f"compute core ({l['core'][0]},{l['core'][1]}) of the {l['grid'][0]}x{l['grid'][1]} grid, element "
                   f"({l['element'][0]},{l['element'][1]}), " if l.get("core") else
                   f"DRAM bank {l['bank']}, " if l.get("bank") is not None and l["split"] not in COMPUTE_SPLITS else "")
                + f"row {l['row']} col {l['col']} (tile {l['tile'][0]},{l['tile'][1]}), {l['words']} words")
    if ld.get("off"):
        out += f"; load stopped: {ld['off']}"
    return out


def repeat_line(rp, s, ch=None):
    """The repeat split's outcome for one failing chip: did the fault come back, after how many launches."""
    if not rp:
        return ""
    load = f"; load {rp.get('load')}" if rp.get("load") else ""
    if (s or {}).get("bad_checks"):
        b = rp.get("back") or {}
        f = repeat_finding({"repeat": s})
        where = f", {label(f[0], f[2], ch)}" if f and f[0] != "repeat" else ""
        if s.get("launch0_bad") and int(s["bad_checks"]) <= 1:   # only launch 0's reference against the majority
            return (f"wrong from launch 0 (its reference against the 32-chip majority){where}; no new fault in the "
                    f"{rp.get('launches', 0)} launches after it ({rp.get('s', 0)} s){load}")
        return (f"fault back within {b.get('launches', rp.get('launches'))} launches ({b.get('s', rp.get('s'))} s)"
                f"{where}{load}")
    return f"not back in {rp.get('launches', 0)} launches ({rp.get('s', 0)} s){load}"


def eth_note(t, any_hit):
    if t.get("c4"):
        return "eth excluded (the C4 matmul that failed uses no ETH link)"
    if t.get("parts") and t["parts"] <= CCL_PARTS and not any_hit:
        return "eth/ccl suspected (only the sustained h3 part with CCL failed; local splits clean)"
    if t.get("parts") & CCL_PARTS:
        return "eth not tested alone (the sustained h3 part has CCL)"
    return "eth excluded (the failing sustained parts use no ETH link)" if t.get("parts") else ""


def report(numeric_lines, c7_lines, res, why=""):
    """The NUMERIC-PINPOINT lines: findings for every chip with hits, one verdict per failing chip."""
    tg = targets(numeric_lines, c7_lines)
    res = res if isinstance(res, dict) else {}
    chips = res.get("chips") or {}
    caps = nc.parse(sum((v for v in (res.get("captures") or {}).values()), []))
    out = [c["line"] for b in sorted(caps) for c in caps[b]]
    for bdf in sorted(set(tg) | set(chips), key=lambda b: (tray_of(b) or 9, b)):
        ch = chips.get(bdf) or {}
        tray = (tg.get(bdf) or {}).get("tray") or ch.get("tray") or tray_of(bdf) or "?"
        fs = findings(ch)
        extra = "" if bdf in tg else " (not in the run's NUMERIC FAIL lines)"
        for comp, split, k, detail in fs:
            out.append(f"NUMERIC-PINPOINT {bdf} tray {tray} component={comp} hits {k} (split {split}; {detail}){extra}")
        if bdf not in tg:
            continue
        note = eth_note(tg[bdf], bool(fs))
        if fs:
            comp, split, k, detail = fs[0]
            v = f"{label(comp, detail, ch)} ({split} {k})"
            if comp.startswith(CONC_PFX):
                v += f"; {wx_note(ch)}"
        elif caps.get(bdf):
            c = caps[bdf][0]
            v = (f"captured in-run: {c['cls']}, recompute {c['recompute']} ({len(caps[bdf])} capture(s); "
                 "NUMERIC-CAPTURE lines)")
        elif not ch:
            v = "not run" + (f" ({why})" if why else " (no result for this chip)")
        else:
            ran = [s for s in ("repeat", "dram", "mm_l1", "mm_dram", "heavy", "heavy_xl", "heavy_read", "heavy_write", "rw_dram", "fpu", "sfpu", "clock")
                   if (ch.get(s) or {}).get("checks")]
            n = sum(int(ch[s]["checks"]) for s in ran)
            v = (f"not isolated (0/{n}: " + ", ".join(f"{s} {kn(ch[s])}" for s in ran) + "): the fault did not come "
                 f"back in {res.get('s', '?')} s; intermittent, rerun the pinpoint or the run")
        tail = "; ".join(x for x in (note, why if fs and why else "") if x)
        out.append(f"NUMERIC-PINPOINT VERDICT {bdf} tray {tray}: {v}" + (f"; {tail}" if tail else ""))
        r = repeat_line(res.get("repeat"), ch.get("repeat"), ch)
        if r:
            out.append(f"NUMERIC-PINPOINT REPEAT {bdf} tray {tray}: {r}")
        r = load_line(res.get("load"), dict(ch, _bdf=bdf))
        if r:
            out.append(f"NUMERIC-PINPOINT LOAD {bdf} tray {tray}: numeric checks {r}")
    for s, e in sorted((res.get("errors") or {}).items()):
        out.append(f"NUMERIC-PINPOINT ERROR split {s}: {e}")
    return out


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    try:
        if argv[0] == "targets":
            for bdf, t in sorted(targets(read_lines(argv[1]), read_lines(argv[2]) if len(argv) > 2 else []).items()):
                print(bdf, t["tray"] or "?", int(t["c4"]), ",".join(sorted(t["parts"])) or "-")
        elif argv[0] == "capture":
            tg, caps = targets(read_lines(argv[1]), read_lines(argv[2])), nc.parse(read_lines(argv[2]))
            if len(argv) > 3 and caps:
                try:
                    with open(argv[3]) as f:
                        res = json.load(f)
                except (OSError, ValueError):
                    res = {}
                res = res if isinstance(res, dict) else {}
                res["captures"] = {b: [c["line"] for c in v] for b, v in caps.items()}
                with open(argv[3] + ".tmp", "w") as f:
                    json.dump(res, f, indent=1)
                os.replace(argv[3] + ".tmp", argv[3])
            print(nc.retrigger(tg, caps), flush=True)
        elif argv[0] == "report":
            try:
                with open(argv[3]) as f:
                    res = json.load(f)
            except (OSError, ValueError, IndexError):
                res = {}
            print("\n".join(report(read_lines(argv[1]), read_lines(argv[2]), res, argv[4] if len(argv) > 4 else "")),
                  flush=True)
    except Exception as e:      # never fail a run
        print(f"NUMERIC-PINPOINT ERROR report: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
