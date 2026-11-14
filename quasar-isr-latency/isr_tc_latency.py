#!/usr/bin/env python3
"""Measure DFB implicit-sync ISR reaction time on Quasar from a Zebu waveform export.

Input: the waves.vcd.gz produced by `start-verdi --export --rc isr_tc.rc` (signal list in isr_tc.rc).

For every ISR firing on the ISR core (DM0) it reports the timeline:
  T0  trid tiles_to_process counter reaches its threshold (risc_to_noc_arb.o_[wr_]tiles_to_process_irq rises)
  T1  DM0 per-trid IP bit set in the RoCC cmd buffer -> rocc_interrupt rises
  T1b Rocket CSR io_interrupt (trap is taken next)
  T2  first retired instruction of the handler (trace iaddr == handler entry)
  T3  llk_intf_write retired (trace iaddr == llk_intf_write PC)
  T3b overlay tile-counter MMIO write (i_local_wren from DM0's client) to the POSTED/ACKED register
  T4  overlay tile counter value changes (o_tiles_available of that counter)
  T5  overlay->Neo forward handshake (Neo u_t6 i_overlay_to_t6_rts && o_t6_to_overlay_rtr) for that counter
  T6  per-trid IP bit cleared; mret retired
"""
import argparse
import gzip
import re
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict

# Overlay LLK tile-counter register layout (tt_overlay_tile_counters_with_comparators.sv):
# 16 counters x 8 regs x 4 B, register 2 = POSTED (tiles available), 3 = ACKED (space available).
REG_POSTED, REG_ACKED = 2, 3


def parse_vcd(path, want):
    """Return (timescale_fs, {name: [(t, int|None)]}) for vars whose full name matches any regex in `want`."""
    opener = gzip.open if path.endswith(".gz") else open
    scope, id2names, changes = [], defaultdict(list), {}
    ts_fs = 1
    with opener(path, "rt", errors="replace") as f:
        it = iter(f)
        for line in it:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "$timescale":
                spec = tok[1] if len(tok) > 1 else next(it).strip()
                m = re.match(r"(\d+)\s*(fs|ps|ns|us|ms|s)", spec)
                ts_fs = int(m.group(1)) * {"fs": 1, "ps": 10**3, "ns": 10**6, "us": 10**9, "ms": 10**12, "s": 10**15}[m.group(2)]
            elif tok[0] == "$scope":
                scope.append(tok[2])
            elif tok[0] == "$upscope":
                scope.pop()
            elif tok[0] == "$var":
                code = tok[3]
                ref = re.sub(r"\[\d+:\d+\]$", "", "".join(tok[4:-1]))  # drop packed bit range, keep array index
                name = ".".join([s for s in scope if s not in ("qiwc_full",) and not s.startswith("unnamed$$")] + [ref])
                if any(re.search(w, name) for w in want):
                    id2names[code].append(name)
                    changes[name] = []
            elif tok[0] == "$enddefinitions":
                break
        t = 0
        for line in it:
            c = line[0] if line else ""
            if c == "#":
                t = int(line[1:])
                continue
            if c in "01xzXZ":
                code, val = line[1:].strip(), (int(c) if c in "01" else None)
            elif c in "bB":
                v, code = line[1:].split()
                val = None if re.search(r"[xzXZ]", v) else int(v, 2)
            else:
                continue
            for n in id2names.get(code, ()):
                ch = changes[n]
                if not ch or ch[-1][1] != val:
                    ch.append((t, val))
    return ts_fs, changes


def find(changes, suffix):
    hits = [n for n in changes if n.endswith(suffix)]
    if len(hits) != 1:
        raise KeyError(f"signal '{suffix}': {len(hits)} matches {hits[:4]}")
    return changes[hits[0]]


def opt(changes, suffix):
    try:
        return find(changes, suffix)
    except KeyError:
        return None


def rises(ch, bit=None):
    """Times where (value[bit] if bit is not None else value) goes 0 -> 1."""
    out, prev = [], 0
    for t, v in ch:
        cur = 0 if v is None else ((v >> bit) & 1 if bit is not None else (1 if v else 0))
        if cur and not prev:
            out.append(t)
        prev = cur
    return out


def falls(ch, bit):
    out, prev = [], 0
    for t, v in ch:
        cur = 0 if v is None else (v >> bit) & 1
        if prev and not cur:
            out.append(t)
        prev = cur
    return out


def first_after(times, t0, t_max=None):
    i = bisect_left(times, t0)
    if i < len(times) and (t_max is None or times[i] <= t_max):
        return times[i]
    return None


def clock_period(ch):
    r = rises(ch)
    if len(r) < 3:
        return None
    d = sorted(b - a for a, b in zip(r, r[1:]))
    return d[len(d) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vcd")
    ap.add_argument("--handler-pc", type=lambda x: int(x, 0), default=0xEE28, help="dfb_implicit_sync_handler entry PC")
    ap.add_argument("--vector-base", type=lambda x: int(x, 0), default=0xED00, help="MEM_INTERRUPT_TABLE_BASE (mtvec)")
    ap.add_argument("--llk-write-pcs", default="0xef3c,0xef9c", help="PCs of tt.rocc.llk_intf_write (posted,acked)")
    ap.add_argument("--clear-pcs", default="0xef04,0xef64", help="PCs of cmdbuf_clear_tiles_to_process_{tr_ack,wr_sent}")
    ap.add_argument("--mret-pc", type=lambda x: int(x, 0), default=0xEEE8)
    ap.add_argument("--tensix", type=int, default=0, help="LLK interface / Neo index of the consumer")
    ap.add_argument("--isr-client", type=int, default=0, help="i_local_* client index of the ISR core (DM0=0)")
    ap.add_argument("--window-ns", type=float, default=5000.0, help="max T0->event search window")
    ap.add_argument("--mark", action="append", default=[], metavar="NAME=PC",
                    help="also report retire time of PC relative to the IRQ (e.g. rdcycle_entry=0xee68); repeatable")
    ap.add_argument("--verbose", action="store_true", help="also print raw T-leg deltas")
    ap.add_argument("--dump-signals", action="store_true", help="list parsed signals and change counts")
    a = ap.parse_args()

    ts_fs, ch = parse_vcd(a.vcd, [r"."])
    ns = lambda t: t * ts_fs / 1e6
    if a.dump_signals:
        for n, c in sorted(ch.items()):
            print(f"{len(c):7d}  {n}")
        return

    missing = []
    req = lambda s: (opt(ch, s) if opt(ch, s) is not None else missing.append(s))
    dm_clk, core_clk, ai_clk = req("tt_tensix_with_l1.i_dm_clk"), req("overlay_wrapper.core_clk"), req("overlay_noc_wrap.i_ai_clk")
    arb_irq = req("risc_to_noc_arb.o_tiles_to_process_irq")
    arb_wirq = req("risc_to_noc_arb.o_wr_tiles_to_process_irq")
    # Aggregate RoCC IRQ line into the ISR core (OR of all per-trid IP bits of all cmd buffers).
    rocc_irq = req("core.io_rocc_interrupt")
    csr_irq = req("core.csr.io_interrupt")
    tr_v, tr_pc = req("core.csr.io_trace_0_valid"), req("core.csr.io_trace_0_iaddr")
    TC = f"g_LLK_IF_GEN[{a.tensix}].overlay_tile_counters_with_comparators."
    wren = req(TC + "i_local_wren")
    waddr = req(TC + f"i_local_addr[{a.isr_client}]")
    wdata = req(TC + f"i_local_wrdata[{a.isr_client}]")
    rts, rtr, rsel, ridx, rinc = (req(TC + s) for s in ("o_remote_rts", "i_remote_rtr", "o_remote_counter_sel", "o_remote_idx", "o_remote_incr"))
    NEO = f"t6[{a.tensix}].neo.u_t6."
    nrts, nrtr, nsel, nidx = (req(NEO + s) for s in ("i_overlay_to_t6_rts", "o_t6_to_overlay_rtr", "i_overlay_to_t6_counter_sel", "i_overlay_to_t6_idx"))
    if missing:
        print("MISSING SIGNALS (not in waveform):", *missing, sep="\n  ")
        sys.exit(2)

    def val_at(c, t):
        i = bisect_right([x[0] for x in c], t) - 1
        return c[i][1] if i >= 0 else None

    periods = {k: clock_period(c) for k, c in (("dm_clk", dm_clk), ("core_clk", core_clk), ("ai_clk", ai_clk))}
    print("clock periods (ns):", {k: (round(ns(v), 3) if v else "not toggling in dump") for k, v in periods.items()})
    dm_p = periods["dm_clk"] or periods["core_clk"]

    # Retired instructions on DM0.
    retired = []
    # trace valid/iaddr are sampled together; walk valid=1 intervals per clock edge using dm clock rises.
    clk_edges = rises(dm_clk) if periods["dm_clk"] else rises(core_clk)
    tv_times = [x[0] for x in tr_v]
    pc_times = [x[0] for x in tr_pc]
    for e in clk_edges:
        i = bisect_right(tv_times, e) - 1
        if i >= 0 and tr_v[i][1] == 1:
            j = bisect_right(pc_times, e) - 1
            if j >= 0 and tr_pc[j][1] is not None:
                retired.append((e, tr_pc[j][1]))
    ret_by_pc = defaultdict(list)
    for t, pc in retired:
        ret_by_pc[pc].append(t)
    llk_pcs = [int(x, 0) for x in a.llk_write_pcs.split(",")]
    clr_pcs = [int(x, 0) for x in a.clear_pcs.split(",")]

    # Overlay TC MMIO writes from the ISR client.
    wr_events = []
    for t in rises(wren, a.isr_client):
        addr, data = val_at(waddr, t), val_at(wdata, t)
        if addr is None:
            continue
        reg = (addr >> 2) & 7
        ctr = (addr >> 5) & 0xF
        wr_events.append((t, ctr, reg, data, addr))
    fwd = [(t, val_at(rsel, t), val_at(ridx, t), val_at(rinc, t)) for t in rises(rts)]
    neo_hs = []
    hs_times = sorted(set([x[0] for x in nrts] + [x[0] for x in nrtr]))
    prev = 0
    for t in hs_times:
        cur = (val_at(nrts, t) or 0) & (val_at(nrtr, t) or 0)
        if cur and not prev:
            neo_hs.append((t, val_at(nsel, t), val_at(nidx, t)))
        prev = cur

    win = int(a.window_ns * 1e6 / ts_fs)
    rocc_rise, rocc_fall = rises(rocc_irq), [t for t, v in rocc_irq if v == 0]
    vec_pc = a.vector_base + 4 * 13  # ROCC interrupt = cause 13, mtvec in vectored mode
    rows = []
    for kind, arb in (("read/tr_ack->POSTED", arb_irq), ("write/wr_sent->ACKED", arb_wirq)):
        want_reg = REG_POSTED if kind.startswith("read") else REG_ACKED
        for trid in range(32):
            for t0 in rises(arb, trid):
                # Skip teardown/reset: thresholds return to 0 so every trid reads "count >= 0" at once.
                if bin(val_at(arb, t0) or 0).count("1") > 8:
                    continue
                t1 = first_after(rocc_rise, t0, t0 + win)
                if t1 is None:
                    continue  # IE not enabled for this trid on the ISR core
                r = dict(kind=kind, trid=trid, T0=t0, T1=t1)
                r["T1b"] = first_after(rises(csr_irq), t1, t1 + win)
                r["Tvec"] = first_after(ret_by_pc.get(vec_pc, []), t1, t1 + win)
                r["T2"] = first_after(ret_by_pc.get(a.handler_pc, []), t1, t1 + win)
                r["T3"] = first_after(sorted(sum((ret_by_pc.get(p, []) for p in llk_pcs), [])), t1, t1 + win)
                w = [e for e in wr_events if t1 <= e[0] <= t1 + win and e[2] == want_reg]
                r["T3b"] = w[0][0] if w else None
                r["ctr"] = w[0][1] if w else None
                r["inc"] = w[0][3] if w else None
                if w:
                    ctr_av = opt(ch, TC + f"tile_counters.o_tiles_available[{w[0][1]}]")
                    r["T4"] = first_after([x[0] for x in ctr_av], w[0][0], w[0][0] + win) if ctr_av else None
                    f = [x for x in fwd if x[0] >= w[0][0] and x[1] == w[0][1]]
                    r["fwd"] = f[0][0] if f else None
                    n = [x for x in neo_hs if x[0] >= w[0][0] and x[1] == w[0][1]]
                    r["T5"] = n[0][0] if n else None
                r["T6clr"] = first_after(rocc_fall, t1, t1 + win)
                r["mret"] = first_after(ret_by_pc.get(a.mret_pc, []), t1, t1 + win)
                for m in a.mark:
                    name, pc = m.split("=")
                    r[name] = first_after(ret_by_pc.get(int(pc, 0), []), t1, r["mret"] or t1 + win)
                rows.append(r)
    rows.sort(key=lambda r: r["T1"])
    if not rows:
        print("No ISR firings found (no per-trid IP bit rose). Check IE enable / trace window.")
        sys.exit(1)

    cyc = lambda dt: dt / dm_p if dm_p else ns(dt)
    unit = "cycles" if dm_p else "ns"

    def at(r, key):
        """Offset of event `key` from the interrupt (T1), as a printable '+N' string."""
        return "     ?" if r.get(key) is None else f"{cyc(r[key] - r['T1']):+6.0f}"

    def span(r, a_, b_):
        return None if r.get(a_) is None or r.get(b_) is None else cyc(r[b_] - r[a_])

    if a.verbose:
        legs = [("T0", "T1"), ("T1", "T1b"), ("T1", "Tvec"), ("Tvec", "T2"), ("T2", "T3b"), ("T3b", "T4"),
                ("T3b", "T5"), ("T0", "T5"), ("T1", "T6clr"), ("T1", "mret")]
        for r in rows:
            print(f"[{r['kind']}] trid={r['trid']} " + " ".join(f"{x}->{y}={span(r, x, y)}" for x, y in legs))

    print(f"\nDFB implicit-sync ISR reaction time  ({len(rows)} ISR firing(s); "
          f"1 cycle = {ns(dm_p) if dm_p else '?'} ns, all clocks equal on the emulator)")
    summary = []
    for i, r in enumerate(rows, 1):
        posted = r["kind"].startswith("read")
        what = "posted (data written into the DFB)" if posted else "acked (space freed in the DFB)"
        print(f"\n=== ISR #{i}: txn id {r['trid']} -> +{r.get('inc')} tiles {what}, Neo{a.tensix} tile counter {r.get('ctr')}")
        if i == 1:
            print("    (first ISR of the program: instruction/data caches are cold)")
        print(f"\n    {unit:>6}  event")
        steps = [
            ("T0", "the last NOC transfer of this txn-id batch completed (txn counter reached its threshold)" if posted
                   else "the last write of this txn-id batch was sent (txn counter reached its threshold)"),
            ("T1", "INTERRUPT: RoCC interrupt line to DM0 goes high                    <-- start"),
            ("Tvec", "DM0 has jumped to the interrupt vector table"),
            ("T2", "DM0 starts dfb_implicit_sync_handler (saves registers, reads IP reg + txn descriptor)"),
            ("T3b", "handler writes '+N tiles' to the tile counter (llk_intf write lands in overlay)"),
            ("T4", "overlay copy of the tile counter is updated"),
            ("T5", f"Neo{a.tensix} receives the update -> consumer can see the tiles   <-- DATA READY" if posted
                   else f"Neo{a.tensix} receives the update -> producer can see free space <-- SPACE READY"),
            ("T6clr", "handler cleared the interrupt"),
            ("mret", "handler returned (mret); DM0 back to what it was doing"),
        ]
        for m in a.mark:
            steps.append((m.split("=")[0], f"(mark) retired PC {m.split('=')[1]}"))
        for key, text in sorted(steps, key=lambda kv: float("inf") if r.get(kv[0]) is None else r[kv[0]]):
            print(f"    {at(r, key)}  {text}")
        total = span(r, "T1", "T5")
        enter, body, prop = span(r, "T1", "T2"), span(r, "T2", "T3b"), span(r, "T3b", "T5")
        print(f"\n    ==> interrupt -> {'data' if posted else 'space'} ready: "
              + (f"{total:.0f} {unit} ({ns(r['T5'] - r['T1']):.0f} ns)" if total is not None else "n/a"))
        if None not in (enter, body, prop):
            print(f"        = {enter:.0f} getting into the ISR  +  {body:.0f} ISR code until the counter write"
                  f"  +  {prop:.0f} counter update reaching Neo{a.tensix}")
        summary.append((i, r["trid"], total, enter, body, prop))

    print(f"\n=== Summary ({unit})")
    print("    ISR  txn  interrupt->ready  = enter ISR + ISR code + counter->Neo")
    for i, trid, total, enter, body, prop in summary:
        f = lambda v: "   n/a" if v is None else f"{v:6.0f}"
        print(f"    #{i:<3} {trid:>3}  {f(total):>16}  = {f(enter):>9} + {f(body):>8} + {f(prop):>12}")
    return rows


if __name__ == "__main__":
    main()
