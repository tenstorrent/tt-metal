#!/usr/bin/env python3
"""Post-process a dfb_isr_timing_bench waveform capture (Zebu -> start-verdi --export -> waves.vcd.gz).

The bench's kernels carry no timing instrumentation; every number here comes from the waveform. The VCD
parser and edge helpers are imported from isr_tc_latency.py (unchanged). Export the signals listed by
gen_rc.py for the same manifest.

For each launch in the capture (matched in order to the bench manifest's JSON lines) it reports:
  * per ISR credit event (one per txn-id threshold crossing):
      T0  txn counter reached its threshold (risc_to_noc_arb o_[wr_]tiles_to_process_irq bit rises)
      T1  DM0 interrupt line high for it (rise after T0, or T0 itself when the ISR was already pending/running)
      Te  the ISR entry that serviced it (vector-table retire on DM0 preceding the credit write)
      T3b DM0's credit write lands on that DFB's tile counter (posted for producer trids, acked for consumer)
      T4  the overlay counter value changes            T5  Neo handshake for that counter (TRISC peers)
    plus whether it was queued behind another trid in the same ISR entry;
  * ISR occupancy: entries (vector -> mret), busy cycles, credit writes per entry;
  * spin time of every producer/consumer DM (tight loops in its retire trace), with the hottest loop lines
    so they can be mapped to source with addr2line on the kernel ELF;
  * throughput: whole-launch bytes/cycle, and steady state from threshold events after the first lap.

Cycles are DM cycles (all clocks are 1:1 on the emulator; see isr_tc_latency.py).

Usage:
  isr_bench_post.py waves.vcd.gz manifest.jsonl --dm0-elf <cache>/firmware/dm0/dm0.elf [--csv events.csv]
      [--tile X,Y=GY,GX ...] [--seq-offset N] [--kernel-pc-min 0x20000] [--launch-gap 8000]
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import isr_tc_latency as L  # noqa: E402  (parse_vcd, rises, first_after, clock_period)

REG_POSTED, REG_ACKED = L.REG_POSTED, L.REG_ACKED
ISR_CLIENT = 0  # i_local_* client index of DM0
DEFAULT_OBJDUMP = os.path.join(HERE, "..", "tt-metal", "runtime", "sfpi", "compiler", "bin", "riscv-tt-elf-objdump")


def dm_domain(dm):
    return "tile_prci_domain" if dm == 0 else f"tile_prci_domain_{dm}"


class Sigs:
    """Signal lookup scoped to one RTL tile (gen_y[gy].gen_x[gx])."""

    def __init__(self, ch, tile):
        self.ch = ch
        self.tile = tile
        gy, gx = tile
        self.names = [n for n in ch if f"gen_y[{gy}]" in n and f"gen_x[{gx}]" in n] or list(ch)

    def get(self, suffix, required=True):
        hits = [n for n in self.names if n.endswith(suffix)]
        if len(hits) == 1:
            return self.ch[hits[0]]
        if required:
            raise KeyError(f"tile {self.tile}: signal '{suffix}': {len(hits)} matches {hits[:3]}")
        return None


def elf_pcs(elf, objdump, vector_base):
    """Handler entry and its mret PC from the DM0 firmware ELF that ran."""
    dis = subprocess.run([objdump, "-d", elf], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^([0-9a-f]+) <_Z25dfb_implicit_sync_handlerv>:", dis, re.M)
    if not m:
        sys.exit("dfb_implicit_sync_handler not found in " + elf)
    handler = int(m.group(1), 16)
    mret = None
    for line in dis[m.end():].splitlines():
        mm = re.match(r"\s*([0-9a-f]+):\s+\S+\s+mret", line)
        if mm:
            mret = int(mm.group(1), 16)
            break
    return {"handler": handler, "mret": mret, "vector": None if vector_base is None else vector_base + 4 * 13}


def retires(edges, valid, iaddr):
    vt, pt = [x[0] for x in valid], [x[0] for x in iaddr]
    out = []
    for e in edges:
        i = bisect_right(vt, e) - 1
        if i >= 0 and valid[i][1] == 1:
            j = bisect_right(pt, e) - 1
            if j >= 0 and iaddr[j][1] is not None:
                out.append((e, iaddr[j][1]))
    return out


def val_at(c, t):
    i = bisect_right([x[0] for x in c], t) - 1
    return c[i][1] if i >= 0 else None


def spin_intervals(ret, min_cycles, period):
    """Runs of retires that stay inside one 64 B line and keep branching back (tight polling loops)."""
    out = []
    i = 0
    n = len(ret)
    while i < n:
        line = ret[i][1] >> 6
        j, backs = i + 1, 0
        while j < n and (ret[j][1] >> 6) == line and ret[j][0] - ret[j - 1][0] <= period:
            if ret[j][1] <= ret[j - 1][1]:
                backs += 1
            j += 1
        if backs >= 2 and ret[j - 1][0] - ret[i][0] >= min_cycles:
            out.append((ret[i][0], ret[j - 1][0], line << 6))
        i = j
    return out


def launch_windows(dm_rets, kernel_pc_min, gap):
    pts = sorted(t for ret in dm_rets.values() for t, pc in ret if pc >= kernel_pc_min)
    wins = []
    for t in pts:
        if wins and t - wins[-1][1] <= gap:
            wins[-1][1] = t
        else:
            wins.append([t, t])
    return [tuple(w) for w in wins]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vcd")
    ap.add_argument("manifest")
    ap.add_argument("--dm0-elf", help="firmware/dm0/dm0.elf from the run's TT_METAL_CACHE (preferred)")
    ap.add_argument("--handler-pc", type=lambda x: int(x, 0), help="instead of --dm0-elf")
    ap.add_argument("--mret-pc", type=lambda x: int(x, 0), help="instead of --dm0-elf")
    ap.add_argument("--objdump", default=DEFAULT_OBJDUMP)
    ap.add_argument("--vector-base", type=lambda x: int(x, 0), default=None,
                    help="DM0 vector table base; default: found from the retire trace (the PC that jumps to the handler)")
    ap.add_argument("--tile", action="append", default=[], metavar="X,Y=GY,GX", help="as in gen_rc.py")
    ap.add_argument("--seq-offset", type=int, default=0, help="manifest seq of the first launch in this capture")
    ap.add_argument("--kernel-pc-min", type=lambda x: int(x, 0), default=0x20000, help="PCs >= this are kernel code")
    ap.add_argument("--launch-gap", type=int, default=8000,
                    help="DM-cycle gap that separates launches (Zebu stops the clock while the host works: ~16.7k cycles between launches)")
    ap.add_argument("--window", type=int, default=20000, help="max cycles from T0 to its credit write")
    ap.add_argument("--spin-min", type=int, default=40, help="shortest spin interval reported (cycles)")
    ap.add_argument("--csv", help="write one row per ISR credit event")
    a = ap.parse_args()

    tiles = {(0, 0): (1, 0)}
    for t in a.tile:
        lhs, rhs = t.split("=")
        tiles[tuple(int(v) for v in lhs.split(","))] = tuple(int(v) for v in rhs.split(","))
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    rows = [r for r in rows if r["seq"] >= a.seq_offset]
    if a.dm0_elf:
        pcs = elf_pcs(a.dm0_elf, a.objdump, a.vector_base)
    elif a.handler_pc is not None and a.mret_pc is not None:
        pcs = {"handler": a.handler_pc, "mret": a.mret_pc, "vector": None if a.vector_base is None else a.vector_base + 4 * 13}
    else:
        sys.exit("give --dm0-elf, or both --handler-pc and --mret-pc")
    print(f"DM0 firmware: handler 0x{pcs['handler']:x}, mret 0x{pcs['mret']:x}" + (f", vector 0x{pcs['vector']:x}" if pcs["vector"] is not None else ", vector: auto"))

    ts_fs, ch = L.parse_vcd(a.vcd, [r"."])
    cores = sorted({tuple(c["logical"]) for r in rows for c in r["cores"]}) or [(0, 0)]
    events_out = []

    for core in cores:
        if core not in tiles:
            sys.exit(f"no RTL tile for logical core {core}; pass --tile")
        S = Sigs(ch, tiles[core])
        dm_clk = S.get("tt_tensix_with_l1.i_dm_clk")
        dm_p = L.clock_period(dm_clk)
        edges = L.rises(dm_clk)
        cyc = lambda dt: dt / dm_p
        OWs = "overlay_wrapper."
        arb_r = S.get("risc_to_noc_arb.o_tiles_to_process_irq")
        arb_w = S.get("risc_to_noc_arb.o_wr_tiles_to_process_irq")
        rocc = S.get("tile_prci_domain.element_reset_domain_rockettile.core.io_rocc_interrupt")
        rocc_rise = L.rises(rocc)

        # Retire traces: DM0 always, plus every DM the manifest uses that was exported.
        used_dms = {0}
        for r in rows:
            for c in r["cores"]:
                if tuple(c["logical"]) == core:
                    for d in c["dfbs"]:
                        used_dms |= {rc["risc"] for rc in d["risc_tcs"] if rc["risc"] < 8}
        dm_rets = {}
        for dm in sorted(used_dms):
            base = f".{dm_domain(dm)}.element_reset_domain_rockettile.core.csr.io_trace_0_"
            v, p = S.get(base + "valid", required=False), S.get(base + "iaddr", required=False)
            if v is None or p is None:
                print(f"  core {core}: DM{dm} retire trace not exported (spin/launch windows skip it)")
                continue
            dm_rets[dm] = retires(edges, v, p)
        dm0 = dm_rets.get(0, [])
        by_pc = defaultdict(list)
        for t, pc in dm0:
            by_pc[pc].append(t)
        if pcs["vector"] is None:
            # The vector-table slot is the PC retired just before each handler entry (0xef34 on W42 firmware).
            pre = defaultdict(int)
            for i in range(1, len(dm0)):
                if dm0[i][1] == pcs["handler"]:
                    pre[dm0[i - 1][1]] += 1
            if pre:
                pcs["vector"] = max(pre, key=pre.get)
                print(f"  core {core}: vector slot 0x{pcs['vector']:x} (precedes {pre[pcs['vector']]} handler entries)")
        vec_t, mret_t = by_pc.get(pcs["vector"], []), by_pc.get(pcs["mret"], [])

        # Credit writes from DM0 into every exported tile-counter bank.
        writes = []  # (t, bank, ctr, reg, data)
        banks = {}
        for k in range(4):
            pre = f"g_LLK_IF_GEN[{k}].overlay_tile_counters_with_comparators."
            wren = S.get(pre + "i_local_wren", required=False)
            if wren is None:
                continue
            addr, data = S.get(pre + f"i_local_addr[{ISR_CLIENT}]"), S.get(pre + f"i_local_wrdata[{ISR_CLIENT}]")
            banks[k] = pre
            for t in L.rises(wren, ISR_CLIENT):
                ad = val_at(addr, t)
                if ad is not None:
                    writes.append((t, k, (ad >> 5) & 0xF, (ad >> 2) & 7, val_at(data, t)))
        writes.sort()
        wtimes = [w[0] for w in writes]

        def neo_handshakes(k):
            pre = f"t6[{k}].neo.u_t6."
            rts, rtr, sel = (S.get(pre + s, required=False) for s in ("i_overlay_to_t6_rts", "o_t6_to_overlay_rtr", "i_overlay_to_t6_counter_sel"))
            if rts is None:
                return []
            out, prev = [], 0
            for t in sorted({x[0] for x in rts} | {x[0] for x in rtr}):
                cur = (val_at(rts, t) or 0) & (val_at(rtr, t) or 0)
                if cur and not prev:
                    out.append((t, val_at(sel, t)))
                prev = cur
            return out

        neo_hs = {k: neo_handshakes(k) for k in banks}

        # Launch windows (kernel code on the worker DMs) in capture order.
        wins = launch_windows({d: r for d, r in dm_rets.items() if d >= 2}, a.kernel_pc_min, a.launch_gap * dm_p)
        core_rows = [r for r in rows if any(tuple(c["logical"]) == core for c in r["cores"])]
        if not wins:
            t_all = [x[0] for x in dm_clk]
            wins = [(t_all[0], t_all[-1])]
            print(f"  core {core}: no worker-DM retire trace; treating the capture as one launch")
        if len(wins) != len(core_rows):
            print(f"  core {core}: WARNING {len(wins)} launch windows vs {len(core_rows)} manifest launches; "
                  f"matching the first {min(len(wins), len(core_rows))} in order")

        for wi, (w0, w1) in enumerate(wins):
            r = core_rows[wi] if wi < len(core_rows) else None
            cfg = next((c for c in r["cores"] if tuple(c["logical"]) == core), None) if r else None
            title = f"seq {r['seq']} {r['name']} launch {r['launch']}" if r else f"window {wi} (no manifest row)"
            print(f"\n=== core {core} {title}: {cyc(w1 - w0):.0f} cycles of kernel activity")
            if r:
                print(f"    {r['config']}")

            # Which tile counters does each (trid, kind) credit?
            targets = defaultdict(set)  # (trid, 'read'|'write') -> {(bank, ctr)}
            ent_per_event = {}
            for d in (cfg["dfbs"] if cfg else []):
                for side, kind in (("producer", "read"), ("consumer", "write")):
                    tx = d[f"{side}_txn"]
                    tcs = {(b, c) for rc in d["risc_tcs"] if rc["producer"] == (side == "producer") for b, c in rc["tcs"]}
                    for trid in tx["txn_ids"]:
                        targets[(trid, kind)] |= tcs
                        ent_per_event[(trid, kind)] = tx["entries_per_txn_id"]

            evs = []
            for kind, arb, reg in (("read", arb_r, REG_POSTED), ("write", arb_w, REG_ACKED)):
                for trid in range(32):
                    if targets and (trid, kind) not in targets:
                        continue
                    for t0 in L.rises(arb, trid):
                        if not (w0 - a.launch_gap * dm_p <= t0 <= w1 + a.window * dm_p):
                            continue
                        if bin(val_at(arb, t0) or 0).count("1") > 8:
                            continue  # teardown: thresholds reset to 0
                        tgt = targets.get((trid, kind))
                        i = bisect_left(wtimes, t0)
                        wr = None
                        while i < len(writes) and writes[i][0] <= t0 + a.window * dm_p:
                            w = writes[i]
                            if w[3] == reg and (tgt is None or (w[1], w[2]) in tgt):
                                wr = w
                                break
                            i += 1
                        pending = (val_at(rocc, t0) or 0) == 1
                        t1 = t0 if pending else L.first_after(rocc_rise, t0, t0 + a.window * dm_p)
                        e = dict(core=core, seq=r["seq"] if r else -1, kind=kind, trid=trid, T0=t0, T1=t1,
                                 queued=pending)
                        if wr:
                            e.update(T3b=wr[0], bank=wr[1], ctr=wr[2], inc=wr[4])
                            j = bisect_right(vec_t, wr[0]) - 1
                            e["Te"] = vec_t[j] if j >= 0 else None
                            av = S.get(banks[wr[1]] + f"tile_counters.o_tiles_available[{wr[2]}]", required=False)
                            e["T4"] = L.first_after([x[0] for x in av], wr[0], wr[0] + a.window * dm_p) if av else None
                            hs = [x[0] for x in neo_hs.get(wr[1], []) if x[0] >= wr[0] and x[1] == wr[2]]
                            e["T5"] = hs[0] if hs else None
                        evs.append(e)
            evs.sort(key=lambda e: e["T0"])

            def d(e, x, y):
                return None if e.get(x) is None or e.get(y) is None else cyc(e[y] - e[x])

            if evs:
                print(f"    {'kind':5} {'trid':>4} {'T0->T1':>7} {'T1->Te':>7} {'Te->T3b':>8} {'T0->T3b':>8} {'T3b->T4':>8} {'T3b->T5':>8}  queued")
                for n, e in enumerate(evs):
                    if n < 12 or n >= len(evs) - 4:
                        f = lambda v: "     -" if v is None else f"{v:6.0f}"
                        print(f"    {e['kind']:5} {e['trid']:>4} {f(d(e,'T0','T1')):>7} {f(d(e,'T1','Te')):>7} {f(d(e,'Te','T3b')):>8} "
                              f"{f(d(e,'T0','T3b')):>8} {f(d(e,'T3b','T4')):>8} {f(d(e,'T3b','T5')):>8}  {'yes' if e['queued'] else ''}")
                    elif n == 12:
                        print(f"    ... {len(evs) - 16} more ...")
                lat = sorted(v for v in (d(e, "T0", "T3b") for e in evs[1:]) if v is not None)
                if lat:
                    print(f"    threshold->credit write, excluding the first (cold) event: n={len(lat)} "
                          f"min {lat[0]:.0f} median {lat[len(lat)//2]:.0f} max {lat[-1]:.0f}; "
                          f"first event {d(evs[0], 'T0', 'T3b')}")
                missing = sum(1 for e in evs if e.get("T3b") is None)
                if missing:
                    print(f"    WARNING: {missing} threshold event(s) without a matching credit write in the window")

            # ISR occupancy in the launch window.
            ent = [t for t in vec_t if w0 - a.launch_gap * dm_p <= t <= w1 + a.window * dm_p]
            busy, per_entry = 0, Counter()
            for t in ent:
                m = L.first_after(mret_t, t)
                if m is not None:
                    busy += m - t
                    per_entry[sum(1 for w in writes if t <= w[0] <= m)] += 1
            span = w1 - w0
            print(f"    ISR: {len(ent)} entries, busy {cyc(busy):.0f} cycles ({100.0 * busy / span:.1f}% of the window); "
                  f"credit writes per entry {dict(sorted(per_entry.items()))}")

            # Spin time per worker DM.
            for dm, ret in sorted(dm_rets.items()):
                if dm < 2:
                    continue
                rr = [x for x in ret if w0 <= x[0] <= w1]
                if not rr:
                    continue
                sp = spin_intervals(rr, a.spin_min * dm_p, 4 * dm_p)
                tot = sum(b - a_ for a_, b, _ in sp)
                hot = Counter()
                for a_, b, line in sp:
                    hot[line] += b - a_
                hot_s = ", ".join(f"0x{line:x}:{cyc(c):.0f}" for line, c in hot.most_common(3))
                print(f"    DM{dm}: active {cyc(rr[-1][0] - rr[0][0]):.0f} cycles, spinning {cyc(tot):.0f} "
                      f"({100.0 * tot / max(1, rr[-1][0] - rr[0][0]):.1f}%); hottest loop lines {hot_s}")

            # Throughput.
            if r and cfg:
                nbytes = r["stream_entries"] * r["entry_size"] * len(cfg["dfbs"])
                print(f"    throughput: {nbytes} B in {cyc(span):.0f} cycles = {nbytes / cyc(span):.2f} B/cycle (whole launch)")
                kind = "write" if r["flow"] == "write" else "read"
                ss = [e for e in evs if e["kind"] == kind]
                lap = max((len(d0[f"{'consumer' if kind == 'write' else 'producer'}_txn"]["txn_ids"]) for d0 in cfg["dfbs"]), default=0)
                if len(ss) > 2 * lap + 1:
                    ss = ss[lap:]
                    ent_bytes = sum(ent_per_event.get((e["trid"], kind), 0) for e in ss[1:]) * r["entry_size"]
                    print(f"    steady state ({len(ss) - 1} threshold events after the first lap): "
                          f"{ent_bytes / cyc(ss[-1]['T0'] - ss[0]['T0']):.2f} B/cycle")
            events_out += evs

    if a.csv:
        keys = ["core", "seq", "kind", "trid", "queued", "T0", "T1", "Te", "T3b", "T4", "T5", "bank", "ctr", "inc"]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys + ["cyc_T0_T3b", "cyc_T1_Te", "cyc_Te_T3b"])
            w.writeheader()
            for e in events_out:
                row = {k: e.get(k) for k in keys}
                for name, x, y in (("cyc_T0_T3b", "T0", "T3b"), ("cyc_T1_Te", "T1", "Te"), ("cyc_Te_T3b", "Te", "T3b")):
                    row[name] = None if e.get(x) is None or e.get(y) is None else round((e[y] - e[x]) / dm_p)
                w.writerow(row)
        print(f"\nwrote {len(events_out)} events to {a.csv}")


if __name__ == "__main__":
    main()
