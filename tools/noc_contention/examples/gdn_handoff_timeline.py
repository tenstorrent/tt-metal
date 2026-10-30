#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer 1 of tools/noc_contention for one op: the per-head hand-off timeline of the fused chunk_gated_delta_rule
prefill from a device-profiler CSV (TT_METAL_DEVICE_PROFILER=1; the zones are the ones in the op's dataflow and
compute kernels: rd_item, tx_wait_cb, tx_wait_credit, tx_issue, tx_barrier, tx_valid on the producers, rx_reserve,
rx_wait_valid, scan_wait_in, scan_step on the receivers).

    gdn_handoff_timeline.py profile_log_device.csv CASES WARM MEAS CASE [--bh 12 --nv 2 --np 7 --depth 5]
                            [--west 4 --west-ep 2] [--head H] [--root <tt-metal root>]

Pairs every producer item (ready = tx_wait_cb end, credit seen, issued, VALID sent) with the receivers' side of the
same chunk (credit sent = rx_reserve end, VALID seen = rx_wait_valid end, the step) through the op's core map
(placement 1, the Python oracle of test_chunk_gdn_fused_geometry.py; --west / --west-ep mirror the producer split of
PR #63364) and prints per head: the receivers' end, the compute's input wait per step, the producers' period,
credit wait and issue time per item, the credit and VALID latencies (mean / max), and with --head the chunk table.
CASES / WARM / MEAS select the run inside a sweep that profiled several cases with warm-up calls (the runs are
taken in order; CASE picks the middle measured run of that case). The pairing is the op-specific part of the
method: for another op, write its own version of `rows` / `per_rx` from its own zones and core map.
"""

import argparse
import csv
import importlib.util
import os
import statistics
import sys
from collections import defaultdict

CLK = 1350.0  # cycles per us


def us(c):
    return c / CLK


def parse_instances(path):
    inst = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # run -> (x,y,risc) -> name -> [(s,e)]
    kernel = defaultdict(dict)
    open_zones = {}
    with open(path) as f:
        next(f)
        next(f)
        for row in csv.reader(f):
            if len(row) < 12:
                continue
            x, y, risc, ts, run = int(row[1]), int(row[2]), row[3], int(row[5]), int(row[7])
            name, typ = row[10], row[11]
            key = (run, x, y, risc, name)
            if typ == "ZONE_START":
                open_zones[key] = ts
            elif typ == "ZONE_END" and key in open_zones:
                s = open_zones.pop(key)
                if name.endswith("-KERNEL"):
                    ks, ke = kernel[run].get((x, y, risc), (s, ts))
                    kernel[run][(x, y, risc)] = (min(ks, s), max(ke, ts))
                else:
                    inst[run][(x, y, risc)][name].append((s, ts))
    return inst, kernel


def load_oracle(root):
    p = os.path.join(root, "tests/ttnn/unit_tests/operations/transformers/test_chunk_gdn_fused_geometry.py")
    spec = importlib.util.spec_from_file_location("gdn_geom", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("cases")
    ap.add_argument("warm", type=int)
    ap.add_argument("meas", type=int)
    ap.add_argument("case")
    ap.add_argument("--bh", type=int, default=12)
    ap.add_argument("--nv", type=int, default=2)
    ap.add_argument("--np", type=int, default=7)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--head", type=int, default=-1, help="print the chunk table of this head")
    ap.add_argument("--root", default=".")
    ap.add_argument("--west", type=int, default=0, help="TT_GDN_FUSED_WEST: producers placed west of the receivers")
    ap.add_argument("--west-ep", type=int, default=-1, help="TT_GDN_FUSED_WEST_EP: the same in logical rows 0, 3, 7, 9")
    a = ap.parse_args()

    inst, kernel = parse_instances(a.csv)
    cases = a.cases.split(",")
    runs = sorted(kernel)
    r = runs[cases.index(a.case) * (a.warm + a.meas) + a.warm + a.meas // 2]
    t0 = min(s for s, _ in kernel[r].values())
    cores = sorted({(x, y) for (x, y, _) in kernel[r]})
    xs = sorted({x for x, _ in cores})
    ys = sorted({y for _, y in cores})
    virt = lambda lx, ly: (xs[lx], ys[ly])  # logical -> virtual, from the cores the run used

    G = load_oracle(a.root)
    rcv, prod = G._placement_row_local(len(xs), len(ys), a.bh, a.nv, a.np)
    NP, NV, D = a.np, a.nv, a.depth
    if a.west:  # the row heads: [west producers][receivers][east producers]
        L = NV + NP
        k_per_row = len(xs) // L
        n_row_heads = min(a.bh, k_per_row * len(ys))
        rcv, prod = list(rcv), list(prod)
        for h in range(n_row_heads):
            row, xs0 = h // k_per_row, (h % k_per_row) * L
            w = a.west_ep if (a.west_ep >= 0 and row in (0, 3, 7, 9)) else a.west
            for v in range(NV):
                rcv[h * NV + v] = (xs0 + w + v, row)
            for j in range(NP):
                prod[h * NP + j] = (xs0 + j if j < w else xs0 + NV + j, row)
    zr = lambda lx, ly, risc: inst[r].get((*virt(lx, ly), risc), {})

    print(f"== {a.case} run {r}: grid {len(xs)}x{len(ys)} (virtual x {xs}, y {ys}); t0 = kernel start")
    print(
        "head | rcv end (us) | starve (rx_wait_valid sum) | prod: items, rd_item/item, period, credit-wait/item, "
        "issue+barrier/item | latencies: credit (last rx credit -> tx seen), valid (tx sent -> rx seen) mean/max"
    )
    for h in range(a.bh):
        rx = [rcv[h * NV + v] for v in range(NV)]
        rx_end = [us(max(e for (xx, yy, rr), (_, e) in kernel[r].items() if (xx, yy) == virt(*c)) - t0) for c in rx]
        rx_wait = [zr(*c, "NCRISC").get("rx_wait_valid", []) for c in rx]
        rx_res = [zr(*c, "NCRISC").get("rx_reserve", []) for c in rx]
        NC = max(len(w) for w in rx_wait)
        starve = [us(sum(e - s for s, e in w)) for w in rx_wait]
        # per chunk: credit sent = the latest receiver's rx_reserve[c] end; valid seen = latest rx_wait_valid[c] end
        credit_sent = [max(w[c][1] for w in rx_res if c < len(w)) for c in range(NC)]
        valid_seen = [max(w[c][1] for w in rx_wait if c < len(w)) for c in range(NC)]
        valid_seen_first = [min(w[c][1] for w in rx_wait if c < len(w)) for c in range(NC)]
        step_end = [max(w[c][0] for w in rx_wait if c < len(w)) for c in range(NC)]  # start of the wait = step done
        rows = []
        prod_stats = []
        for j in range(NP):
            p = h * NP + j
            pc = prod[p]
            bz = zr(*pc, "BRISC")
            nz = zr(*pc, "NCRISC")
            n_items = len(bz.get("tx_valid", []))
            rd = nz.get("rd_item", [])
            period = []
            for n in range(n_items):
                c = j + n * NP
                ready = bz["tx_wait_cb"][n][1]
                cred = bz["tx_wait_credit"][n][1]
                issued = bz["tx_issue"][n][1]
                sent = bz["tx_valid"][n][1]
                rows.append(
                    dict(
                        c=c,
                        p=j,
                        n=n,
                        ready=ready,
                        credit_sent=credit_sent[c] if c < NC else None,
                        credit_seen=cred,
                        issued=issued,
                        sent=sent,
                        seen=valid_seen[c] if c < NC else None,
                        seen_first=valid_seen_first[c] if c < NC else None,
                        prev_step_end=step_end[c] if c < NC else None,
                    )
                )
                if n:
                    period.append(ready - bz["tx_wait_cb"][n - 1][1])
            prod_stats.append(
                dict(
                    j=j,
                    items=n_items,
                    rd=us(sum(e - s for s, e in rd)) / max(1, len(rd)),
                    period=us(statistics.mean(period)) if period else 0.0,
                    credit_wait=us(sum(e - s for s, e in bz.get("tx_wait_credit", []))) / max(1, n_items),
                    issue=us(sum(e - s for s, e in bz.get("tx_issue", []))) / max(1, n_items),
                    barrier=us(sum(e - s for s, e in bz.get("tx_barrier", []))) / max(1, n_items),
                    end=us(max(e for (xx, yy, rr), (_, e) in kernel[r].items() if (xx, yy) == virt(*pc)) - t0),
                )
            )
        rows.sort(key=lambda d: d["c"])
        lat_credit = [us(d["credit_seen"] - d["credit_sent"]) for d in rows if d["credit_sent"] is not None]
        lat_valid = [us(d["seen"] - d["sent"]) for d in rows if d["seen"] is not None]
        # Receiver-side per chunk (steady state, chunks 8..NC-2): the reader's gap between seeing VALID(c) and
        # reaching the wait for c+1, the part of it spent in rx_reserve (waiting for compute), the compute's own
        # input wait (TRISC_0 scan_wait_in, the zoned steps only), and the hand-off slack = seen(c) - sent(c).
        per_rx = []
        for v, c in enumerate(rx):
            w = rx_wait[v]
            res = rx_res[v]
            gaps = [us(w[k + 1][0] - w[k][1]) for k in range(8, len(w) - 2)]
            resv = [us(res[k + D][1] - res[k + D][0]) for k in range(8, len(w) - 2) if k + D < len(res)]
            tz = zr(*c, "TRISC_0").get("scan_wait_in", [])
            cwait = [us(e - s) for s, e in tz[3:]]
            slack = [us(d["seen_first" if v == 0 else "seen"] - d["sent"]) for d in rows[8:-2] if d["seen"] is not None]
            per_rx.append(
                f"v{v}: gap {statistics.mean(gaps):.2f} (reserve {statistics.mean(resv) if resv else 0:.2f}) "
                f"cwait {statistics.mean(cwait) if cwait else 0:.2f} slack {statistics.mean(slack):.2f}"
            )
        print(f"      rx steady: {' | '.join(per_rx)}")
        print(
            f"h{h:2d} rcv {','.join(f'({c[0]},{c[1]})' for c in rx)} end {'/'.join(f'{e:.1f}' for e in rx_end)} "
            f"starve {'/'.join(f'{s:.1f}' for s in starve)} | prod end {min(s['end'] for s in prod_stats):.0f}-"
            f"{max(s['end'] for s in prod_stats):.0f}, items {'/'.join(str(s['items']) for s in prod_stats)}, "
            f"rd {statistics.mean(s['rd'] for s in prod_stats):.2f}, period {statistics.mean(s['period'] for s in prod_stats):.2f}, "
            f"credit-wait {statistics.mean(s['credit_wait'] for s in prod_stats):.2f}, "
            f"issue {statistics.mean(s['issue'] for s in prod_stats):.2f} + barrier {statistics.mean(s['barrier'] for s in prod_stats):.2f} "
            f"| lat credit {statistics.mean(lat_credit):.2f}/{max(lat_credit):.2f}, valid {statistics.mean(lat_valid):.2f}/{max(lat_valid):.2f}"
        )
        if h == a.head:
            print(
                "   c  p n | ready  credit_sent credit_seen issued  sent  seen(first/last) | step_end-before | starve"
            )
            for d in rows:
                st = us(d["seen"] - d["prev_step_end"]) if d["seen"] is not None else float("nan")
                print(
                    f"  {d['c']:3d} {d['p']:2d} {d['n']:2d} | {us(d['ready']-t0):6.1f} {us(d['credit_sent']-t0) if d['credit_sent'] else float('nan'):6.1f} "
                    f"{us(d['credit_seen']-t0):6.1f} {us(d['issued']-t0):6.1f} {us(d['sent']-t0):6.1f} "
                    f"{us(d['seen_first']-t0) if d['seen_first'] else float('nan'):6.1f}/{us(d['seen']-t0) if d['seen'] else float('nan'):6.1f} | "
                    f"{us(d['prev_step_end']-t0) if d['prev_step_end'] else float('nan'):6.1f} | {st:5.2f}"
                )


if __name__ == "__main__":
    main()
