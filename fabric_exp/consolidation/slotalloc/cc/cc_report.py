#!/usr/bin/env python3
"""cc_report.py [--results DIR] [--tag cc]: the credit-cadence A/B's numbers (credit_cadence_runs.sh).

Per run (LABEL_txq.json from txq_snap.py, LABEL_tests.csv from the test):
- data links: Ethernet cores whose TXQ0 TRANSFER_CNT moved by at least 10% of the busiest core's (the router's E0 data);
  each one's credits come back on its peer core's TXQ1 (the far router's E1);
- steady window, per data core: the longest run of consecutive snapshot intervals whose TXQ0 rate is at least 95% of
  the median of the faster half of its intervals;
- over that window: data transfers / s, transfers per data packet k = data transfers / (window s x packets / s from
  the test's cycles/packet at 1.35 GHz), and credit transfers per data packet (the peer's TXQ1 delta / data packets),
  the resend deltas of both queues.
Then per row and arm the mean over passes, and the safety runs (cycles/packet with and without the snapshots)."""
import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics as st

CLK = 1.35e9
WRAP = 1 << 32


def rate_cpp(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    cpps = [CLK / float(r["packets_per_second"]) for r in rows if float(r["packets_per_second"]) > 0]
    return st.mean(cpps) if cpps else None


def series(snaps, i, key):
    out, prev, add = [], None, 0
    for s in snaps:
        c = s["cores"][i]
        v = c.get(key)
        if v is None:
            out.append((c["t"], None))
            continue
        if prev is not None and v < prev:
            add += WRAP
        prev = v
        out.append((c["t"], v + add))
    return out


def steady(ser):
    pts = [(t, v) for t, v in ser if v is not None]
    rates = [((pts[j + 1][1] - pts[j][1]) / (pts[j + 1][0] - pts[j][0]), j) for j in range(len(pts) - 1)
             if pts[j + 1][0] > pts[j][0]]
    if not rates:
        return None
    fast = sorted(r for r, _ in rates)[len(rates) // 2:]
    thr = 0.95 * st.median(fast)
    best, cur = (0, None, None), None
    for r, j in rates:
        if r >= thr:
            cur = (cur[0], j) if cur else (j, j)
            if cur[1] - cur[0] + 1 > best[0]:
                best = (cur[1] - cur[0] + 1, cur[0], cur[1] + 1)
        else:
            cur = None
    if not best[0]:
        return None
    return pts[best[1]], pts[best[2]], best[1], best[2]


def run_numbers(label, d):
    snap = json.load(open(os.path.join(d, label + "_txq.json")))
    cpp = rate_cpp(os.path.join(d, label + "_tests.csv"))
    cores = snap["cores"]
    snaps = snap["snapshots"]
    idx = {(c["chip"], c["chan"]): i for i, c in enumerate(cores)}
    tot = {}
    for i in range(len(cores)):
        s = [v for _, v in series(snaps, i, "q0_transfer") if v is not None]
        tot[i] = (s[-1] - s[0]) if len(s) > 1 else 0
    busiest = max(tot.values()) if tot else 0
    links = []
    for i, c in enumerate(cores):
        if busiest == 0 or tot[i] < 0.1 * busiest:
            continue
        w = steady(series(snaps, i, "q0_transfer"))
        if not w:
            continue
        (t0, v0), (t1, v1), j0, j1 = w
        peer = idx.get(tuple(c["peer"]))
        if peer is None:
            continue
        p = series(snaps, peer, "q1_transfer")
        pr0 = series(snaps, i, "q0_resend")
        pr1 = series(snaps, peer, "q1_resend")
        dt = t1 - t0
        packets = dt * CLK / cpp if cpp else None
        cr = p[j1][1] - p[j0][1] if p[j1][1] is not None and p[j0][1] is not None else None
        links.append({
            "data": f"{c['chip']}/{c['chan']}", "peer": f"{c['peer'][0]}/{c['peer'][1]}", "window_s": round(dt, 2),
            "data_per_s": (v1 - v0) / dt, "k": (v1 - v0) / packets if packets else None,
            "credits_per_packet": cr / packets if packets and cr is not None else None,
            "credits_per_data_transfer": cr / (v1 - v0) if cr is not None and v1 > v0 else None,
            "resend_data": (pr0[j1][1] or 0) - (pr0[j0][1] or 0), "resend_credit": (pr1[j1][1] or 0) - (pr1[j0][1] or 0),
        })
    return cpp, links, len(snaps), snap.get("errors", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/home/aagarwal/tt-metal/fabric_exp/results")
    ap.add_argument("--tag", default="cc")
    a = ap.parse_args()
    d = a.results
    safety = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, f"{a.tag}s_*_tests.csv"))):
        lab = os.path.basename(f)[:-len("_tests.csv")]
        kind = "snap" if re.search(r"_s\d$", lab) else "plain"
        safety[kind].append((lab, rate_cpp(f)))
    if safety:
        print("## Safety: idoff ns_4352, cycles/packet without and with the snapshot loop")
        for kind in ("plain", "snap"):
            print(f"  {kind}: " + ", ".join(f"{lab.rsplit('_', 1)[1]} {v:.2f}" for lab, v in safety[kind]))
        if safety["plain"] and safety["snap"]:
            dp = st.mean(v for _, v in safety["snap"]) - st.mean(v for _, v in safety["plain"])
            print(f"  snap - plain: {dp:+.2f} cycles/packet")
    agg = collections.defaultdict(list)
    print("\n## Per run (steady window per data link)")
    print("| run | cycles/packet | data link -> credit peer | window s | k (data transfers / packet) | credits / packet | "
          "resends data / credit |")
    print("| --- | ---: | --- | ---: | ---: | ---: | --- |")
    for f in sorted(glob.glob(os.path.join(d, f"{a.tag}[ab]_*_txq.json"))):
        lab = os.path.basename(f)[:-len("_txq.json")]
        m = re.match(rf"{a.tag}([ab])_(idoff|id)_(.+)$", lab)
        if not m:
            continue
        p, arm, row = m.groups()
        cpp, links, n, errs = run_numbers(lab, d)
        for li in links:
            print(f"| {lab} | {cpp:.2f} | {li['data']} -> {li['peer']} | {li['window_s']} | {li['k']:.3f} | "
                  f"{li['credits_per_packet']:.3f} | {li['resend_data']} / {li['resend_credit']} |")
            agg[(row, arm)].append((cpp, li["credits_per_packet"], li["k"]))
        if errs:
            print(f"| {lab} | read errors: {sum(errs.values())} |")
    print("\n## Per row: mean over passes and links")
    print("| row | idoff cycles/packet | id cycles/packet | idoff credits / packet | id credits / packet | k (idoff / id) |")
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in sorted({r for r, _ in agg}):
        o, i = agg.get((row, "idoff"), []), agg.get((row, "id"), [])
        if not o or not i:
            continue
        mean = lambda xs, j: st.mean(x[j] for x in xs)  # noqa: E731
        print(f"| {row} | {mean(o, 0):.2f} | {mean(i, 0):.2f} | {mean(o, 1):.3f} | {mean(i, 1):.3f} | "
              f"{mean(o, 2):.2f} / {mean(i, 2):.2f} |")


if __name__ == "__main__":
    main()
