#!/usr/bin/env python3
"""slack_report.py [--tag T] [--prefix P] [--ladder D,D,...] [--json OUT]: the slack study's table (NOTES.md) from
fabric_exp/results/slack/s<T><pass>_<router>_<arm>.csv. Defaults: the wall-clock ladder (tag "", arms tx<D>/rx<D>, D 4..128).
The mcycle settling A/B: --tag m --prefix m (arms mtx<D>/mrx<D>); its wall-clock floor arms: --tag m (tx/rx 1, 2, 4).

Per row (test, links, size) and router: the base rate (cycles/packet = 1.35e9 / the test's avg_packets_per_s; base and
base2 runs of both passes), and per side (E0 = tx spins, E1 = rx spins) and D the rate per pass, their mean and the
change against the base mean. The noise band of a router and test is the largest spread of its rows' base runs (max -
min over the base and base2 runs of both passes; pooled over the test's rows because main's NeighborExchange rows switch
between two modes ~3 cycles apart; and over the day's sets, TAGS), at least FLOOR cycles/packet. A ladder point "drops" when the mean of its passes is
above the base mean by more than the band. Slack (the user's definition) = the largest D on the ladder with no drop at it
or any smaller D (0 when D=4 drops already); the knee lies between it and the next ladder D. A side with slack 0 bounds
the row; "neither" when both sides have slack (the row is bound elsewhere: wire, worker, a window). Also, per side, the
same reading at CLEAR cycles/packet ("clear drop", above any noise seen) for the hard knee."""
import csv, glob, json, os, re, statistics, sys

RES = "/home/aagarwal/tt-metal/fabric_exp/results/slack"
LADDER = [4, 8, 16, 32, 64, 128]
FLOOR = 0.5
CLEAR = 4.0
ROUTERS = ["main", "port"]
TAGS = ["", "m"]  # the day's sets: the wall-clock ladder, the mcycle settling A/B (their base runs pool the noise band)


def rates(path):
    out = {}
    for r in csv.DictReader(open(path)):
        k = (r["test_name"], int(r["num_links"]), int(r["packet_size"]), r["ntype"].replace("NOC_UNICAST_", "").lower())
        out[k] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def load(tag=""):
    data = {}  # (router, arm) -> {pass: {row: rate}}
    for f in glob.glob(f"{RES}/s{tag}[ab]_*_*.csv"):
        m = re.match(rf"s{tag}([ab])_(main|port)_(\w+?)\.csv$", os.path.basename(f))
        if not m or f.endswith("_detail.csv"):
            continue
        p, router, arm = m.groups()
        data.setdefault((router, arm), {})[p] = rates(f)
    return data


def analyse(data, prefix="", ladder=LADDER, band_sets=None):
    """band_sets: the data sets whose base runs set the noise band (default: every set of the day, TAGS)."""
    rows = sorted({row for v in data.values() for d in v.values() for row in d},
                  key=lambda r: (r[0] != "Mesh1HopEW", r[2], r[1]))

    def base_runs(router, row, dd=None):
        dd = data if dd is None else dd
        return [d[row] for arm in ("base", "base2") for d in dd.get((router, arm), {}).values() if row in d]

    group_band = {}
    for dd in (band_sets if band_sets is not None else [load(t) for t in TAGS]):
        for router in ROUTERS:
            for row in rows:
                b = base_runs(router, row, dd)
                if b:
                    k = (router, row[0])
                    group_band[k] = max(group_band.get(k, 0.0), max(b) - min(b))
    out = []
    for row in rows:
        for router in ROUTERS:
            base = base_runs(router, row)
            if not base:
                continue
            bm = statistics.mean(base)
            band = max(FLOOR, group_band[(router, row[0])])
            rec = {"test": row[0], "links": row[1], "size": row[2], "ntype": row[3], "router": router,
                   "base": round(bm, 2), "base_runs": [round(b, 2) for b in base], "band": round(band, 2), "sides": {}}
            for side, pre in (("E0", prefix + "tx"), ("E1", prefix + "rx")):
                pts, slack, broke, clear, cbroke = [], 0, False, 0, False
                for d in ladder:
                    per = {p: v[row] for p, v in data.get((router, f"{pre}{d}"), {}).items() if row in v}
                    if not per:
                        continue
                    mean = statistics.mean(per.values())
                    drop = mean - bm > band
                    pts.append({"D": d, "passes": {p: round(x, 2) for p, x in sorted(per.items())}, "mean": round(mean, 2),
                                "delta": round(mean - bm, 2), "drop": drop})
                    if not broke:
                        if drop:
                            broke = True
                        else:
                            slack = d
                    if not cbroke:
                        if mean - bm >= CLEAR:
                            cbroke = True
                        else:
                            clear = d
                tested = [p["D"] for p in pts]
                nxt = next((p["D"] for p in pts if p["drop"]), None)
                cnxt = next((p["D"] for p in pts if p["delta"] >= CLEAR), None)
                rec["sides"][side] = {"points": pts, "slack": slack, "knee_next": nxt, "clear_slack": clear,
                                      "clear_next": cnxt, "open": nxt is None,
                                      "max_tested": max(tested) if tested else None}
            e0, e1 = rec["sides"]["E0"], rec["sides"]["E1"]
            b0 = e0["points"] and e0["slack"] == 0 and e0["knee_next"] is not None
            b1 = e1["points"] and e1["slack"] == 0 and e1["knee_next"] is not None
            rec["bound"] = "E0 and E1" if b0 and b1 else "E0" if b0 else "E1" if b1 else "neither"
            out.append(rec)
    return out


def fmt(recs, ladder=LADDER):
    lines = []
    for rec in recs:
        lines.append(f"### {rec['test']} {rec['size']} B, {rec['links']} link(s), {rec['ntype']}: {rec['router']}")
        lines.append(f"base {rec['base']:.2f} cycles/packet (runs {', '.join(f'{b:.2f}' for b in rec['base_runs'])};"
                     f" noise band {rec['band']:.2f})")
        lines.append("| side | " + " | ".join(f"D={d}" for d in ladder) + " | slack |")
        lines.append("|---|" + "---:|" * (len(ladder) + 1))
        for side in ("E0", "E1"):
            s = rec["sides"][side]
            by = {p["D"]: p for p in s["points"]}
            cells = []
            for d in ladder:
                p = by.get(d)
                if not p:
                    cells.append("-")
                else:
                    pv = " / ".join(f"{x:.1f}" for x in p["passes"].values())
                    cells.append(f"{p['delta']:+.1f}{' *' if p['drop'] else ''} ({pv})")
            sl = (f"{s['slack']} (knee {s['slack']}-{s['knee_next']})" if s["knee_next"] is not None
                  else f">= {s['max_tested']} (no drop)")
            if s["clear_next"] != s["knee_next"]:
                sl += (f"; clear drop: {s['clear_slack']}-{s['clear_next']}" if s["clear_next"] is not None
                       else f"; no clear drop up to {s['max_tested']}")
            lines.append(f"| {side} | " + " | ".join(cells) + f" | {sl} |")
        lines.append(f"bound: {rec['bound']}")
        lines.append("")
    return "\n".join(lines)


def main():
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    ladder = [int(x) for x in arg("--ladder", ",".join(map(str, LADDER))).split(",")]
    recs = analyse(load(arg("--tag", "")), arg("--prefix", ""), ladder)
    print(fmt(recs, ladder))
    if "--json" in sys.argv:
        json.dump(recs, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
