#!/usr/bin/env python3
"""sls2k_report.py RESULTS_DIR [lr|arms|2d|vc1|vc2] [--md]: tables for the 2 KiB-slot Galaxy study and the N/S floor arms
(SLOT_STUDY.md; Galaxy jobs run with fabric_exp/slotstudy/sls_glx2k.sbatch, labels <kind>_<pass>_<tag>).

  lr    line rate against payload: cycles/packet and payload GB/s per row, size and slot configuration, against the wire
        line 12.80 + P / 18.50 (+3.0 above 4 KiB: a second Ethernet frame; RESULTS §98, job 155281)
  arms  the N/S floor's one-side A/B: per row, size and configuration (2dk: 4352 B slots; 2d2k: 2 KiB slots) the base
        rate and the cost of each arm (tx16 / tx32 / rx16 / rx32, sp<N> / rp<N> per pass); a spin of N on a side with no
        slack costs about N + 12, so slack = (N + 12) - cost, floored at 0 (V4_BOUNDS.md §1's reading); side "TX" (E0) or
        "RX" (E1) when only that side has under 8 cycles of slack at 32, "both", or "-" (neither: a window, the link)
  2d    2D rows per slot split (2dk_* at 4352 B slots, 2d2k_* at 2 KiB), per payload
  vc1   the VC1 split rows per allocation (vc12k_* at 2 KiB slots, vc1k_* the 4352 B-slot control)
  vc2   VC2-on rows: per sender (VC0 / VC2) from the profiler's log lines
Cycles/packet = a sender row's cycles / its packets (1.35 GHz on both systems); merges are per link.
"""
import csv, glob, gzip, os, re, statistics, sys
from collections import defaultdict

CLK = 1.35e9
SENDER = re.compile(r"Sender core \S+ \(VC(\d)\) on device (\d+) direction \S+ link \d+: (\d+) packets in (\d+) cycles"
                    r"|another sender core \S+ \(VC(\d)\) with (\d+) packets in (\d+) cycles")


def wire(p):
    return 12.80 + p / 18.50 + (3.0 if p + 96 > 4096 else 0.0)


def lines(path):
    if os.path.exists(path):
        return open(path, errors="replace")
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", errors="replace")
    return []


def rows(f):
    """{(test, size): [cycles/packet of each sender row]}"""
    out = defaultdict(list)
    if not os.path.exists(f):
        return out
    for r in csv.DictReader(open(f)):
        name = r["test_name"].split("_size_")[0].split("_ntype_")[0]
        n = int(r["num_packets"]) * max(1, int(r.get("total_traffic_count") or 1))
        if n and int(r["cycles"]):
            out[(name, int(r["packet_size"]))].append(int(r["cycles"]) / n)
    return out


def runs(res, kind):
    """{tag: {pass: base path}}"""
    out = defaultdict(dict)
    for f in glob.glob(f"{res}/{kind}_*_tests.csv"):
        m = re.match(rf"{kind}_([a-z0-9]+?)_(.+)_tests\.csv$", os.path.basename(f))
        if m:
            out[m.group(2)][m.group(1)] = f[: -len("_tests.csv")]
    return out


def mean_rows(bases):
    acc = defaultdict(list)
    for b in bases:
        for k, v in rows(b + "_tests.csv").items():
            acc[k].append(statistics.mean(v))
    return {k: (statistics.mean(v), max(v) - min(v) if len(v) > 1 else 0.0) for k, v in acc.items()}


def table(head, body, md):
    if md:
        print("| " + " | ".join(head) + " |")
        print("| --- |" + " ---: |" * (len(head) - 1))
        for r in body:
            print("| " + " | ".join(r) + " |")
    else:
        print("  ".join(f"{h:>12}" for h in head))
        for r in body:
            print("  ".join(f"{c:>12}" for c in r))


def cmd_lr(res, md):
    rr = runs(res, "lr")
    data = mean_rows([b for t in rr.values() for b in t.values()])
    body = []
    for name in sorted({k[0] for k in data}, key=lambda n: (n[:4], -int(n.split("_s")[1]))):
        for size in sorted({k[1] for k in data if k[0] == name}):
            c, d = data[(name, size)]
            w = wire(size)
            body.append([name, str(size), f"{c:.1f}", f"{size / c * CLK / 1e9:.2f}", f"{w:.1f}", f"{100 * w / c:.1f}%"])
    table(["row (slots)", "payload B", "cyc/pkt", "GB/s", "wire cyc", "of wire"], body, md)


def arm_cost(arm):
    m = re.match(r"(tx|rx|sp|rp)(\d+)", arm)
    return (m.group(1), int(m.group(2))) if m else (None, 0)


def cmd_arms(res, md):
    for kind, label in (("2dk", "4352 B slots (17,16,16)"), ("2d2k", "2 KiB slots (default 41,32,32)")):
        rr = runs(res, kind)
        base = mean_rows([b for t, p in rr.items() if t == "base" for b in p.values()])
        arms = {t: mean_rows(list(p.values())) for t, p in rr.items() if arm_cost(t)[0]}
        if not base:
            continue
        print(f"\n## {kind}: {label}; base = mean of {sum(len(p) for t, p in rr.items() if t == 'base')} runs")
        order = sorted(arms, key=lambda a: (a[:2], int(re.sub(r"\D", "", a) or 0)))
        head = ["row", "B", "base"] + [f"c_{a}" for a in order] + ["slack TX", "slack RX", "side"]
        body = []
        for (name, size), (b, d) in sorted(base.items(), key=lambda x: (x[0][0], x[0][1])):
            cells, slack = [], {}
            for a in order:
                v = arms[a].get((name, size))
                cost = v[0] - b if v else None
                cells.append("-" if cost is None else f"{cost:+.1f}")
                side, n = arm_cost(a)
                if cost is not None and n == 32 and side in ("tx", "rx"):
                    slack[side] = max(0.0, 44 - cost)
            st, sr = slack.get("tx"), slack.get("rx")
            s = "?" if st is None or sr is None else ("both" if st < 8 and sr < 8 else "TX" if st < 8 else "RX" if sr < 8 else "-")
            body.append([name, str(size), f"{b:.1f}"] + cells + [f"{st:.1f}" if st is not None else "-",
                                                               f"{sr:.1f}" if sr is not None else "-", s])
        table(head, body, md)


def cmd_grid(res, kinds, md):
    for kind in kinds:
        rr = runs(res, kind)
        if not rr:
            continue
        tags = sorted(rr)
        data = {t: mean_rows(list(rr[t].values())) for t in tags}
        slots = {}
        for t in tags:
            for line in lines(next(iter(rr[t].values())) + ".log"):
                m = re.search(r"VC0 sender slots \{(\d+), (\d+)[^}]*\} receiver slots \{(\d+)\}", re.sub(r"\x1b\[[0-9;]*m", "", line))
                if m and m.group(2) != "0":
                    slots[t] = f"{m.group(1)},{m.group(2)},{m.group(3)}"
                    m1 = None
                    break
            for line in lines(next(iter(rr[t].values())) + ".log"):
                m1 = re.search(r"VC([12]) sender slots \{(\d+)[^}]*\} receiver slots \{(\d+)\}", re.sub(r"\x1b\[[0-9;]*m", "", line))
                if m1 and m1.group(2) != "0":
                    slots[t] += f" +VC{m1.group(1)} {m1.group(2)},{m1.group(3)}"
                    break
        keys = sorted({k for t in tags for k in data[t]})
        print(f"\n## {kind}")
        body = [[k[0], str(k[1])] + [f"{data[t][k][0]:.1f}" if k in data[t] else "-" for t in tags] for k in keys]
        table(["row", "B"] + [f"{t} {slots.get(t, '?')}" for t in tags], body, md)


def cmd_vc2(res, md):
    for kind in ("vc2k", "vc22k"):
        rr = runs(res, kind)
        print(f"\n## {kind}")
        for t in sorted(rr):
            for p, b in sorted(rr[t].items()):
                per = defaultdict(list)
                test = None
                for line in lines(b + ".log"):
                    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
                    m = re.search(r"Running Test: (\S+)", line)
                    if m:
                        test = m.group(1).split("_size_")[0]
                    m = SENDER.search(line)
                    if m and test:
                        vc, pk, cy = (m.group(1), m.group(3), m.group(4)) if m.group(1) else (m.group(5), m.group(6), m.group(7))
                        per[(test, vc)].append(int(cy) / int(pk))
                cells = ", ".join(f"{k[0]} VC{k[1]} {statistics.mean(v):.1f}" for k, v in sorted(per.items()))
                print(f"  {t} ({p}): {cells}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    md = "--md" in sys.argv
    res, what = a[0], (a[1] if len(a) > 1 else "lr")
    {"lr": lambda: cmd_lr(res, md), "arms": lambda: cmd_arms(res, md), "2d": lambda: cmd_grid(res, ("2dk", "2d2k"), md),
     "vc1": lambda: cmd_grid(res, ("vc1k", "vc12k"), md), "vc2": lambda: cmd_vc2(res, md)}[what]()
