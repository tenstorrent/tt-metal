#!/usr/bin/env python3
"""cisp_router_tally.py RUN [RUN ...], RUN = label,log,csv,off_log,off_csv (off_log may be "-": the arm is its own
baseline): per router and NX unicast_write iteration, the router's telemetry cycles against its baseline (median over
the baseline log's iterations of the same size / num_links / num_packets). A router is "low-rate" in an iteration when
its cycles are more than --thresh (default 0.02) above the baseline. Slow iterations: the sender row below 97% of the
off CSV's median for its size / ntype / num_links.

Router naming: physical chip and eth channel (cisp_router_map.topology), the peer it sends to, direction (row 0 north,
column 0 west), plane (P0: used by the num_links 1 rows; P1: the second link, num_links 2 only), link-sync role
(T: chip id below the peer's, else R; on arm only). A telemetry line is the router's own sending (eth_words_sent), so a
low-rate router is the sender side of the flow chip -> peer; the peer router is its receiver side."""
import collections, csv, re, statistics, sys

sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/port_tune/cisp")
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
import cisp_router_map as RM  # noqa: E402

THR = 0.02
if "--thresh" in ARGS:
    i = ARGS.index("--thresh")
    THR = float(ARGS[i + 1])
    del ARGS[i:i + 2]
RUN = re.compile(r"Running Test: (\S+)")
TEL = re.compile(r"Telemetry from MeshCoordinate\(\[(\d+), (\d+)\]\) core 0-(\d+): .*cycles=(\d+), eth_words_sent=\d+, "
                 r"packets_sent=(\d+)")
KEY = re.compile(r"size_(\d+)_ntype_(\w+?)_num_links_(\d)$")
peer, c2chip = RM.topology()
NEAR7 = {7, 3, 6}  # chip 7 ([1,3], the CSV row's sender) and its NeighborExchange sync partners [0,3] = 3, [1,2] = 6
chip2c = {v: k for k, v in c2chip.items()}


def parse(log):
    out, cur = [], None
    for line in open(log, errors="replace"):
        if m := RUN.search(line):
            cur = m.group(1)
            out.append((cur, {}))
        elif cur and (m := TEL.search(line)):
            r, c, ch, cyc, pk = map(int, m.groups())
            out[-1][1][(c2chip[(r, c)], ch)] = cyc
    return out


def rowkey(name, rows_csv_row):
    m = KEY.search(name)
    return (m.group(1), m.group(2), m.group(3), rows_csv_row["num_packets"])


plane = {}


def rname(rt):
    chip, ch = rt
    pc, pch = peer[(chip, ch)]
    (r, c), (r2, c2) = chip2c[chip], chip2c[pc]
    d = "N" if r2 < r else "S" if r2 > r else "W" if c2 < c else "E"
    return f"c{chip}/e{ch}->{d} c{pc}/e{pch} {plane.get(rt, 'P?')} {'T' if chip < pc else 'R'}"


def main():
    runs = []
    for spec in ARGS:
        label, log, csvp, off_log, off_csv = spec.split(",")
        runs.append((label, parse(log), list(csv.DictReader(open(csvp))), parse(off_log) if off_log != "-" else None,
                     list(csv.DictReader(open(off_csv))), off_log != "-"))
    # planes: routers seen in num_links 1 write iterations are P0
    for label, it, rows, offit, offrows, _ in runs:
        for (name, tel), r in zip(it, rows):
            if KEY.search(name).group(3) == "1":
                for rt in tel:
                    plane[rt] = "P0"
    for label, it, rows, offit, offrows, _ in runs:
        for (name, tel), r in zip(it, rows):
            for rt in tel:
                plane.setdefault(rt, "P1")
    census = collections.defaultdict(lambda: collections.Counter())  # router -> counts
    stable = {}
    levels = collections.defaultdict(collections.Counter)
    grp = collections.Counter()  # grouping of low routers within an iteration (all runs)
    for label, it, rows, offit, offrows, has_off in runs:
        assert len(it) == len(rows), (label, len(it), len(rows))
        med = collections.defaultdict(list)
        for r in offrows:
            m = KEY.search(r["test_name"])
            med[m.group(0)].append(float(r["bandwidth_GB_s"]))
        med = {k: statistics.median(v) for k, v in med.items()}
        base_it = list(zip(offit if offit else it, offrows if offit else rows))
        base = collections.defaultdict(list)
        for (name, tel), r in base_it:
            k = rowkey(name, r)
            for rt, cyc in tel.items():
                base[(k, rt)].append(cyc)
        base = {k: statistics.median(v) for k, v in base.items()}
        nlow = {True: collections.Counter(), False: collections.Counter()}
        conf = collections.Counter()  # (slow, a low router sends from chip 7 or a sync neighbor of it, any low router)
        per_router = collections.defaultdict(lambda: [0, 0, 0, 0])  # low in slow, slow its, low in normal, normal its
        examples = []
        for j, ((name, tel), r) in enumerate(zip(it, rows)):
            k = rowkey(name, r)
            if k[1] != "unicast_write":
                continue
            slow = float(r["bandwidth_GB_s"]) < 0.97 * med[KEY.search(r["test_name"]).group(0)]
            low = []
            for rt, cyc in tel.items():
                b = base.get((k, rt))
                if not b:
                    continue
                dx = cyc / b - 1
                levels[(k[0], k[2])][round(dx * 100)] += 1
                pr = per_router[rt]
                if slow:
                    pr[1] += 1
                else:
                    pr[3] += 1
                if dx > THR:
                    low.append((rt, dx))
                    pr[0 if slow else 2] += 1
            nlow[slow][len(low)] += 1
            lowset = {rt for rt, _ in low}
            if low:
                # by source chip: all of the chip's active routers low, or only some
                for chip in {rt[0] for rt in lowset}:
                    act = [rt for rt in tel if rt[0] == chip and (k, rt) in base]
                    grp["src chip: all low" if all(rt in lowset for rt in act) else "src chip: some low"] += 1
                # by destination chip (the peer receiving the flow)
                for dst in {peer[rt][0] for rt in lowset}:
                    inb = [rt for rt in tel if peer[rt][0] == dst and (k, rt) in base]
                    grp["dst chip: all inbound low" if all(rt in lowset for rt in inb) else "dst chip: some inbound low"] += 1
                # by link: the reverse direction (peer -> chip) also low
                for rt in lowset:
                    grp["link: reverse also low" if peer[rt] in lowset else "link: reverse not low"] += 1
                grp["iterations with low routers"] += 1
                grp["iterations: all active routers low"] += all(rt in lowset for rt in tel if (k, rt) in base)
                vals = sorted(round(100 * dx, 1) for _, dx in low)
                grp["iterations: low routers within 0.3% of each other"] += (vals[-1] - vals[0] <= 0.3)
            conf[(slow, any(rt[0] in NEAR7 for rt, _ in low), bool(low))] += 1
            if slow and len(examples) < 6:
                examples.append((j, name.split("Unicast")[-1], sorted(low)))
        print(f"\n== {label} (baseline: {'off arm' if has_off else 'own iterations'}); low-rate routers per write iteration:")
        for s in (True, False):
            c = nlow[s]
            n = sum(c.values())
            if n:
                print(f"   {'slow  ' if s else 'normal'} {n:3d} iterations: " + ", ".join(f"{k} low: {v}" for k, v in sorted(c.items())))
        print("   CSV row (chip 7 W, includes the end sync with chips 3 and 6) against the low routers' source chips:")
        for s_ in (True, False):
            print(f"      {'slow  ' if s_ else 'normal'}: a low router on chip 7/3/6: {conf[(s_, True, True)]}; low routers "
                  f"only elsewhere: {conf[(s_, False, True)]}; none: {conf[(s_, False, False)]}")
        for j, nm, low in examples:
            print(f"   e.g. slow it {j} {nm}: " + "; ".join(f"{rname(rt)} +{100 * dx:.1f}%" for rt, dx in low))
        ranked = sorted(per_router.items(), key=lambda x: -(x[1][0] / max(1, x[1][1])))
        stable[label] = {rt for rt, v in per_router.items() if v[1] and v[0] / v[1] >= 0.5}
        print("   routers by share of slow iterations at low rate (low in slow / slow; low in normal / normal):")
        for rt, v in ranked[:10]:
            print(f"      {rname(rt):34s} {v[0]:3d}/{v[1]:3d}   {v[2]:3d}/{v[3]:3d}")
        for rt, v in per_router.items():
            census[rt]["ls"] += v[0]; census[rt]["s"] += v[1]; census[rt]["ln"] += v[2]; census[rt]["n"] += v[3]
            if v[1] and v[0] / v[1] >= 0.5:
                census[rt]["runs_major"] += 1
            if v[1]:
                census[rt]["runs_with_slow"] += 1
    print("\n== grouping of low routers within an iteration (all runs, write iterations):")
    for kk in sorted(grp):
        print(f"   {kk}: {grp[kk]}")
    print("\n== router delta levels (% above baseline, write iterations, all runs): per size / num_links")
    for k in sorted(levels):
        c = levels[k]
        print(f"   size {k[0]} L{k[1]}: " + ", ".join(f"{d:+d}%:{n}" for d, n in sorted(c.items()) if n >= 3))
    print("\n== pooled census (all runs): router, low in slow / slow, low in normal / normal, runs where low in >= half the slow")
    for rt, c in sorted(census.items(), key=lambda x: -(x[1]["ls"] / max(1, x[1]["s"]))):
        print(f"   {rname(rt):34s} {c['ls']:4d}/{c['s']:4d} ({100 * c['ls'] / max(1, c['s']):5.1f}%)   {c['ln']:4d}/{c['n']:4d} "
              f"({100 * c['ln'] / max(1, c['n']):4.1f}%)   {c['runs_major']}/{c['runs_with_slow']}")
    labs = list(stable)
    print("\n== overlap of each run's majority set (routers low in >= half its slow iterations): size, Jaccard vs the first")
    for l in labs:
        a, b = stable[l], stable[labs[0]]
        print(f"   {l:14s} {len(a):2d}  J={len(a & b) / max(1, len(a | b)):.2f}  " + " ".join(sorted(rname(rt).split()[0] for rt in a)))


if __name__ == "__main__":
    main()
