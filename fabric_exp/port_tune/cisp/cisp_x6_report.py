#!/usr/bin/env python3
"""cisp_x6_report.py RESULTS_DIR: the num_links discriminator (cisp_flow6_ab.sh): per arm (A = stock L1/L2 alternation,
L1-only, L2-only) and profiler arm, the iterations with a low-rate router (router telemetry +2% over the off arm's
per-router median for the row type), the low router-iterations by plane (P0 = link 0, P1 = link 1), direction and source
chip, and the slow iterations' positions inside each test's 6 write iterations (size 2048 x3, then 4096 x3; for A each
size's L1 x3 then L2 x3, 12 per test)."""
import collections, csv, statistics, sys

R = sys.argv[1]
sys.argv = ["x"]
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/port_tune/cisp")
import cisp_router_tally as T  # noqa: E402

# planes: link 0 routers = those used by num_links 1 rows (from the A arm's L1 iterations)
for (n, tel), r in zip(T.parse(f"{R}/flow_A_off.log"), csv.DictReader(open(f"{R}/flow_A_off_bandwidth_results.csv"))):
    if T.rowkey(n, r)[2] == "1":
        for rt in tel:
            T.plane[rt] = "P0"


def dirn(rt):
    return T.rname(rt).split("->")[1][0]


for arm in ("A", "L1", "L2"):
    bit = T.parse(f"{R}/flow_{arm}_off.log"); brows = list(csv.DictReader(open(f"{R}/flow_{arm}_off_bandwidth_results.csv")))
    base = collections.defaultdict(list)
    for (n, tel), r in zip(bit, brows):
        for rt, c in tel.items():
            base[(T.rowkey(n, r), rt)].append(c)
    base = {k: statistics.median(v) for k, v in base.items()}
    for pa in ("on", "off"):
        it = T.parse(f"{R}/flow_{arm}_{pa}.log"); rows = list(csv.DictReader(open(f"{R}/flow_{arm}_{pa}_bandwidth_results.csv")))
        per_test = 12 if arm == "A" else 6
        lowit, cnt, pos, chips = 0, collections.Counter(), collections.Counter(), collections.Counter()
        pattern = ""
        for j, ((n, tel), r) in enumerate(zip(it, rows)):
            k = T.rowkey(n, r)
            low = [rt for rt, c in tel.items() if c / base[(k, rt)] - 1 > 0.02]
            for rt in tel:
                pl = T.plane.get(rt, "P1")
                cnt[(pl, "n")] += 1; cnt[("dir" + dirn(rt), "n")] += 1
            for rt in low:
                pl = T.plane.get(rt, "P1")
                cnt[(pl, "low")] += 1; cnt[("dir" + dirn(rt), "low")] += 1; chips[rt[0]] += 1
            lowit += bool(low)
            if low:
                pos[j % per_test] += 1
            pattern += "S" if low else "."
        print(f"== {arm} {pa}: {lowit}/{len(it)} iterations with a low router")
        if lowit:
            print("   low router-iterations by plane: " + ", ".join(
                f"{p} {cnt[(p,'low')]}/{cnt[(p,'n')]}" for p in ("P0", "P1") if cnt[(p, 'n')]))
            print("   by direction: " + ", ".join(f"{d[3]} {cnt[(d,'low')]}/{cnt[(d,'n')]}" for d in ("dirN", "dirE", "dirS", "dirW") if cnt[(d, 'n')]))
            print("   by source chip: " + ", ".join(f"c{c} {chips[c]}" for c in sorted(chips)))
            print(f"   positions in each test's {per_test} iterations: " + " ".join(f"{p}:{pos[p]}" for p in range(per_test)))
            print("   " + "\n   ".join(pattern[i:i + per_test] for i in range(0, len(pattern), per_test)))
