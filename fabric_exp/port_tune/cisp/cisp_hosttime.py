#!/usr/bin/env python3
"""cisp_hosttime.py LOG CSV OFF_CSV [LOG CSV OFF_CSV ...]: does a slow unicast_write iteration take longer in host
time? Per iteration: the host window from "Waiting for programs" to "Test ... Finished." (log timestamps, 1 ms
resolution) and the sender row's device cycles (CSV). Slow: the row below 97% of OFF_CSV's median for its size / ntype /
num_links. Per run and row type (size, num_links): mean window of slow and of normal iterations, the difference, and the
difference the cycles predict at a fixed 1.35 GHz (what a real slowdown would add). A clock change that leaves real time
alone would show a window difference near 0 while the cycles predict ~+0.44 ms per 100k packets."""
import collections, csv, datetime, re, statistics, sys

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})")
KEY = re.compile(r"size_(\d+)_ntype_(\w+?)_num_links_(\d)$")


def t_of(line):
    m = TS.match(line)
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() if m else None


def iters(log):
    out, cur, tl, tw = [], None, None, None
    for line in open(log, errors="replace"):
        if "Running Test: " in line:
            cur = line.split("Running Test: ")[1].split(" (")[0].strip()
        elif "Launching programs" in line:
            tl = t_of(line)
        elif "Waiting for programs" in line:
            tw = t_of(line)
        elif cur and f"Test {cur} Finished." in line and tw is not None:
            out.append((cur, tl, tw, t_of(line)))
            tl = tw = None
    return out


def main(a):
    agg = collections.defaultdict(lambda: [[], [], [], []])
    for i in range(0, len(a), 3):
        log, on_csv, off_csv = a[i:i + 3]
        med = collections.defaultdict(list)
        for r in csv.DictReader(open(off_csv)):
            med[KEY.search(r["test_name"]).group(0)].append(float(r["bandwidth_GB_s"]))
        med = {k: statistics.median(v) for k, v in med.items()}
        rows = list(csv.DictReader(open(on_csv)))
        it = iters(log)
        assert len(it) == len(rows), (log, len(it), len(rows))
        per = collections.defaultdict(lambda: [[], [], [], []])  # key -> slow windows, normal windows, slow cycles, normal cycles
        for (name, tl, tw, tf), r in zip(it, rows):
            assert r["test_name"] in name, (name, r["test_name"])
            m = KEY.search(r["test_name"])
            if m.group(2) != "unicast_write":
                continue
            slow = float(r["bandwidth_GB_s"]) < 0.97 * med[m.group(0)]
            k = (m.group(1), m.group(3), r["num_packets"])
            g = 0 if slow else 1
            per[k][g].append((tf - tw) * 1e3)
            per[k][g + 2].append(int(r["cycles"]) / 1.35e6)
            for q in range(4):
                pass
            agg[k][g].append((tf - tw) * 1e3)
            agg[k][g + 2].append(int(r["cycles"]) / 1.35e6)
        print(f"== {log.split('/')[-2]}/{log.split('/')[-1]}")
        for k in sorted(per):
            s, n, sc, nc = per[k]
            if not s or not n:
                print(f"   size {k[0]} L{k[1]} pk {k[2]}: slow {len(s)} normal {len(n)}")
                continue
            print(f"   size {k[0]} L{k[1]} pk {k[2]}: slow {len(s):3d} normal {len(n):3d}  window slow {statistics.mean(s):6.2f} "
                  f"normal {statistics.mean(n):6.2f} ms, diff {statistics.mean(s) - statistics.mean(n):+.2f} ms "
                  f"(se {(statistics.pstdev(s) ** 2 / len(s) + statistics.pstdev(n) ** 2 / len(n)) ** 0.5:.2f}); "
                  f"cycles predict {statistics.mean(sc) - statistics.mean(nc):+.2f} ms")
    print("== pooled over runs")
    for k in sorted(agg):
        s, n, sc, nc = agg[k]
        if s and n:
            print(f"   size {k[0]} L{k[1]} pk {k[2]}: slow {len(s):3d} normal {len(n):3d}  window diff "
                  f"{statistics.mean(s) - statistics.mean(n):+.3f} ms (se {(statistics.pstdev(s) ** 2 / len(s) + statistics.pstdev(n) ** 2 / len(n)) ** 0.5:.3f}); "
                  f"cycles predict {statistics.mean(sc) - statistics.mean(nc):+.3f} ms")


main(sys.argv[1:])
