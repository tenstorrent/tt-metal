#!/usr/bin/env python3
"""sb_rows_report.py TAG [TAG ...] [--senders] [--lat]: sb_rows_runs.sh results (labels sb<TAG>_<router>_<row>_<rep>).

Default: cycles/packet (1.35 GHz) of each row's summary CSV per router (mean over repetitions and tags, each value in
parentheses), and the change against the first router.
--senders: per-sender cycles/packet from the bandwidth profiler's log lines (sync-off rows: each sender's own run).
--lat: median net round trip (ns) per latency case (lat rows)."""
import collections, csv, glob, os, re, statistics, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
mode = "senders" if "--senders" in sys.argv else "lat" if "--lat" in sys.argv else "rows"
LINE = re.compile(r"(?:Sender core (\S+) \(VC(\d)\) on device (\d+) direction \S+ link \d+: (\d+) packets in (\d+) cycles"
                  r"|another sender core (\S+) \(VC(\d)\) with (\d+) packets in (\d+) cycles)")
data = collections.defaultdict(lambda: collections.defaultdict(list))
routers = []
for tag in args:
    ext = {"rows": ".csv", "senders": ".log", "lat": "_latency.csv"}[mode]
    for f in sorted(glob.glob(f"{R}/sb{tag}_*{ext}")):
        b = os.path.basename(f)
        if mode == "rows" and (b.endswith("_detail.csv") or b.endswith("_latency.csv")):
            continue
        m = re.match(rf"sb{tag}_(v[^_]+)_(.+)_(\d+){re.escape(ext)}$", b)
        if not m:
            continue
        router, row, rep = m.groups()
        if router not in routers:
            routers.append(router)
        if mode == "rows":
            for r in csv.DictReader(open(f)):
                data[row][router].append(1.35e9 / float(r["avg_packets_per_s"]))
        elif mode == "lat":
            g = collections.defaultdict(list)
            for r in csv.DictReader(open(f)):
                g[re.sub(r"_sample\d+$", "", r["test_name"])].append(float(r["net_avg_ns"]))
            for k, v in g.items():
                data[k][router].append(statistics.median(v))
        else:
            text = re.sub(r"\x1b\[[0-9;]*m", "", open(f, errors="replace").read())
            for g in LINE.findall(text):
                if g[0]:
                    key = f"{row} dev{g[2]} VC{g[1]}"
                    packets, cycles = int(g[3]), int(g[4])
                else:
                    key = f"{row} shared VC{g[6]}"
                    packets, cycles = int(g[7]), int(g[8])
                data[key][router].append(cycles / packets)
unit = "ns" if mode == "lat" else "cycles/packet"
print(f"{'row':34}" + "".join(f"{r:>26}" for r in routers) + f"   ({unit}; change vs {routers[0] if routers else '-'})")
for row in sorted(data):
    per = data[row]
    cells = []
    base = statistics.mean(per[routers[0]]) if per.get(routers[0]) else None
    for r in routers:
        v = per.get(r)
        if not v:
            cells.append(f"{'-':>26}")
            continue
        m = statistics.mean(v)
        d = f" {m - base:+6.1f}" if base is not None and r != routers[0] else ""
        cells.append(f"{m:8.1f}{d} ({' / '.join(f'{x:.1f}' for x in v)})".rjust(26))
    print(f"{row:34}" + "".join(cells))
