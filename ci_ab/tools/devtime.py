# Per case: device kernel time summed over the ops between signposts "<case>#k" (k >= 1), median over k per run; per side
# the median of its runs, deltas against H with the A/A (H2 against H) and the sign of every block. usage: devtime.py <dir>
import csv, glob, os, re, statistics, sys
from collections import defaultdict

O = sys.argv[1]
VS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["V1", "V2"]
per = defaultdict(lambda: defaultdict(dict))  # case -> side -> run -> ns
names = []
for d in sorted(glob.glob(f"{O}/prof_*_*")):
    if d.endswith(".log"):
        continue
    side, run = os.path.basename(d).split("_")[1:3]
    fs = glob.glob(f"{d}/reports/*/ops_perf_results_*.csv")
    if not fs:
        print("no report", d)
        continue
    acc = defaultdict(lambda: defaultdict(float))
    cur = None
    for row in csv.DictReader(open(fs[0])):
        if row.get("OP TYPE") == "signpost":
            n, _, k = row["OP CODE"].rpartition("#")
            cur = (n, int(k)) if k.isdigit() else None
            if cur and n not in names:
                names.append(n)
            continue
        if cur is None or cur[1] == 0:
            continue
        try:
            acc[cur[0]][cur[1]] += float(row["DEVICE KERNEL DURATION [ns]"])
        except (ValueError, KeyError):
            pass
    for n, reps in acc.items():
        per[n][side][int(run)] = statistics.median(reps.values())
for n in names:
    for side, runs in sorted(per[n].items()):
        print("RAW", n, side, " ".join(f"{k}:{v:.0f}" for k, v in sorted(runs.items())))
print("| case | H ns (runs) | H2 A/A % | " + " | ".join(f"{v} ns | {v} % | {v} blocks" for v in VS) + " |")
print("|---|---|---|" + "---|---|---|" * len(VS))
for n in names:
    s = per[n]
    if "H" not in s:
        continue
    mh = statistics.median(s["H"].values())
    cells = [f"{mh:.0f} ({', '.join(f'{v:.0f}' for _, v in sorted(s['H'].items()))})"]
    cells.append(f"{100 * (statistics.median(s['H2'].values()) - mh) / mh:+.2f}" if "H2" in s else "")
    for v in VS:
        if v not in s:
            cells += ["", "", ""]
            continue
        mv = statistics.median(s[v].values())
        hs, vs = sorted(s["H"].items()), sorted(s[v].items())
        # block b holds runs 2b-1 and 2b of every side
        signs = []
        for b in sorted({(k + 1) // 2 for k, _ in hs}):
            a = [x for k, x in hs if (k + 1) // 2 == b]
            c = [x for k, x in vs if (k + 1) // 2 == b]
            if a and c:
                signs.append("-" if statistics.mean(c) < statistics.mean(a) else "+")
        cells += [f"{mv:.0f} ({', '.join(f'{x:.0f}' for _, x in vs)})", f"{100 * (mv - mh) / mh:+.2f}", "".join(signs)]
    print(f"| {n} | " + " | ".join(cells) + " |")
