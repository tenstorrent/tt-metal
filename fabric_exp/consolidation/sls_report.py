#!/usr/bin/env python3
"""sls_report.py: slot study tables (SLOT_STUDY.md). Cycles/packet per row and allocation, from per-test CSVs
(cycles / packets of each sender row; = 1.35e9 / pps on the P150).

  sls_report.py p150 [TAG ...]            P150 VC1 split runs results/sls<pass>_<tag>_p1_o3 (mean of the passes; the
                                          largest pass-to-pass difference per allocation in the last column)
  sls_report.py glx RESULTS_DIR [KIND]    a Galaxy job's results (labels <kind>_<pass>_<tag>; KIND vc1, 2d or 1d)
  sls_report.py pred                      sls_predictions.txt against the P1 means (prov and v4gi)
  sls_report.py p150vc0 KIND [TAG ...]    P150 single-mesh runs results/sls2d_<pass>_<tag> / sls1d_<pass>_<tag> (run_cap)
Options: --md (markdown), --sizes 4352 (default 16,1024,4352).
"""
import csv, glob, os, re, statistics, sys
from collections import defaultdict

RES = "/home/aagarwal/tt-metal/fabric_exp/results"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sls_model import log_slots  # noqa: E402

# (row label, test-name prefix, packets of the sender row or None)
VC1_ROWS = [
    ("1-hop E/W", "VC1Mesh1HopEW", None), ("1-hop N/S", "VC1Mesh1HopNS", None),
    ("bursts 8", "VC1BurstEW_N8_G500", None), ("bursts 32", "VC1BurstEW_N32_G500", None),
    ("2-hop", "VC1Mesh2Hop", None), ("2-hop merged", "VC1Mesh2HopMerge", None), ("multicast", "VC1MeshMulticast", None),
    ("inter direct (1_0)", "VC1Inter_0_0_to_1_0", None), ("inter +1 VC1 (1_1)", "VC1Inter_0_0_to_1_1", None),
    ("inter +1 VC1 (1_2)", "VC1Inter_0_0_to_1_2", None), ("inter +2 VC1 (1_3)", "VC1Inter_0_0_to_1_3", None),
    ("mix: VC0 worker", "VC1MixWorker", 100000), ("mix: its VC1 flow", "VC1MixWorker", 40000),
    ("mix: VC0 forwarded", "VC1MixForward", 60000), ("mix: its VC1 flow ", "VC1MixForward", 40000),
    ("solo VC0 worker", "VC1SoloWorker", None), ("solo VC0 forwarded", "VC1SoloForward", None),
]
GLX_VC1_ROWS = VC1_ROWS[:7] + [
    (f"inter {s}->1_{c}", f"GlxInter_{s}_to_1_{c}", None) for s in ("0_1", "0_0") for c in range(4)] + [
    ("mix: VC0 worker", "GlxMixWorker", 100000), ("mix: its VC1 flow", "GlxMixWorker", 40000),
    ("mix: VC0 forwarded", "GlxMixForward", 60000), ("mix: its VC1 flow ", "GlxMixForward", 40000),
    ("solo VC0 worker", "GlxSoloWorker", None), ("solo VC0 forwarded", "GlxSoloForward", None)]
VC0_ROWS = [(n, n, None) for n in ("Sls1HopEW", "Sls1HopNS", "Sls1HopEWBidir", "Sls3HopEW", "SlsRowMerge", "SlsTurn4",
                                   "SlsMulticast", "Sls3HopNS", "Sls7HopNS", "SlsCorner10", "SlsColMerge",
                                   "SlsLin1HopEW", "SlsLin3HopEW", "SlsLinRowMerge", "SlsLin7HopNS", "SlsLinColMerge")]


def rows_of(tests_csv, spec):
    """{(row, size): [cycles/packet of each matching sender row]}"""
    out = defaultdict(list)
    if not os.path.exists(tests_csv):
        return out
    for r in csv.DictReader(open(tests_csv)):
        name = r["test_name"].split("_size_")[0].split("_ntype_")[0]
        size = r["packet_size"]
        n = int(r["num_packets"]) * max(1, int(r.get("total_traffic_count") or 1))
        if not n or not int(r["cycles"]):
            continue
        for label, prefix, pk in spec:
            if name == prefix and (pk is None or int(r["num_packets"]) == pk):
                out[(label, size)].append(int(r["cycles"]) / n)
    return out


def table(columns, spec, sizes, md):
    """columns: [(header, {pass: tests_csv})]"""
    data = []
    for hdr, runs in columns:
        per = {p: rows_of(f, spec) for p, f in runs.items()}
        data.append((hdr, per))
    keys = []
    for label, _, _ in spec:
        for s in sizes:
            if any((label, s) in per[p] for _, per in data for p in per):
                keys.append((label, s))
    head = ["row", "B"] + [h for h, _ in data] + ["max pass diff"]
    lines = []
    if md:
        lines.append("| " + " | ".join(head) + " |")
        lines.append("| --- | ---: |" + " ---: |" * (len(head) - 2))
    else:
        lines.append(" ".join(f"{h:>14}" for h in head))
    for k in keys:
        cells, diff = [], 0.0
        for hdr, per in data:
            vals = [statistics.mean(per[p][k]) for p in sorted(per) if per[p].get(k)]
            cells.append(f"{statistics.mean(vals):.1f}" if vals else "-")
            if len(vals) > 1:
                diff = max(diff, max(vals) - min(vals))
        row = [k[0], k[1]] + cells + [f"{diff:.1f}"]
        lines.append(("| " + " | ".join(row) + " |") if md else " ".join(f"{c:>14}" for c in row))
    return "\n".join(lines)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    md = "--md" in sys.argv
    sizes = ["16", "1024", "4352"]
    for a in sys.argv:
        if a.startswith("--sizes="):
            sizes = a.split("=", 1)[1].split(",")
    mode = args[0]
    if mode == "p150":
        runs = defaultdict(dict)
        for f in glob.glob(f"{RES}/sls*_*_p1_o3_13129088_vc1_tests.csv"):
            m = re.match(r"sls([a-z0-9]+?)_(.+)_p1_o3_13129088_vc1_tests\.csv", os.path.basename(f))
            if m:
                runs[m.group(2)][m.group(1)] = f
        tags = args[1:] or sorted(runs)
        cols = []
        for t in tags:
            if t not in runs:
                continue
            a = log_slots(next(iter(runs[t].values()))[: -len("_tests.csv")] + ".log")
            cols.append((f"{t} {','.join(map(str, a)) if a else '?'}", runs[t]))
        print(table(cols, VC1_ROWS, sizes, md))
    elif mode == "glx":
        res, kind = args[1], (args[2] if len(args) > 2 else "vc1")
        runs = defaultdict(dict)
        for f in glob.glob(f"{res}/{kind}_*_tests.csv"):
            m = re.match(rf"{kind}_([a-z0-9]+?)_(.+)_tests\.csv", os.path.basename(f))
            if m:
                runs[m.group(2)][m.group(1)] = f
        spec = GLX_VC1_ROWS if kind == "vc1" else VC0_ROWS
        cols = [(t, runs[t]) for t in (args[3:] or sorted(runs))]
        print(table(cols, spec, sizes, md))
    elif mode == "p150vc0":
        kind = args[1]
        runs = defaultdict(dict)
        for f in glob.glob(f"{RES}/sls{kind}_*.log"):
            m = re.match(rf"sls{kind}_([a-z0-9]+?)_(.+)\.log", os.path.basename(f))
            if m:
                t = f[:-4] + "_tests.csv"
                runs[m.group(2)][m.group(1)] = t
        cols = [(t, runs[t]) for t in (args[2:] or sorted(runs))]
        print(table(cols, VC0_ROWS, sizes, md))


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] != "pred"):
    main()


def pred_check(md=False):
    """sls_predictions.txt (written before P1) against the P1 means: per allocation and class, measured / prov / v4gi."""
    import statistics as st
    pred = defaultdict(dict)  # (tag, spec, size) -> {model: {class: value}}
    for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sls_predictions.txt")):
        m = re.match(r"(\S+)\s+(\S+)\s+(prov|v4gi)\s+(\d+) B\s+(.*)", line)
        if not m:
            continue
        cells = re.findall(r"(1-hop|2-hop|inter direct|inter \+ VC1|inter \+ 2 VC1)\s+([\d.]+) \[(\w+)\]", m.group(5))
        pred[(m.group(1), m.group(2), m.group(4))][m.group(3)] = {c: (float(v), w) for c, v, w in cells}
    cls_rows = {"1-hop": ("1-hop E/W", "1-hop N/S"), "2-hop": ("2-hop",), "inter direct": ("inter direct (1_0)",),
                "inter + VC1": ("inter +1 VC1 (1_1)", "inter +1 VC1 (1_2)"), "inter + 2 VC1": ("inter +2 VC1 (1_3)",)}
    errs = defaultdict(list)
    out = ["| allocation | B | class | measured | prov (binds) | err | v4gi | err |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (tag, spec, size), models in pred.items():
        files = sorted(glob.glob(f"{RES}/sls*_{tag}_p1_o3_13129088_vc1_tests.csv"))
        if not files:
            continue
        meas = defaultdict(list)
        for f in files:
            for (label, s), v in rows_of(f, VC1_ROWS).items():
                if s == size:
                    meas[label] += v
        for c, labels in cls_rows.items():
            vals = [x for l in labels for x in meas.get(l, [])]
            if not vals or c not in models.get("prov", {}):
                continue
            mv = st.mean(vals)
            pv, pw = models["prov"][c]
            gv = models.get("v4gi", {}).get(c, (None, None))[0]
            ep = 100 * (pv - mv) / mv
            errs["prov"].append(abs(ep))
            eg = 100 * (gv - mv) / mv if gv else None
            if eg is not None:
                errs["v4gi"].append(abs(eg))
            out.append(f"| {tag} {spec} | {size} | {c} | {mv:.1f} | {pv:.1f} ({pw}) | {ep:+.1f}% | "
                       f"{gv if gv is None else f'{gv:.1f}'} | {'' if eg is None else f'{eg:+.1f}%'} |")
    print("\n".join(out))
    for k, v in errs.items():
        print(f"{k}: {len(v)} points, mean |error| {st.mean(v):.1f}%, max {max(v):.1f}%, within 3%: {sum(x <= 3 for x in v)}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "pred":
    pred_check()
