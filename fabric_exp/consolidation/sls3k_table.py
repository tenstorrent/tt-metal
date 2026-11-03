#!/usr/bin/env python3
"""sls3k_table.py [G3_RESULTS]: one table per path across slot sizes on the Galaxy (SLOT_STUDY.md "3-3.5 KiB slots").
Each cell: cycles/packet at the slot's largest payload (P = 2048 / 3072 / 3584 / 4352 B), its payload GB/s at 1.35 GHz, and
the rate at a small payload (1 KiB; 256 B where run), for the allocations named in COLS (mean of every pass found).
Results: 4352 B slots jobs 167464 (P2 / P3) and 167648 (2dk, vc1k, vc2k); 2 KiB jobs 167648 + 167673; 3 / 3.5 KiB the
G3 job's results (argument, default fabric_exp/results/glx_g3)."""
import glob, os, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), "results")
sys.path.insert(0, HERE)
sys.argv, args = ["x"], sys.argv[1:]
import sls2k_report as r2  # noqa: E402
import sls_report as r1  # noqa: E402

G3 = args[0] if args else f"{R}/glx_g3"
CLK = 1.35e9


def wire(p):
    return 12.80 + p / 18.50 + (3.0 if p + 96 > 4096 else 0.0)


def vc0(dirs, kind, tag):
    return r2.mean_rows([b[:-10] for d in dirs for b in glob.glob(f"{d}/{kind}_*_{tag}_tests.csv")])


def vc1(dirs, kind, tag):
    acc = defaultdict(list)
    for d in dirs:
        for f in glob.glob(f"{d}/{kind}_*_{tag}_tests.csv"):
            for k, v in r1.rows_of(f, r1.GLX_VC1_ROWS).items():
                acc[(k[0], int(k[1]))].append(statistics.mean(v))
    return {k: statistics.mean(v) for k, v in acc.items()}


D2K = [f"{R}/glx167648", f"{R}/glx167673"]
SIZES = [  # (slot payload, 2D (dirs, kind, tag, label), VC1 (dirs, kind, tag, label))
    (2048, [(D2K, "2d2k", "base", "41,32,32"), (D2K, "2d2k", "r1", "20,39,32")],
     [(D2K, "vc12k", "base", "107,9,16,5,4"), (D2K, "vc12k", "k3", "32,18,32,12,10")]),
    (3072, [([G3], "2d3k", "d", "35,16,32"), ([G3], "2d3k", "r", "20,21,32")],
     [([G3], "vc13k", "d", "53,9,16,5,4"), ([G3], "vc13k", "rule", "24,15,16,8,6"), ([G3], "vc13k", "sp", "20,15,16,9,7")]),
    (3584, [([G3], "2d35k", "d", "19,16,32"), ([G3], "2d35k", "r1", "20,21,16"), ([G3], "2d35k", "r2", "22,15,32")],
     [([G3], "vc135k", "d", "37,9,16,5,4"), ([G3], "vc135k", "rule", "14,13,16,8,6"), ([G3], "vc135k", "sp", "20,11,16,8,6")]),
    (4352, [([f"{R}/glx167648"], "2dk", "base", "17,16,16"), ([f"{R}/glx167464"], "2d", "s20", "20,15,16")],
     [([f"{R}/glx167464"], "vc1", "today", "19,9,16,5,4"), ([f"{R}/glx167464"], "vc1", "w31", "20,11,16,3,3"),
      ([f"{R}/glx167464"], "vc1", "w12", "20,6,16,7,6"), ([f"{R}/glx167464"], "vc1", "cp", "13,7,16,8,7")]),
]
ROWS2D = [("2D 1-hop E/W", "Sls1HopEW"), ("2D 3-hop E/W", "Sls3HopEW"), ("2D 3-hop N/S", "Sls3HopNS"),
          ("2D 7-hop N/S", "Sls7HopNS"), ("2D 10-hop corner", "SlsCorner10"), ("2D turn", "SlsTurn4"),
          ("2D row merge (link)", "SlsRowMerge"), ("2D column merge (link)", "SlsColMerge"), ("2D multicast", "SlsMulticast")]
ROWSV = [("VC1 split: VC0 1-hop", "1-hop E/W"), ("VC1 split: VC0 2-hop", "2-hop"), ("VC1 split: multicast", "multicast"),
         ("inter direct", "inter 0_1->1_0"), ("VC0 hop + inter direct", "inter 0_0->1_0"),
         ("inter + 1 VC1 hop E/W", "inter 0_1->1_1"), ("inter + 1 VC1 hop N/S", "inter 0_1->1_2"),
         ("inter + 2 VC1 hops", "inter 0_1->1_3")]


def cell(v, p, small):
    if v is None:
        return "-"
    gb = p / v * CLK / 1e9
    s = f"{v:.1f} ({gb:.1f} GB/s)" + (" wire" if v < wire(p) * 1.01 else "")
    return s + (f"; 1 KiB {small:.1f}" if small else "")


def main():
    print("Wire (cycles/packet): " + ", ".join(f"{p} B {wire(p):.1f} ({p / wire(p) * CLK / 1e9:.1f} GB/s)"
                                               for p in (2048, 3072, 3584, 4352)))
    for name, test in ROWS2D:
        print(f"\n### {name}")
        print("| slot payload | allocation | large payload: cycles (GB/s) | 1 KiB | 256 B |")
        print("| ---: | --- | --- | ---: | ---: |")
        for p, a2d, _ in SIZES:
            for dirs, kind, tag, lab in a2d:
                d = vc0(dirs, kind, tag)
                v = d.get((test, p), (None,))[0]
                k1 = d.get((test, 1024), (None,))[0]
                k256 = d.get((test, 256), (None,))[0]
                print(f"| {p} | {lab} | {cell(v, p, None)} | {k1 and f'{k1:.1f}' or '-'} | {k256 and f'{k256:.1f}' or '-'} |")
    for name, row in ROWSV:
        print(f"\n### {name}")
        print("| slot payload | allocation | large payload: cycles (GB/s) | small payload (16 / 256 B) |")
        print("| ---: | --- | --- | ---: |")
        for p, _, av in SIZES:
            for dirs, kind, tag, lab in av:
                d = vc1(dirs, kind, tag)
                v = d.get((row, p))
                sm = d.get((row, 256)) or d.get((row, 16))
                print(f"| {p} | {lab} | {cell(v, p, None)} | {sm and f'{sm:.1f}' or '-'} |")


if __name__ == "__main__":
    main()
