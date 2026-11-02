#!/usr/bin/env python3
"""Phase 4 report (VC2_LINE_BRINGUP.md): one vc2l_glx_mm.sbatch job's allvc*/perf* steps on the Galaxy 2x2 split.

    allvc_report.py /data/aagarwal/frs-glx/jobs/<id>/results [--yaml fabric_exp/vc2line/glx_allvc_perf.yaml]

Steps: per-rank test counts and the slot rows each step logged (TT_FABRIC_EXP_LOG_SLOTS). Perf steps: the profiler's
per-sender log lines (the main tree's profiler change: "Sender core ... (VCn) on device d direction D link l: P packets in
C cycles", and "Shared link: ..." for a second sender core on the same chip and direction), each under the test its rank
(= mesh: [1,0] mesh 0, [1,1] mesh 1) was running: per sender its packets over its own cycles, as cycles per packet and
payload GB/s in the device clock (recovered from the step's per-sender CSV: pps x cycles / packets; else 1.35 GHz).
Shared paths: the flows of a two-flow test (VC0 and VC2 onto one intermesh exit and one landing router's VC1; VC0
forwarded and VC2 injected onto one link inside a mesh) with all packets over the slowest member's cycles. PerfAllFlows:
per VC, the spread of its senders' cycles per packet. perfab*: the window A/B table. A step without per-sender lines
(job 167441's profiler) falls back to its per-sender CSV, one row per test and rank, labelled from the yaml.
"""
import argparse
import csv
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
COORD = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}  # 2x2 chip id -> (row, col)
EXIT = {(0, 1): "N", (0, 3): "E", (1, 0): "N", (1, 2): "S", (0, 2): "E"}  # (mesh, chip) -> straight exit direction
MESH_PKTS = {20000: 0, 16000: 1}
# Paths shared by two flows: test -> ([(mesh, chip, vc) of each sender], description)
SHARED = {"PerfInterVC0VC2": ([(0, 2, 0), (0, 3, 2)], "m0 chip 3 east exit + m1 landing VC1"),
          "PerfIntraVC0VC2": ([(0, 2, 0), (0, 0, 2)], "m0 chip 0 -> chip 1 link"),
          "NsInterVC0VC2": ([(0, 2, 0), (0, 3, 2)], "m0 chip 3 east exit + m1 landing VC1"),
          "NsIntraVC0VC2": ([(0, 2, 0), (0, 0, 2)], "m0 chip 0 -> chip 1 link")}


def strip(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def step_status(results, label):
    log = os.path.join(results, f"{label}.log")
    if not os.path.exists(log):
        return None, []
    text = strip(open(log, errors="replace").read())
    counts = re.findall(r"Tests: (\d+) executed, (\d+) failed", text)
    slots = re.findall(r"(VC\d sender slots \{[^}]*\} receiver slots \{[^}]*\})", text)
    rows = {}
    for s in slots:
        rows[s] = rows.get(s, 0) + 1
    fatal = sorted(set(re.findall(r"(TT_FATAL[^\n]{0,160}|TT_THROW[^\n]{0,160})", text)))[:3]
    st = "; ".join(f"{e} executed, {f} failed" for e, f in counts) or "no summary"
    return st + ("" if not fatal else "  [" + " | ".join(fatal) + "]"), sorted(rows.items())


def flow_dir(mesh, chip, vc, dst):
    """First hop of a 1-hop or straight flow (None where routing decides and the sender is alone on its chip)."""
    if dst[0] != mesh:
        return EXIT.get((mesh, chip)) if vc == 2 else None
    (r0, c0), (r1, c1) = COORD[chip], COORD[dst[1]]
    if r0 != r1 and c0 != c1:
        return None
    if r0 != r1:
        return "S" if r1 > r0 else "N"
    return "E" if c1 > c0 else "W"


def senders(yaml_path):
    """{test: [(mesh, chip, vc, dst, packets, first-hop or None)]}."""
    out = {}
    for t in yaml.safe_load(open(yaml_path))["Tests"]:
        dflt = t.get("defaults", {}).get("num_packets")
        lst = []
        for s in t["senders"]:
            mesh, chip = s["device"]
            for p in s["patterns"]:
                d = p["destination"]
                if "device" in d:
                    dst = tuple(d["device"])
                    lst.append((mesh, chip, s["vc_id"], dst, p.get("num_packets", dflt),
                                flow_dir(mesh, chip, s["vc_id"], dst)))
                else:  # multicast by hops: shown as mesh -1, chip = the hop counts
                    lst.append((mesh, chip, s["vc_id"], (-1, "+".join(f"{k}{v}" for k, v in d["hops"].items())),
                                p.get("num_packets", dflt), None))
        out[t["name"]] = lst
    return out


def label_row(flows, mesh, chip, direction):
    on_chip = [f for f in flows if f[0] == mesh and f[1] == chip]
    if len(on_chip) == 1:
        return on_chip[0]
    for f in on_chip:
        if f[5] == direction:
            return f
    rest = [f for f in on_chip if f[5] is None]
    return rest[0] if len(rest) == 1 else None


def parse_name(name):
    m = re.match(r"(.+?)_(?:ntype_\w+?_)?size_(\d+)", name) or re.match(r"(.+?)_size_(\d+)", name)
    base = re.sub(r"_ntype_\w+$", "", m.group(1)) if m else name
    return base, int(m.group(2)) if m else None


SENDER = re.compile(r"\[1,(\d)\]<stdout>:.*Sender core (\S+) \(VC(\d)\) on device (\d+) direction (\S+) link (\d+): "
                    r"(\d+) packets in (\d+) cycles")
SHARED_LINE = re.compile(r"\[1,(\d)\]<stdout>:.*Shared link: device (\d+) direction (\S+) link (\d+): another sender core "
                         r"(\S+) \(VC(\d)\) with (\d+) packets in (\d+) cycles")
RUNNING = re.compile(r"\[1,(\d)\]<stdout>:.*Running Test: (\S+)")


def sender_lines(log):
    """Per-sender records from the profiler's log lines, each under the test its rank was running."""
    cur, out = {}, []
    if not os.path.exists(log):
        return out
    for line in strip(open(log, errors="replace").read()).splitlines():
        m = RUNNING.search(line)
        if m:
            cur[int(m.group(1))] = m.group(2)
            continue
        m = SENDER.search(line)
        if m:
            rk, core, vc, dev, d, link, pk, cyc = m.groups()
            out.append(dict(test=cur.get(int(rk), "?"), mesh=int(rk), chip=int(dev), dir=d.split("::")[-1], vc=int(vc),
                            core=core, packets=int(pk), cycles=int(cyc), shared=False))
            continue
        m = SHARED_LINE.search(line)
        if m:
            rk, dev, d, link, core, vc, pk, cyc = m.groups()
            out.append(dict(test=cur.get(int(rk), "?"), mesh=int(rk), chip=int(dev), dir=d.split("::")[-1], vc=int(vc),
                            core=core, packets=int(pk), cycles=int(cyc), shared=True))
    return out


def csv_clock(path):
    if not os.path.exists(path):
        return None
    clocks = []
    for r in csv.DictReader(open(path)):
        n = int(r["num_packets"]) * int(r["total_traffic_count"])
        if n and int(r["cycles"]):
            clocks.append(float(r["packets_per_second"]) * int(r["cycles"]) / n)
    return sorted(clocks)[len(clocks) // 2] if clocks else None


def csv_rows(results, label, flows):
    """Job 167441's form: the per-sender CSV (one row per test and rank) labelled from the yaml."""
    path = os.path.join(results, f"{label}_tests.csv")
    if not os.path.exists(path):
        return
    print(f"  {'test':<18}{'size':>6}  {'flow':<34}{'dir':>4}{'cycles':>10}{'cyc/pkt':>9}{'GB/s':>8}"
          f"{'boundary GB/s':>15}{'clock MHz':>10}")
    for r in csv.DictReader(open(path)):
        test, size = parse_name(r["test_name"])
        n, tc, cyc = int(r["num_packets"]), int(r["total_traffic_count"]), int(r["cycles"])
        clock = float(r["packets_per_second"]) * cyc / (n * tc) if n and tc else None
        mesh = MESH_PKTS.get(n, 0)
        f = label_row(flows.get(test, []), mesh, int(r["device"]), r["direction"])
        own = f[4] if f else None
        gbs = own * size / cyc * clock / 1e9 if own and clock else None
        name = f"VC{f[2]} m{f[0]}c{f[1]}->m{f[3][0]}c{f[3][1]}" if f else f"? m{mesh}c{r['device']}"
        print(f"  {test:<18}{size:>6}  {name:<34}{r['direction']:>4}{cyc:>10}"
              f"{(cyc / own if own else float('nan')):>9.1f}{(gbs or float('nan')):>8.2f}"
              f"{float(r['bandwidth_GB_s']):>15.2f}{(clock or 0) / 1e6:>10.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--yaml", default=os.path.join(HERE, "glx_allvc_perf.yaml"))
    a = ap.parse_args()

    print("## Steps")
    labels = sorted({f[:-4] for f in os.listdir(a.results) if f.endswith(".log")
                     and not f.startswith(("glx_reset", "preflight")) and "hangcap" not in f})
    slot_rows = {}
    for l in labels:
        st, rows = step_status(a.results, l)
        print(f"  {l:<14} {st}")
        slot_rows[l] = rows
    print("\n## Slot rows (distinct rows per step; count = router configs that logged it, both ranks: the router config\n"
          "   once per rank, and the five per-direction MUX-mode configs, fabric_builder_context.cpp, unused here)")
    seen = {}
    for l, rows in slot_rows.items():
        key = tuple(r for r, _ in rows)
        seen.setdefault(key, []).append(l)
    for key, ls in seen.items():
        print(f"  {', '.join(ls)}:")
        for r in key:
            print(f"     {r}  x{dict(slot_rows[ls[0]])[r]}")

    logs = [l for l in labels if l.startswith("perf")]
    if not logs:
        return 0
    flows = {**senders(a.yaml), **senders(os.path.join(HERE, "glx_allvc_perf_ab.yaml")),
             **senders(os.path.join(HERE, "glx_allvc_perf_nosync.yaml"))}
    per = {l: sender_lines(os.path.join(a.results, f"{l}.log")) for l in logs}
    for l in logs:
        clock = csv_clock(os.path.join(a.results, f"{l}_tests.csv")) or 1.35e9
        recs = per[l]
        if not recs:
            print(f"\n## {l}: no per-sender lines (profiler before the main tree's change); CSV rows only")
            csv_rows(a.results, l, flows)
            continue
        print(f"\n## {l}: per-sender lines ({len(recs)}; clock {clock / 1e6:.0f} MHz)")
        print(f"  {'test':<16}{'size':>6}  {'sender':<10}{'dir':>4}  {'flows':<30}{'packets':>8}{'cycles':>11}"
              f"{'cyc/pkt':>9}{'GB/s':>8}")
        for r in recs:
            test, size = parse_name(r["test"])
            dsts = [(f"m{f[3][0]}c{f[3][1]}" if f[3][0] >= 0 else f"mcast {f[3][1]}") for f in flows.get(test, []) if (f[0], f[1], f[2]) == (r["mesh"], r["chip"], r["vc"])]
            print(f"  {test:<16}{size:>6}  VC{r['vc']} m{r['mesh']}c{r['chip']:<3}{r['dir']:>4}  {'->' + ','.join(dsts):<30}"
                  f"{r['packets']:>8}{r['cycles']:>11}{r['cycles'] / r['packets']:>9.1f}"
                  f"{r['packets'] * size / r['cycles'] * clock / 1e9:>8.2f}{'  (shares the link)' if r['shared'] else ''}")
        print("  shared paths (all members' packets over the slowest member's cycles):")
        for test_name in sorted({r["test"] for r in recs}):
            test, size = parse_name(test_name)
            for name, (members, desc) in SHARED.items():
                if name != test:
                    continue
                got = [r for r in recs if r["test"] == test_name and (r["mesh"], r["chip"], r["vc"]) in members]
                if len(got) == len(members):
                    pk, cyc = sum(r["packets"] for r in got), max(r["cycles"] for r in got)
                    print(f"    {test:<16}{size:>6}  {desc:<40} {pk} pkts / {cyc} cyc: {cyc / pk:.1f} cyc/pkt, "
                          f"{pk * size / cyc * clock / 1e9:.2f} GB/s; members finish within "
                          f"{(max(r['cycles'] for r in got) / min(r['cycles'] for r in got) - 1) * 100:.2f}%")
            if test in ("PerfAll", "NsAll"):
                by_vc = {}
                for r in recs:
                    if r["test"] == test_name:
                        by_vc.setdefault(r["vc"], []).append(r["cycles"] / r["packets"])
                print(f"    {test:<16}{size:>6}  per VC, cycles/packet of its senders: " + "; ".join(
                    f"VC{v} n={len(c)} min {min(c):.0f} median {sorted(c)[len(c) // 2]:.0f} max {max(c):.0f}"
                    for v, c in sorted(by_vc.items())))
    ab = [l for l in logs if l.startswith("perfab")]
    if ab:
        print("\n## Window A/B (cycles/packet per flow; rows: default, VC1 8/8, VC2 8/8/8)")
        print(f"  {'test':<14}{'size':>6}" + "".join(f"{l:>16}" for l in ab))
        keys = sorted({(parse_name(r["test"])) for l in ab for r in per[l]})
        for test, size in keys:
            cells = []
            for l in ab:
                got = [r for r in per[l] if parse_name(r["test"]) == (test, size)]
                cells.append(f"{got[0]['cycles'] / got[0]['packets']:>16.1f}" if got else f"{'-':>16}")
            print(f"  {test:<14}{size:>6}" + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
