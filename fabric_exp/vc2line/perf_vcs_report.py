#!/usr/bin/env python3
"""Idle- and full-VC perf tables (VC2_LINE_BRINGUP.md, "Perf: idle and full VCs") from vc2l_glx.sbatch / vc2l_glx_mm.sbatch
job results.

    perf_vcs_report.py RESULTS_DIR [RESULTS_DIR ...]

Reads the profiler's per-sender log lines ("Sender core ... (VCn) on device d direction D link l: P packets in C
cycles"; benchmark mode without sync, so C is the sender's own send loop) of the steps idle_*, split_idle_* and full*,
each under the test its process was running (MPI rank r = mesh r on the 2x2 split; single-mesh logs have no rank tag).
Cycles per packet = C / P in the device clock (1,350 MHz).
  Idle: per test and size, VC0's cycles per packet with VC2 off, neighbour (idle) and line (idle), per ERISC mode, and
        the deltas against off; on the split VC1 is compiled in (idle) in all three.
  Full: per test and size, each flow's VC, cycles per packet, packets per us and payload GB/s; the test's aggregate
        (every flow's packets over the slowest flow's cycles) against the best one-VC test of the same family and size
        and against the wire (248 cycles per 4,352 B packet, scaled by payload + 96 B header: an estimate that ignores
        per-packet wire overhead, so it flatters the wire at small sizes); then the window rows side by side.
"""
import collections
import os
import re
import sys

CLOCK = 1.35e9
HEADER = 96
WIRE_BYTES_PER_CYCLE = (4352 + HEADER) / 248.0

SENDER = re.compile(r"(?:\[1,(\d)\]<stdout>:)?.*Sender core (\S+) \(VC(\d)\) on device (\d+) direction (\S+) link (\d+): "
                    r"(\d+) packets in (\d+) cycles")
SHARED = re.compile(r"(?:\[1,(\d)\]<stdout>:)?.*Shared link: device (\d+) direction (\S+) link (\d+): another sender core "
                    r"(\S+) \(VC(\d)\) with (\d+) packets in (\d+) cycles")
RUNNING = re.compile(r"(?:\[1,(\d)\]<stdout>:)?.*Running Test: (\S+)")


def parse_name(name):
    m = re.match(r"(.+?)_(?:ntype_\w+?_)?size_(\d+)", name)
    return (re.sub(r"_ntype_\w+$", "", m.group(1)), int(m.group(2))) if m else (name, None)


def senders(log):
    cur, out = {}, []
    for line in re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="replace").read()).splitlines():
        m = RUNNING.search(line)
        if m:
            cur[int(m.group(1) or 0)] = m.group(2)
            continue
        m = SENDER.search(line)
        if m:
            rk, core, vc, dev, d, link, pk, cyc = m.groups()
        else:
            m = SHARED.search(line)
            if not m:
                continue
            rk, dev, d, link, core, vc, pk, cyc = m.groups()
        rk = int(rk or 0)
        test, size = parse_name(cur.get(rk, "?"))
        out.append(dict(test=test, size=size, mesh=rk, chip=int(dev), dir=d.split("::")[-1], vc=int(vc),
                        packets=int(pk), cycles=int(cyc)))
    return out


def cpp(r):
    return r["cycles"] / r["packets"]


def step_logs(dirs):
    logs = {}
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.endswith(".log") and (f.startswith("idle_") or f.startswith("split_idle_") or f.startswith("full")):
                logs[f[:-4]] = os.path.join(d, f)
    return logs


def idle_tables(logs):
    for prefix in ("idle_", "split_idle_"):
        steps = [s for s in logs if s.startswith(prefix)]
        if not steps:
            continue
        data = {s: {(r["test"], r["size"]): cpp(r) for r in senders(logs[s])} for s in steps}
        for one in ("", "_1e"):
            cols = [f"{prefix}{c}{one}" for c in ("off", "nbr", "line")]
            reps = [s for s in steps if re.fullmatch(rf"{prefix}(off|nbr|line){one}_r\d", s)]
            cols = [c for c in cols if c in data] + sorted(reps)
            if not cols:
                continue
            base = cols[0]
            print(f"\n## Idle VCs: {prefix[:-1]} {'one ERISC' if one else 'two ERISCs'} (VC0 cycles per packet; "
                  f"deltas against {base})")
            print(f"  {'test':<14}{'size':>6}" + "".join(f"{c:>20}" for c in cols))
            keys = sorted({k for c in cols for k in data[c]})
            for k in keys:
                b = data[base].get(k)
                cells = []
                for c in cols:
                    v = data[c].get(k)
                    if v is None:
                        cells.append(f"{'-':>20}")
                    elif c == base or b is None:
                        cells.append(f"{v:>20.1f}")
                    else:
                        cells.append(f"{v:>11.1f} ({v - b:+6.1f})")
                print(f"  {k[0]:<14}{k[1]:>6}" + "".join(cells))


FAMILY = {"L": ("LVC0", "LVC2", "LVC1"), "I": ("IVC0", "IVC2"), "R": ("LVC0", "LVC2", "LVC1")}


def full_tables(logs):
    steps = [s for s in logs if s.startswith("full")]
    per = {s: senders(logs[s]) for s in steps}
    for s in steps:
        recs = per[s]
        if not recs:
            continue
        print(f"\n## Full VCs: {s} (per flow: VC, sender, cycles per packet, packets per us, GB/s; aggregate = all "
              f"flows' packets over the slowest flow's cycles)")
        by_test = collections.OrderedDict()
        for r in recs:
            by_test.setdefault((r["test"], r["size"]), []).append(r)
        best = {}
        for (t, z), rs in by_test.items():
            if len(rs) == 1:
                best[(t[0], z)] = min(best.get((t[0], z), 1e18), cpp(rs[0]))
        for (t, z), rs in by_test.items():
            pk = sum(r["packets"] for r in rs)
            cyc = max(r["cycles"] for r in rs)
            agg_cpp = cyc / pk
            flows = "; ".join(f"VC{r['vc']} m{r['mesh']}c{r['chip']}{r['dir']} {cpp(r):.1f}" for r in rs)
            wire = (z + HEADER) / WIRE_BYTES_PER_CYCLE
            b = best.get((t[0], z))
            print(f"  {t:<9}{z:>6}  aggregate {agg_cpp:7.1f} cyc/pkt = {CLOCK / agg_cpp / 1e6:6.2f} pkt/us = "
                  f"{z * CLOCK / agg_cpp / 1e9:6.2f} GB/s | best one-VC {b if b is None else round(b, 1)} | wire est "
                  f"{wire:.1f} | {flows}")
    if len(steps) > 1:
        print("\n## Window rows (cycles per packet per flow)")
        keys = []
        for s in steps:
            for r in per[s]:
                k = (r["test"], r["size"], r["vc"], r["mesh"], r["chip"], r["dir"])
                if k not in keys:
                    keys.append(k)
        print(f"  {'test':<9}{'size':>6} {'flow':<14}" + "".join(f"{s:>13}" for s in steps))
        for k in keys:
            cells = []
            for s in steps:
                got = [r for r in per[s] if (r["test"], r["size"], r["vc"], r["mesh"], r["chip"], r["dir"]) == k]
                cells.append(f"{cpp(got[0]):>13.1f}" if got else f"{'-':>13}")
            print(f"  {k[0]:<9}{k[1]:>6} VC{k[2]} m{k[3]}c{k[4]}{k[5]:<4}" + "".join(cells))


def main():
    logs = step_logs(sys.argv[1:])
    idle_tables(logs)
    full_tables(logs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
