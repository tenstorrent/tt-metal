#!/usr/bin/env python3
"""E7 readout: pipeline throughput per producer run, per-stage chunk compute and the stage hop.

  e7_analyze.py results/e7

Throughput = tokens / (producer push wall + final layer-ack drain): the drain ends when the last chunk's
last layer has acked, so the sum is the time to complete every pushed chunk. Hop (sync session only,
PREFILL_SYNC_PER_CHUNK=1): rank1 CHUNK_START(c) - (rank0 CHUNK_START(c) + rank0 compute_ms(c)), i.e. from
stage A finishing chunk c to stage B starting it (includes the D2D send / receive).
"""
import glob, os, re, statistics, sys

DONE = re.compile(r"DONE wall=([\d.]+)s pushes=(\d+) requests=(\d+) tokens=(\d+)")
DRAIN = re.compile(r"drained (\d+)/(\d+) layer acks in ([\d.]+)s")
START = re.compile(r"\[pp rank (\d)\] CHUNK_START c=(\d+) compute_start=([\d.]+)")
COMPUTE = re.compile(r"\[pp rank (\d)\] CHUNK_COMPUTE c=(\d+) compute_ms=([\d.]+)")


def producers(d):
    rows = []
    for log in sorted(glob.glob(os.path.join(d, "*.log"))):
        name = os.path.basename(log)[:-4]
        if name.startswith(("runner", "reset")):
            continue
        txt = open(log, errors="replace").read()
        m = DONE.search(txt)
        drains = DRAIN.findall(txt)
        if not m or not drains:
            rows.append((name, None))
            continue
        wall, pushes, reqs, toks = float(m[1]), int(m[2]), int(m[3]), int(m[4])
        drain_s = float(drains[-1][2])  # the post-schedule drain is the last one logged
        rows.append((name, (pushes, toks, wall, drain_s, toks / (wall + drain_s))))
    return rows


def hops(runner_log):
    start, comp = {}, {}
    for line in open(runner_log, errors="replace"):
        if m := START.search(line):
            start[(int(m[1]), int(m[2]))] = float(m[3])
        elif m := COMPUTE.search(line):
            comp[(int(m[1]), int(m[2]))] = float(m[3])
    out = []
    for (r, c), t0 in sorted(start.items()):
        if r == 0 and (0, c) in comp and (1, c) in start:
            out.append((c, comp[(0, c)], comp.get((1, c)), (start[(1, c)] - (t0 + comp[(0, c)] / 1000)) * 1000))
    return out


def main(d):
    print(f"{'run':12} {'chunks':>6} {'tokens':>7} {'push s':>7} {'drain s':>7} {'tok/s':>8}")
    for name, v in producers(d):
        if v is None:
            print(f"{name:12} (no DONE / drain line)")
        else:
            print(f"{name:12} {v[0]:6d} {v[1]:7d} {v[2]:7.2f} {v[3]:7.2f} {v[4]:8.0f}")
    rl = os.path.join(d, "runner_sync1.log")
    if os.path.exists(rl):
        h = hops(rl)
        print(f"\nsync session: {len(h)} chunks with both stages timed")
        for lo, hi, label in ((0, 10**9, "all"),):
            sel = [x for x in h if lo <= x[0] < hi]
            if sel:
                a = [x[1] for x in sel]
                b = [x[2] for x in sel if x[2] is not None]
                hp = [x[3] for x in sel]
                print(f"  stage A compute ms  median {statistics.median(a):.1f}  (min {min(a):.1f} max {max(a):.1f})")
                if b:
                    print(
                        f"  stage B compute ms  median {statistics.median(b):.1f}  (min {min(b):.1f} max {max(b):.1f})"
                    )
                print(
                    f"  hop ms (A done -> B start) median {statistics.median(hp):.2f}  p10 {sorted(hp)[len(hp) // 10]:.2f}"
                    f"  p90 {sorted(hp)[9 * len(hp) // 10]:.2f}"
                )
        for c, a, b, hp in h:
            print(f"    c={c:4d} A {a:7.1f} B {b if b is None else round(b, 1)} hop {hp:7.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
