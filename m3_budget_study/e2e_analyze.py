#!/usr/bin/env python3
"""Part B readout: per session, tok/s per producer stream; for sync sessions, per-stage compute, the bottleneck
stage, and the hop at every stage boundary (only chunks where the next stage was idle, so no queueing).

  e2e_analyze.py results_sp2/e2e > results_sp2/e2e/summary.txt

tok/s = tokens / (push wall + final layer-ack drain). Acks fire at host issue, so the end of the run is
known to within about one chunk.
"""
import glob, os, re, statistics, sys

DONE = re.compile(r"DONE wall=([\d.]+)s pushes=(\d+) requests=(\d+) tokens=(\d+)")
DRAIN = re.compile(r"drained (\d+)/(\d+) layer acks in ([\d.]+)s")
START = re.compile(r"\[pp rank (\d+)\] CHUNK_START c=(\d+) compute_start=([\d.]+)")
COMPUTE = re.compile(r"\[pp rank (\d+)\] CHUNK_COMPUTE c=(\d+) compute_ms=([\d.]+)")
PUSH = re.compile(r"\[producer\] push slot=\d+ cidx=\d+ start=(\d+)")


def streams(d):
    out = []
    for log in sorted(glob.glob(os.path.join(d, "*.log"))):
        name = os.path.basename(log)[:-4]
        if name in ("runner", "reset") or "warm" in name:
            continue
        txt = open(log, errors="replace").read()
        m, dr = DONE.search(txt), DRAIN.findall(txt)
        if not m or not dr:
            out.append((name, None))
            continue
        wall, toks, drain = float(m[1]), int(m[4]), float(dr[-1][2])
        out.append((name, (int(m[2]), toks, wall + drain, toks / (wall + drain))))
    return out


def sync_stages(d):
    """Per-stage compute and boundary hops, split by stream (history of each chunk from the producer logs)."""
    st, cp = {}, {}
    for line in open(os.path.join(d, "runner.log"), errors="replace"):
        if m := START.search(line):
            st[(int(m[1]), int(m[2]))] = float(m[3])
        elif m := COMPUTE.search(line):
            cp[(int(m[1]), int(m[2]))] = float(m[3]) / 1000
    ranks = sorted({r for r, _ in st})
    # chunk -> stream label, in push order across the producer logs (the runner counts chunks globally)
    order = []
    for log in sorted(glob.glob(os.path.join(d, "*.log")), key=os.path.getmtime):
        name = os.path.basename(log)[:-4]
        if name in ("runner", "reset"):
            continue
        txt = open(log, errors="replace").read()
        order += [name] * len(PUSH.findall(txt))
    res = {}
    for label in sorted(set(order)):
        cs = [c for c, lab in enumerate(order) if lab == label]
        comp = {r: [cp[(r, c)] * 1000 for c in cs if (r, c) in cp] for r in ranks}
        hops = {}
        for r in ranks[:-1]:
            h = []
            for c in cs:
                if (r, c) in st and (r, c) in cp and (r + 1, c) in st:
                    end_r = st[(r, c)] + cp[(r, c)]
                    prev_next = st.get((r + 1, c - 1), 0) + cp.get((r + 1, c - 1), 0)
                    if prev_next <= end_r:
                        h.append((st[(r + 1, c)] - end_r) * 1000)
            hops[r] = h
        res[label] = (comp, hops)
    return ranks, res


def main(root):
    for d in sorted(glob.glob(os.path.join(root, "r*_w*"))):
        print(f"## {os.path.basename(d)}")
        for name, v in streams(d):
            print(
                f"  {name:12} "
                + (
                    "(no DONE / drain line)"
                    if v is None
                    else f"chunks {v[0]:3d} tokens {v[1]:7d} time {v[2]:6.2f} s  tok/s {v[3]:7.0f}"
                )
            )
        if "sync1" in d and os.path.exists(os.path.join(d, "runner.log")):
            ranks, res = sync_stages(d)
            for label, (comp, hops) in res.items():
                if "warm" in label:
                    continue
                med = {r: statistics.median(v) for r, v in comp.items() if v}
                if not med:
                    continue
                bott = max(med, key=med.get)
                print(
                    f"  [{label}] per-stage compute ms (median): "
                    + "  ".join(f"s{r} {med[r]:.1f}" for r in ranks if r in med)
                    + f"   bottleneck = stage {bott}"
                )
                for r, h in hops.items():
                    if h:
                        print(
                            f"  [{label}] hop s{r}->s{r + 1}: median {statistics.median(h):.2f} ms"
                            f" (min {min(h):.2f}, n={len(h)})"
                        )
        print()


if __name__ == "__main__":
    main(sys.argv[1])
