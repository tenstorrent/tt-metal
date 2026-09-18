#!/usr/bin/env python3
"""Per-phase wall time of tt-train bench runs from the naive profiler's markers (each marker syncs the device, so a
phase's wall time is its device time). Usage: S_summarize.py <log or glob> ...   (steps 1-2 are warm-up; ms/step)."""
import glob, os, re, sys

MARKERS = [
    "dataloader_step_done",
    "forward_pass_done",
    "backward_pass_done",
    "gradient_sync_done",
    "optimizer_step_done",
]
PHASES = ["data", "forward", "backward", "gradsync", "optimizer"]


def steps_of(path):
    events = re.findall(r"(\w+) timestamp_us=(\d+)", open(path, errors="replace").read())
    steps, cur, prev_t = [], {}, None
    for name, t in events:
        t = int(t)
        if name not in MARKERS:
            continue
        idx = MARKERS.index(name)
        if idx == 0:
            cur = {"data": (t - prev_t) / 1e3 if prev_t else 0.0}
        elif prev_t is not None and cur:
            cur[PHASES[idx]] = (t - prev_t) / 1e3
            if name == "optimizer_step_done":
                cur["total"] = sum(cur.get(p, 0.0) for p in PHASES)
                steps.append(cur)
                cur = {}
        prev_t = t
    return steps


print(
    f"{'run':62s} {'n':>2s} {'total':>8s} {'forward':>8s} {'backward':>9s} {'gradsync':>9s} {'optim':>7s} {'data':>6s}"
)
for pattern in sys.argv[1:]:
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)[:-4]
        steps = steps_of(path)[2:]
        if not steps:
            print(f"{name:62s} (incomplete)")
            continue
        m = {k: sum(s.get(k, 0.0) for s in steps) / len(steps) for k in PHASES + ["total"]}
        print(
            f"{name:62s} {len(steps):2d} {m['total']:8.1f} {m['forward']:8.1f} {m['backward']:9.1f} {m['gradsync']:9.1f} {m['optimizer']:7.1f} {m['data']:6.1f}"
        )
