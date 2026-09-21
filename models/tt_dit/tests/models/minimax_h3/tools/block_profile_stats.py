# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Statistics of the per-op transformer-block breakdown across Tracy ops CSVs (the profile
`test_minimax_h3_transformer_block_perf` writes under `generated/profiler/reports/<ts>/`), using the same
warm-iteration isolation and device merge as `project_block_perf.py` (mean over devices for collectives, max
otherwise). Not a test; pytest leaves it alone.

    python block_profile_stats.py compare doc=<csv> today=<csv> [...]   # per-op side by side, deltas vs the first
    python block_profile_stats.py runs r0=<csv> r1=<csv> [...]           # per-op mean / std / min / max / CoV across runs
    python block_profile_stats.py devices <csv> [<csv> ...]              # per-op spread across the 32 devices within a run

Ops under 0.5 ms in every input are summed as "remaining". `device only` sums the merged per-op device durations
(no dispatch gaps, the underestimate); `device + op gap` adds the OP TO OP LATENCY column (the overestimate).
"""

from __future__ import annotations

import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_block_perf import _per_op, _warm_rows  # noqa: E402

MAJOR_MS = 0.5


def _short(code: str) -> str:
    return code.replace("DeviceOperation", "")


def _load(arg: str) -> tuple[str, dict[str, float], float, int]:
    """'label=path' -> (label, {op: merged ms}, total op-to-op gap ms, calls-per-op dict folded into ms map's keys)."""
    label, path = arg.split("=", 1) if "=" in arg else (os.path.basename(os.path.dirname(arg)), arg)
    rows, ok = _warm_rows(path)
    if not ok:
        print(f"warning: start/stop signposts not found in {path}; using every row")
    per = _per_op(rows)
    ms = {k: v["device_ns"] / 1e6 for k, v in per.items()}
    calls = {k: int(v["calls"]) for k, v in per.items()}
    return label, ms, sum(v["gap_ns"] for v in per.values()) / 1e6, calls


def compare(args: list[str]) -> None:
    loaded = [_load(a) for a in args]
    labels = [l for l, _, _, _ in loaded]
    tables = [t for _, t, _, _ in loaded]
    gaps = [g for _, _, g, _ in loaded]
    calls = loaded[0][3]
    ops = sorted(set().union(*tables), key=lambda o: -max(t.get(o, 0.0) for t in tables))
    print(
        "| op | calls | "
        + " | ".join(f"{l} ms" for l in labels)
        + " | "
        + " | ".join(f"Δ {l}" for l in labels[1:])
        + " |"
    )
    print("|" + "---|" * (1 + len(labels) + len(labels)))
    small = [0.0] * len(tables)
    n_small = 0
    for o in ops:
        vals = [t.get(o, 0.0) for t in tables]
        if max(vals) < MAJOR_MS:
            n_small += 1
            small = [s + v for s, v in zip(small, vals)]
            continue
        deltas = [f"{v - vals[0]:+.2f}" if vals[0] else "—" for v in vals[1:]]
        cells = " | ".join(f"{v:.2f}" if v else "—" for v in vals)
        print(
            f"| {_short(o)} | {calls.get(o, max(l[3].get(o, 0) for l in loaded))} | {cells} | "
            + " | ".join(deltas)
            + " |"
        )
    tot = [sum(t.values()) for t in tables]
    print(
        f"| *(remaining {n_small} ops)* | | "
        + " | ".join(f"{v:.2f}" for v in small)
        + " | "
        + " | ".join(f"{v - small[0]:+.2f}" for v in small[1:])
        + " |"
    )
    print(
        "| **device only** | | "
        + " | ".join(f"**{v:.2f}**" for v in tot)
        + " | "
        + " | ".join(f"**{v - tot[0]:+.2f} ({(v - tot[0]) / tot[0] * 100:+.1f}%)**" for v in tot[1:])
        + " |"
    )
    print(
        "| device + op gap | | "
        + " | ".join(f"{t + g:.2f}" for t, g in zip(tot, gaps))
        + " | "
        + " | ".join(f"{(t + g) - (tot[0] + gaps[0]):+.2f}" for t, g in zip(tot[1:], gaps[1:]))
        + " |"
    )


def runs(args: list[str]) -> None:
    loaded = [_load(a) for a in args]
    labels = [l for l, _, _, _ in loaded]
    tables = [t for _, t, _, _ in loaded]
    gaps = [g for _, _, g, _ in loaded]
    n = len(tables)
    ops = sorted(set().union(*tables), key=lambda o: -max(t.get(o, 0.0) for t in tables))
    print(f"{n} runs: {', '.join(labels)}")
    print("| op | mean ms | std ms | min | max | max-min | CoV |")
    print("|---|---|---|---|---|---|---|")

    def line(name: str, v: list[float]) -> None:
        sd = st.stdev(v) if n > 1 else 0.0
        m = st.mean(v)
        print(
            f"| {name} | {m:.2f} | {sd:.3f} | {min(v):.2f} | {max(v):.2f} | {max(v) - min(v):.2f} | {sd / m * 100:.2f}% |"
        )

    small = [0.0] * n
    for o in ops:
        v = [t.get(o, 0.0) for t in tables]
        if max(v) < MAJOR_MS:
            small = [s + x for s, x in zip(small, v)]
            continue
        line(_short(o), v)
    tot = [sum(t.values()) for t in tables]
    line("remaining small ops", small)
    line("**device only**", tot)
    line("device + op gap", [t + g for t, g in zip(tot, gaps)])
    print("\nper-run device-only totals: " + ", ".join(f"{l} {t:.2f}" for l, t in zip(labels, tot)))


def devices(paths: list[str]) -> None:
    for path in paths:
        rows, _ = _warm_rows(path)
        by_dev: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r.get("OP TYPE") != "signpost":
                by_dev[r["DEVICE ID"]].append(r)
        devs = list(by_dev.values())
        n_ops = min(len(d) for d in devs)
        per_op: dict[str, list[list[float]]] = defaultdict(list)  # op -> per call -> per device ms
        for i in range(n_ops):
            code = devs[0][i]["OP CODE"]
            vals = [int(d[i]["DEVICE KERNEL DURATION [ns]"]) / 1e6 for d in devs if d[i]["DEVICE KERNEL DURATION [ns]"]]
            per_op[code].append(vals)
        print(f"\n{path}: {len(devs)} devices, {n_ops} ops per device")
        print("| op | calls | mean over devices ms | slowest device | fastest device | max-min | max/min |")
        print("|---|---|---|---|---|---|---|")
        for code, calls in sorted(per_op.items(), key=lambda kv: -sum(max(c) for c in kv[1])):
            mean_sum = sum(st.mean(c) for c in calls)
            mx, mn = sum(max(c) for c in calls), sum(min(c) for c in calls)
            if mx < MAJOR_MS:
                continue
            print(
                f"| {_short(code)} | {len(calls)} | {mean_sum:.2f} | {mx:.2f} | {mn:.2f} | {mx - mn:.2f} | {mx / mn:.3f} |"
            )


def main() -> None:
    modes = {"compare": compare, "runs": runs, "devices": devices}
    if len(sys.argv) < 3 or sys.argv[1] not in modes:
        print(__doc__)
        raise SystemExit(2)
    modes[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
