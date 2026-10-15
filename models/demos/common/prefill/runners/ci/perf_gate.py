# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import argparse
import json
import math
import os
import sys

METRICS = {
    "throughput_tok_s": ("TPS", "tok/s", ",.0f", True),
    "ttft_s": ("TTFT", "s", ",.2f", False),
    "chunk_time_ms": ("CHUNK TIME", "ms", ",.0f", False),
}


def power_class():
    from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import get_tdp_limit_max, is_high_power

    if is_high_power():
        return "14kw"
    try:
        tdp = get_tdp_limit_max()
    except Exception as e:
        tdp = f"unreadable ({type(e).__name__})"
    print(f"perf baseline: TDP_LIMIT_MAX={tdp} is below 130 W; using 8kw", file=sys.stderr)
    return "8kw"


def lookup(table, model, config, cls):
    entry = table["baselines"].get(model, {}).get(config)
    if entry is None:
        return None
    cls = cls or power_class()
    if cls not in entry:
        print(f"perf baseline: none for {model} {config} on {cls}", file=sys.stderr)
        return None
    print(f"perf baseline: {model} {config} on {cls} {entry['probe']}: {entry[cls]}", file=sys.stderr)
    return entry["probe"], {metric: (value, table["margin"][metric]) for metric, value in entry[cls].items()}


def check(rec, path, label, metric, expected, margin, fast_fails):
    name, unit, fmt, higher_is_better = METRICS[metric]
    value = (rec.get(metric) or {}).get(label)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        print(f"{name} GATE FAIL: {path} has no finite {metric} at {label}: {value!r}", file=sys.stderr)
        return False
    lo, hi = expected * (1 - margin), expected * (1 + margin)
    print(
        f"{name} GATE {label}: {value:{fmt}} {unit} vs {expected:{fmt}} +/- {margin * 100:.0f}% "
        f"[{lo:{fmt}}, {hi:{fmt}}]"
    )
    slow = value < lo if higher_is_better else value > hi
    fast = value > hi if higher_is_better else value < lo
    if fast_fails:
        if slow or fast:
            print(f"{name} GATE FAIL: {value:{fmt}} outside [{lo:{fmt}}, {hi:{fmt}}]", file=sys.stderr)
            return False
    elif slow:
        side, bound = ("below", lo) if higher_is_better else ("above", hi)
        print(f"{name} GATE FAIL: {value:{fmt}} {side} {bound:{fmt}}", file=sys.stderr)
        return False
    elif fast:
        side, bound = ("above", hi) if higher_is_better else ("below", lo)
        print(f"{name} GATE WARN: {value:{fmt}} {side} {bound:{fmt}}; update the baseline if this holds")
    print(f"{name} GATE PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description="Gate a prefill run's perf JSON against its baseline.")
    parser.add_argument("perf_json")
    parser.add_argument("--table", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--power", default="", choices=["", "8kw", "14kw"], help="skip the tt-smi TDP probe")
    parser.add_argument("--expected-tps", default="", help="explicit tok/s target; overrides the table")
    parser.add_argument(
        "--margin", default="", help="margin for --expected-tps (default 0.15); the table carries its own"
    )
    parser.add_argument("--require", default="0", help="1 fails the run if nothing was measured")
    args = parser.parse_args()

    probe, baseline, fast_fails = None, None, False
    if args.expected_tps:
        baseline, fast_fails = {"throughput_tok_s": (float(args.expected_tps), float(args.margin or 0.15))}, True
    elif args.table and os.path.isfile(args.table):
        with open(args.table) as f:
            found = lookup(json.load(f), args.model, args.config, args.power)
        if found is not None:
            probe, baseline = found
    if baseline is None and args.require != "1":
        return 0

    try:
        with open(args.perf_json) as f:
            rec = json.load(f)
    except Exception as e:
        print(f"TPS GATE FAIL: no perf metrics at {args.perf_json}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    probes, index = rec.get("throughput_tok_s") or {}, rec.get("chunk_index") or {}
    label = max(probes, key=lambda k: index.get(k, -1), default=None)
    if label is None:
        print(f"TPS GATE FAIL: {args.perf_json} has no throughput probes; the run measured nothing", file=sys.stderr)
        return 1

    if baseline is None:
        print(f"TPS: {probes[label]:,.0f} tok/s {label} (no PREFILL_EXPECTED_TPS set -- reporting, not gating)")
        return 0

    if probe and label != probe:
        print(
            f"TPS GATE FAIL: baseline is for {probe} but the deepest probe is {label}; "
            f"the context length changed, so re-measure and update perf_baselines.json",
            file=sys.stderr,
        )
        return 1

    results = [
        check(rec, args.perf_json, label, metric, expected, margin, fast_fails)
        for metric, (expected, margin) in baseline.items()
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
