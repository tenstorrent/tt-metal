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


def fail(msg):
    """Print a gate failure; under GitHub Actions also annotate it so log summaries see it."""
    print(msg, file=sys.stderr)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        escaped = msg.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error title=Prefill perf gate::{escaped}")


def power_class():
    """This host's power class: 14kw, 8kw, or unknown when tt-smi cannot report the TDP (no tt-smi here,
    or it timed out). An unreadable TDP is not evidence of a low-power host."""
    try:
        from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import get_tdp_limit_max

        tdp = get_tdp_limit_max()
    except Exception as e:
        tdp = f"unreadable ({type(e).__name__})"
    cls = "unknown" if not isinstance(tdp, int) else "14kw" if tdp >= 130 else "8kw"
    print(f"perf baseline: TDP_LIMIT_MAX={tdp} -> {cls} (14kw needs >= 130 W)", file=sys.stderr)
    return cls


class GateError(Exception):
    """The leg is baselined but cannot be gated on this host."""


def lookup(table, model, config, cls):
    """[(probe label, {metric: (expected, margin)})] for this model/config/power class, or None when the
    leg has no baseline. Raises GateError when it has one but not for this host's power class.

    An entry either names one "probe" (the deepest) with its metrics flat under the power class, or keys
    the power class by probe label to gate every probe. A per-entry "margin" {label: {metric: margin}}
    overrides the table-wide per-metric margin at that probe."""
    entry = table["baselines"].get(model, {}).get(config)
    if entry is None:
        return None
    cls = cls or power_class()
    classes = ", ".join(sorted(k for k in entry if k in ("8kw", "14kw")))
    if cls == "unknown":
        raise GateError(
            f"{model} {config} is baselined on {classes} but this host's power class could not be determined "
            "(TDP unreadable: tt-smi missing or timed out)"
        )
    if cls not in entry:
        # Baselined, just not for this power class: the leg ran on the wrong pool (an sc1 perf leg on an
        # 8kw host), so fail rather than pass ungated.
        raise GateError(
            f"{model} {config} is baselined on {classes} only but this host is {cls}; "
            "the runner is misrouted or mislabeled"
        )
    by_probe = {entry["probe"]: entry[cls]} if "probe" in entry else entry[cls]
    margins = entry.get("margin", {})
    print(f"perf baseline: {model} {config} on {cls}: {by_probe}", file=sys.stderr)
    return [
        (label, {m: (v, {**table["margin"], **margins.get(label, {})}[m]) for m, v in metrics.items()})
        for label, metrics in by_probe.items()
    ]


def check(rec, path, label, metric, expected, margin):
    """Gate one metric at one probe. Returns (ok, job-summary table row)."""
    name, unit, fmt, higher_is_better = METRICS[metric]
    value = (rec.get(metric) or {}).get(label)
    lo, hi = expected * (1 - margin), expected * (1 + margin)
    band = f"{expected:{fmt}} +/- {margin * 100:g}% [{lo:{fmt}}, {hi:{fmt}}]"
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{name} GATE FAIL: {path} has no finite {metric} at {label}: {value!r}")
        return False, [label, f"{name} ({unit})", "missing", band, "FAIL"]
    print(f"{name} GATE {label}: {value:{fmt}} {unit} vs {band}")
    if value < lo or value > hi:
        faster = (value > hi) == higher_is_better
        hint = "; faster than the baseline, update the configured baseline in this change" if faster else ""
        fail(f"{name} GATE FAIL: {value:{fmt}} outside [{lo:{fmt}}, {hi:{fmt}}]{hint}")
        return False, [label, f"{name} ({unit})", f"{value:{fmt}}", band, "FAIL (faster)" if faster else "FAIL"]
    print(f"{name} GATE PASS")
    return True, [label, f"{name} ({unit})", f"{value:{fmt}}", band, "PASS"]


def publish_verdicts(perf_json, rows):
    """Write the gate table to PREFILL_SUMMARIES/perf/<run>_gate.md, which CI adds to the job summary."""
    root = os.environ.get("PREFILL_SUMMARIES")
    if not root or not rows:
        return
    run = os.path.splitext(os.path.basename(perf_json))[0]
    table = ["| probe | metric | measured | expected | verdict |", "|---|---|---|---|---|"]
    table += ["| " + " | ".join(row) + " |" for row in rows]
    try:
        out = os.path.join(root, "perf")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"{run}_gate.md"), "w") as fh:
            fh.write(f"### perf gate -- {run}\n\n" + "\n".join(table) + "\n")
    except OSError as exc:
        print(f"perf gate summary not persisted ({exc})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Gate a prefill run's perf JSON against its baseline.")
    parser.add_argument("perf_json", nargs="?")
    parser.add_argument("--print-power", action="store_true", help="print this host's power class and exit")
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
    if args.print_power:
        print(power_class())
        return 0

    baselines = None
    if args.expected_tps:
        baselines = [(None, {"throughput_tok_s": (float(args.expected_tps), float(args.margin or 0.15))})]
    elif args.table and os.path.isfile(args.table):
        with open(args.table) as f:
            table = json.load(f)
        try:
            baselines = lookup(table, args.model, args.config, args.power)
        except GateError as e:
            fail(f"PERF GATE FAIL: {e}")
            publish_verdicts(args.perf_json, [["-", "-", "-", "-", f"FAIL: {e}"]])
            return 1
    if baselines is None and args.require != "1":
        return 0

    try:
        with open(args.perf_json) as f:
            rec = json.load(f)
    except Exception as e:
        fail(f"TPS GATE FAIL: no perf metrics at {args.perf_json}: {type(e).__name__}: {e}")
        return 1

    probes, index = rec.get("throughput_tok_s") or {}, rec.get("chunk_index") or {}
    label = max(probes, key=lambda k: index.get(k, -1), default=None)
    if label is None:
        fail(f"TPS GATE FAIL: {args.perf_json} has no throughput probes; the run measured nothing")
        return 1

    if baselines is None:
        print(f"TPS: {probes[label]:,.0f} tok/s {label} (no PREFILL_EXPECTED_TPS set -- reporting, not gating)")
        return 0

    gated = [probe for probe, _ in baselines if probe is not None]
    if gated and label not in gated:
        fail(
            f"TPS GATE FAIL: baseline covers {', '.join(gated)} but the deepest probe is {label}; "
            f"the context length changed, so re-measure and update perf_baselines.json"
        )
        return 1

    results = [
        check(rec, args.perf_json, probe or label, metric, expected, margin)
        for probe, baseline in baselines
        for metric, (expected, margin) in baseline.items()
    ]
    publish_verdicts(args.perf_json, [row for _, row in results])
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
