#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare a perf run against a committed baseline and judge it by intent.

The issue solver runs a scoped subset of the existing perf tests
(`tests/python_tests/perf_*.py`), which emit per-variant cycle-count CSVs to
`perf_data/<module>/<module>.post.csv`. This helper diffs the freshly produced
CSV ("current") against the baseline CSV captured from `origin/main`, and
returns an **intent-aware** verdict:

  - goal=no_regress (bug fixes / features): a fix must NOT get slower.
  - goal=improve    (optimization issues):  a fix SHOULD get faster.

It is schema-agnostic: every perf module has a different set of parameter
columns, so the variant key is "all columns that are not a metric column"
(`mean(...)`, `std(...)`, `TEXT_SIZE(...)`). The headline metric is
`mean(L1_TO_L1)` (total L1->L1 cycles), measured on the `TILE_LOOP` marker
(per-tile, the most comparable number) and falling back to `KERNEL`.

The perf tests' run-to-run noise is ~0.5% (per the perf team), so deltas within
+/-0.5% are treated as noise (neutral) by default — see --regress-pct /
--improve-pct.

Stdlib-only (no pandas) so the unit tests stay dependency-free.

Exit codes (consumed by the perf-tester agent):
  0  goal met        (no_regress: not slower; improve: faster)
  1  perf miss        (no_regress: regressed; improve: regressed or not improved)
  2  not comparable   (no baseline, or no matching variants)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PRIMARY_METRIC = "mean(L1_TO_L1)"
MARKER_PREFERENCE = ("TILE_LOOP", "KERNEL")
METRIC_PREFIXES = ("mean(", "std(", "TEXT_SIZE(")
# Extra context metrics surfaced in the report when present.
CONTEXT_METRICS = (
    "mean(UNPACK_ISOLATE)",
    "mean(MATH_ISOLATE)",
    "mean(PACK_ISOLATE)",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    return list(csv.DictReader(text.splitlines()))


def _key_columns(fieldnames: list[str]) -> list[str]:
    return [c for c in fieldnames if not c.startswith(METRIC_PREFIXES)]


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _select_marker(rows: list[dict[str, str]]) -> str | None:
    markers = {r.get("marker") for r in rows if r.get("marker")}
    for preferred in MARKER_PREFERENCE:
        if preferred in markers:
            return preferred
    # No known marker column / values — compare across whatever exists.
    return None


def _filter_op(rows: list[dict[str, str]], op: str | None) -> list[dict[str, str]]:
    if not op:
        return rows
    op_l = op.lower()
    # Only filter when the rows actually carry a mathop column; otherwise the
    # perf module is single-op (matmul, tilize, ...) and every row is relevant.
    if not rows or "mathop" not in rows[0]:
        return rows
    return [r for r in rows if op_l in (r.get("mathop") or "").lower()]


def _index_by_key(
    rows: list[dict[str, str]], key_cols: list[str]
) -> dict[tuple[str, ...], dict[str, str]]:
    index: dict[tuple[str, ...], dict[str, str]] = {}
    for r in rows:
        key = tuple(r.get(c, "") for c in key_cols)
        index[key] = r
    return index


def evaluate(
    current_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    *,
    op: str | None,
    goal: str,
    noise_pct: float,
    regress_pct: float,
    improve_pct: float,
) -> dict[str, Any]:
    """Return a `perf` result dict. Pure function for easy unit testing."""

    if not current_rows:
        return {
            "measured": False,
            "goal": goal,
            "op": op,
            "verdict": "not_measured",
            "reason": "no current perf rows",
            "exit_code": 2,
        }

    key_cols = _key_columns(list(current_rows[0].keys()))
    current_rows = _filter_op(current_rows, op)
    baseline_rows = _filter_op(baseline_rows, op)

    marker = _select_marker(current_rows)
    if marker:
        current_rows = [r for r in current_rows if r.get("marker") == marker]
        baseline_rows = [r for r in baseline_rows if r.get("marker") == marker]

    primary_label = f"{PRIMARY_METRIC} @ {marker or 'all-markers'}"

    base_index = _index_by_key(baseline_rows, key_cols)

    per_variant: list[dict[str, Any]] = []
    deltas: list[float] = []
    for cur in current_rows:
        cur_val = _to_float(cur.get(PRIMARY_METRIC))
        if cur_val is None or cur_val == 0:
            continue
        key = tuple(cur.get(c, "") for c in key_cols)
        base = base_index.get(key)
        base_val = _to_float(base.get(PRIMARY_METRIC)) if base else None
        entry: dict[str, Any] = {
            "key": {c: cur.get(c, "") for c in key_cols},
            "current_cycles": cur_val,
            "baseline_cycles": base_val,
            # Raw rows kept internally so we can build a per-thread breakdown for
            # the worst variant only; stripped before output.
            "_cur": cur,
            "_base": base,
        }
        if base_val is not None and base_val != 0:
            delta = (cur_val - base_val) / base_val * 100.0
            entry["delta_pct"] = round(delta, 3)
            deltas.append(delta)
        per_variant.append(entry)

    if not deltas:
        return {
            "measured": True,
            "goal": goal,
            "op": op,
            "test": None,
            "primary_metric": primary_label,
            "verdict": "no_baseline",
            "reason": "no matching baseline variants to compare against",
            "variants_measured": len(per_variant),
            "exit_code": 2,
        }

    worst = max(deltas)  # most positive == worst regression
    best = min(deltas)  # most negative == best improvement
    median = statistics.median(deltas)

    # Base verdict from thresholds, independent of goal.
    if worst > regress_pct:
        base_verdict = "regressed"
    elif best < -improve_pct:
        base_verdict = "improved"
    else:
        base_verdict = "neutral"

    # Map to goal-aware verdict + exit code.
    if base_verdict == "regressed":
        verdict, exit_code = "regressed", 1  # a regression is a miss for both goals
    elif goal == "improve":
        if base_verdict == "improved":
            verdict, exit_code = "improved", 0
        else:
            verdict, exit_code = "not_improved", 1
    else:  # goal == no_regress
        verdict, exit_code = base_verdict, 0  # improved or neutral both pass

    worst_variant = max(
        (e for e in per_variant if "delta_pct" in e),
        key=lambda e: e["delta_pct"],
    )

    # Localize the regression: for the worst variant, break the cycle change down
    # per Tensix thread (UNPACK / MATH / PACK isolates) so the fixer immediately
    # knows which thread the change slowed down, instead of only the L1->L1 total.
    breakdown: dict[str, Any] = {}
    cur_row, base_row = worst_variant.get("_cur"), worst_variant.get("_base")
    for metric in CONTEXT_METRICS:
        cur_m = _to_float(cur_row.get(metric)) if cur_row else None
        base_m = _to_float(base_row.get(metric)) if base_row else None
        if cur_m is None or base_m is None or base_m == 0:
            continue
        breakdown[metric] = {
            "baseline": base_m,
            "current": cur_m,
            "delta_pct": round((cur_m - base_m) / base_m * 100.0, 3),
        }
    # Drop the internal raw-row refs from every entry; attach the breakdown.
    for e in per_variant:
        e.pop("_cur", None)
        e.pop("_base", None)
    if breakdown:
        worst_variant["thread_breakdown"] = breakdown

    # Keep the result compact: the headline aggregates plus the single worst
    # variant (with its per-thread breakdown) are enough for run.json /
    # runs.jsonl. Full per-variant detail lives in the archived
    # perf_current_*/perf_baseline_* CSVs, not here.
    return {
        "measured": True,
        "goal": goal,
        "op": op,
        "primary_metric": primary_label,
        "noise_pct": noise_pct,
        "regress_pct": regress_pct,
        "improve_pct": improve_pct,
        "variants_compared": len(deltas),
        "delta_pct_median": round(median, 3),
        "delta_pct_worst": round(worst, 3),
        "delta_pct_best": round(best, 3),
        "verdict": verdict,
        "worst_variant": worst_variant,
        "exit_code": exit_code,
    }


def _format_summary(result: dict[str, Any]) -> str:
    lines = [
        f"perf verdict: {result.get('verdict')}  (goal={result.get('goal')})",
    ]
    if result.get("op"):
        lines.append(f"  op: {result['op']}")
    if result.get("primary_metric"):
        lines.append(f"  metric: {result['primary_metric']}")
    if "delta_pct_median" in result:
        lines.append(
            "  delta%%: median=%.2f  worst=%.2f  best=%.2f  (variants=%d)"
            % (
                result["delta_pct_median"],
                result["delta_pct_worst"],
                result["delta_pct_best"],
                result["variants_compared"],
            )
        )
        wv = result.get("worst_variant") or {}
        if wv:
            lines.append(
                "  worst variant: base=%s -> cur=%s (%.2f%%)"
                % (
                    wv.get("baseline_cycles"),
                    wv.get("current_cycles"),
                    wv.get("delta_pct", 0.0),
                )
            )
            for metric, b in (wv.get("thread_breakdown") or {}).items():
                lines.append(
                    "    %s: %s -> %s (%.2f%%)"
                    % (metric, b["baseline"], b["current"], b["delta_pct"])
                )
    if result.get("reason"):
        lines.append(f"  reason: {result['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--current", required=True, help="Path to the current .post.csv")
    p.add_argument(
        "--baseline", default=None, help="Path to the baseline .post.csv (optional)"
    )
    p.add_argument(
        "--op",
        default=None,
        help="Filter to a single op (substring match on the mathop column)",
    )
    p.add_argument(
        "--test", default=None, help="Perf test module name (for the report)"
    )
    p.add_argument("--goal", choices=["improve", "no_regress"], default="no_regress")
    # The perf team measured the perf tests' run-to-run noise at ~0.5%, so a
    # delta within +/-0.5% is treated as noise (neutral), not a regression or a
    # real improvement.
    p.add_argument("--noise-pct", type=float, default=0.5)
    p.add_argument(
        "--regress-pct",
        type=float,
        default=0.5,
        help="Delta%% above which a variant counts as a regression (noise floor)",
    )
    p.add_argument(
        "--improve-pct",
        type=float,
        default=0.5,
        help="Delta%% below which (faster) a variant counts as an improvement",
    )
    p.add_argument(
        "--json-out", default=None, help="Write the perf result JSON to this path"
    )
    args = p.parse_args(argv)

    current_rows = _read_csv(Path(args.current))
    baseline_rows = _read_csv(Path(args.baseline)) if args.baseline else []

    result = evaluate(
        current_rows,
        baseline_rows,
        op=args.op,
        goal=args.goal,
        noise_pct=args.noise_pct,
        regress_pct=args.regress_pct,
        improve_pct=args.improve_pct,
    )
    if args.test:
        result["test"] = args.test
    if args.baseline:
        result.setdefault("baseline_source", args.baseline)

    exit_code = int(result.pop("exit_code", 0))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
    print(_format_summary(result))
    # Also emit the JSON to stderr so a caller can capture it without a temp file.
    print(json.dumps(result), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
