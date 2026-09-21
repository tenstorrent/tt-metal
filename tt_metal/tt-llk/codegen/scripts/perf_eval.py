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
  - goal=measure    (benchmark infrastructure): exact planned current measurements,
    without a baseline or any speedup/no-regression claim. Requires a sealed v2
    cycle_measurement leaf via --required-manifest/--requirement-id, plus raw CSV
    via --raw-current. Every planned TILE_LOOP variant must have positive finite
    cycles and post cycles must equal raw/(loop_factor*tile_cnt). No op filtering.

Legacy v1 manifests/comparisons retain their contract. New measurement leaves
require a v2-aware solver and dispatcher/executor with raw CSV publication; do
not resume them with old consumers or silently convert existing comparisons.

It is schema-agnostic: every perf module has a different set of parameter
columns, so the variant key is "all columns that are not a metric column"
(`mean(...)`, `std(...)`, `TEXT_SIZE(...)`). The default headline metric is
`mean(L1_TO_L1)` (total L1->L1 cycles). Isolate-only tests require an explicit
--primary-metric selection, agreed before inspecting results; the evaluator
never substitutes a more favorable metric. It uses the `TILE_LOOP` marker
(per-tile, the most comparable number), falling back to `KERNEL`.

Configuration columns must agree and variant keys must be unique. A broader
baseline is allowed, but every selected current variant needs a baseline before
the whole comparison can pass. New unmatched variants are not regressions: they
remain measured but make a favorable comparison incomplete (exit 2).

The perf tests' run-to-run noise is ~0.5% (per the perf team), so deltas within
+/-0.5% are treated as noise (neutral) by default — see --regress-pct /
--improve-pct.

Stdlib-only (no pandas) so the unit tests stay dependency-free.

Exit codes (consumed by the perf-tester agent):
  0  goal met        (no_regress: not slower; improve: faster; measure: measured)
  1  perf miss        (no_regress: regressed; improve: regressed or not improved)
  2  missing/invalid evidence (comparison baseline, coverage or measurements)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from decimal import Decimal, InvalidOperation
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
SUPPORTED_PRIMARY_METRICS = (PRIMARY_METRIC, *CONTEXT_METRICS)


def _read_csv(
    path: Path, *, strict: bool = False, data: bytes | None = None
) -> list[dict[str, str]]:
    if data is None:
        if not path or not path.exists():
            return []
        data = path.read_bytes()
    text = data.decode("utf-8").strip()
    if not text:
        return []
    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    if strict and (
        not reader.fieldnames
        or len(set(reader.fieldnames)) != len(reader.fieldnames)
        or any(
            None in row or any(value is None for value in row.values()) for row in rows
        )
    ):
        raise ValueError("measurement CSV has duplicate headers or malformed rows")
    return rows


def _key_columns(fieldnames: list[str]) -> list[str]:
    return [c for c in fieldnames if not c.startswith(METRIC_PREFIXES)]


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
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


def _measurement_key_columns(fieldnames: list[str]) -> list[str]:
    # Perf export_metrics prefixes percentage metrics with the run type; they
    # are measurements, not configuration keys, and are not tile-normalized.
    return [
        key
        for key in _key_columns(fieldnames)
        if not re.match(r"^[A-Z0-9_]+_(?:mean|std)\(", key)
    ]


def _canonical_measurement_variant(variant: dict[str, str]) -> dict[str, str]:
    """Only loop/tile counts have numeric identity; other CSV keys stay exact."""
    normalized = dict(variant)
    for field in ("loop_factor", "tile_cnt"):
        try:
            value = Decimal(variant[field])
            if (
                not value.is_finite()
                or value <= 0
                or value != value.to_integral_value()
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"measurement {field} must be a positive finite integer"
                )
            normalized[field] = str(int(value))
        except (InvalidOperation, TypeError, KeyError) as exc:
            raise ValueError(
                f"measurement {field} must be a positive finite integer"
            ) from exc
    return normalized


def canonical_measurement_contract(contract: Any) -> dict[str, Any]:
    validate_measurement_contract(contract)
    return {
        **contract,
        "variants": [_canonical_measurement_variant(v) for v in contract["variants"]],
    }


def validate_measurement_contract(contract: Any) -> None:
    """Validate a predeclared current-only benchmark, never infer its coverage."""
    if (
        not isinstance(contract, dict)
        or set(contract) != {"primary_metric", "marker", "normalization", "variants"}
        or contract["primary_metric"] not in SUPPORTED_PRIMARY_METRICS
        or contract["marker"] != "TILE_LOOP"
        or contract["normalization"] != "loop_factor*tile_cnt"
    ):
        raise ValueError("invalid measurement_contract schema/metric/normalization")
    variants = contract["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("measurement_contract requires exact nonempty variants")
    keys = None
    seen = set()
    for variant in variants:
        if (
            not isinstance(variant, dict)
            or not {"marker", "loop_factor", "tile_cnt"}.issubset(variant)
            or any(
                not isinstance(k, str) or not isinstance(v, str) or not v
                for k, v in variant.items()
            )
            or set(_measurement_key_columns(list(variant))) != set(variant)
            or variant["marker"] != "TILE_LOOP"
        ):
            raise ValueError(
                "measurement_contract variants must contain exact CSV keys"
            )
        normalized = _canonical_measurement_variant(variant)
        if keys is not None and set(variant) != keys:
            raise ValueError("measurement_contract variant schemas differ")
        keys = set(variant)
        identity = tuple(sorted(normalized.items()))
        if identity in seen:
            raise ValueError("measurement_contract contains duplicate variants")
        seen.add(identity)


def evaluate_measurement(current_rows, raw_rows, contract) -> dict[str, Any]:
    """Validate exact coverage and raw-to-per-tile normalization, without a baseline."""
    contract = canonical_measurement_contract(contract)
    result = {
        "goal": "measure",
        "measured": False,
        "verdict": "not_measured",
        "primary_metric": contract["primary_metric"],
        "marker": "TILE_LOOP",
        "units": "cycles_per_tile",
        "exit_code": 2,
    }
    keys = sorted(contract["variants"][0])
    expected = {tuple(variant[k] for k in keys) for variant in contract["variants"]}
    metric = contract["primary_metric"]
    indexed = []
    for label, rows in (("current", current_rows), ("raw", raw_rows)):
        selected = [row for row in rows if row.get("marker") == "TILE_LOOP"]
        index = {}
        for row in selected:
            if (
                set(_measurement_key_columns(list(row))) != set(keys)
                or metric not in row
            ):
                return {**result, "reason": f"{label}_measurement_schema_mismatch"}
            try:
                normalized = _canonical_measurement_variant(row)
            except ValueError:
                return {**result, "reason": f"{label}_invalid_measurement_divisor"}
            key = tuple(normalized[k] for k in keys)
            value = _to_float(row[metric])
            if key in index:
                return {**result, "reason": f"{label}_duplicate_measurement_variant"}
            if value is None or value <= 0:
                return {**result, "reason": f"{label}_invalid_measurement_cycles"}
            index[key] = value
        if set(index) != expected:
            return {**result, "reason": f"{label}_measurement_coverage_mismatch"}
        indexed.append(index)
    current, raw = indexed
    variants = []
    for variant in contract["variants"]:
        key = tuple(variant[k] for k in keys)
        divisor = float(variant["loop_factor"]) * float(variant["tile_cnt"])
        normalized = raw[key] / divisor
        if (
            not math.isfinite(divisor)
            or not math.isfinite(normalized)
            or normalized <= 0
            or not math.isclose(current[key], normalized, rel_tol=1e-9, abs_tol=0.0)
        ):
            return {**result, "reason": "measurement_normalization_mismatch"}
        variants.append(
            {"key": variant, "current_cycles": current[key], "raw_cycles": raw[key]}
        )
    return {
        **result,
        "measured": True,
        "verdict": "measured",
        "exit_code": 0,
        "variants_measured": len(variants),
        "variants": variants,
        "measurements": {"cycle_measurement": {"measured": True}},
    }


def evaluate_bound_measurement(
    current_path: Path, raw_path: Path, contract, receipt
) -> dict[str, Any]:
    """Parse exactly the CSV bytes attested by the selected hardware receipt."""
    if (
        receipt.get("version") != 4
        or receipt.get("suite") != "perf"
        or receipt.get("backend") != "silicon"
        or receipt.get("classification") != "success"
    ):
        raise ValueError("measurement requires a successful v4 silicon perf receipt")
    rows, digests = [], {}
    for name, path in (("current", current_path), ("raw_current", raw_path)):
        data = path.read_bytes()
        expected = receipt["measurement_artifacts"][name]
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != expected["size"] or digest != expected["sha256"]:
            raise ValueError(f"{name} CSV differs from selected hardware receipt")
        rows.append(_read_csv(path, strict=True, data=data))
        digests[f"{name}_sha256"] = digest
    return {**evaluate_measurement(rows[0], rows[1], contract), **digests}


def _record_measurement_result(path: Path, result: dict[str, Any]) -> None:
    """Preserve independent leaves in the existing per-run perf result artifact."""
    document = {"schema": "tt.issue-solver.perf-results", "version": 1, "results": {}}
    if path.exists():
        previous = json.loads(path.read_text())
        if (
            isinstance(previous, dict)
            and previous.get("schema") == document["schema"]
            and previous.get("version") == 1
            and set(previous) == set(document)
            and isinstance(previous.get("results"), dict)
        ):
            document = previous
        elif isinstance(previous, dict) and isinstance(
            previous.get("requirement_id"), str
        ):
            document["results"][previous["requirement_id"]] = previous
        else:
            raise ValueError(
                "existing perf results have no valid requirement-keyed identity"
            )
    document["results"][result["requirement_id"]] = result
    # Existing atomic writer; role dispatch is serial, so no independent writer
    # is permitted to update the envelope concurrently.
    from run_json_writer import _atomic_write

    _atomic_write(path.parent, document, destination=path)


def evaluate(
    current_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    *,
    op: str | None,
    goal: str,
    noise_pct: float,
    regress_pct: float,
    improve_pct: float,
    primary_metric: str = PRIMARY_METRIC,
) -> dict[str, Any]:
    """Return a `perf` result dict. Pure function for easy unit testing."""

    if primary_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ValueError(f"unsupported primary metric: {primary_metric}")
    for name, value in (
        ("noise_pct", noise_pct),
        ("regress_pct", regress_pct),
        ("improve_pct", improve_pct),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    if not current_rows:
        return {
            "measured": False,
            "goal": goal,
            "op": op,
            "primary_metric": primary_metric,
            "verdict": "not_measured",
            "reason_code": "current_rows_missing",
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

    primary_label = f"{primary_metric} @ {marker or 'all-markers'}"
    coverage = {
        "current_rows": len(current_rows),
        "baseline_rows": len(baseline_rows),
        # Schema errors make cross-file identity unknown, not an empty match.
        "current_variants": None,
        "baseline_variants": None,
        "matched_variants": None,
        "current_only_variants": None,
        "baseline_only_variants": None,
        "comparison_complete": False,
    }

    def not_comparable(
        verdict: str, reason: str, *, reason_code: str | None = None
    ) -> dict[str, Any]:
        return {
            "measured": False,
            "goal": goal,
            "op": op,
            "primary_metric": primary_label,
            "verdict": verdict,
            "reason_code": reason_code or verdict,
            "reason": reason,
            "exit_code": 2,
            "coverage": coverage,
        }

    if not current_rows:
        return not_comparable(
            "not_measured",
            "current perf rows do not match the requested operation/marker selection",
            reason_code="current_selection_empty",
        )

    for source, rows in (("current", current_rows), ("baseline", baseline_rows)):
        if rows and any(primary_metric not in row for row in rows):
            return not_comparable(
                "missing_metric", f"{source} rows lack selected metric {primary_metric}"
            )

    expected_columns = set(key_cols)
    for source, rows in (("current", current_rows), ("baseline", baseline_rows)):
        if any(set(_key_columns(list(row))) != expected_columns for row in rows):
            return not_comparable(
                "no_baseline",
                f"{source} variant configuration columns differ from current schema",
                reason_code="variant_schema_mismatch",
            )

    current_keys = [tuple(row.get(c, "") for c in key_cols) for row in current_rows]
    baseline_keys = [tuple(row.get(c, "") for c in key_cols) for row in baseline_rows]
    current_set, baseline_set = set(current_keys), set(baseline_keys)
    coverage.update(
        current_variants=len(current_set),
        baseline_variants=len(baseline_set),
        matched_variants=len(current_set & baseline_set),
        current_only_variants=len(current_set - baseline_set),
        baseline_only_variants=len(baseline_set - current_set),
        duplicate_current_rows=len(current_keys) - len(current_set),
        duplicate_baseline_rows=len(baseline_keys) - len(baseline_set),
    )
    if coverage["duplicate_current_rows"] or coverage["duplicate_baseline_rows"]:
        return not_comparable(
            "no_baseline",
            "duplicate variant keys make current/baseline matching ambiguous",
            reason_code="duplicate_variant_keys",
        )
    # A cached baseline may contain a wider sweep; those extra rows do not
    # weaken coverage of the selected current sweep and need not be rerun.
    base_index = _index_by_key(baseline_rows, key_cols)

    per_variant: list[dict[str, Any]] = []
    deltas: list[float] = []
    for cur in current_rows:
        cur_val = _to_float(cur.get(primary_metric))
        if cur_val is None or cur_val <= 0:
            return not_comparable(
                "invalid_measurement",
                f"current {primary_metric} must contain finite positive cycle counts",
            )
        key = tuple(cur.get(c, "") for c in key_cols)
        base = base_index.get(key)
        base_val = _to_float(base.get(primary_metric)) if base else None
        if base is not None and (base_val is None or base_val <= 0):
            return not_comparable(
                "invalid_measurement",
                f"matching baseline {primary_metric} must contain finite positive cycle counts",
            )
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
            if not math.isfinite(delta):
                return not_comparable(
                    "invalid_measurement",
                    f"{primary_metric} percentage delta is not finite",
                )
            entry["delta_pct"] = round(delta, 3)
            deltas.append(delta)
        per_variant.append(entry)

    coverage["comparison_complete"] = bool(current_set) and current_set <= baseline_set

    if not deltas:
        return {
            "measured": True,
            "goal": goal,
            "op": op,
            "test": None,
            "primary_metric": primary_label,
            "verdict": "no_baseline",
            "reason_code": (
                "no_matching_baseline_variants"
                if baseline_rows
                else "baseline_rows_missing"
            ),
            "reason": "no matching baseline variants to compare against",
            "variants_measured": len(per_variant),
            "exit_code": 2,
            "coverage": coverage,
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

    matched_verdict = verdict
    incomplete_reason = None
    if coverage["current_only_variants"]:
        incomplete_reason = (
            f"{coverage['current_only_variants']} current variants have no matching baseline; "
            "matched-subset deltas do not certify the complete current sweep"
        )
        # A proven regression remains a failure even if other variants lack a
        # baseline. A favorable or inconclusive subset cannot certify the whole.
        if verdict != "regressed":
            verdict, exit_code = "no_baseline", 2

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
        if cur_m is None or base_m is None or base_m <= 0:
            continue
        context_delta = (cur_m - base_m) / base_m * 100.0
        if not math.isfinite(context_delta):
            continue
        breakdown[metric] = {
            "baseline": base_m,
            "current": cur_m,
            "delta_pct": round(context_delta, 3),
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
    result = {
        "measured": True,
        "goal": goal,
        "op": op,
        "primary_metric": primary_label,
        "noise_pct": noise_pct,
        "regress_pct": regress_pct,
        "improve_pct": improve_pct,
        "variants_compared": len(deltas),
        "coverage": coverage,
        "delta_pct_median": round(median, 3),
        "delta_pct_worst": round(worst, 3),
        "delta_pct_best": round(best, 3),
        "verdict": verdict,
        "worst_variant": worst_variant,
        "exit_code": exit_code,
    }
    if incomplete_reason:
        result.update(
            reason=incomplete_reason,
            reason_code="incomplete_baseline_coverage",
            matched_verdict=matched_verdict,
        )
    return result


def _format_summary(result: dict[str, Any]) -> str:
    lines = [
        f"perf verdict: {result.get('verdict')}  (goal={result.get('goal')})",
    ]
    if result.get("op"):
        lines.append(f"  op: {result['op']}")
    if result.get("primary_metric"):
        lines.append(f"  metric: {result['primary_metric']}")
    coverage = result.get("coverage") or {}
    if coverage:
        lines.append(
            "  coverage: matched=%s current-only=%s baseline-only=%s complete=%s"
            % (
                coverage.get("matched_variants"),
                coverage.get("current_only_variants"),
                coverage.get("baseline_only_variants"),
                coverage.get("comparison_complete"),
            )
        )
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
    p.add_argument(
        "--goal", choices=["improve", "no_regress", "measure"], default="no_regress"
    )
    p.add_argument(
        "--raw-current", help="Raw CSV before loop/tile normalization (measure only)"
    )
    p.add_argument(
        "--required-manifest", help="Sealed verification manifest (measure only)"
    )
    p.add_argument("--requirement-id", help="Exact measurement leaf in the manifest")
    p.add_argument(
        "--verification-result", help="Exact copied hardware receipt (measure only)"
    )
    p.add_argument(
        "--results-out",
        help="Retain measurement by requirement in this per-run perf result JSON",
    )
    p.add_argument(
        "--primary-metric",
        choices=SUPPORTED_PRIMARY_METRICS,
        default=PRIMARY_METRIC,
        help="Cycle metric selected by the test contract; no automatic fallback",
    )
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
    for name in ("noise_pct", "regress_pct", "improve_pct"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            p.error(f"--{name.replace('_', '-')} must be finite and non-negative")

    if (
        args.results_out
        and args.json_out
        and Path(args.results_out).resolve() == Path(args.json_out).resolve()
    ):
        p.error("--results-out must differ from the per-leaf --json-out")
    try:
        current_rows = [] if args.goal == "measure" else _read_csv(Path(args.current))
    except ValueError as exc:
        p.error(str(exc))
    baseline_rows = _read_csv(Path(args.baseline)) if args.baseline else []

    if args.goal == "measure":
        if (
            not all(
                (
                    args.raw_current,
                    args.required_manifest,
                    args.requirement_id,
                    args.verification_result,
                )
            )
            or args.baseline
            or args.op
        ):
            p.error(
                "measure requires --raw-current, --required-manifest, --requirement-id, --verification-result and forbids baseline/op filtering"
            )
        # Reuse the sealer's strict schema/identity validation, including v2 fields.
        from run_json_writer import _load_required_manifest, _load_verification_result

        manifest = _load_required_manifest(Path(args.required_manifest))
        leaves = [
            r
            for r in manifest["requirements"]
            if r["requirement_id"] == args.requirement_id
        ]
        if (
            len(leaves) != 1
            or "cycle_measurement" not in leaves[0]["required_measurements"]
        ):
            p.error("requirement is not a sealed current-only measurement")
        leaf = leaves[0]
        if args.test and args.test != leaf["selector"]["test"]:
            p.error("test does not match sealed measurement selector")
        receipt = _load_verification_result(Path(args.verification_result))
        if (
            any(receipt[k] != manifest[k] for k in ("run_id", "attempt_id"))
            or receipt["requirement_id"] != leaf["requirement_id"]
            or receipt["architecture"] != leaf["architecture"]
            or receipt["selector"] != leaf["selector"]
            or receipt["provenance"]["expected_base_sha"]
            != manifest["expected_base_sha"]
            or receipt["provenance"]["actual_base_sha"] != manifest["expected_base_sha"]
        ):
            p.error("hardware receipt does not match sealed measurement identity")
        try:
            result = evaluate_bound_measurement(
                Path(args.current),
                Path(args.raw_current),
                leaf["measurement_contract"],
                receipt,
            )
        except (ValueError, OSError, KeyError) as exc:
            p.error(str(exc))
        result.update(
            outcome="PERF_OK" if result["exit_code"] == 0 else "PERF_ENV_ERROR",
            job_id=receipt["job_id"],
            verification_result_id=receipt["result_id"],
            patch_sha256=receipt["provenance"]["patch_sha256"],
        )
        result.update(
            {
                "test": leaf["selector"]["test"],
                "arch": leaf["architecture"],
                "run_id": manifest["run_id"],
                "attempt_id": manifest["attempt_id"],
                "requirement_id": leaf["requirement_id"],
                "base_commit": manifest["expected_base_sha"],
                "current_source": str(Path(args.current).resolve()),
                "raw_current_source": str(Path(args.raw_current).resolve()),
            }
        )
    else:
        if (
            args.raw_current
            or args.required_manifest
            or args.requirement_id
            or args.verification_result
            or args.results_out
        ):
            p.error("measurement contract arguments require --goal measure")
        result = evaluate(
            current_rows,
            baseline_rows,
            op=args.op,
            goal=args.goal,
            noise_pct=args.noise_pct,
            regress_pct=args.regress_pct,
            improve_pct=args.improve_pct,
            primary_metric=args.primary_metric,
        )
    if args.test:
        result["test"] = args.test
    if args.baseline:
        result.setdefault("baseline_source", args.baseline)

    exit_code = int(result.pop("exit_code", 0))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
    if args.results_out:
        _record_measurement_result(Path(args.results_out), result)
    print(_format_summary(result))
    # Also emit the JSON to stderr so a caller can capture it without a temp file.
    print(json.dumps(result), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
