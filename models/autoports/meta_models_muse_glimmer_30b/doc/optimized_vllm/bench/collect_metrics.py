# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fold the arms' raw ``vllm bench serve`` JSON into one comparison artifact.

Every number the optimized-vLLM README quotes comes from here rather than from a
hand-copied console line, so the report and the artifacts cannot drift.  The two
workload profiles are kept apart on purpose -- primary single-user 128/128/1 is the
headline, CI serving-burst 100/100/32 is capacity/nightly parity -- and they are
never compared against each other.

Each arm ran the benchmark stage N times back-to-back as the first traffic after a
server start.  That repetition is not noise-averaging for its own sake: the first
requests after a start are measurably slower than the fourth, so a single sample
per arm would compare two points on a warm-up curve.  Both ``per_run`` and the
``warm`` aggregate (the last ``--warm-from`` runs) are reported.

Usage::

    python doc/optimized_vllm/bench/collect_metrics.py \
        --arm before=doc/optimized_vllm/before --arm after=doc/optimized_vllm/after \
        --out doc/optimized_vllm/metrics.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

#: (raw file, profile key, workload description) for the two profiles the readiness
#: runner writes on every ``benchmark`` stage.
PROFILES = (
    (
        "vllm_result.json",
        "primary_single_user",
        {"prompt_len": 128, "gen_len": 128, "requests": 1, "max_concurrency": 1, "temperature": 0.0},
    ),
    (
        "vllm_ci_serving_result.json",
        "ci_serving_burst",
        {"prompt_len": 100, "gen_len": 100, "requests": 32, "max_concurrency": None, "temperature": 0.0},
    ),
)

#: Raw vLLM fields carried through verbatim.  ``decode_tps_u`` is derived, and it is
#: derived one way only: ``1000 / mean_tpot_ms``.
FIELDS = (
    "completed",
    "total_output_tokens",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "std_itl_ms",
    "output_throughput",
    "request_throughput",
    "median_e2el_ms",
)


def _load(path: pathlib.Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except OSError:
        return None


def _row(raw: dict) -> dict:
    row = {field: raw.get(field) for field in FIELDS}
    row["decode_tps_u"] = 1000.0 / raw["mean_tpot_ms"] if raw.get("mean_tpot_ms") else None
    return row


def _aggregate(rows: list[dict]) -> dict:
    out: dict = {"runs": len(rows)}
    for field in (*FIELDS, "decode_tps_u"):
        values = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
        if not values:
            continue
        out[field] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", default=[], help="name=directory")
    parser.add_argument(
        "--warm-from",
        type=int,
        default=4,
        help="first run index counted as warm; runs below it are kept but reported separately",
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    report: dict = {"warm_from_run": args.warm_from, "arms": {}}
    for spec in args.arm:
        name, _, directory = spec.partition("=")
        base = pathlib.Path(directory)
        arm: dict = {"dir": str(base), "profiles": {}}
        for filename, profile, workload in PROFILES:
            per_run = {}
            for run_dir in sorted(base.glob("run*")):
                raw = _load(run_dir / filename)
                if raw:
                    per_run[int(run_dir.name[3:])] = _row(raw)
            if not per_run:
                continue
            warm = [row for index, row in sorted(per_run.items()) if index >= args.warm_from]
            arm["profiles"][profile] = {
                "workload": workload,
                "per_run": {str(k): v for k, v in sorted(per_run.items())},
                "all": _aggregate([row for _, row in sorted(per_run.items())]),
                "warm": _aggregate(warm) if warm else None,
            }
        report["arms"][name] = arm

    # Deltas, warm-aggregate median against warm-aggregate median, per profile.  Only
    # emitted when both arms have a warm aggregate, so a partial sweep cannot produce
    # a comparison that looks complete.
    names = list(report["arms"])
    if "before" in names and "after" in names:
        deltas = {}
        for profile in ("primary_single_user", "ci_serving_burst"):
            before = report["arms"]["before"]["profiles"].get(profile, {}).get("warm")
            after = report["arms"]["after"]["profiles"].get(profile, {}).get("warm")
            if not before or not after:
                continue
            row = {}
            for field in (
                "median_ttft_ms",
                "p99_ttft_ms",
                "mean_tpot_ms",
                "median_itl_ms",
                "output_throughput",
                "decode_tps_u",
                "median_e2el_ms",
            ):
                if field in before and field in after:
                    b, a = before[field]["median"], after[field]["median"]
                    row[field] = {
                        "before": b,
                        "after": a,
                        "delta": a - b,
                        "pct": (a - b) / b * 100.0 if b else None,
                        "speedup_before_over_after": b / a if a else None,
                    }
            deltas[profile] = row
        report["deltas_warm_median"] = deltas

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report.get("deltas_warm_median", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
