# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare a run's Parquet batch to historical batches (regression detection).

For every comparable data point — one (test, sweep config, marker, run type) with
a ``mean(<run_type>)`` value — the current run is compared to a baseline built
from the same point across the historical batches (median, robust to noise). A
point that is more than ``threshold`` slower than its baseline is flagged as a
regression. Points with no matching history (a new config) are reported
separately, not as regressions.

This is the "compare to history" step of the real use case; full regression
semantics (noise modelling, approved baselines) is a later milestone. Needs
pyarrow + pandas; no device libraries.
"""

from statistics import median

import pandas as pd
import pyarrow.parquet as pq

from .perf_wide_schema import DB_SCHEMA

_SWEEP_CATEGORIES = {"formats", "flags", "key", "configuration"}
_SWEEP_COLUMNS = {c.name for c in DB_SCHEMA if c.category in _SWEEP_CATEGORIES}


def _load(path):
    return pq.read_table(path).to_pandas()


def _point_key(row, sweep_columns):
    """Stable identity of a data point across runs: test + arch + marker + config.

    arch is part of the identity so a run is only ever compared to history from
    the SAME architecture — comparing wormhole cycles to blackhole cycles is
    meaningless.
    """
    config = tuple(
        sorted((c, row[c]) for c in sweep_columns if pd.notna(row[c]))  # skip NaN
    )
    return (row["test_name"], row["arch"], row["marker"], config)


def compare_to_history(current_parquet, history_parquets, *, threshold=0.05):
    """Compare current to history (from Parquet files). Returns a dict:

    {
      "records":    [ {test, marker, run_type, current, baseline, delta, regression} ],
      "regressions":[ subset of records where regression is True ],
      "new_points": [ {test, marker, run_type, current} with no history baseline ],
    }
    ``delta`` is the fractional change vs baseline (0.12 = 12% slower).
    """
    current = _load(current_parquet)
    history = [_load(p) for p in history_parquets]
    return _compare_frames(current, history, threshold=threshold)


def compare_run_to_history(
    warehouse, run_id, *, pipeline="nightly", threshold=0.05, table="llk_perf"
):
    """Same comparison, but sourced from a PerfWarehouse instead of Parquet files.

    Compares one run (by ``run_id``) to every other run of the same ``pipeline``
    already in the warehouse. This is the seam that lets compare run against the
    real table later with no logic change — swap ``PERF_WAREHOUSE=snowflake``.
    """
    current = warehouse.query(
        f"SELECT * FROM {table} WHERE run_id = '{run_id}'"  # noqa: S608 - run_id from CI
    )
    history = warehouse.query(
        f"SELECT * FROM {table} WHERE run_id <> '{run_id}' AND pipeline = '{pipeline}'"
    )
    # history is one frame holding all prior runs' rows; the baseline median over a
    # point key is the same whether the values arrive split per run or together.
    return _compare_frames(current, [history], threshold=threshold)


def _compare_frames(current, history, *, threshold=0.05):
    """Core comparison over DataFrames (shared by the file- and warehouse-based paths)."""
    mean_columns = [c for c in current.columns if c.startswith("mean(")]
    sweep_columns = [c for c in current.columns if c in _SWEEP_COLUMNS]

    # baseline[(key, mean_col)] = median of the historical values for that point
    samples = {}
    for frame in history:
        frame_sweep = [c for c in frame.columns if c in _SWEEP_COLUMNS]
        for _, row in frame.iterrows():
            key = _point_key(row, frame_sweep)
            for col in mean_columns:
                val = row.get(col)
                if pd.notna(val):
                    samples.setdefault((key, col), []).append(float(val))
    baseline = {k: median(v) for k, v in samples.items()}

    records, regressions, new_points = [], [], []
    for _, row in current.iterrows():
        key = _point_key(row, sweep_columns)
        for col in mean_columns:
            val = row.get(col)
            if pd.isna(val):
                continue
            run_type = col[len("mean(") : -1]
            base = baseline.get((key, col))
            point = {
                "test": row["test_name"],
                "marker": row["marker"],
                "run_type": run_type,
                "current": float(val),
            }
            if base is None:
                new_points.append(point)
                continue
            delta = (float(val) - base) / base if base else 0.0
            record = {
                **point,
                "baseline": base,
                "delta": delta,
                "regression": delta > threshold,
            }
            records.append(record)
            if record["regression"]:
                regressions.append(record)

    return {"records": records, "regressions": regressions, "new_points": new_points}


def summarize_comparison(result) -> str:
    """Human summary: regression count + the worst offenders."""
    regs = result["regressions"]
    lines = [
        f"{len(result['records'])} points compared, "
        f"{len(regs)} regression(s), {len(result['new_points'])} new point(s)."
    ]
    for r in sorted(regs, key=lambda x: -x["delta"])[:10]:
        lines.append(
            f"  REGRESSION {r['test']} [{r['marker']}] {r['run_type']}: "
            f"{r['current']:.1f} vs {r['baseline']:.1f} baseline "
            f"(+{r['delta'] * 100:.1f}%)"
        )
    return "\n".join(lines)
