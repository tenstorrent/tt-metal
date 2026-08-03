# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Historical CSV -> Parquet migration (Milestone 3).

Converts an archive of past runs into one immutable Parquet batch per run,
reusing the live converter (perf_parquet.convert_csvs_to_parquet) so migrated
historical data lands in the exact same shared schema as live runs.

  - Deterministic: same archive -> same batches + same report (no clocks, no
    run-order effects; runs and CSVs are processed in sorted order).
  - Lenient: historical CSVs may not fit today's schema, so conversion does not
    fail on them; instead each run's dropped columns and coerced values are
    recorded in a coverage report, so nothing is lost silently.
  - Idempotent: a run whose Parquet already exists is skipped (resumable).

Provenance per run is derived from the run's folder (arch from the directory
name, run_id = the directory name) plus an optional run_meta.json sidecar.

"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .perf_parquet import convert_csvs_to_parquet

_ARCH_RE = re.compile(r"(blackhole|wormhole|quasar)")


@dataclass(frozen=True)
class MigrationRun:
    """One historical run: its provenance plus the per-test CSVs it produced."""

    run_id: str
    arch: str
    commit_sha: str
    timestamp: str
    pipeline: str
    csv_paths: tuple
    pr_number: str = None


def migrate_runs(runs, out_dir, *, compression="zstd", overwrite=False):
    """Convert each run's CSVs into one Parquet batch; return a coverage report.

    Writes out_dir/<run_id>.parquet, one per run. Lenient (never raises on a
    dirty CSV). A run whose output already exists is skipped unless overwrite.
    Deterministic: runs and CSVs are processed sorted, so the output is stable.
    Report: {run_id -> {arch, csv_count, dropped_columns, coerced_values, skipped}}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for run in sorted(runs, key=lambda r: r.run_id):
        out_path = out_dir / f"{run.run_id}.parquet"
        if out_path.exists() and not overwrite:
            report[run.run_id] = {"arch": run.arch, "skipped": True}
            continue
        diagnostics = convert_csvs_to_parquet(
            sorted(str(p) for p in run.csv_paths),
            out_path,
            strict=False,
            compression=compression,
            commit_sha=run.commit_sha,
            arch=run.arch,
            run_id=run.run_id,
            timestamp=run.timestamp,
            pipeline=run.pipeline,
            pr_number=run.pr_number,
        )
        report[run.run_id] = {
            "arch": run.arch,
            "csv_count": len(run.csv_paths),
            "dropped_columns": diagnostics["unknown_columns"],
            "coerced_values": diagnostics["coerced_values"],
            "skipped": False,
        }
    return report


def discover_runs(archive_root, *, pipeline="nightly"):
    """Enumerate MigrationRun objects from an archive laid out as
    <archive_root>/<run_dir>/**/<test>.csv — one run per top-level run_dir.

    Provenance per run: arch parsed from the run_dir name; run_id = run_dir name;
    commit_sha/timestamp/pipeline from an optional run_meta.json sidecar in the
    run_dir, else "unknown"/the given pipeline. ``.post.csv`` files are excluded.
    Deterministic: depends only on the tree.
    """
    archive_root = Path(archive_root)
    runs = []
    for run_dir in sorted(p for p in archive_root.iterdir() if p.is_dir()):
        csvs = tuple(
            sorted(
                p for p in run_dir.rglob("*.csv") if not p.name.endswith(".post.csv")
            )
        )
        if not csvs:
            continue
        arch_match = _ARCH_RE.search(run_dir.name)
        meta_file = run_dir / "run_meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        runs.append(
            MigrationRun(
                run_id=run_dir.name,
                arch=meta.get("arch", arch_match.group(1) if arch_match else "unknown"),
                commit_sha=meta.get("commit_sha", "unknown"),
                timestamp=meta.get("timestamp", "unknown"),
                pipeline=meta.get("pipeline", pipeline),
                pr_number=meta.get("pr_number"),
                csv_paths=csvs,
            )
        )
    return runs


def summarize_coverage(report) -> str:
    """One-line-per-run human summary of what migrated cleanly vs. needed attention."""
    lines = []
    for run_id in sorted(report):
        r = report[run_id]
        if r.get("skipped"):
            lines.append(f"  {run_id}: skipped (already migrated)")
            continue
        dropped = sum(len(v) for v in r["dropped_columns"].values())
        coerced = sum(len(v) for v in r["coerced_values"].values())
        status = (
            "clean"
            if not dropped and not coerced
            else f"{dropped} dropped col(s), {coerced} coerced field(s)"
        )
        lines.append(f"  {run_id} [{r['arch']}]: {r['csv_count']} csv(s) -> {status}")
    return "\n".join(lines)
