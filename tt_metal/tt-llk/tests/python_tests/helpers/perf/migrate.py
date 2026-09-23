# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Historical CSV -> Parquet migration (Milestone 3).

Converts an archive of past runs into one immutable Parquet batch per run,
reusing the live converter (perf_parquet.convert_csvs_to_parquet) so migrated
historical data lands in the exact same shared schema as live runs.

  - Deterministic: same archive -> same batches + same report (no clocks, no
    run-order effects; runs and CSVs are processed in sorted order).
  - Lenient: a dirty run never aborts the migration. Unknown columns are dropped
    and coercible-to-NULL values are nulled (both recorded in the coverage
    report); a run that raises for any reason (empty CSV, NULL mandatory column,
    ...) is recorded as ``failed`` and skipped, so every other run still
    migrates and the accumulated report survives.
  - Idempotent: each batch is written atomically (temp file + rename), so a run
    whose output exists is safely skipped (resumable) and a crash mid-write never
    leaves a half-written file that a later pass mistakes for complete.

Provenance per run comes from an optional ``run_meta.json`` sidecar in the
run_dir (``arch``/``commit_sha``/``timestamp``/``pipeline``/``pr_number``);
anything the sidecar omits (or sets null) falls back to the folder — ``arch``
parsed from the run_dir name, ``run_id`` = the dir name — or a default.

Raw vs post-processed: only the raw per-test CSVs migrate; the post-processed
twin (``<base>.post.csv``) and per-worker ``<base>.counters.csv`` are excluded.
Raw is the canonical stored form both here and in the live publish path
(``perf.combine_perf_reports``): a ``TILE_LOOP`` row carries loop totals, and the
per-tile figures are derived downstream by dividing mean/std by
``loop_factor * tile_cnt`` (both are columns). Storing raw keeps the table
lossless without a redundant per-tile copy, so no ``variant`` column is needed.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .parquet import convert_csvs_to_parquet

_UNKNOWN = "unknown"  # provenance placeholder when nothing derivable

# NOTE: this is a 3rd copy of the arch value set (chip_architecture.py has it).
# The enum can't be imported here — it pulls in ttexalens and would break this
# stack's hardware-free property. TODO(#51249): extract the arch strings into a
# device-free module and import from there.
_ARCHS = ("blackhole", "wormhole", "quasar")
# Match a whole arch token (delimited by non-lowercase letters) so an arch name
# embedded in a larger word can't false-match. A dir naming two arches still
# takes the leftmost — the run_meta.json ``arch`` sidecar disambiguates.
_ARCH_RE = re.compile(r"(?<![a-z])(" + "|".join(_ARCHS) + r")(?![a-z])")


@dataclass(frozen=True)
class MigrationRun:
    """One historical run: its provenance plus the per-test CSVs it produced."""

    run_id: str
    arch: str
    commit_sha: str
    timestamp: str
    pipeline: str
    csv_paths: tuple[Path, ...]
    pr_number: str | None = None


def _read_meta(meta_file: Path) -> dict:
    """Read a run_meta.json sidecar leniently: missing / invalid JSON / non-object
    all resolve to an empty dict, so a hand-written sidecar can never abort discovery.
    """
    if not meta_file.exists():
        return {}
    try:
        data = json.loads(meta_file.read_text())
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def migrate_runs(runs, out_dir, *, compression="zstd", overwrite=False):
    """Convert each run's CSVs into one Parquet batch; return a coverage report.

    Writes ``out_dir/<run_id>.parquet``, one per run, atomically (temp + rename).
    Lenient: a run that raises is recorded ``failed`` and skipped, never aborting
    the others. A run whose output already exists is skipped unless ``overwrite``.
    Deterministic: runs and CSVs are processed sorted, so output is stable.

    Report: ``{run_id -> entry}`` where every entry has ``arch``, ``csv_count``,
    ``skipped``; a migrated run adds ``dropped_columns`` + ``coerced_values``; a
    run that raised adds ``failed=True`` + ``error``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for run in sorted(runs, key=lambda r: r.run_id):
        out_path = out_dir / f"{run.run_id}.parquet"
        base = {"arch": run.arch, "csv_count": len(run.csv_paths)}
        # exists() is trustworthy because writes are atomic (below): a crashed
        # write leaves only a .tmp, never a footer-less final file.
        if out_path.exists() and not overwrite:
            report[run.run_id] = {
                **base,
                "dropped_columns": {},
                "coerced_values": {},
                "skipped": True,
            }
            continue
        tmp_path = out_dir / f"{run.run_id}.parquet.tmp"
        try:
            diagnostics = convert_csvs_to_parquet(
                sorted(str(p) for p in run.csv_paths),
                tmp_path,
                strict=False,
                compression=compression,
                commit_sha=run.commit_sha,
                arch=run.arch,
                run_id=run.run_id,
                timestamp=run.timestamp,
                pipeline=run.pipeline,
                pr_number=run.pr_number,
            )
            os.replace(tmp_path, out_path)  # atomic publish
        except Exception as exc:  # noqa: BLE001 — a dirty run must not abort the rest
            # remove any partial temp so a resumed pass retries this run cleanly
            Path(tmp_path).unlink(missing_ok=True)
            report[run.run_id] = {
                **base,
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
                "skipped": False,
            }
            continue
        report[run.run_id] = {
            **base,
            "dropped_columns": diagnostics["unknown_columns"],
            "coerced_values": diagnostics["coerced_values"],
            "skipped": False,
        }
    return report


def discover_runs(archive_root, *, pipeline="nightly"):
    """Enumerate MigrationRun objects from an archive laid out as
    ``<archive_root>/<run_dir>/**/<test>.csv`` — one run per top-level run_dir.

    Excludes ``.post.csv`` (post-processed twins) and ``.counters.csv`` (per-worker
    counter dumps) and zero-byte files. Provenance per run: a ``run_meta.json``
    sidecar in the run_dir supplies any of ``arch``/``commit_sha``/``timestamp``/
    ``pipeline``/``pr_number``; whatever it omits falls back to the folder (arch
    parsed from the run_dir name, run_id = the dir name) or a default.
    Deterministic: depends only on the tree.
    """
    archive_root = Path(archive_root)
    runs = []
    for run_dir in sorted(p for p in archive_root.iterdir() if p.is_dir()):
        csvs = tuple(
            sorted(
                p
                for p in run_dir.rglob("*.csv")
                # .post.csv would double each test's rows with mixed normalization;
                # .counters.csv collapses to the same test_name and injects
                # NULL-metric phantom rows. Skip both, and skip empty files.
                if not p.name.endswith((".post.csv", ".counters.csv"))
                and p.stat().st_size > 1
            )
        )
        if not csvs:
            continue
        meta = _read_meta(run_dir / "run_meta.json")
        arch_match = _ARCH_RE.search(run_dir.name)
        folder_arch = arch_match.group(1) if arch_match else _UNKNOWN
        # ``or`` (not dict.get default) so an explicit null in the sidecar still
        # falls back rather than poisoning a non-nullable provenance column.
        runs.append(
            MigrationRun(
                run_id=run_dir.name,
                arch=meta.get("arch") or folder_arch,
                commit_sha=meta.get("commit_sha") or _UNKNOWN,
                timestamp=meta.get("timestamp") or _UNKNOWN,
                pipeline=meta.get("pipeline") or pipeline,
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
        if r.get("failed"):
            lines.append(f"  {run_id} [{r['arch']}]: FAILED — {r['error']}")
            continue
        if r.get("skipped"):
            lines.append(f"  {run_id}: skipped (already migrated)")
            continue
        dropped = sum(len(v) for v in r["dropped_columns"].values())
        # coerced_values[test] is {column -> {type, bad, example}}; count the
        # nulled VALUES (info["bad"]), not the number of affected columns.
        coerced = sum(
            info["bad"]
            for cols in r["coerced_values"].values()
            for info in cols.values()
        )
        status = (
            "clean"
            if not dropped and not coerced
            else f"{dropped} dropped col(s), {coerced} coerced value(s)"
        )
        lines.append(f"  {run_id} [{r['arch']}]: {r['csv_count']} csv(s) -> {status}")
    return "\n".join(lines)
