# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CLI: turn one run's per-test CSVs into a single typed ``run.parquet``.
It gathers the combined per-test CSVs, stamps run provenance (from CLI args + env
set by CI), and reuses the tested ``convert_csvs_to_parquet`` converter. One
Parquet per architecture per run.

TILE_LOOP timing columns are RAW loop totals (as emitted in ``<test>.csv``), NOT
divided by ``loop_factor * tile_cnt``; that per-tile normalization lives only in
``<test>.post.csv``, which this tool does not read. Divide by
``loop_factor * tile_cnt`` for cycles-per-tile.
"""

import argparse
import datetime
import glob
import os

from .parquet import convert_csvs_to_parquet

_VALID_ARCHES = ("wormhole", "blackhole", "quasar")
# Lowercase, as the warehouse's RUNS.PIPELINE stores them.
_VALID_PIPELINES = ("pr", "nightly")


def _run_csvs(csv_dir):
    """The combined per-test CSVs, excluding the .post / .counters side files.

    Real runs nest one directory per test
    (``perf_data/runs/<tag>/<base>/<base>.csv``), so the glob is recursive. Point
    ``csv_dir`` at ONE run — ``perf_data/latest`` or a specific ``runs/<tag>`` —
    never at ``perf_data`` itself, or every retained run is swept into one batch.
    """
    return sorted(
        p
        for p in glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True)
        if not p.endswith((".post.csv", ".counters.csv"))
    )


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _require_env(name):
    """Return a required, non-empty environment value, or raise ValueError.

    ``os.environ.get(name, default)`` only applies the default when the var is
    *absent*; a defined-but-empty var — the usual Actions outcome when a
    ``${{ ... }}`` expression resolves empty — yields "". These values land in
    ROW_KEY columns, so an empty one must fail loud, not publish a colliding key.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"publish_run: required env {name} is unset or empty")
    return value


def publish(csv_dir, out_path, arch, *, strict=False):
    """Convert the run's CSVs under ``csv_dir`` to ``out_path``. Returns diagnostics.

    Raises ``ValueError`` on bad input (unknown arch/pipeline, missing
    provenance, no CSVs, or — with ``strict=True`` — schema drift). ``strict``
    False = drop/coerce unknown-or-mistyped columns and log, never block; True =
    fail loud on drift.
    """
    if arch not in _VALID_ARCHES:
        raise ValueError(
            f"publish_run: --arch must be one of {_VALID_ARCHES}, got {arch!r}"
        )
    pipeline = os.environ.get("PIPELINE", "").strip()
    if pipeline not in _VALID_PIPELINES:
        raise ValueError(
            f"publish_run: PIPELINE must be one of {_VALID_PIPELINES}, got {pipeline!r}"
        )

    csvs = _run_csvs(csv_dir)
    if not csvs:
        raise ValueError(f"publish_run: no CSVs found under {csv_dir!r}")

    diagnostics = convert_csvs_to_parquet(
        csvs,
        out_path,
        strict=strict,
        commit_sha=_require_env("COMMIT_SHA"),
        arch=arch,
        run_id=_require_env("RUN_ID"),
        timestamp=os.environ.get("RUN_TIMESTAMP") or _utcnow(),
        pipeline=pipeline,
        pr_number=os.environ.get("PR_NUMBER") or None,
    )
    print(f"publish_run: wrote {out_path} from {len(csvs)} CSV(s)")
    for test, cols in sorted(diagnostics.get("unknown_columns", {}).items()):
        print(f"  note: {test} dropped unknown columns: {cols}")
    # coerced_values is the silent-data-loss path under strict=False: a value that
    # failed its declared type was NULLed (indistinguishable from a never-emitted
    # column downstream). The docstring promises to log it, so log it.
    for test, cols in sorted(diagnostics.get("coerced_values", {}).items()):
        print(f"  warning: {test} coerced value(s) to NULL (type mismatch): {cols}")
    return diagnostics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv-dir", required=True, help="dir of combined per-test CSVs")
    ap.add_argument("--out", required=True, help="output run parquet path")
    ap.add_argument(
        "--arch",
        required=True,
        choices=_VALID_ARCHES,
        help="wormhole | blackhole | quasar",
    )
    ap.add_argument("--strict", action="store_true", help="fail on schema drift")
    a = ap.parse_args(argv)
    try:
        publish(a.csv_dir, a.out, a.arch, strict=a.strict)
    except ValueError as e:
        # publish() is the importable half and raises ValueError; the CLI turns
        # that into a non-zero exit (so a caller's `except Exception` still works).
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
