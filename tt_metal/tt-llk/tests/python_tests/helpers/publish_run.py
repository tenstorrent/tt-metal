# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CLI: turn one run's per-test CSVs into a single typed ``run.parquet``.

This is the entry point the CI ``publish-parquet`` job calls after a perf sweep.
It gathers the combined per-test CSVs, stamps run provenance (from CLI args + env
set by CI), and reuses the tested ``convert_csvs_to_parquet`` converter. One
Parquet per architecture per run.

    python -m helpers.publish_run --csv-dir perf_data --out run-wormhole.parquet --arch wormhole

Provenance comes from the environment (CI sets these); sensible local defaults so
it also runs by hand:
    COMMIT_SHA, RUN_ID, PIPELINE ("PR"|"nightly"), PR_NUMBER, RUN_TIMESTAMP
"""

import argparse
import datetime
import glob
import os

from .perf_parquet import convert_csvs_to_parquet


def _run_csvs(csv_dir):
    """The combined per-test CSVs, excluding the .post / .counters side files."""
    return sorted(
        p
        for p in glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True)
        if not p.endswith((".post.csv", ".counters.csv"))
    )


def _utcnow():
    return datetime.datetime.utcnow().isoformat() + "Z"


def publish(csv_dir, out_path, arch, *, strict=False):
    """Convert the run's CSVs under ``csv_dir`` to ``out_path``. Returns diagnostics.

    ``strict`` (Q8, open): False = drop/coerce unknown columns and log, never block
    the publish; True = fail loud on drift. Prototype default is lenient.
    """
    csvs = _run_csvs(csv_dir)
    if not csvs:
        raise SystemExit(f"publish_run: no CSVs found under {csv_dir!r}")
    diagnostics = convert_csvs_to_parquet(
        csvs,
        out_path,
        strict=strict,
        commit_sha=os.environ.get("COMMIT_SHA", "unknown"),
        arch=arch,
        run_id=os.environ.get("RUN_ID", "local"),
        timestamp=os.environ.get("RUN_TIMESTAMP") or _utcnow(),
        pipeline=os.environ.get("PIPELINE", "PR"),
        pr_number=os.environ.get("PR_NUMBER") or None,
    )
    print(f"publish_run: wrote {out_path} from {len(csvs)} CSV(s)")
    for test, cols in sorted(diagnostics.get("unknown_columns", {}).items()):
        print(f"  note: {test} dropped unknown columns: {cols}")
    return diagnostics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv-dir", required=True, help="dir of combined per-test CSVs")
    ap.add_argument("--out", required=True, help="output run parquet path")
    ap.add_argument("--arch", required=True, help="wormhole | blackhole")
    ap.add_argument("--strict", action="store_true", help="fail on schema drift")
    a = ap.parse_args(argv)
    publish(a.csv_dir, a.out, a.arch, strict=a.strict)


if __name__ == "__main__":
    main()
