# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Merge a run's per-shard Parquets into one file per architecture.

Merging is safe with respect to the loader's other hard rule -- one
``(configuration, marker)`` pair per run. pytest-split partitions by test item,
so no configuration appears in two shards. Verified across three nightlies, six
architecture groups, 649k rows: zero repeats.
"""

import datetime
import os

RUN_ID_TEMPLATE = "{pipeline}-{date}-{workflow_run_id}-{arch}"


def merged_run_id(*, pipeline, timestamp, workflow_run_id, arch, attempt=""):
    """The run_id one merged file carries. Derived only from the rows.

    Deriving it from the file contents rather than the CI environment is what
    lets the backfill produce exactly what the nightly produces.
    """
    date = _date_of(timestamp)
    run_id = RUN_ID_TEMPLATE.format(
        pipeline=pipeline, date=date, workflow_run_id=workflow_run_id, arch=arch
    )
    return f"{run_id}-{attempt}" if attempt else run_id


def _date_of(timestamp):
    """``2026-09-01T03:53:29+00:00`` -> ``20260901``.

    Falls back to the leading 10 characters if the value is not ISO-8601, so a
    producer that changes format degrades to a readable id instead of raising.
    """
    try:
        return datetime.datetime.fromisoformat(timestamp).strftime("%Y%m%d")
    except (TypeError, ValueError):
        return str(timestamp)[:10].replace("-", "")


def workflow_run_id_of(run_id):
    """The workflow run id inside a per-shard run_id.

    ``core._run_id`` writes the run tag, whose first component is
    ``github.run_id`` (``33465181016-wormhole-4`` -> ``33465181016``). Nights
    archived before that change carry the bare workflow id, which is its own
    first component, so one rule covers both.
    """
    return run_id.split("-", 1)[0]


def attempt_of(run_id, workflow_run_id):
    """The re-run attempt in a per-shard run_id, or "" for attempt 1.

    A run_id is one of four shapes, across the two eras of the archive::

        <wf>                        legacy, attempt 1
        <wf>-<attempt>              legacy, re-run
        <wf>-<arch>-<shard>         current, attempt 1
        <wf>-<arch>-<shard>-<att>   current, re-run

    So the attempt is the trailing component only when stripping the workflow
    id leaves one or three parts. Two parts is ``<arch>-<shard>``, whose shard
    index is a trailing number that is not an attempt -- which is what a
    "trailing digits" rule gets wrong.
    """
    if not run_id.startswith(f"{workflow_run_id}-"):
        return ""
    parts = run_id[len(workflow_run_id) + 1 :].split("-")
    return parts[-1] if len(parts) in (1, 3) and parts[-1].isdigit() else ""


def group_by_arch(paths):
    """Map arch -> its shard paths, reading arch from the rows, not the name.

    The name is a filesystem convention and has been three different things
    across the archive; the column is the same in every file ever written.
    """
    import pyarrow.parquet as pq

    groups = {}
    for path in sorted(paths):
        arch = set(pq.read_table(path, columns=["arch"]).column("arch").to_pylist())
        if len(arch) != 1:
            raise ValueError(f"{path}: arch is not constant: {sorted(arch)}")
        groups.setdefault(arch.pop(), []).append(path)
    return groups


def merge_run(paths, out_dir, *, prefix="llk_perf_"):
    """Merge shards into one Parquet per arch under ``out_dir``.

    Returns a list of ``{"file", "run_id", "arch", "shards", "rows"}``, one per
    architecture. Raises ValueError if a group's rows disagree on a run column
    that merging does not unify.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    merged = []
    for arch, group in sorted(group_by_arch(paths).items()):
        tables = [pq.read_table(p) for p in group]
        table = pa.concat_tables(tables, promote_options="permissive")

        constant = {}
        for column in ("commit_sha", "pipeline", "pr_number"):
            values = {v for v in table.column(column).to_pylist() if v is not None}
            if len(values) > 1:
                raise ValueError(
                    f"{arch}: {column} is not constant across shards: "
                    f"{sorted(values)[:5]}"
                )
            constant[column] = values.pop() if values else None

        timestamps = [t for t in table.column("timestamp").to_pylist() if t]
        earliest = min(timestamps)
        shard_ids = table.column("run_id").to_pylist()
        workflow_run_id = workflow_run_id_of(shard_ids[0])
        run_id = merged_run_id(
            pipeline=constant["pipeline"],
            timestamp=earliest,
            workflow_run_id=workflow_run_id,
            arch=arch,
            attempt=attempt_of(shard_ids[0], workflow_run_id),
        )

        for column, value in (("run_id", run_id), ("timestamp", earliest)):
            table = table.set_column(
                table.schema.get_field_index(column),
                column,
                pa.array([value] * table.num_rows, type=pa.string()),
            )

        name = f"{prefix}{run_id}.parquet"
        pq.write_table(table, os.path.join(out_dir, name), compression="zstd")
        merged.append(
            {
                "file": name,
                "run_id": run_id,
                "arch": arch,
                "shards": len(group),
                "rows": table.num_rows,
            }
        )
    return merged


def main(argv=None):
    """CLI: merge every Parquet under --in-dir into one file per arch in --out-dir."""
    import argparse
    import glob
    import sys

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--in-dir", required=True, help="dir of downloaded per-shard artefacts"
    )
    ap.add_argument("--out-dir", required=True, help="dir to write the merged files to")
    ap.add_argument("--prefix", default="llk_perf_", help="uploaded object name prefix")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(a.in_dir, "**", "*.parquet"), recursive=True))
    if not paths:
        print(f"merge: no Parquet under {a.in_dir!r}", file=sys.stderr)
        return 1
    try:
        merged = merge_run(paths, a.out_dir, prefix=a.prefix)
    except ValueError as e:
        print(f"merge: {e}", file=sys.stderr)
        return 1
    for row in merged:
        print(f"merge: {row['file']} <- {row['shards']} shard(s), {row['rows']} row(s)")
    print(f"merge: {len(paths)} shard file(s) -> {len(merged)} run file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
