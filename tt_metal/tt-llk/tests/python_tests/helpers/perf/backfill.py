# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CLI: load nights already archived in GitHub artefacts into the perf warehouse.

Produces **exactly what the nightly upload job produces** -- one Parquet per
run per architecture, merged through ``helpers.perf.merge`` -- so a backfilled
night and a live one are indistinguishable in ``LLK_PERF.RUNS``. That is the
whole design constraint: any difference here becomes a discontinuity in the
history the regression gate reads.

    collect   gh run download  ->  runs/<run id>/**/*.parquet
    stage     verify + merge   ->  llk_perf_<pipeline>-<date>-<run>-<arch>.parquet
    upload    sftp -b batchfile

Two layouts, one output
-----------------------
The archive spans a change in how CI packages its artefacts, so collection has
to read both:

=================  =========================================================
2026-08-28 on      ``perf-data-<arch>-<n>/<tag>/<tag>.parquet`` -- one Parquet
                   per shard, named after the run tag.
2026-08-15..08-27  ``perf-data-<arch>-<n>/perf_data-<arch>-<n>.zip``, holding
                   ``perf_data/<run id>.parquet`` **and**
                   ``perf_data/run-<arch>-<n>.parquet`` -- the same rows twice
                   (the double write #53928 removed). Unzip, keep one.
before 2026-08-15  CSV only. The Parquet writer's import was broken until
                   2026-08-14 11:40, so those nights cannot be loaded.
=================  =========================================================

Both eras carry the full run-provenance column set, and the merge reads
``arch`` and ``run_id`` from the rows rather than the file name, so the naming
difference does not reach the output.

Schema drift is expected and harmless
-------------------------------------
Audited over every night from 2026-08-15: the column count moves 81 -> 85 in
three steps (``formats.sfpu_math`` becomes ``formats.sfpu_src``/``_dst`` on
08-21; ``alpha_bits``/``beta_bits`` on 08-29; ``relu_config`` on 08-30). None
is a promoted TEST_EXECUTIONS column, so all of it lands in the warehouse's
``PARAMS`` VARIANT with no DDL change. Only ``mean`` and ``std`` appear as
stats, and only the six real counters as metrics, so nothing is silently
ignored by the loader.

The one analytical wrinkle is the rename: a query on
``params:"formats.sfpu_src"`` returns NULL for nights before 2026-08-21, where
the same thing was called ``formats.sfpu_math``.

Examples::

    # Check the credentials first. Nothing else works until this does.
    python3 -m helpers.perf.backfill --check \
        --host s-xxxx.server.transfer.us-east-2.amazonaws.com \
        --user llk-perf-run-writer --key ~/.ssh/llk_perf_run

    # Rehearse: stage the last 20 nightlies, upload nothing.
    python3 -m helpers.perf.backfill --last 20 --stage-dir /tmp/stage --dry-run

    # Backfill for real.
    python3 -m helpers.perf.backfill --last 20 --stage-dir /tmp/stage --upload \
        --host ... --user llk-perf-run-writer --key ~/.ssh/llk_perf_run

    # Re-upload an already collected directory (needs no GitHub access).
    python3 -m helpers.perf.backfill --from-dir /tmp/stage/runs --stage-dir /tmp/stage --upload ...

``manifest.csv`` records what was sent. Query the warehouse against it to
confirm the ingest.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

from .merge import merge_run

# The workflow that produces the nightly perf artefacts, and the artefact name
# prefix it uploads them under. Both live in .github/workflows/llk-perf-impl.yaml.
NIGHTLY_WORKFLOW = "llk-perf.yaml"
ARTIFACT_PATTERN = "perf-data-*"

DEFAULT_REMOTE_PREFIX = "llk_perf_"

MANIFEST_NAME = "manifest.csv"
MANIFEST_COLUMNS = (
    "file",
    "run_id",
    "arch",
    "shards",
    "rows",
    "bytes",
    "commit_sha",
    "timestamp",
    "pipeline",
)


class BackfillError(Exception):
    """Bad input or a failed external command. The CLI turns it into exit 1."""


# ---------------------------------------------------------------- collect


def _gh(args):
    """Run a ``gh`` subcommand and return stdout. Raises BackfillError on failure."""
    try:
        done = subprocess.run(
            ["gh", *args], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        raise BackfillError(
            "backfill: `gh` is not installed; run `gh auth login` first"
        )
    if done.returncode != 0:
        raise BackfillError(
            f"backfill: gh {' '.join(args)} failed: {done.stderr.strip()}"
        )
    return done.stdout


def nightly_run_ids(limit, *, search_depth=100):
    """The ids of the last ``limit`` scheduled nightly runs, newest first.

    Only ``schedule`` runs are nightlies; a ``workflow_dispatch`` run is
    somebody testing and must not enter the baseline history. Failed runs are
    kept: the artefact upload is ``if: always()``, so a night that lost some
    shards still archived the ones that finished, and partial coverage is real
    data. ``search_depth`` is how far back to look, because dispatched runs are
    frequent enough that the newest ``limit`` runs are rarely ``limit`` nights.
    """
    out = _gh(
        [
            "run",
            "list",
            "--workflow",
            NIGHTLY_WORKFLOW,
            "--limit",
            str(search_depth),
            "--json",
            "databaseId,conclusion,createdAt,event",
        ]
    )
    nights = [r for r in json.loads(out) if r["event"] == "schedule"]
    return [str(r["databaseId"]) for r in nights[:limit]]


def download_run(run_id, runs_dir):
    """Download one run's perf-data artefacts into ``runs_dir/<run id>``.

    One directory per run, never a shared one: every night reuses the same
    artefact names (``perf-data-<arch>-<shard>``), so a shared destination
    makes two nights collide on the first repeated name.
    """
    dest = os.path.join(runs_dir, str(run_id))
    os.makedirs(dest, exist_ok=True)
    _gh(["run", "download", str(run_id), "--pattern", ARTIFACT_PATTERN, "-D", dest])
    return dest


def _expand_zips(root):
    """Unpack any ``*.zip`` under ``root`` in place. Returns how many it opened.

    Artefacts from before 2026-08-28 wrap the report tree in a second zip, so
    the Parquets are invisible to a plain glob until this runs.
    """
    opened = 0
    for archive in sorted(glob.glob(os.path.join(root, "**", "*.zip"), recursive=True)):
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(os.path.join(os.path.dirname(archive), "unzipped"))
            opened += 1
        except zipfile.BadZipFile:
            # Not fatal: a corrupt archive costs one shard, not the night.
            print(f"  WARNING {archive}: not a zip, skipped")
    return opened


def collect_shards(root):
    """Every shard's Parquet under ``root``, one per shard, sorted.

    Unpacks the legacy inner zips first, then resolves the legacy double write:
    a shard that holds both ``<run id>.parquet`` and ``run-<arch>-<n>.parquet``
    holds the same rows twice, so only one is kept. ``run-<arch>-<n>`` wins --
    identical rows, and it is the copy whose name still says which shard it is,
    which keeps the two eras diagnosable from a file listing alone.
    """
    _expand_zips(root)
    by_dir = {}
    for path in sorted(
        glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
    ):
        by_dir.setdefault(os.path.dirname(path), []).append(path)

    shards = []
    for directory, paths in sorted(by_dir.items()):
        if len(paths) == 1:
            shards.append(paths[0])
            continue
        named = [p for p in paths if os.path.basename(p).startswith("run-")]
        if len(named) == 1:
            shards.append(named[0])
        else:
            raise BackfillError(
                f"{directory}: {len(paths)} Parquets and no single "
                f"run-<arch>-<n> copy to prefer: "
                f"{sorted(os.path.basename(p) for p in paths)}"
            )
    return shards


# ---------------------------------------------------------------- stage


def verify_shard(path):
    """Read ``path`` and return its provenance, or raise BackfillError.

    These are the warehouse's own preconditions, applied before the upload
    rather than after it: a row of NULL provenance lands in the table and
    cannot be attributed to a night afterwards.
    """
    # Imported here, not at module scope, so --help and the offline unit tests
    # do not need pyarrow.
    import pyarrow.parquet as pq

    from .wide_schema import MANDATORY

    try:
        table = pq.read_table(path)
    except Exception as e:  # corrupt or truncated artefact
        raise BackfillError(f"{path}: cannot read Parquet: {e}")

    if table.num_rows == 0:
        raise BackfillError(f"{path}: no rows")

    missing = [c for c in MANDATORY if c not in table.column_names]
    if missing:
        raise BackfillError(f"{path}: missing mandatory column(s): {missing}")
    nulled = [c for c in MANDATORY if table.column(c).null_count]
    if nulled:
        raise BackfillError(f"{path}: NULL in mandatory column(s): {nulled}")

    def one(column):
        values = set(table.column(column).to_pylist())
        if len(values) != 1:
            raise BackfillError(
                f"{path}: {column} is not constant: {sorted(values)[:5]}"
            )
        return values.pop()

    return {
        "arch": one("arch"),
        "run_id": one("run_id"),
        "commit_sha": one("commit_sha"),
        "pipeline": one("pipeline"),
        "rows": table.num_rows,
    }


def stage(runs_dir, stage_dir, *, prefix=DEFAULT_REMOTE_PREFIX):
    """Verify and merge each run under ``runs_dir`` into ``stage_dir``.

    Returns ``(staged, rejected)``: manifest rows, and ``(path, reason)`` for
    shards that failed verification. A run is merged from the shards that pass;
    one corrupt shard costs its own rows, not the night.
    """
    import pyarrow.parquet as pq

    os.makedirs(stage_dir, exist_ok=True)
    staged, rejected = [], []
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(d)
    ) or [runs_dir]

    for run_dir in run_dirs:
        good = []
        for path in collect_shards(run_dir):
            try:
                verify_shard(path)
                good.append(path)
            except BackfillError as e:
                rejected.append((path, str(e)))
        if not good:
            print(f"  SKIPPED {os.path.basename(run_dir)}: no loadable Parquet")
            continue
        try:
            merged = merge_run(good, stage_dir, prefix=prefix)
        except ValueError as e:
            rejected.append((run_dir, f"{run_dir}: merge failed: {e}"))
            continue
        for row in merged:
            table = pq.read_table(
                os.path.join(stage_dir, row["file"]),
                columns=["commit_sha", "timestamp", "pipeline"],
            )
            staged.append(
                {
                    **row,
                    "bytes": os.path.getsize(os.path.join(stage_dir, row["file"])),
                    "commit_sha": table.column("commit_sha")[0].as_py(),
                    "timestamp": table.column("timestamp")[0].as_py(),
                    "pipeline": table.column("pipeline")[0].as_py(),
                }
            )
    return staged, rejected


def write_manifest(staged, stage_dir):
    """Write ``manifest.csv`` beside the staged files. Returns its path.

    The manifest is the checklist for the read-back: every row here must appear
    in the warehouse, with the same shard and row counts, after the ingest.
    """
    path = os.path.join(stage_dir, MANIFEST_NAME)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(staged)
    return path


# ---------------------------------------------------------------- upload


def batchfile_lines(names, *, remote_dir=None):
    """The sftp -b script that puts ``names`` and then lists what arrived.

    ``ls -hal`` last is deliberate: sftp -b stops at the first failed command,
    so a listing in the log is proof that every put before it succeeded.
    """
    lines = []
    if remote_dir:
        lines.append(f"cd {remote_dir}")
    lines.extend(f"put {name}" for name in names)
    lines.append("ls -hal")
    return lines


def _require_credentials(host, user, key, what):
    """Fail before any work when a credential is missing. ``what`` is the action."""
    for value, flag in ((host, "--host"), (user, "--user"), (key, "--key")):
        if not value:
            raise BackfillError(f"backfill: {flag} is required to {what}")


def _sftp_command(batchfile, *, host, user, key):
    """The sftp argv. ``BatchMode=yes`` fails at once instead of prompting.

    Without it an unauthorised key waits for a password, which in CI is a job
    that hangs until its timeout rather than one that fails with a reason.
    """
    return [
        "sftp",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-i",
        os.path.expanduser(key),
        "-b",
        batchfile,
        f"{user}@{host}",
    ]


def key_fingerprint(key):
    """The SHA256 fingerprint of ``key``, or None if ssh-keygen cannot read it.

    The one value the warehouse owner needs to answer "is the key you installed
    the key I am sending?", so the failure path prints it.
    """
    done = subprocess.run(
        ["ssh-keygen", "-lf", os.path.expanduser(key)],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip() if done.returncode == 0 else None


def check_connection(*, host, user, key, remote_dir=None):
    """Log in, print the remote listing, and return 0. Raises BackfillError if not.

    Run this first. Authentication is the leg that fails, and it fails for
    reasons outside this repo: a key the server has not installed, the wrong
    user, the wrong server. The user name in particular is
    ``<name>-<permission>`` in the warehouse's terraform, so it is
    ``llk-perf-run-writer``, not ``llk-perf-run``.
    """
    _require_credentials(host, user, key, "check the connection")

    work = tempfile.mkdtemp(prefix="llk-perf-check-")
    batchfile = os.path.join(work, "sftp_batchfile.txt")
    lines = ([f"cd {remote_dir}"] if remote_dir else []) + ["pwd", "ls -hal"]
    with open(batchfile, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"backfill: checking {user}@{host}")
    done = subprocess.run(
        _sftp_command(batchfile, host=host, user=user, key=key), cwd=work, text=True
    )
    shutil.rmtree(work, ignore_errors=True)
    if done.returncode != 0:
        raise BackfillError(
            f"backfill: cannot log in to {user}@{host} (sftp exited {done.returncode}).\n"
            "  Either the server has not installed this key for that user, or the\n"
            "  host or the user name is wrong -- note the user is\n"
            "  <name>-<permission>, e.g. llk-perf-run-writer. Send the warehouse\n"
            "  owner this fingerprint and ask which key they installed:\n"
            f"    {key_fingerprint(key) or f'(ssh-keygen cannot read {key})'}"
        )
    print("backfill: login OK")
    return 0


def upload(stage_dir, names, *, host, user, key, remote_dir=None, dry_run=False):
    """Upload the named files from ``stage_dir`` over SFTP. Returns the batchfile."""
    if not names:
        raise BackfillError("backfill: nothing to upload")
    _require_credentials(host, user, key, "upload")

    batchfile = os.path.join(stage_dir, "sftp_batchfile.txt")
    with open(batchfile, "w") as fh:
        fh.write("\n".join(batchfile_lines(names, remote_dir=remote_dir)) + "\n")

    command = _sftp_command(batchfile, host=host, user=user, key=key)
    if dry_run:
        print("backfill: would run:", " ".join(command))
        return batchfile

    done = subprocess.run(command, cwd=stage_dir, text=True)
    if done.returncode != 0:
        raise BackfillError(f"backfill: sftp exited {done.returncode}")
    return batchfile


# ---------------------------------------------------------------- CLI


def run(args):
    """Do the collect -> stage -> upload sequence the parsed args ask for."""
    if args.check:
        return check_connection(
            host=args.host, user=args.user, key=args.key, remote_dir=args.remote_dir
        )
    if not args.stage_dir:
        raise BackfillError("backfill: --stage-dir is required")

    runs_dir = args.from_dir or os.path.join(args.stage_dir, "runs")

    if not args.from_dir:
        run_ids = list(args.run_id)
        if args.last:
            run_ids += nightly_run_ids(args.last)
        if not run_ids:
            raise BackfillError("backfill: give --run-id, --last or --from-dir")
        for run_id in run_ids:
            print(f"backfill: downloading run {run_id}")
            try:
                download_run(run_id, runs_dir)
            except BackfillError as e:
                # A night whose failure predates any artefact has nothing to
                # download. Saying so beats failing the whole backfill.
                print(f"  SKIPPED run {run_id}: {e}")

    staged, rejected = stage(runs_dir, args.stage_dir, prefix=args.remote_prefix)
    for _, reason in rejected:
        print(f"  REJECTED {reason}")
    if not staged:
        raise BackfillError(f"backfill: nothing loadable under {runs_dir!r}")

    manifest = write_manifest(staged, args.stage_dir)
    print(
        f"backfill: staged {len(staged)} run file(s) from "
        f"{sum(r['shards'] for r in staged)} shard(s), "
        f"{sum(r['rows'] for r in staged)} row(s), {len(rejected)} rejected"
    )
    print(f"backfill: manifest {manifest}")

    if args.upload or args.dry_run:
        upload(
            args.stage_dir,
            [row["file"] for row in staged],
            host=args.host,
            user=args.user,
            key=args.key,
            remote_dir=args.remote_dir,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(
                f"backfill: uploaded {len(staged)} file(s) to {args.user}@{args.host}"
            )
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = ap.add_argument_group("what to load")
    source.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="a llk-perf workflow run id; repeatable",
    )
    source.add_argument(
        "--last", type=int, help="also take the last N scheduled nightlies"
    )
    source.add_argument(
        "--from-dir",
        help="skip the download and read already-downloaded runs from here",
    )
    ap.add_argument(
        "--stage-dir",
        help="working dir for the merged copies (not needed with --check)",
    )
    ap.add_argument(
        "--remote-prefix",
        default=DEFAULT_REMOTE_PREFIX,
        help=f"prefix for the uploaded object names (default {DEFAULT_REMOTE_PREFIX!r})",
    )
    dest = ap.add_argument_group("where to send it")
    dest.add_argument(
        "--check",
        action="store_true",
        help="only test the login and list the remote directory, then stop",
    )
    dest.add_argument("--upload", action="store_true", help="actually upload")
    dest.add_argument(
        "--dry-run",
        action="store_true",
        help="stage and print the sftp command, but do not connect",
    )
    dest.add_argument("--host", default=os.environ.get("LLK_PERF_SFTP_HOST"))
    dest.add_argument("--user", default=os.environ.get("LLK_PERF_SFTP_USER"))
    dest.add_argument(
        "--key",
        default=os.environ.get("LLK_PERF_SFTP_KEY"),
        help="private key file (never the .pub)",
    )
    dest.add_argument("--remote-dir", help="cd here on the server before putting")
    args = ap.parse_args(argv)

    try:
        return run(args)
    except BackfillError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
