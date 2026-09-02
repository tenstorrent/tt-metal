# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CLI: send finished nightly run Parquets to the perf warehouse over SFTP.

The nightly (``llk-perf.yaml`` -> ``llk-perf-impl.yaml``) already writes one
typed Parquet per shard, ``perf_data/runs/<tag>/<tag>.parquet``, and uploads it
as the ``perf-data-<arch>-<split_group>`` artifact. This tool moves those
artifacts into the warehouse. It exists for two jobs:

  1. Backfill  — load nights that ran before the warehouse existed.
  2. Rehearsal — prove the naming, the verification and the SFTP leg on real
                 data before the same upload runs inside CI.

Three stages, each usable on its own::

    collect   gh run download  ->  runs/<tag>/<tag>.parquet
    stage     verify + rename  ->  <prefix><tag>.parquet  + manifest.csv
    upload    sftp -b batchfile

The run tag ``<run_id>-<arch>-<shard>`` is already unique across nights, so the
remote layout is flat: every file lands in the SFTP home directory under one
prefix. Flat means no ``mkdir`` on the server and no ordering rules between
uploads, which is what a Snowflake external stage or Snowpipe wants.

Nothing is uploaded that did not pass ``_verify_parquet``: the file must read,
hold rows, carry every mandatory column non-NULL, and its ``run_id`` and
``arch`` must agree with the tag in its name. A file that fails is reported and
skipped; the rest still go.

Examples::

    # Check the credentials first. Nothing else works until this does.
    python3 -m helpers.perf.backfill --check \
        --host s-xxxx.server.transfer.us-east-2.amazonaws.com \
        --user llk-perf-run --key ~/.ssh/id_ed25519

    # Rehearse offline: stage the last 5 nightlies, upload nothing.
    python3 -m helpers.perf.backfill --last 5 --stage-dir /tmp/stage --dry-run

    # Backfill three named runs for real.
    python3 -m helpers.perf.backfill \
        --run-id 33145544147 --run-id 33230616532 --run-id 33465181016 \
        --stage-dir /tmp/stage --upload \
        --host s-xxxx.server.transfer.us-east-2.amazonaws.com \
        --user llk-perf-run --key ~/.ssh/id_ed25519

    # Re-upload an already staged directory (no GitHub access needed).
    python3 -m helpers.perf.backfill --from-dir /tmp/stage/runs --stage-dir /tmp/stage --upload ...

``manifest.csv`` records what was sent (tag, arch, run_id, commit_sha,
timestamp, rows, bytes). Query the warehouse against it to confirm the ingest.
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

# The workflow that produces the nightly perf artifacts, and the artifact name
# prefix it uploads them under. Both live in .github/workflows/llk-perf-impl.yaml.
NIGHTLY_WORKFLOW = "llk-perf.yaml"
ARTIFACT_PATTERN = "perf-data-*"

# Prefix every uploaded object, so LLK perf files stay identifiable in a
# directory the warehouse shares with other producers.
DEFAULT_REMOTE_PREFIX = "llk_perf_"

MANIFEST_NAME = "manifest.csv"
MANIFEST_COLUMNS = (
    "file",
    "tag",
    "arch",
    "run_id",
    "commit_sha",
    "timestamp",
    "pipeline",
    "rows",
    "bytes",
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


def nightly_run_ids(limit, *, search_depth=60):
    """The ids of the last ``limit`` successful scheduled nightly runs, newest first.

    Only ``schedule`` runs are nightlies; ``workflow_dispatch`` runs are
    somebody testing and must not enter the baseline history. ``search_depth``
    is how far back to look, because failed and dispatched runs are common
    enough that the newest ``limit`` runs are rarely ``limit`` nightlies.
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
            "databaseId,conclusion,createdAt,event,headSha",
        ]
    )
    runs = json.loads(out)
    nightlies = [
        r for r in runs if r["event"] == "schedule" and r["conclusion"] == "success"
    ]
    return [str(r["databaseId"]) for r in nightlies[:limit]]


def download_run(run_id, runs_dir):
    """Download one run's perf-data artifacts into ``runs_dir/<run_id>``.

    One directory per run, never a shared one. ``gh run download`` unpacks each
    artifact into a directory named after the artifact, and every night reuses
    the same artifact names (``perf-data-<arch>-<shard>``), so a shared
    destination makes two nights collide on the first file whose name repeats.
    Returns the directory.
    """
    dest = os.path.join(runs_dir, str(run_id))
    os.makedirs(dest, exist_ok=True)
    _gh(["run", "download", str(run_id), "--pattern", ARTIFACT_PATTERN, "-D", dest])
    return dest


def collect_parquets(root):
    """Every run Parquet under ``root``, sorted. Ignores the CSVs beside them."""
    return sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))


# ---------------------------------------------------------------- stage


def tag_of(parquet_path):
    """The run tag ``<run_id>-<arch>-<shard>`` a run Parquet is named after."""
    return os.path.basename(parquet_path)[: -len(".parquet")]


def shard_of(tag):
    """The shard index at the end of a run tag."""
    return tag.rsplit("-", 1)[-1]


def bare_run_id(run_id):
    """``33465181016-2`` (re-run attempt 2) -> ``33465181016``.

    ``core._run_id`` appends the attempt number to the row-level ``run_id`` but
    the workflow builds the run tag from ``github.run_id`` alone, so the two
    disagree on every re-run. This is the one place that knows it.
    """
    return run_id.split("-", 1)[0]


def _verify_parquet(path):
    """Read ``path`` and return its provenance, or raise BackfillError.

    A row of NULL provenance is worse than a missing file: it lands in the
    warehouse and cannot be attributed to a night afterwards. So the checks are
    the warehouse's own preconditions, applied before the upload rather than
    after it.
    """
    # Imported here, not at module scope, so --help and the offline unit tests
    # do not need pyarrow.
    import pyarrow.parquet as pq

    from .wide_schema import MANDATORY

    try:
        table = pq.read_table(path)
    except Exception as e:  # corrupt or truncated artifact
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

    provenance = {
        "arch": one("arch"),
        "run_id": one("run_id"),
        "commit_sha": one("commit_sha"),
        "pipeline": one("pipeline"),
        # timestamp varies per test within a shard; keep the earliest.
        "timestamp": min(table.column("timestamp").to_pylist()),
        "rows": table.num_rows,
    }

    # The tag is what makes the remote name unique, so it must be the same run
    # the rows claim to come from. A mismatch means a renamed or hand-edited
    # file, and hand-edited files are exactly what must not reach the warehouse.
    tag = tag_of(path)
    expected = (
        f"{bare_run_id(provenance['run_id'])}-{provenance['arch']}-{shard_of(tag)}"
    )
    if tag != expected:
        raise BackfillError(
            f"{path}: name {tag!r} disagrees with its rows "
            f"(run_id={provenance['run_id']}, arch={provenance['arch']})"
        )
    return provenance


def remote_name(tag, provenance, prefix=DEFAULT_REMOTE_PREFIX):
    """The flat object name this Parquet takes in the SFTP home directory.

    Built from the rows' own ``run_id``, not from the file name, because the
    file name drops the re-run attempt: attempt 1 and attempt 2 of the same
    shard share a run tag. Two attempts are two different measurements, so they
    must be two different objects -- ``...-33465181016-blackhole-7`` and
    ``...-33465181016-2-blackhole-7``.
    """
    return (
        f"{prefix}{provenance['run_id']}-{provenance['arch']}-{shard_of(tag)}.parquet"
    )


def stage(parquets, stage_dir, *, prefix=DEFAULT_REMOTE_PREFIX):
    """Verify each Parquet and copy it into ``stage_dir`` under its remote name.

    Returns ``(staged, rejected)``: a list of manifest rows, and a list of
    ``(path, reason)`` for the files that failed verification. Verification
    failure never stops the run -- one corrupt shard must not block the other
    nine.
    """
    os.makedirs(stage_dir, exist_ok=True)
    staged, rejected = [], []
    for path in parquets:
        try:
            provenance = _verify_parquet(path)
        except BackfillError as e:
            rejected.append((path, str(e)))
            continue
        name = remote_name(tag_of(path), provenance, prefix)
        target = os.path.join(stage_dir, name)
        shutil.copyfile(path, target)
        staged.append(
            {
                "file": name,
                "tag": tag_of(path),
                "arch": provenance["arch"],
                "run_id": provenance["run_id"],
                "commit_sha": provenance["commit_sha"],
                "timestamp": provenance["timestamp"],
                "pipeline": provenance["pipeline"],
                "rows": provenance["rows"],
                "bytes": os.path.getsize(target),
            }
        )
    return staged, rejected


def write_manifest(staged, stage_dir):
    """Write ``manifest.csv`` beside the staged files. Returns its path.

    The manifest is the checklist for the read-back: every row here must appear
    in the warehouse with the same row count after the ingest.
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
    that hangs until its timeout rather than a job that fails with a reason.
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

    This is the one value the warehouse owner needs to answer "is the key you
    installed the key I am sending?", so the failure path prints it.
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
    user, the wrong server. Learning that in one second beats learning it after
    staging thirty files.
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
        fingerprint = key_fingerprint(key) or f"(ssh-keygen cannot read {key})"
        raise BackfillError(
            f"backfill: cannot log in to {user}@{host} (sftp exited {done.returncode}).\n"
            "  Either the server has not installed this key for that user, or the\n"
            "  host or the user name is wrong. Send the warehouse owner this\n"
            "  fingerprint and ask which key they installed:\n"
            f"    {fingerprint}"
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
                dest = download_run(run_id, runs_dir)
            except BackfillError as e:
                print(f"  SKIPPED run {run_id}: {e}")
                continue
            # Nights older than the in-run Parquet writer ship CSVs only. They
            # are not an error -- they are simply outside what the warehouse
            # can take, and saying so is better than a confusing empty result.
            if not collect_parquets(dest):
                print(f"  SKIPPED run {run_id}: no run Parquet (predates the writer?)")

    parquets = collect_parquets(runs_dir)
    if not parquets:
        raise BackfillError(f"backfill: no run Parquets under {runs_dir!r}")

    staged, rejected = stage(parquets, args.stage_dir, prefix=args.remote_prefix)
    for path, reason in rejected:
        print(f"  REJECTED {reason}")
    if not staged:
        raise BackfillError("backfill: every Parquet failed verification")

    manifest = write_manifest(staged, args.stage_dir)
    total_rows = sum(row["rows"] for row in staged)
    print(
        f"backfill: staged {len(staged)} file(s), {total_rows} row(s) "
        f"({len(rejected)} rejected) -> {args.stage_dir}"
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
        "--last",
        type=int,
        help="also take the last N successful scheduled nightlies",
    )
    source.add_argument(
        "--from-dir",
        help="skip the download and read already-downloaded runs from here",
    )
    ap.add_argument(
        "--stage-dir",
        help="working dir for the staged copies (not needed with --check)",
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
