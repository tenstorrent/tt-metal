# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CLI: publish a perf run made by hand (Quasar today) to the perf warehouse.

Quasar has no CI runner. People run the suite by hand on the emulator, so the
CI path (shard Parquet -> merge -> SFTP) does not apply. Three subcommands:

  publish   one local run (``perf_data/latest`` by default) -> one Parquet file,
            with provenance from this checkout. ``--upload`` also sends it.
  backfill  an archive of old runs, one directory per run with a
            ``run_meta.json``, -> one Parquet file per run. Reuses ``migrate``.
  upload    send Parquet files that ``publish`` or ``backfill`` wrote.

Every file carries ``pipeline = "manual"`` and a run_id of the CI shape,
``manual-<date>-<stamp>-<arch>``, where ``stamp`` is the run's UTC start
(``20261005T143012Z``). The stamp is not all digits, so the dashboard does not
mistake it for a GitHub workflow run. The id comes from the run, not from the
time of publishing, so publishing one run twice replaces it in the warehouse
(the loader replays by RUN_ID) instead of adding a second copy.

Every file also says what executed the run: ``platform`` (emulator,
simulator or silicon) and ``platform_version`` (its build). Numbers from two
platforms must never share a trend, so ``platform`` is mandatory here.

Before a file leaves the machine, ``check_file`` applies the rules of the
warehouse loader (data_airflow ``dags/pipelines/llk_perf_run``): the run
columns are present and constant, ``arch`` and ``pipeline`` are lowercase and
not blank, and each (configuration, marker) pair occurs once. The loader
replaces a run by RUN_ID, and it refuses a RUN_ID that another (arch,
pipeline) owns. A manual run_id holds its pipeline and its arch, so the check
also makes sure that the run_id agrees with both. ``upload`` sends only files
that pass, and it names each remote object from the run_id in its rows, not
from the local file name.

Only the holder of a private key that data_iac lists for the ``llk-perf-run``
SFTP user can upload. Without ``--upload`` nothing leaves the machine.
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq

from .migrate import discover_runs, migrate_runs, summarize_coverage
from .parquet import convert_csvs_to_parquet
from .publish_run import _VALID_ARCHES, _run_csvs

PIPELINE = "manual"

PLATFORMS = ("emulator", "simulator", "silicon")

# Same shape as core.RUN_ID_TEMPLATE. Not imported: core reaches the device
# libraries, and this tool must run where they are absent.
RUN_ID_TEMPLATE = "{pipeline}-{date}-{stamp}-{arch}"

# The same name prefix the CI merge gives its files.
OBJECT_PREFIX = "llk_perf_"

# The CI upload in .github/workflows/llk-perf-impl.yaml uses the same endpoint.
SFTP_HOST = "s-dbd4b8a190fa40a4b.server.transfer.us-east-2.amazonaws.com"
SFTP_USER = "llk-perf-run-writer"
KEY_ENV = "LLK_PERF_SFTP_KEY"

# The loader's RUN_COLUMNS: a file without any one of them is quarantined.
# OPTIONAL_RUN_COLUMNS may be absent, but must be constant when present.
LOADER_RUN_COLUMNS = (
    "run_id",
    "pipeline",
    "commit_sha",
    "pr_number",
    "arch",
    "timestamp",
)
LOADER_OPTIONAL_RUN_COLUMNS = ("platform", "platform_version")
MARKER_COLUMN = "marker"
# The loader's STAT_COLUMN_RE: "mean(X)" / "std(X)" are measurements, not config.
_STAT_COLUMN_RE = re.compile(r"^(?P<stat>[A-Za-z_]\w*)\((?P<metric>.+)\)$")

# TestConfig.perf_run_tag() writes local-%Y%m%dT%H%M%SZ off CI.
_STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")

# helpers/perf/publish_manual.py -> tt_metal/tt-llk
_LLK_ROOT = Path(__file__).resolve().parents[4]


def manual_run_id(timestamp, arch):
    """``2026-10-05T14:30:12+00:00``, quasar -> ``manual-20261005-20261005T143012Z-quasar``."""
    ts = _parse_ts(timestamp).astimezone(datetime.timezone.utc)
    return RUN_ID_TEMPLATE.format(
        pipeline=PIPELINE,
        date=ts.strftime("%Y%m%d"),
        stamp=ts.strftime("%Y%m%dT%H%M%SZ"),
        arch=arch,
    )


def _parse_ts(timestamp):
    """ISO-8601 -> aware datetime. A naive value is taken as UTC."""
    try:
        ts = datetime.datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"timestamp is not ISO-8601: {timestamp!r}") from None
    return ts if ts.tzinfo else ts.replace(tzinfo=datetime.timezone.utc)


def run_start(run_dir):
    """The UTC start of a local run, from its tag, else the directory mtime.

    ``perf_data/latest`` is a symlink to ``runs/<tag>``; resolve it first so
    the tag is the directory name.
    """
    run_dir = Path(run_dir).resolve()
    match = _STAMP_RE.search(run_dir.name)
    if match:
        ts = datetime.datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
        return ts.replace(tzinfo=datetime.timezone.utc)
    return datetime.datetime.fromtimestamp(
        run_dir.stat().st_mtime, tz=datetime.timezone.utc
    ).replace(microsecond=0)


def _git(*args, cwd=_LLK_ROOT):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def checkout_state(cwd=_LLK_ROOT):
    """``(commit_sha, problems)`` for the checkout the run was built from.

    A problem is a reason the run must not reach the warehouse: the numbers
    would be filed under a commit that does not hold the code that made them,
    or under a commit that main never had, which the trend would then show
    as a step on main.
    """
    head = _git("rev-parse", "HEAD", cwd=cwd)
    if head.returncode != 0:
        return None, [f"not a git checkout: {head.stderr.strip()}"]
    sha = head.stdout.strip()
    problems = []
    dirty = _git("status", "--porcelain", "--untracked-files=no", cwd=cwd)
    if dirty.stdout.strip():
        problems.append("the checkout has uncommitted changes")
    on_main = _git("merge-base", "--is-ancestor", "HEAD", "origin/main", cwd=cwd)
    if on_main.returncode != 0:
        problems.append(
            "HEAD is not on origin/main (run `git fetch origin main` if it is)"
        )
    return sha, problems


def _normalize_arch(arch):
    """Lowercase and strip, as the loader stores ARCH; refuse an unknown arch."""
    value = str(arch or "").strip().lower()
    if value not in _VALID_ARCHES:
        raise ValueError(f"arch must be one of {_VALID_ARCHES}, got {arch!r}")
    return value


def _check_platform(platform):
    if platform not in PLATFORMS:
        raise ValueError(f"platform must be one of {PLATFORMS}, got {platform!r}")


def publish(
    run_dir,
    out_dir,
    arch,
    *,
    commit_sha,
    platform,
    platform_version=None,
    timestamp=None,
):
    """Convert one local run's CSVs to ``out_dir/llk_perf_<run_id>.parquet``.

    Strict, as the CI path is: a column the schema lacks, or a value it cannot
    type, fails the publish instead of writing a lossy file.
    """
    if arch not in _VALID_ARCHES:
        raise ValueError(f"arch must be one of {_VALID_ARCHES}, got {arch!r}")
    _check_platform(platform)
    csvs = _run_csvs(str(run_dir))
    if not csvs:
        raise ValueError(f"no CSVs under {str(run_dir)!r}")
    start = _parse_ts(timestamp) if timestamp else run_start(run_dir)
    run_id = manual_run_id(start.isoformat(), arch)
    out = Path(out_dir) / f"{OBJECT_PREFIX}{run_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    convert_csvs_to_parquet(
        csvs,
        out,
        strict=True,
        commit_sha=commit_sha,
        arch=arch,
        run_id=run_id,
        timestamp=start.isoformat(),
        pipeline=PIPELINE,
        pr_number=None,
        platform=platform,
        platform_version=platform_version,
    )
    return out, len(csvs)


def plan_backfill(archive_root, arch, platform=None, platform_version=None):
    """The runs under ``archive_root``, each with a ``manual`` run_id.

    Every run directory needs a ``run_meta.json`` with ``timestamp``: it fixes
    the run_id and the run's place in the trend. ``commit_sha`` should be
    there too; without it the run loads as commit ``unknown``. ``arch``
    defaults to ``arch`` when neither the sidecar nor the directory name says,
    and ``platform`` / ``platform_version`` to the given values when the
    sidecar omits them. A run must end up with a platform.
    """
    runs = []
    for run in discover_runs(archive_root, pipeline=PIPELINE):
        if run.timestamp == "unknown":
            raise ValueError(
                f"{run.run_id}: run_meta.json has no timestamp; the run_id needs it"
            )
        try:
            run_arch = _normalize_arch(arch if run.arch == "unknown" else run.arch)
        except ValueError as e:
            raise ValueError(f"{run.run_id}: {e}") from None
        run_platform = (run.platform or platform or "").strip().lower() or None
        try:
            _check_platform(run_platform)
        except ValueError as e:
            raise ValueError(
                f"{run.run_id}: {e} (set it in run_meta.json or pass --platform)"
            ) from None
        runs.append(
            replace(
                run,
                arch=run_arch,
                pipeline=PIPELINE,
                run_id=manual_run_id(run.timestamp, run_arch),
                platform=run_platform,
                platform_version=run.platform_version or platform_version,
            )
        )
    ids = [r.run_id for r in runs]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(
            f"two runs share a timestamp, so one would replace the other: {duplicates}"
        )
    return runs


def object_name(run_id):
    """The remote object for a run. One run is one object, as in CI."""
    return f"{OBJECT_PREFIX}{run_id}.parquet"


def _distinct(series):
    return series.dropna().unique().tolist()


def check_file(path):
    """``(run_id, rows, problems)`` for one Parquet file.

    A problem is a reason the warehouse loader would quarantine the file, or
    would file it under the wrong run. The rules copy the loader's (see the
    module docstring); keep them in step with data_airflow.
    """
    path = Path(path)
    if not path.is_file() or path.suffix != ".parquet":
        return None, 0, [f"not a Parquet file: {path}"]
    try:
        df = pq.read_table(path).to_pandas()
    except Exception as e:  # noqa: BLE001 - any read failure is a problem to report
        return None, 0, [f"cannot read: {type(e).__name__}: {e}"]
    if df.empty:
        return None, 0, ["the file has no rows"]

    problems = []
    missing = [c for c in LOADER_RUN_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing run columns: {missing}")
    if MARKER_COLUMN not in df.columns:
        problems.append(f"no '{MARKER_COLUMN}' column")
    stats = [c for c in df.columns if _STAT_COLUMN_RE.match(c)]
    if not any(_STAT_COLUMN_RE.match(c).group("stat").lower() == "mean" for c in stats):
        problems.append("no mean(<METRIC>) column")

    run = {}
    for column in LOADER_RUN_COLUMNS + LOADER_OPTIONAL_RUN_COLUMNS:
        if column not in df.columns:
            continue
        values = _distinct(df[column])
        if len(values) > 1:
            problems.append(f"'{column}' is not constant: {values[:5]}")
        run[column] = values[0] if values else None

    run_id = run.get("run_id")
    for column in ("run_id", "arch", "pipeline"):
        if column in df.columns and not str(run.get(column) or "").strip():
            problems.append(f"'{column}' is blank")
    for column in ("arch", "pipeline"):
        value = run.get(column)
        if isinstance(value, str) and value and value != value.strip().lower():
            problems.append(f"'{column}' is not lowercase: {value!r}")
    if run.get("pipeline") and run["pipeline"] != PIPELINE:
        problems.append(f"pipeline is {run['pipeline']!r}, not {PIPELINE!r}")
    if run.get("arch") and run["arch"] not in _VALID_ARCHES:
        problems.append(f"arch {run['arch']!r} is not one of {_VALID_ARCHES}")
    platform = run.get("platform")
    if platform not in PLATFORMS:
        problems.append(f"platform {platform!r} is not one of {PLATFORMS}")

    timestamp = run.get("timestamp")
    try:
        start = _parse_ts(timestamp)
        if not 2020 <= start.year <= 2100:
            problems.append(f"timestamp {timestamp!r} is outside 2020..2100")
        # The ownership rule: the id holds the pipeline and the arch, so no
        # wormhole or blackhole run, and no other manual arch, can own it.
        if run.get("arch") and run_id != manual_run_id(timestamp, run["arch"]):
            problems.append(
                f"run_id {run_id!r} does not agree with its timestamp and arch "
                f"(expected {manual_run_id(timestamp, run['arch'])!r})"
            )
    except ValueError as e:
        problems.append(str(e))

    if MARKER_COLUMN in df.columns:
        run_columns = set(LOADER_RUN_COLUMNS) | set(LOADER_OPTIONAL_RUN_COLUMNS)
        grain = [c for c in df.columns if c not in run_columns and c not in stats]
        repeated = int(df.duplicated(subset=grain, keep=False).sum())
        if repeated:
            problems.append(
                f"{repeated} rows repeat a (configuration, marker) pair; "
                "the loader refuses the whole file"
            )
    return run_id, len(df), problems


def check_files(paths):
    """``{path: (run_id, rows, problems)}``; two files with one run_id fail both."""
    results = {Path(p): check_file(p) for p in paths}
    owners = {}
    for p, (run_id, _, _) in results.items():
        if run_id:
            owners.setdefault(run_id, []).append(p)
    for run_id, files in owners.items():
        if len(files) > 1:
            for p in files:
                results[p][2].append(
                    f"run_id {run_id!r} is also in {[str(f) for f in files if f != p]}; "
                    "the later file would replace the earlier one"
                )
    return results


def _report_checks(results):
    bad = 0
    for p, (run_id, rows, problems) in results.items():
        if problems:
            bad += 1
            print(f"FAIL {p}")
            for problem in problems:
                print(f"     {problem}")
        else:
            print(f"ok   {p}: {rows} rows -> {object_name(run_id)}")
    return bad


def upload(paths, key, *, dry_run=False):
    """Send ``paths`` to the warehouse ingest bucket over SFTP.

    Every file must pass ``check_files`` first; one failure stops the whole
    batch before a connection opens. Each file goes up as
    ``llk_perf_<run_id>.parquet``, from the run_id in its rows. Then ``ls`` of
    each object, so the log shows that the server holds it. The batch aborts
    on the first failed command.
    """
    results = check_files(paths)
    for run_id, _, problems in results.values():
        if problems:
            raise ValueError("; ".join(problems))
    lines = []
    for p, (run_id, _, _) in results.items():
        lines.append(f'put "{p}" "{object_name(run_id)}"')
    lines += [f'ls -l "{object_name(r)}"' for r, _, _ in results.values()]
    cmd = [
        "sftp",
        "-i",
        str(key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-b",
        "-",
        f"{SFTP_USER}@{SFTP_HOST}",
    ]
    if dry_run:
        print(" ".join(cmd))
        print("\n".join(lines))
        return 0
    return subprocess.run(cmd, input="\n".join(lines) + "\n", text=True).returncode


def _key(path):
    key = path or os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise SystemExit(
            f"--upload needs --key or {KEY_ENV}: the private key data_iac lists "
            "for the llk-perf-run SFTP user"
        )
    if not Path(key).expanduser().is_file():
        raise SystemExit(f"key file not found: {key}")
    return Path(key).expanduser()


def _cmd_publish(a):
    # --commit is for a run built from another checkout, so this one says
    # nothing about it; the caller vouches for the commit instead.
    sha, problems = (a.commit, []) if a.commit else checkout_state()
    if not sha:
        raise SystemExit("; ".join(problems))
    if problems and a.upload:
        raise SystemExit("not uploading: " + "; ".join(problems))
    for p in problems:
        print(f"warning: {p}. The file is written but must not be uploaded.")
    try:
        out, n = publish(
            a.run_dir,
            a.out_dir,
            a.arch,
            commit_sha=sha,
            platform=a.platform,
            platform_version=a.platform_version,
            timestamp=a.timestamp,
        )
    except ValueError as e:
        raise SystemExit(f"publish: {e}")
    print(f"publish: wrote {out} from {n} CSV(s), commit {sha}, {a.platform}")
    if not a.platform_version:
        print("warning: no --platform-version; runs on two builds will look alike")
    if not a.upload:
        print("publish: not uploaded (pass --upload to send it)")
        return 0
    return upload([out], _key(a.key), dry_run=a.dry_run)


def _cmd_backfill(a):
    try:
        runs = plan_backfill(a.archive, a.arch, a.platform, a.platform_version)
    except ValueError as e:
        raise SystemExit(f"backfill: {e}")
    if not runs:
        raise SystemExit(f"backfill: no run directories with CSVs under {a.archive}")
    report = migrate_runs(runs, a.out_dir, overwrite=a.overwrite)
    print(summarize_coverage(report))
    for run in runs:
        if run.commit_sha == "unknown":
            print(f"warning: {run.run_id}: no commit_sha; it loads as 'unknown'")
        if not run.platform_version:
            print(f"warning: {run.run_id}: no platform_version")
    failed = [r for r, e in report.items() if e.get("failed")]
    print(
        f"backfill: {len(runs) - len(failed)} of {len(runs)} run(s) written to "
        f"{a.out_dir}. Read the report before you upload: a dropped column is data "
        "the warehouse will not get."
    )
    return 1 if failed else 0


def _cmd_upload(a):
    key = _key(a.key)
    results = check_files(a.files)
    if _report_checks(results):
        raise SystemExit("upload: not sending anything; fix the files above")
    return upload(a.files, key, dry_run=a.dry_run)


def _cmd_check(a):
    results = check_files(a.files)
    bad = _report_checks(results)
    rows = sum(r for _, r, _ in results.values())
    print(f"check: {len(results) - bad} of {len(results)} file(s) pass, {rows} rows")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("publish", help="publish one local run")
    p.add_argument(
        "--run-dir",
        default=str(_LLK_ROOT / "perf_data" / "latest"),
        help="one run's directory (default: perf_data/latest)",
    )
    p.add_argument("--arch", default="quasar", choices=_VALID_ARCHES)
    p.add_argument(
        "--out-dir",
        default=str(_LLK_ROOT / "perf_data" / "publish"),
        help="where to write",
    )
    p.add_argument(
        "--commit", help="override the commit (default: HEAD of this checkout)"
    )
    p.add_argument("--timestamp", help="override the run start, ISO-8601 UTC")
    p.add_argument(
        "--platform", required=True, choices=PLATFORMS, help="what executed the run"
    )
    p.add_argument("--platform-version", help="its build, e.g. the emulator image")
    p.add_argument(
        "--upload", action="store_true", help="also send it to the warehouse"
    )
    p.add_argument("--key", help=f"SFTP private key (default: ${KEY_ENV})")
    p.add_argument(
        "--dry-run", action="store_true", help="print the sftp batch, send nothing"
    )
    p.set_defaults(func=_cmd_publish)

    b = sub.add_parser("backfill", help="convert an archive of old runs")
    b.add_argument("--archive", required=True, help="one sub-directory per run")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--arch", default="quasar", choices=_VALID_ARCHES)
    b.add_argument("--overwrite", action="store_true", help="rewrite existing files")
    b.add_argument(
        "--platform", choices=PLATFORMS, help="for runs whose run_meta.json omits it"
    )
    b.add_argument("--platform-version", help="for runs whose run_meta.json omits it")
    b.set_defaults(func=_cmd_backfill)

    c = sub.add_parser("check", help="apply the loader's rules; send nothing")
    c.add_argument("files", nargs="+")
    c.set_defaults(func=_cmd_check)

    u = sub.add_parser("upload", help="send Parquet files to the warehouse")
    u.add_argument("files", nargs="+")
    u.add_argument("--key", help=f"SFTP private key (default: ${KEY_ENV})")
    u.add_argument(
        "--dry-run", action="store_true", help="print the sftp batch, send nothing"
    )
    u.set_defaults(func=_cmd_upload)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
