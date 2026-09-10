# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unpack a GitHub run-attempt log archive into the <job_id>.log layout this pipeline expects.

GitHub serves every job log for a run attempt as one zip:

    GET /repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/logs

which replaces the one-request-per-job loop download_cicd_logs_and_artifacts.sh used to
run. That loop made this pipeline's API cost scale with the size of the run being
analyzed -- 129 requests for a 129-job merge-gate run, against the repository's shared
15,000/hr GITHUB_TOKEN budget, ~173 times an hour.

Everything downstream reads <job_id>.log, so archive entries have to be mapped back onto
job ids. Entry names are the job name with "/" rewritten to "_" and nothing else changed
-- measured exact on 85/85 entries of run 34360282367 -- so that substitution is the
primary match. Emoji and characters like "[", "]", "(" and "," survive byte-for-byte in
the central directory; the "?" that `unzip -l` shows for them is its own display
rendering, not archive content.

A normalized comparison (lowercased, everything but [a-z0-9] dropped) runs as a second
tier in case GitHub ever changes the substitution. It is deliberately not the primary
match: it maps distinct job names onto the same key ("a/b" and "a-b" both become "ab"),
and a wrong match here is worse than a missing one, because it files a job's failure
signature and runner telemetry under a different job.

So any name -- at either tier -- that does not resolve to exactly one job is left
unmapped on purpose. The caller falls back to a per-job request for those, which costs
one request and makes misattribution impossible rather than merely unobserved.

Reading the zip directly, rather than shelling out to `unzip`, also keeps archive-supplied
names off the filesystem entirely: the only paths written are <job_id>.log. `unzip` can
fail to create an entry whose name contains emoji, and having no tty to answer the
"continue?" prompt that follows, it aborts and leaves a silently partial extraction --
observed truncating at 46 of 85 job logs.
"""

import argparse
import json
import pathlib
import re
import sys
import zipfile
from collections import Counter, defaultdict

# Top-level archive entries are the whole-job logs, named "<ordinal>_<job name>.txt".
# The per-job subdirectories below them hold the same content split per step.
TOP_LEVEL_JOB_LOG = re.compile(r"^(?P<ordinal>\d+)_(?P<name>.*)\.txt$")

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def archive_name_for(job_name: str) -> str:
    """The entry name GitHub gives a job's log."""
    return job_name.replace("/", "_")


def normalize(name: str) -> str:
    return _NON_ALNUM.sub("", name.lower())


def _unique_index(jobs: list, key) -> dict:
    """key(job name) -> job id, holding only keys that exactly one job produces.

    Keys claimed by more than one job are dropped rather than resolved, so an ambiguous
    name can never be silently attributed to whichever job happened to be listed first.
    """
    grouped = defaultdict(list)
    for job in jobs:
        job_id = job.get("id")
        name = job.get("name")
        if job_id is None or not name:
            continue
        grouped[key(str(name))].append(int(job_id))
    return {k: ids[0] for k, ids in grouped.items() if len(ids) == 1}


def resolve_entries(archive: zipfile.ZipFile, jobs: list) -> dict:
    """Archive entry name -> job id, for entries that map to exactly one job.

    Both directions are checked for ambiguity: a name matching several jobs is dropped by
    _unique_index, and a job claimed by several entries is dropped here. What survives is
    a strict one-to-one mapping.
    """
    by_archive_name = _unique_index(jobs, archive_name_for)
    by_normalized = _unique_index(jobs, normalize)

    resolved = {}
    for entry in archive.infolist():
        if entry.is_dir() or "/" in entry.filename:
            continue
        matched = TOP_LEVEL_JOB_LOG.match(entry.filename)
        if not matched:
            continue
        name = matched.group("name")

        job_id = by_archive_name.get(name)
        if job_id is None:
            job_id = by_normalized.get(normalize(name))
        if job_id is None:
            continue
        resolved[entry.filename] = job_id

    claimed_more_than_once = {job_id for job_id, count in Counter(resolved.values()).items() if count > 1}
    return {entry: job_id for entry, job_id in resolved.items() if job_id not in claimed_more_than_once}


def extract(archive_path: pathlib.Path, jobs: list, logs_dir: pathlib.Path) -> int:
    with zipfile.ZipFile(archive_path) as archive:
        resolved = resolve_entries(archive, jobs)
        for entry_name, job_id in resolved.items():
            # Written under an id we chose, so no archive-supplied name -- and no emoji
            # or overlong path -- ever reaches the filesystem.
            with archive.open(entry_name) as source, open(logs_dir / f"{job_id}.log", "wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    return len(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=pathlib.Path)
    parser.add_argument("--jobs-json", required=True, type=pathlib.Path)
    parser.add_argument("--logs-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if not args.logs_dir.is_dir():
        print(f"[Warning] logs dir does not exist: {args.logs_dir}", file=sys.stderr)
        return 1

    try:
        jobs = json.loads(args.jobs_json.read_text()).get("jobs") or []
    except (OSError, ValueError) as exc:
        print(f"[Warning] could not read jobs payload: {exc}", file=sys.stderr)
        return 1

    try:
        written = extract(args.archive, jobs, args.logs_dir)
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"[Warning] could not unpack the attempt log archive: {exc}", file=sys.stderr)
        return 1

    # No logs at all means the archive was not usable; let the caller fall back rather
    # than proceed with an empty logs dir, which the parsers downstream assert against.
    if written == 0:
        print("[Warning] attempt log archive contained no recognizable job logs", file=sys.stderr)
        return 1

    print(f"[info] recovered {written} job logs from the archive in 1 request")
    return 0


if __name__ == "__main__":
    sys.exit(main())
