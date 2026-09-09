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

The archive names entries after job and step names, but everything downstream reads
<job_id>.log, so entries have to be mapped back onto job ids. Two properties of the
archive make that worth doing in Python rather than with `unzip`:

  * Step names appear verbatim in entry paths, emoji included. `unzip` can fail to
    create such a name, and because it then prompts to continue and gets no tty in CI,
    it aborts the whole extraction and leaves a silently partial result.
  * Entry names contain characters that are glob-special to `unzip`'s own pattern
    matching (`[`, `]`), so selecting entries by name is unreliable.

Reading the zip directly avoids both: names are matched in memory and the only paths
created on disk are ones we choose.

Job names are matched to entry names on an aggressively normalized form (lowercased,
everything but [a-z0-9] dropped) because the archive flattens characters that are
illegal in a filename -- most visibly the "/" in "<caller job> / <called job>" -- and
guessing GitHub's exact substitution table would be a standing source of breakage.
"""

import argparse
import json
import pathlib
import re
import sys
import zipfile
from collections import defaultdict

# Top-level archive entries are the whole-job logs, named "<ordinal>_<job name>.txt".
# The per-job subdirectories below them hold the same content split per step.
TOP_LEVEL_JOB_LOG = re.compile(r"^(\d+)_(?P<name>.*)\.txt$")

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def normalize(name: str) -> str:
    return _NON_ALNUM.sub("", name.lower())


def build_name_index(jobs: list) -> dict:
    """normalized job name -> list of job ids, most recent id last.

    A list rather than a single id because matrix legs can share a name; ids are handed
    out one per matching entry so two same-named jobs land in two different files.
    """
    index = defaultdict(list)
    for job in jobs:
        job_id = job.get("id")
        name = job.get("name")
        if job_id is None or not name:
            continue
        index[normalize(str(name))].append(int(job_id))
    return index


def extract(archive_path: pathlib.Path, jobs: list, logs_dir: pathlib.Path) -> int:
    index = build_name_index(jobs)
    written = 0

    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or "/" in entry.filename:
                continue
            matched = TOP_LEVEL_JOB_LOG.match(entry.filename)
            if not matched:
                continue

            candidates = index.get(normalize(matched.group("name")))
            if not candidates:
                # Left for the per-job fallback in the shell script, so an unmapped
                # entry costs one request rather than losing the log.
                continue
            job_id = candidates.pop(0)

            # Written by us under an id we chose, so no archive-supplied name -- and no
            # emoji or overlong path -- ever reaches the filesystem.
            with archive.open(entry) as source, open(logs_dir / f"{job_id}.log", "wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            written += 1

    return written


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
