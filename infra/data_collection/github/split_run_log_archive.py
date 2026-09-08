#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Split a GitHub Actions run log archive into the per-job log files the analysis expects.

Downloading job logs one at a time is the single largest consumer of tt-metal's
GITHUB_TOKEN budget: one API call per job, against a 15,000 calls/hour/repository limit
that the produce-data workflow alone was using roughly two thirds of at peak.

`GET /repos/{owner}/{repo}/actions/runs/{run_id}/attempts/{n}/logs` returns every job log
for a run in a single ZIP, for a single API call. This script maps the entries in that
archive back to job ids so that everything downstream -- which keys on job id -- is
unchanged.

Archive layout, verified against run 34261922962 (125 jobs, 80 of which ran):

    <index>_<job name>.txt      one per job that produced a log
    <job name>/<step>.txt       per-step logs, which we do not use

Only "/" is rewritten, to "_"; every other character, including non-ASCII, is preserved
verbatim. (`unzip -l` renders non-ASCII as "?", which is a display artifact of unzip and
not what the archive contains.) Jobs that were skipped never produced a log and are simply
absent from the archive, which matches what we want to collect anyway.

Job ids that could not be resolved from the archive are written to stdout, one per line,
so the caller can fall back to a per-job download for just those and stay correct even if
GitHub changes the naming rules.
"""

import argparse
import json
import pathlib
import re
import sys
import zipfile

# Entries look like "12_build / 🛠️ Build Release ubuntu 22.04.txt"; the numeric prefix is
# the job's position in the run and is not stable enough to key on, so it is stripped and
# the remainder matched against the job name.
_ENTRY_PREFIX = re.compile(r"^\d+_")


def _normalize(job_name):
    """Render a job name the way GitHub names its archive entry."""
    return job_name.replace("/", "_")


def _jobs_expected_to_have_logs(jobs_json_path):
    """Job id -> name, for jobs that actually ran.

    Skipped jobs (and jobs with no conclusion) never produced a log, so they are neither
    expected in the archive nor worth falling back to a per-job download for.
    """
    with open(jobs_json_path, encoding="utf-8") as f:
        payload = json.load(f)

    jobs = {}
    for job in payload.get("jobs", []):
        conclusion = job.get("conclusion")
        if conclusion is None or conclusion == "skipped":
            continue
        jobs[int(job["id"])] = job["name"]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="run log archive ZIP")
    parser.add_argument("--jobs-json", required=True, help="workflow_jobs.json for the same run+attempt")
    parser.add_argument("--out-dir", required=True, help="directory to write <job_id>.log into")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = _jobs_expected_to_have_logs(args.jobs_json)

    # Names are matched, not positions, so a name shared by two jobs in one run is
    # ambiguous. It has not been observed (0 collisions across 1,127 jobs in 10 runs), but
    # if it happens both jobs fall back to a per-job download rather than risk attributing
    # a log to the wrong job.
    name_to_ids = {}
    for job_id, name in jobs.items():
        name_to_ids.setdefault(_normalize(name), []).append(job_id)

    try:
        archive = zipfile.ZipFile(args.archive)
    except (zipfile.BadZipFile, OSError) as e:
        # A non-ZIP body here usually means an API error was written to the file. Report
        # every job as unresolved so the caller downloads them individually.
        print(f"[Warning] Unusable run log archive ({e}); falling back to per-job downloads", file=sys.stderr)
        for job_id in sorted(jobs):
            print(job_id)
        return 0

    resolved = set()
    with archive:
        for entry in archive.namelist():
            # Per-step logs live under "<job name>/"; only the top-level per-job files are
            # the whole-job log that the per-job API endpoint would have returned.
            if "/" in entry or not entry.endswith(".txt"):
                continue

            name = _ENTRY_PREFIX.sub("", entry[: -len(".txt")])
            candidates = name_to_ids.get(name)
            if not candidates or len(candidates) > 1:
                continue

            job_id = candidates[0]
            if job_id in resolved:
                continue

            # Written as bytes so the file is byte-identical to what the per-job endpoint
            # returns, including the UTF-8 BOM and CRLF line endings the log carries.
            (out_dir / f"{job_id}.log").write_bytes(archive.read(entry))
            resolved.add(job_id)

    unresolved = sorted(set(jobs) - resolved)
    print(
        f"[info] run log archive supplied {len(resolved)} of {len(jobs)} job logs in 1 API call"
        + (f"; {len(unresolved)} need a per-job download" if unresolved else ""),
        file=sys.stderr,
    )
    for job_id in unresolved:
        print(job_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
