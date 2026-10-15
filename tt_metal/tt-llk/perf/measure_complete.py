# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decide whether one arch's measurement is complete, from its own legs.

Run it by path: ``tt_metal/tt-llk`` is not an importable package.
"""

import argparse
import json
import re
from datetime import datetime

_OK = ("", "success", "skipped")


def legs_of(jobs, arch):
    """The measuring legs of ``arch``: jobs named ``llk_perf[_<suite>]_<arch> group i/n``."""
    leg = re.compile(rf"\bllk_perf(?:_[a-z_]+)?_{re.escape(arch)} group \d+/\d+")
    return [j for j in jobs if leg.search(j.get("name") or "")]


def _minutes(job):
    """Whole minutes the job ran, or None when GitHub did not give both times."""
    try:
        start, end = (
            datetime.fromisoformat(job[k].replace("Z", "+00:00"))
            for k in ("started_at", "completed_at")
        )
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return int((end - start).total_seconds() // 60)


def _describe(job):
    """``<leg>: <conclusion> after N min``, without the caller's ``... / `` prefix."""
    conclusion = job.get("conclusion") or "in_progress"
    line = f"{job['name'].rsplit(' / ', 1)[-1]}: {conclusion}"
    minutes = _minutes(job)
    if minutes is not None:
        line += f" after {minutes} min"
    # The gate does not run in a cancelled run, so a cancelled leg most likely ran out of time.
    if conclusion == "cancelled":
        line += ", most likely at its timeout"
    return line


def measure_complete(results, jobs, arch):
    """``(complete, reason)``: the arch's own legs decide; without legs, the job results."""
    legs = legs_of(jobs or [], arch)
    if legs:
        failed = [_describe(j) for j in legs if j.get("conclusion") != "success"]
        if failed:
            return (
                False,
                f"{len(failed)} of {len(legs)} {arch} legs did not pass: "
                + "; ".join(failed),
            )
        return True, f"all {len(legs)} {arch} legs passed"
    bad = [
        r.strip() for r in (results or "").split(",") if r.strip().lower() not in _OK
    ]
    if bad:
        return (
            False,
            f"no {arch} legs found, and the measuring jobs ended {', '.join(bad)}",
        )
    return True, "the measuring jobs succeeded or were skipped"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", required=True)
    ap.add_argument(
        "--results", default="", help="measuring job results, comma-separated"
    )
    ap.add_argument(
        "--jobs",
        help="JSON lines of {name, conclusion[, started_at, completed_at]}; missing = unknown",
    )
    a = ap.parse_args(argv)
    jobs = None
    if a.jobs:
        try:
            with open(a.jobs) as fh:
                jobs = [json.loads(line) for line in fh if line.strip()]
        except (OSError, ValueError) as e:
            print(
                f"::warning::Could not read the jobs ({e}); using the job results only."
            )
    complete, reason = measure_complete(a.results, jobs, a.arch)
    print(f"complete={'true' if complete else 'false'}")
    print(f"legs={len(legs_of(jobs or [], a.arch))}")
    print(f"reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
