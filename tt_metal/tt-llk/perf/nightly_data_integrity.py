# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check the baseline each nightly gate read, and the perf data each nightly wrote.

Reads only GitHub artifacts and gate logs: the health job has no warehouse secret.
"""

import argparse
import collections
import csv
import datetime as dt
import io
import json
import os
import pathlib
import re
import runpy
import subprocess
import sys
import time
import zipfile

from nightly_health import (
    ARCHES,
    NIGHTLIES,
    _gh,
    _is_gate,
    _json_lines,
    gate_result,
    job_log,
    run_jobs,
    scheduled_run,
)

COUNTS = re.compile(r"(\d+) points compared, .*?(\d+) new point")
# baseline_from_warehouse.py: "baseline baseline-20261020-39432951782-wormhole (85b69f4..., ...".
BASELINE_RUN = re.compile(r"Z baseline \S*?-(\d{6,})-[a-z]+ \(")
MAX_AGE_DAYS = 7
MIN_COMPARED = 0.95
# Non-SoL more than 5 % and 100 cycles faster than SoL at the same point.
SOL_SLACK = 0.05
SOL_MIN_CYCLES = 100
# One setup cost per kernel: these markers are the same in every config by design.
SETUP_MARKERS = ("INIT", "UNINIT")
STUCK_MIN_CONFIGS = 10
NOT_KEY = ("mean(", "std(", "TEXT_SIZE(")
SHORT = {"blackhole": "BH", "wormhole": "WH"}
SHOW = 4


def artifacts(repo, run_id):
    return {
        a["name"]: a["id"]
        for a in _json_lines(
            _gh(
                "api",
                "--paginate",
                f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
                "--jq",
                ".artifacts[] | select(.expired | not) | {name, id} | @json",
            )
        )
    }


def artifact_zip(repo, artifact_id):
    # Blob storage sometimes refuses a download ("Egress is over the account limit").
    for wait in (30, None):
        got = subprocess.run(
            ["gh", "api", f"repos/{repo}/actions/artifacts/{artifact_id}/zip"],
            capture_output=True,
        )
        if got.returncode == 0:
            return zipfile.ZipFile(io.BytesIO(got.stdout))
        print(f"artifact {artifact_id}: {got.stderr.decode().strip()}", file=sys.stderr)
        if wait:
            time.sleep(wait)
    return None


def perf_zips(repo, arts, arch):
    """The arch's perf-data zips, and the names of those that did not download."""
    zips = {
        n: artifact_zip(repo, i)
        for n, i in sorted(arts.items())
        if n.startswith(f"perf-data-{arch}-")
    }
    return [z for z in zips.values() if z], [n for n, z in zips.items() if not z]


# Renamed tests keep their old name in the baseline; regression_compare.py maps it.
_ALIASES = pathlib.Path(__file__).with_name("test_name_aliases.py")
canonical = (
    runpy.run_path(str(_ALIASES))["canonical_test_name"]
    if _ALIASES.exists()
    else lambda name: name
)


def _module(name):
    # The rule regression_compare.py and the warehouse use for TEST_NAME.
    base = re.sub(r"\.(?:post|counters)?\.?csv$", "", os.path.basename(name))
    return canonical(base.rstrip("."))


def _csv_names(zips):
    return [
        (zf, n)
        for zf in zips
        for n in zf.namelist()
        if n.endswith(".csv") and not n.endswith(".post.csv")
    ]


def _cycles(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_perf(zips):
    """``{(module, key): {metric: text}}`` plus the bad values and duplicate keys per module."""
    data, bad, dups = {}, collections.Counter(), collections.Counter()
    for zf, name in _csv_names(zips):
        module = _module(name)
        with zf.open(name) as fh:
            for row in csv.DictReader(io.TextIOWrapper(fh, newline="")):
                metrics = {k: v for k, v in row.items() if k and k.startswith("mean(")}
                key = (
                    module,
                    tuple(
                        sorted(
                            (k, v)
                            for k, v in row.items()
                            if k and v and not k.startswith(NOT_KEY)
                            # The two nightlies differ only in this column.
                            and k != "speed_of_light"
                        )
                    ),
                )
                if key in data:
                    dups[module] += 1
                data[key] = metrics
                bad[module] += sum(
                    1 for v in metrics.values() if v and not (_cycles(v) or 0) > 0
                )
    return data, +bad, dups


def stuck(data):
    """Modules where one marker and run type give one value in every config."""
    series = collections.defaultdict(set)
    sizes = collections.Counter()
    for (module, key), metrics in data.items():
        marker = dict(key).get("marker")
        if marker in SETUP_MARKERS:
            continue
        for metric, v in metrics.items():
            if v:
                series[(module, marker, metric)].add(v)
                sizes[(module, marker, metric)] += 1
    return collections.Counter(
        s[0]
        for s, values in series.items()
        if len(values) == 1 and sizes[s] >= STUCK_MIN_CONFIGS
    )


def below_sol(sol, non_sol):
    hits = collections.Counter()
    for key, metrics in non_sol.items():
        bound = sol.get(key)
        if not bound:
            continue
        for metric, v in metrics.items():
            v, s = _cycles(v), _cycles(bound.get(metric))
            if v and s and v < s * (1 - SOL_SLACK) and s - v > SOL_MIN_CYCLES:
                hits[key[0]] += 1
    return hits


def _top(counter):
    shown = ", ".join(f"{m} {n:,}" for m, n in counter.most_common(SHOW))
    return shown + (", …" if len(counter) > SHOW else "")


def _names(names):
    names = sorted(names)
    return ", ".join(names[:SHOW]) + (", …" if len(names) > SHOW else "")


def gate_report(zf):
    """(compared, new) from the report, and the modules with a compared point."""
    names = zf.namelist()
    counts = None
    if "perf_gate_report.md" in names:
        m = COUNTS.search(zf.read("perf_gate_report.md").decode())
        counts = (int(m.group(1)), int(m.group(2))) if m else None
    modules = None
    if "perf_gate_report.points.csv" in names:
        with zf.open("perf_gate_report.points.csv") as fh:
            rows = csv.reader(io.TextIOWrapper(fh, newline=""))
            col = next(rows).index("test_module")
            modules = {canonical(r[col]) for r in rows}
    return counts, modules


def baseline_check(label, gate, counts, compared, measured, baseline_modules):
    """One detail line, and one problem line that names every shortfall."""
    wrong = []
    age = gate["baseline_age"]
    if age is not None and age > MAX_AGE_DAYS:
        wrong.append(f"{age} days old")
    line = f"{label}: baseline `{gate['baseline'] or '?'}`, " + (
        f"{age} days" if age is not None else "age unknown"
    )
    if counts:
        done, new = counts
        share = done / (done + new) if done + new else 0.0
        line += f" · {done:,} of {done + new:,} points compared ({share:.0%})"
        if share < MIN_COMPARED:
            wrong.append(f"{share:.0%} of points compared")
    else:
        line += " · no gate report"
        wrong.append("no gate report")
    if compared is not None:
        unmatched = measured - compared
        line += (
            f" · {len(unmatched)} of {len(measured)} modules without a baseline point"
        )
        if unmatched:
            wrong.append(
                f"{len(unmatched)} module(s) without a baseline point ({_names(unmatched)})"
            )
    if baseline_modules is None:
        line += " · baseline run's data not found"
    else:
        gone = baseline_modules - measured
        line += f" · {len(gone)} baseline module(s) not measured"
        if gone:
            wrong.append(
                f"{len(gone)} baseline module(s) not measured ({_names(gone)})"
            )
    return [f"{label}: " + "; ".join(wrong)] if wrong else [], line


def data_check(label, data, bad, dups):
    problems = []
    for what, counter in (
        ("non-positive or non-numeric value(s)", bad),
        ("duplicate key(s)", dups),
        ("series with one value in every config", stuck(data)),
    ):
        if counter:
            problems.append(
                f"{label}: {sum(counter.values()):,} {what}: {_top(counter)}"
            )
    return problems


def check(repo, day):
    problems, lines = [], []
    runs = {}
    for name, workflow in NIGHTLIES:
        run = scheduled_run(repo, workflow, day.isoformat())
        if run is None:
            lines.append(f"{name}: no scheduled run, nothing to check")
            continue
        logs = {
            j["name"]: job_log(repo, j["id"])
            for j in run_jobs(repo, run["id"])
            if _is_gate(j)
        }
        runs[name] = (artifacts(repo, run["id"]), logs)
    baseline_cache = {}
    for arch in ARCHES:
        perf = {}
        for name, (arts, logs) in runs.items():
            label = f"{name} {SHORT[arch]}"
            zips, failed = perf_zips(repo, arts, arch)
            if failed:
                problems.append(f"{label}: could not download {', '.join(failed)}")
            data, bad, dups = read_perf(zips)
            if not data:
                problems.append(f"{label}: no perf data")
                continue
            perf[name] = data
            problems += data_check(label, data, bad, dups)
            log = next((l for n, l in logs.items() if arch in n), "")
            run_id = BASELINE_RUN.search(log)
            if run_id:
                key = (run_id.group(1), arch)
                if key not in baseline_cache:
                    zips, failed = perf_zips(repo, artifacts(repo, key[0]), arch)
                    baseline_cache[key] = (
                        None
                        if failed
                        else {_module(n) for _, n in _csv_names(zips)} or None
                    )
                baseline_modules = baseline_cache[key]
            else:
                baseline_modules = None
            report = f"llk-perf-gate-report-{arch}"
            zf = artifact_zip(repo, arts[report]) if report in arts else None
            counts, compared = gate_report(zf) if zf else (None, None)
            p, line = baseline_check(
                label,
                gate_result(log, day),
                counts,
                compared,
                {m for m, _ in data},
                baseline_modules,
            )
            problems += p
            lines.append(line)
        if len(perf) == len(NIGHTLIES):
            hits = below_sol(*(perf[n] for n, _ in NIGHTLIES))
            lines.append(
                f"{SHORT[arch]}: {sum(hits.values()):,} non-SoL point(s) more than "
                f"{SOL_SLACK:.0%} below SoL" + (f": {_top(hits)}" if hits else "")
            )
    return {"name": "Perf data integrity", "problems": problems, "lines": lines}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument(
        "--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat()
    )
    ap.add_argument("--out", default="nightly_checks/data_integrity.json")
    a = ap.parse_args(argv)
    report = check(a.repo, dt.date.fromisoformat(a.date))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh)
    print("\n".join(report["problems"] + [""] + report["lines"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
