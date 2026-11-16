# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stability checks from nightly perf data that already exists.

Quiet day: when no perf input changed between the previous good nightly and
today's, both runs measured the same code, so every point must be bit-exact.
Drift: per module, the median ratio of today's points to a pinned reference
run, which catches many small moves that each pass the gate.
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
import statistics
import subprocess
import zipfile

NIGHTLIES = (
    ("SoL", "llk-perf.yaml"),
    ("non-SoL", "llk-perf-gate.yaml"),
)
ARCHES = (("blackhole", "BH"), ("wormhole", "WH"))
REFERENCE = pathlib.Path(__file__).with_name("drift_reference.json")
INPUTS_SCRIPT = ".github/scripts/llk-perf-inputs-changed.sh"
SKIP_PARTS = (".gw", ".master", ".post.", ".counters.")
COUNTER = re.compile(
    r"^(L1_TO_L1|UNPACK_ISOLATE|MATH_ISOLATE|PACK_ISOLATE|L1_CONGESTION)_"
)
# The gate's run type and rule; the isolate run types are not gated.
RUN_TYPE, THRESHOLD, MIN_CYCLES = "L1_TO_L1", 0.02, 30
# Same-code WH runs leave a few points off: 2-4 of ~106k on the merge gate of 11-14..16.
WH_FLOOR = 10
DRIFT = 0.03
REPIN_DAYS, ARTIFACT_DAYS = 80, 90
TOP = 3


def _gh(*args):
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout


def _git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def _json_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def good_runs(repo, workflow, day):
    """Successful scheduled runs created on or before ``day``, newest first."""
    return _json_lines(
        _gh(
            "api",
            f"repos/{repo}/actions/workflows/{workflow}/runs?event=schedule&status=success&created=<={day}&per_page=10",
            "--jq",
            ".workflow_runs[] | {id, head_sha, created_at} | @json",
        )
    )


def fetch(repo, run_id, cache):
    """Unzip the perf CSVs of every perf-data-* artifact of a run; reuse the cache."""
    root = pathlib.Path(cache) / str(run_id)
    if (root / "complete").exists():
        return root
    arts = _json_lines(
        _gh(
            "api",
            f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
            "--jq",
            '.artifacts[] | select(.name | startswith("perf-data-")) | {id, name, expired} | @json',
        )
    )
    if not arts or any(a["expired"] for a in arts):
        return None
    for a in arts:
        data = subprocess.run(
            ["gh", "api", f"repos/{repo}/actions/artifacts/{a['id']}/zip"],
            capture_output=True,
            check=True,
        ).stdout
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base.endswith(".csv") and not any(p in base for p in SKIP_PARTS):
                    out = root / a["name"] / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(name))
    (root / "complete").touch()
    return root


def load(root, arch):
    """``{(module, marker, config): (mean, text_size)}`` of one arch, gate run type only."""
    points = {}
    for path in sorted(pathlib.Path(root).glob(f"perf-data-{arch}-*/**/*.csv")):
        if any(p in path.name for p in SKIP_PARTS):
            continue
        module = path.stem
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            mean = f"mean({RUN_TYPE})"
            if "marker" not in cols or mean not in cols:
                continue
            keys = [
                c
                for c in cols
                if not c.startswith(("mean(", "std(", "min(", "max(", "median("))
                and not c.startswith("TEXT_SIZE(")
                and c != "marker"
                and not COUNTER.match(c)
            ]
            for row in reader:
                if row[mean] in ("", "nan", "NaN"):
                    continue
                ident = "|".join(row[c] for c in keys)
                points[(module, row["marker"], ident)] = (
                    float(row[mean]),
                    row.get(f"TEXT_SIZE({RUN_TYPE})", ""),
                )
    return points


def same_code(a, b):
    common = a.keys() & b.keys()
    off = collections.Counter()
    over = 0
    text = 0
    largest = (0.0, None, 0, 0)
    for k in common:
        (va, ta), (vb, tb) = a[k], b[k]
        if ta and tb and ta != tb:
            text += 1
        if va == vb:
            continue
        off[k[0]] += 1
        if abs(vb - va) > MIN_CYCLES and abs(vb - va) > THRESHOLD * va:
            over += 1
        pct = abs(vb - va) / va * 100 if va else float("inf")
        if pct > largest[0]:
            largest = (pct, k, va, vb)
    return {
        "points": len(common),
        "off": sum(off.values()),
        "over": over,
        "text": text,
        "largest": largest,
        "modules": off.most_common(TOP),
    }


def drift(ref, cur):
    ratios = collections.defaultdict(list)
    for k in ref.keys() & cur.keys():
        if ref[k][0] > 0:
            ratios[k[0]].append(cur[k][0] / ref[k][0])
    return {m: (statistics.median(r), len(r)) for m, r in ratios.items()}


def perf_input_commits(base, head, since, script=INPUTS_SCRIPT):
    """Commits in ``base..head`` that change a WH/BH perf input, as the gate counts them."""
    _git(
        "fetch",
        "--quiet",
        "--no-tags",
        "--no-recurse-submodules",
        "--filter=blob:none",
        f"--shallow-since={since}",
        "origin",
        base,
        head,
    )
    log = _git(
        "log",
        "--format=%x00%H",
        "--name-only",
        "--diff-filter=ACMRT",
        f"{base}..{head}",
    )
    commits = [c.split("\n", 1) for c in log.split("\0") if c.strip()]
    changed = 0
    for c in commits:
        files = c[1] if len(c) > 1 else ""
        verdict = subprocess.run(
            ["bash", script], input=files, capture_output=True, text=True, check=True
        ).stdout
        changed += verdict.strip() == "true"
    return changed, len(commits)


def _key(k):
    return f"{k[0]} {k[1]}"


def quiet_day(name, prev, today, perf_inputs, points):
    """Section lines and problems of the quiet-day check for one nightly."""
    head = f"{name} {prev['created_at'][:10]} → {today['created_at'][:10]}"
    changed, total = perf_inputs
    if changed:
        return [], [
            f"{head}: not a quiet day, {changed} of {total} commits change perf inputs"
        ]
    problems = []
    lines = [f"{head}: quiet day ({total} commits, none change perf inputs)"]
    for arch, short in ARCHES:
        s = points.get(arch)
        if s is None:
            problems.append(f"same code {name} {short}: no perf data")
            continue
        pct, k, va, vb = s["largest"]
        line = f"{short}: {s['off']:,} of {s['points']:,} points not bit-exact, {s['over']:,} over the gate rule"
        if k:
            line += f"; largest {pct:.3g} % ({va:.0f} → {vb:.0f}, {_key(k)})"
        if s["modules"]:
            line += "; " + ", ".join(f"{m} {n}" for m, n in s["modules"])
        lines.append(line)
        if s["text"]:
            problems.append(
                f"same code {name} {short}: {s['text']:,} points have another TEXT_SIZE, so the code differs"
            )
        if short == "BH" and s["over"]:
            problems.append(
                f"same code {name} BH: {s['over']:,} points over the gate rule"
            )
        if short == "WH" and s["off"] > WH_FLOOR:
            problems.append(
                f"same code {name} WH: {s['off']:,} points not bit-exact (floor {WH_FLOOR})"
            )
    return problems, lines


def drift_report(name, ref, day, medians):
    """Section lines and problems of the drift check for one nightly."""
    age = (day - dt.date.fromisoformat(ref["date"])).days
    lines = [
        f"{name}: today vs reference {ref['date']} (`{ref['sha'][:11]}`, {age} days old)"
    ]
    problems = []
    if age >= REPIN_DAYS:
        problems.append(
            f"drift {name}: reference run {ref['run_id']} is {age} days old, its artifacts expire at {ARTIFACT_DAYS}; re-pin {REFERENCE.name}"
        )
    for arch, short in ARCHES:
        m = medians.get(arch)
        if not m:
            problems.append(f"drift {name} {short}: no matched points")
            continue
        movers = sorted(m.items(), key=lambda x: -abs(x[1][0] - 1))[:TOP]
        slow = sorted(
            (mod for mod, (r, _) in m.items() if r > 1 + DRIFT),
            key=lambda mod: -m[mod][0],
        )
        lines.append(
            f"{short}: {len(m)} modules, {sum(n for _, n in m.values()):,} points matched; top movers "
            + ", ".join(f"{mod} {100 * (r - 1):+.1f} %" for mod, (r, _) in movers)
        )
        if slow:
            problems.append(
                f"drift {name} {short}: {len(slow)} module(s) slower than {DRIFT:.0%} by median: "
                + ", ".join(
                    f"{mod} {100 * (m[mod][0] - 1):+.1f} %" for mod in slow[:TOP]
                )
            )
    return problems, lines


def _write(path, name, problems, lines):
    with open(path, "w") as fh:
        json.dump({"name": name, "problems": problems, "lines": lines}, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument(
        "--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat()
    )
    ap.add_argument("--cache", default="perf_data_cache")
    ap.add_argument("--out-dir", default="nightly_checks")
    ap.add_argument("--reference", default=str(REFERENCE))
    a = ap.parse_args(argv)
    day = dt.date.fromisoformat(a.date)
    with open(a.reference) as fh:
        refs = json.load(fh)
    q_problems, q_lines, d_problems, d_lines = [], [], [], []
    for name, workflow in NIGHTLIES:
        runs = good_runs(a.repo, workflow, a.date)
        if not runs or runs[0]["created_at"][:10] != a.date:
            q_lines.append(f"{name}: no good nightly today")
            d_lines.append(f"{name}: no good nightly today")
            continue
        today = runs[0]
        root = fetch(a.repo, today["id"], a.cache)
        if root is None:
            q_problems.append(f"{name}: no perf data in run {today['id']}")
            continue
        if len(runs) > 1:
            prev = runs[1]
            since = (
                dt.date.fromisoformat(prev["created_at"][:10]) - dt.timedelta(days=1)
            ).isoformat()
            inputs = perf_input_commits(prev["head_sha"], today["head_sha"], since)
            stats = {}
            if not inputs[0]:
                prev_root = fetch(a.repo, prev["id"], a.cache)
                for arch, _ in ARCHES:
                    if prev_root:
                        stats[arch] = same_code(load(prev_root, arch), load(root, arch))
            p, l = quiet_day(name, prev, today, inputs, stats)
            q_problems += p
            q_lines += l
        ref = refs.get(workflow)
        ref_root = fetch(a.repo, ref["run_id"], a.cache) if ref else None
        if ref_root is None:
            d_problems.append(f"drift {name}: reference run has no perf data")
            continue
        medians = {
            arch: drift(load(ref_root, arch), load(root, arch)) for arch, _ in ARCHES
        }
        p, l = drift_report(name, ref, day, medians)
        d_problems += p
        d_lines += l
    os.makedirs(a.out_dir, exist_ok=True)
    _write(f"{a.out_dir}/quiet_day.json", "Same code, two nights", q_problems, q_lines)
    _write(f"{a.out_dir}/drift.json", "Drift vs pinned reference", d_problems, d_lines)
    print("\n".join([*q_problems, *q_lines, *d_problems, *d_lines]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
