# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Measure the same build again on the board and report what moved.

``nodes`` writes the sentinel node list and a second fixed order for one arch,
``compare`` counts the points of two perf run directories that are not bit-exact,
and ``report`` turns the nightly's results into a nightly health section.
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zlib

SENTINELS = (
    pathlib.Path(__file__).resolve().parents[1] / "tests/perf_stability_sentinels.json"
)
SKIP = (".post.", ".counters.", ".gw", ".master")
VALUE = ("mean(", "std(", "min(", "max(", "median(", "TEXT_SIZE(")
RUN_TYPES = ("L1_TO_L1_", "UNPACK_ISOLATE_", "MATH_ISOLATE_", "PACK_ISOLATE_")
# Moves past this many keep no test parameters, which keeps a bad night's JSON small.
MAX_PARAMS = 20
# The nightly gate's rules (llk-perf-gate.yaml, nightly-gate): threshold, module thresholds.
RULES = {
    "blackhole": (0.02, {}),
    "wormhole": (0.10, {"perf_matmul": 0.25, "perf_math_matmul": 0.25}),
}
MIN_CYCLES = 30
# Points over the rule that a same-code measure may show before it is a problem.
FLOOR = {"blackhole": 0, "wormhole": 0}
SHORT = {"blackhole": "BH", "wormhole": "WH"}
WORKFLOW = "llk-perf-gate.yaml"
ARTIFACTS = "llk-perf-stability-*"


def write_nodes(arch, out):
    nodes = json.loads(SENTINELS.read_text())[arch]
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "nodes.txt").write_text("".join(f"{n}\n" for n in nodes))
    # The plugin's --test-order-file format; crc32 gives a fixed order unrelated to collection order.
    order = sorted(nodes, key=lambda n: zlib.crc32(n.encode()))
    (out / "order.json").write_text(
        json.dumps({"sentinel": [{"test": n} for n in order]})
    )
    return nodes


def load(root):
    points = {}
    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        if any(s in os.path.basename(path) for s in SKIP):
            continue
        module = os.path.splitext(os.path.basename(path))[0]
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            if "marker" not in cols:
                continue
            keys = [
                c
                for c in cols
                if c != "marker"
                and not c.startswith(VALUE)
                and not c.startswith(RUN_TYPES)
            ]
            for row in reader:
                params = "|".join(f"{c}={row[c]}" for c in keys)
                for c in cols:
                    if c.startswith("mean(") and row[c] not in ("", "nan", "NaN"):
                        run_type = c[5:-1]
                        points[(module, run_type, row["marker"], params)] = (
                            float(row[c]),
                            row.get(f"TEXT_SIZE({run_type})", ""),
                        )
    return points


def compare(a_dir, b_dir):
    a, b = load(a_dir), load(b_dir)
    common = a.keys() & b.keys()
    moved = [k for k in common if a[k][0] != b[k][0]]
    # A move of MIN_CYCLES or less is under every rule; it is counted, not listed.
    moves = sorted(
        ([*k, a[k][0], b[k][0]] for k in moved if abs(b[k][0] - a[k][0]) > MIN_CYCLES),
        key=lambda m: -abs(m[5] - m[4]) / max(m[4], 1e-9),
    )
    return {
        "points": len(common),
        "only_a": len(a.keys() - b.keys()),
        "only_b": len(b.keys() - a.keys()),
        "text_differs": sum(1 for k in common if a[k][1] != b[k][1]),
        "not_bit_exact": len(moved),
        "moves": moves[:MAX_PARAMS]
        + [[*m[:3], "", *m[4:]] for m in moves[MAX_PARAMS:]],
    }


def over_rule(arch, moves):
    threshold, per_module = RULES[arch]
    return [
        m
        for m in moves
        if abs(m[5] - m[4]) > MIN_CYCLES
        and abs(m[5] - m[4]) / max(m[4], 1e-9) > per_module.get(m[0], threshold)
    ]


def passed(junit):
    try:
        root = ET.parse(junit).getroot()
    except (OSError, ET.ParseError):
        return 0
    return sum(
        1
        for case in root.iter("testcase")
        if all(case.find(x) is None for x in ("failure", "error", "skipped"))
    )


def _read(path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return None


def _describe(arch, what, result):
    over = over_rule(arch, result["moves"])
    line = f"{what}: {result['not_bit_exact']:,} of {result['points']:,} points not bit-exact, {len(over)} over the gate rule"
    if over:
        m = over[0]
        line += f" (largest {abs(m[5] - m[4]) / max(m[4], 1e-9) * 100:.1f} %, {m[0]} {m[1]} {m[2]})"
    problems = []
    if len(over) > FLOOR[arch]:
        problems.append(
            f"stability {SHORT[arch]}: {what}, {len(over)} points over the gate rule"
        )
    if result["only_a"] or result["only_b"]:
        problems.append(
            f"stability {SHORT[arch]}: {what}, {result['only_a'] + result['only_b']:,} points measured in one pass only"
        )
    if result["text_differs"]:
        problems.append(
            f"stability {SHORT[arch]}: {what}, {result['text_differs']} points with another TEXT_SIZE"
        )
    return line, problems


def assess(found, day, legs):
    """``found`` maps an arch to ``{group: {file: content}}``; ``legs`` maps an arch to its leg count."""
    lines, problems = [], []
    for arch in RULES:
        short = SHORT[arch]
        groups = found.get(arch, {})
        sentinel = next((g for g in groups.values() if "nodes.txt" in g), None)
        if sentinel is None:
            problems.append(f"stability {short}: no sentinel result")
        else:
            listed = len(sentinel["nodes.txt"].split())
            runs = {p: sentinel.get(f"junit-{p}.xml", 0) for p in "abc"}
            ran = min(runs.values())
            if ran < listed:
                problems.append(
                    f"stability {short}: sentinel ran {ran} of {listed} listed tests"
                )
            for what, key in (
                ("sentinel measured twice", "repeat.json"),
                ("sentinel in another order", "reorder.json"),
            ):
                if sentinel.get(key) is None:
                    problems.append(f"stability {short}: {what}, no result")
                    continue
                line, p = _describe(arch, what, sentinel[key])
                lines.append(f"{short} {line}")
                problems += p
        controls = {
            g: c["control.json"] for g, c in groups.items() if c.get("control.json")
        }
        if controls:
            total = {
                k: sum(c[k] for c in controls.values())
                for k in ("points", "only_a", "only_b", "text_differs", "not_bit_exact")
            }
            total["moves"] = sorted(
                (m for c in controls.values() for m in c["moves"]),
                key=lambda m: -abs(m[5] - m[4]) / max(m[4], 1e-9),
            )
            line, p = _describe(
                arch,
                f"full suite measured twice ({len(controls)}/{legs[arch]} legs)",
                total,
            )
            lines.append(f"{short} {line}")
            problems += p
        if day.isoweekday() == 7 and len(controls) < legs[arch]:
            problems.append(
                f"stability {short}: weekly full-suite control from {len(controls)} of {legs[arch]} legs"
            )
    return {"name": "Stability on the board", "problems": problems, "lines": lines}


def collect(root):
    """Artifacts ``llk-perf-stability-<arch>-<group>`` under ``root`` as ``{arch: {group: {file: content}}}``."""
    found = {}
    for d in sorted(pathlib.Path(root).glob("llk-perf-stability-*")):
        arch, group = d.name[len("llk-perf-stability-") :].split("-", 1)
        files = {}
        for f in d.iterdir():
            if f.name == "nodes.txt":
                files[f.name] = f.read_text()
            elif f.name.startswith("junit-"):
                files[f.name] = passed(f)
            elif f.suffix == ".json" and f.name != "order.json":
                files[f.name] = _read(f)
        found.setdefault(arch, {})[group] = files
    return found


def _health():
    # Only the health job needs it, and it imports yaml, which a leg container may lack.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import nightly_health

    return nightly_health


def download(repo, run_id, dest):
    try:
        _health()._gh(
            "run", "download", str(run_id), "-R", repo, "-p", ARTIFACTS, "-D", str(dest)
        )
    except subprocess.CalledProcessError as e:
        print(
            f"no stability artifacts in run {run_id}: {e.stderr.strip()}",
            file=sys.stderr,
        )


def report(repo, day, tests_yaml):
    health = _health()
    run = health.scheduled_run(repo, WORKFLOW, day.isoformat())
    if run is None or run["status"] != "completed":
        # The nightly's own section already reports a missing or unfinished run.
        return {
            "name": "Stability on the board",
            "problems": [],
            "lines": ["no finished non-SoL nightly to read"],
        }
    legs = {
        a: sum(1 for e in health._entries(tests_yaml) if f"_{a} " in e["name"])
        for a in RULES
    }
    with tempfile.TemporaryDirectory() as tmp:
        download(repo, run["id"], tmp)
        return assess(collect(tmp), day, legs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nodes", help="write nodes.txt and order.json for one arch")
    n.add_argument("--arch", required=True, choices=sorted(RULES))
    n.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="compare two perf run directories")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", required=True)
    r = sub.add_parser("report", help="write the nightly health section")
    r.add_argument("--repo", required=True)
    r.add_argument(
        "--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat()
    )
    r.add_argument("--tests-yaml", default="tests/pipeline_reorg/llk_perf_tests.yaml")
    r.add_argument("--out", default="nightly_checks/stability.json")
    a = ap.parse_args(argv)
    if a.cmd == "nodes":
        print(f"{len(write_nodes(a.arch, a.out))} sentinel tests for {a.arch}")
        return 0
    if a.cmd == "compare":
        result = compare(a.a, a.b)
        pathlib.Path(a.out).write_text(json.dumps(result))
        print(f"{result['not_bit_exact']} of {result['points']} points not bit-exact")
        return 0
    result = report(a.repo, dt.date.fromisoformat(a.date), a.tests_yaml)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(result))
    print("\n".join([*result["problems"], *result["lines"]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
