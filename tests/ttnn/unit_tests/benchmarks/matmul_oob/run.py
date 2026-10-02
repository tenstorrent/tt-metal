#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Matmul default-config benchmarks by suite: legacy selection vs ttnn.CONFIG.matmul_auto_config_v2.

  python tests/ttnn/unit_tests/benchmarks/matmul_oob/run.py --suite SUITE [--out DIR]

Suites (the fast ones take a few minutes each on Wormhole):
  gist             the #57884 gist sweeps (gist/): 116 2D-routed + 64 1D-routed Llama shapes, wall time, with PCC
  gist-fast        the same without the PCC check
  gist-device      the gist sweeps' 180 shapes as benchmark cases (device kernel time, PCC), like validation
  validation       every case in cases.csv (device kernel time, PCC)
  validation-fast  the cases in cases_fast.csv (41 cases across the tiers)
  pytest           the matmul pytest directory, flag off and on (outcome and device time per test)
  pytest-fast      every 10th test of it
  pytest-auto      the tests in pytest_auto_tests.txt: the matmul pytest tests where some matmul uses the default
                   config selection (the rest pass their own program configs, so the flag cannot change them)
  all              validation, gist-device and pytest (every suite that reports device kernel time)

Results go to generated/matmul_oob/<arch>_<git rev>/<suite>/, with a summary.txt. Rerunning a suite skips the
parts that finished. Run from the repo root with python_env active. Environment variables such as MM_KCAP are
passed through to the runs.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = [
    "gist",
    "gist-fast",
    "gist-device",
    "validation",
    "validation-fast",
    "pytest",
    "pytest-fast",
    "pytest-auto",
    "all",
]
AUTO_TESTS = f"{HERE}/pytest_auto_tests.txt"


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def arch_and_rev():
    probe = (
        "import ttnn; d = ttnn.open_device(device_id=0); "
        "print('ARCH=' + str(d.arch()).split('.')[-1].lower()); ttnn.close_device(d)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout
    arch = next((l.split("=", 1)[1].strip() for l in out.splitlines() if l.startswith("ARCH=")), "unknown")
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return arch, rev


def run(cmd, log, env=None):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    zone_log = "generated/profiler/.logs/zone_src_locations.log"  # stale copies make the profiler abort
    if os.path.exists(zone_log):
        os.remove(zone_log)
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}).returncode


def gist(out, fast):
    lines = []
    for suite in ("2d", "1d"):
        rows = {}
        for mode in ("legacy", "v2"):
            csv_path = f"{out}/gist_{suite}_{mode}.csv"
            if not os.path.exists(csv_path):
                cmd = [sys.executable, f"{HERE}/gist/time_default.py", suite, csv_path]
                cmd += (["--v2"] if mode == "v2" else []) + (["--no-pcc"] if fast else [])
                if run(cmd, f"{out}/gist_{suite}_{mode}.log") != 0:
                    lines.append(f"{suite} {mode}: failed, see gist_{suite}_{mode}.log")
                    continue
            rows[mode] = {r["operands"]: r for r in csv.DictReader(open(csv_path))}
        if len(rows) < 2:
            continue
        legacy, v2 = rows["legacy"], rows["v2"]
        common = [k for k in v2 if k in legacy and legacy[k]["ms"] and v2[k]["ms"]]
        sp = [float(legacy[k]["ms"]) / float(v2[k]["ms"]) for k in common]
        worst = sorted(zip(sp, common))[:5]
        fb = sum(v2[k].get("fell_back") == "True" for k in v2)
        pccs = [float(v2[k]["pcc"]) for k in v2 if v2[k]["pcc"] not in ("", "nan")]
        lines.append(
            f"{suite}: {len(sp)} shapes, v2 over legacy geomean {geomean(sp):.3f}x, worst {min(sp):.2f}x, "
            f"faster >5% {sum(x > 1.05 for x in sp)}, slower >5% {sum(x < 0.95 for x in sp)}, v2 fallbacks {fb}"
            + (f", min v2 PCC {min(pccs):.5f}" if pccs else "")
        )
        lines += [f"    {x:.2f}x {k}" for x, k in worst]
    return lines


def validation(out, fast):
    cases = f"{HERE}/cases_fast.csv" if fast else f"{HERE}/cases.csv"
    return run_cases(out, ["--cases-csv", cases])


def run_cases(out, selection):
    csv_path = f"{out}/suite.csv"
    rc = run(
        [
            sys.executable,
            f"{HERE}/run_suite.py",
            *selection,
            "--modes",
            "oob",
            "v2",
            "--out",
            csv_path,
            "--resume",
        ],
        f"{out}/suite.log",
    )
    summary = subprocess.run(
        [sys.executable, f"{HERE}/summarize.py", csv_path, "--base-mode", "oob", "--new-mode", "v2"],
        capture_output=True,
        text=True,
    ).stdout
    return ([f"run_suite exit {rc}"] if rc else []) + summary.splitlines()


def pytest(out, sample=1, tests_file=None):
    lines = []
    for mode, overrides in (("off", "{}"), ("on", '{"matmul_auto_config_v2": true}')):
        jsonl = f"{out}/pytest_{mode}.jsonl"
        done = f"{out}/pytest_{mode}.done"
        if os.path.exists(done):
            continue
        if os.path.exists(jsonl):
            os.remove(jsonl)
        env = {
            "PYTHONPATH": f"{HERE}:{os.environ.get('PYTHONPATH', '')}",
            "TTNN_CONFIG_OVERRIDES": overrides,
            "DEVICE_TIME_OUT": jsonl,
            "DEVICE_TIME_SAMPLE": str(sample),
        }
        if tests_file:
            env["DEVICE_TIME_TESTS"] = tests_file
        rc = run(
            [
                "pytest",
                "-q",
                "-p",
                "no:logging",
                "-p",
                "pytest_device_time",
                "tests/ttnn/unit_tests/operations/matmul/",
            ],
            f"{out}/pytest_{mode}.log",
            env,
        )
        open(done, "w").write(f"exit {rc}\n")
    compare = [sys.executable, f"{HERE}/compare_pytest_times.py", f"{out}/pytest_off.jsonl", f"{out}/pytest_on.jsonl"]
    every = subprocess.run(compare, capture_output=True, text=True).stdout
    auto = subprocess.run(
        compare + ["--auto-only", "--write-auto-list", f"{out}/auto_tests.txt"], capture_output=True, text=True
    ).stdout
    if tests_file:
        return lines + auto.splitlines()[:18]
    return lines + ["every test:"] + every.splitlines()[:16] + ["", "default selection only:"] + auto.splitlines()[:18]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=SUITES)
    ap.add_argument("--out", help="default generated/matmul_oob/<arch>_<rev>")
    args = ap.parse_args()
    arch, rev = arch_and_rev()
    base = args.out or f"generated/matmul_oob/{arch}_{rev}"
    parts = {
        "gist": lambda o: gist(o, False),
        "gist-fast": lambda o: gist(o, True),
        "gist-device": lambda o: run_cases(o, ["--tiers", "gist"]),
        "validation": lambda o: validation(o, False),
        "validation-fast": lambda o: validation(o, True),
        "pytest": lambda o: pytest(o),
        "pytest-fast": lambda o: pytest(o, sample=10),
        "pytest-auto": lambda o: pytest(o, tests_file=AUTO_TESTS),
    }
    if args.suite == "pytest-auto" and not os.path.exists(AUTO_TESTS):
        sys.exit(f"{AUTO_TESTS} is missing: run --suite pytest and copy its pytest/auto_tests.txt there")
    names = ["validation", "gist-device", "pytest"] if args.suite == "all" else [args.suite]
    for name in names:
        out = f"{base}/{name}"
        os.makedirs(out, exist_ok=True)
        extra = {k: v for k, v in os.environ.items() if k.startswith("MM_")}
        header = [f"suite {name}, arch {arch}, rev {rev}" + (f", env {json.dumps(extra)}" if extra else "")]
        lines = header + parts[name](out)
        open(f"{out}/summary.txt", "w").write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
        print(f"results: {out}", flush=True)


if __name__ == "__main__":
    main()
