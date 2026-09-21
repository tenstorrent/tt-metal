# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Serial, fresh-process Gemma SDPA search. Without --run-device, only print the plan."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .sdpa_perf_utils import PREFIXES, Candidate, baseline, family_candidates, tilings

ROOT = Path(__file__).resolve().parents[4]
TARGET = "models/demos/gemma4_d_p/tests/test_sdpa_perf.py::test_sdpa_perf"
FAMILIES = ("exp_approx", "math_approx", "fp32_dest", "packer_l1", "fidelity")
COLUMNS = (
    "layer",
    "isl",
    "phase",
    "config",
    "median_us",
    "p90_us",
    "pcc",
    "rmse",
    "status",
    "error",
    "edge_status",
    "seed",
    "commit",
    "worktree_diff_sha256",
    "harness_sha256",
    "hardware",
    "log",
)


def append_csv(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        record = dict(row)
        record["config"] = json.dumps(row["config"], sort_keys=True)
        writer.writerow(record)


def run_process(command, env, log_path, timeout):
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill only this candidate's process group. Never reset hardware or continue after a hang.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return "timeout"
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def best(rows):
    passing = [row for row in rows if row["status"] == "passed"]
    return min(passing, key=lambda row: row["median_us"]) if passing else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-device", action="store_true", help="Explicitly authorize device use on a reserved idle Galaxy"
    )
    parser.add_argument("--layers", nargs="+", choices=("global", "swa"), default=["global", "swa"])
    parser.add_argument("--prefixes", nargs="+", type=int, choices=PREFIXES, default=list(PREFIXES))
    parser.add_argument("--output", type=Path, default=Path("generated/gemma4_sdpa/sweep"))
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Seconds per candidate, including staging and CPU reference"
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 20 or args.timeout <= 0:
        parser.error("Require at least 20 repetitions and a positive timeout")
    if not args.run_device:
        for layer in args.layers:
            configs = [baseline(layer)] if args.baseline_only else tilings(layer)
            for prefix in args.prefixes:
                print(f"{layer} ISL={prefix}: {', '.join(config.id for config in configs)}")
        if not args.baseline_only:
            print(
                f"Then refine each per-ISL winner through: {', '.join(FAMILIES)}; rerun baseline and winner in fresh processes."
            )
        print("Plan only. No device discovery, opening, reset, or execution performed.")
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "results.csv"
    if csv_path.exists():
        parser.error("Use a fresh --output directory to avoid mixing runs")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "models/demos/gemma4_d_p", "ttnn", "tt_metal"], cwd=ROOT
    )
    harness = b"".join(path.read_bytes() for path in sorted(Path(__file__).parent.glob("*sdpa_perf*.py")))
    provenance = {
        "commit": commit,
        "worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "harness_sha256": hashlib.sha256(harness).hexdigest(),
        "hardware": f"{platform.node()}:Blackhole:8x4:CP8:TP4:Linear:links2",
    }
    (output / "manifest.json").write_text(json.dumps({**provenance, "argv": sys.argv}, indent=2) + "\n")
    env = dict(os.environ, GEMMA4_SDPA_REPEATS=str(args.repeats))
    for name in ("TT_METAL_WATCHER", "TT_METAL_LLK_ASSERTS", "TT_METAL_LLK_SANITIZER"):
        env.pop(name, None)
    for name in (
        "GEMMA4_SDPA_CANDIDATE",
        "GEMMA4_SDPA_LAYER",
        "GEMMA4_SDPA_ISL",
        "GEMMA4_SDPA_RESULT",
        "PYTEST_ADDOPTS",
    ):
        env.pop(name, None)
    rows, summaries = [], []

    def run(layer, prefix, config, phase, seed=1234):
        number = len(rows)
        stem = f"{number:04d}-{layer}-isl{prefix}-{phase}-{config.id}"
        result_path, log_path, xml_path = [output / f"{stem}.{ext}" for ext in ("json", "log", "xml")]
        candidate_env = dict(
            env,
            GEMMA4_SDPA_LAYER=layer,
            GEMMA4_SDPA_ISL=str(prefix),
            GEMMA4_SDPA_CANDIDATE=config.to_json(),
            GEMMA4_SDPA_RESULT=str(result_path),
            GEMMA4_SDPA_SEED=str(seed),
        )
        print(f"Running {stem}", flush=True)
        code = run_process(
            [
                sys.executable,
                "-m",
                "pytest",
                TARGET,
                "-sv",
                "--timeout=0",
                "-o",
                f"cache_dir={output / 'pytest-cache'}",
                "--junitxml",
                str(xml_path),
            ],
            candidate_env,
            log_path,
            args.timeout,
        )
        row = {
            **provenance,
            "layer": layer,
            "isl": prefix,
            "config": json.loads(config.to_json()),
            "status": "error",
            "seed": seed,
        }
        if result_path.exists():
            row.update(json.loads(result_path.read_text()))
        if code == "timeout":
            row.update(status="timeout", error=f"Process exceeded {args.timeout}s; hardware state needs inspection")
        elif not result_path.exists():
            row["error"] = f"No candidate result; pytest exit={code}. See {log_path}"
            if xml_path.exists() and ET.parse(xml_path).find(".//testcase/skipped") is not None:
                row["status"] = "skipped"
        elif code not in (0, 1):
            row.update(status="error", error=f"pytest infrastructure failed, exit={code}; see log")
        elif code != 0 and row["status"] == "passed":
            row.update(status="error", error=f"pytest teardown failed, exit={code}")
        row.update(phase=phase, log=str(log_path))
        rows.append(row)
        append_csv(csv_path, row)
        print(f"  {row['status']}: median_us={row.get('median_us')} PCC={row.get('pcc')}", flush=True)
        if code == "timeout" or not result_path.exists() or row["status"] == "error":
            raise RuntimeError(f"Stopping sweep after {row['status']}; inspect {log_path}. No device reset attempted.")
        return row

    # This prevents overlap with another instance of this driver, not unrelated workloads.
    with open(f"/tmp/gemma4-sdpa-{platform.node()}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for layer in args.layers:
            for prefix in args.prefixes:
                base = baseline(layer)
                shape_rows = [run(layer, prefix, base, "baseline")]
                if not args.baseline_only:
                    shape_rows.extend(
                        run(layer, prefix, config, "tiling") for config in tilings(layer) if config != base
                    )
                    winner = best(shape_rows)
                    for family in FAMILIES:
                        if winner is None:
                            break
                        config = Candidate(**winner["config"])
                        seen = {json.dumps(row["config"], sort_keys=True) for row in shape_rows}
                        for variant in family_candidates(config, family):
                            if variant.to_json() not in seen:
                                shape_rows.append(run(layer, prefix, variant, family))
                        winner = best(shape_rows)
                winner = best(shape_rows)
                if winner is None:
                    summaries.append({"layer": layer, "isl": prefix, "status": "no_accurate_candidate"})
                    continue
                # Independently revalidate baseline and selected winner on a second seed.
                fresh_base = run(layer, prefix, base, "confirm_baseline", seed=5678)
                fresh_winner = run(layer, prefix, Candidate(**winner["config"]), "confirm_winner", seed=5678)
                confirmed = fresh_winner["status"] == "passed"
                summary = {
                    "layer": layer,
                    "isl": prefix,
                    "status": "confirmed" if confirmed else "confirmation_failed",
                    "config": winner["config"],
                    "edge_status": fresh_winner.get("edge_status"),
                    "baseline_median_us": fresh_base.get("median_us"),
                    "baseline_p90_us": fresh_base.get("p90_us"),
                    "winner_median_us": fresh_winner.get("median_us"),
                    "winner_p90_us": fresh_winner.get("p90_us"),
                    "production_validation": "pending; no production settings changed",
                }
                if confirmed and fresh_base["status"] == "passed":
                    summary["speedup"] = fresh_base["median_us"] / fresh_winner["median_us"]
                summaries.append(summary)
                (output / "winners.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (output / "winners.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    return 0 if all(row["status"] == "confirmed" for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
