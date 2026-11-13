#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect headline accounting and remaining rows before a server profile closes."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from collect_benchmark_roofline import check, collect

MODEL = Path(__file__).resolve().parents[1]
BASE = MODEL / "doc/benchmark"
TOOLS = MODEL / "tools"


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--action", required=True)
    args = parser.parse_args()
    b = args.concurrency
    server = json.loads((args.run_dir / f"perf-b{b}-server.json").read_text())
    if args.action == "check":
        check(args, server)
        return
    if (BASE / "stop_after_row").exists():
        raise RuntimeError("Coordinator stopped chain between requests for measurement repair")
    collect(args, server)
    timing = json.loads((BASE / "stage_timing.json").read_text())
    plan = json.loads((BASE / "plan.json").read_text())
    deadline = timing["started_monotonic"] + 4800
    for row in plan["rows"]:
        if row["batch"] != b or row.get("missing_reason") or row.get("headline"):
            continue
        dest = args.run_dir / "grid" / row["id"]
        if (dest / "complete.json").exists():
            continue
        if time.monotonic() >= deadline:
            save(BASE / "budget_exhausted.json", {"time": time.time(), "next_row": row["id"]})
            return
        dest.mkdir(parents=True, exist_ok=True)
        save(dest / "prediction.json", dict(row, written_unix=time.time()))
        save(dest / f"perf-b{b}-server.json", server)
        shutil.copy2(args.run_dir / server["configuration_evidence"], dest / server["configuration_evidence"])
        env = dict(
            os.environ,
            BENCHMARK_SERVER_BATCH=str(b),
            BENCHMARK_PHASE_DIR=str(args.run_dir.resolve()),
            BENCHMARK_SAMPLING=row["sampling"],
        )
        for warmup in (True, False):
            name = f"perf-b{b}" + ("-warmup" if warmup else "")
            count = row["concurrency"] if warmup else row["requests"]
            argv = [
                sys.executable,
                str(TOOLS / "benchmark_eval_client.py"),
                "--model",
                server["model"],
                "--base-url",
                server["base_url"],
                "--result-dir",
                str(dest),
                "--result-filename",
                name + ".json",
                "--max-concurrency",
                str(row["concurrency"]),
                "--num-prompts",
                str(count),
                "--random-input-len",
                str(row["input_tokens"]),
                "--random-output-len",
                str(row["output_tokens"]),
                "--seed",
                str(7000 + plan["rows"].index(row) * 2 + int(warmup)),
            ]
            with (dest / (name + ".log")).open("w") as log:
                subprocess.run(
                    argv,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=max(60, deadline - time.monotonic()),
                )
            raw = json.loads((dest / (name + ".json")).read_text())
            if (
                raw["failed"]
                or raw["completed"] != count
                or raw["input_lens"] != [row["input_tokens"]] * count
                or raw["output_lens"] != [row["output_tokens"]] * count
            ):
                raise ValueError(f'Incomplete/mismatched workload: {row["id"]} {name}')
        collect(argparse.Namespace(run_dir=dest, concurrency=b), server)
        save(dest / "complete.json", {"row": row, "finished_unix": time.time()})
        print(json.dumps({"completed": row["id"], "remaining_seconds": deadline - time.monotonic()}), flush=True)


if __name__ == "__main__":
    main()
