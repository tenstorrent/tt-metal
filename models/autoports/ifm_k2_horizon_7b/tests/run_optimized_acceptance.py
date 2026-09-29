"""Serialize optimized correctness, stress, context and separate watcher runs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parents[1] / "doc/optimized_decoder"
MODULE = "models.autoports.ifm_k2_horizon_7b.tests."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["basic", "long", "watcher"], default="basic")
    args = p.parse_args()
    cases = {
        "basic": [
            (
                "boundaries_final",
                "run_optimized",
                [
                    "--lengths",
                    "1",
                    "31",
                    "32",
                    "33",
                    "127",
                    "128",
                    "129",
                    "255",
                    "256",
                    "257",
                    "777",
                    "4095",
                    "4096",
                    "4097",
                    "--decode-steps",
                    "3",
                    "--audit",
                ],
            ),
            (
                "continuation",
                "run_optimized",
                ["--lengths", "777", "--batch", "2", "--split", "31", "--remap", "--decode-steps", "8", "--audit"],
            ),
            (
                "batch32",
                "run_optimized",
                ["--lengths", "33", "--batch", "32", "--decode-steps", "4", "--remap", "--audit"],
            ),
            ("positions", "run_optimized_positions", []),
            ("positions32", "run_optimized_positions", ["--batch", "32"]),
            ("positions_accurate", "run_optimized_positions", ["--pages", "1025", "--batch", "4"]),
            ("public_prefill_boundaries", "probe_optimized_prefill_boundaries", []),
            ("stock_boundary", "run_optimized", ["--lengths", "32766", "--decode-steps", "2", "--audit"]),
            ("accurate_boundary", "run_optimized", ["--lengths", "32767", "--decode-steps", "2", "--audit"]),
        ],
        "long": [
            ("long_tails", "probe_optimized_long_layer", ["--tails", "--continuation"]),
            ("long_context", "run_optimized_long_context", []),
        ],
        "watcher": [
            ("watcher_b1", "run_optimized", ["--lengths", "4097", "--decode-steps", "4", "--audit"]),
            (
                "watcher_continuation",
                "run_optimized",
                ["--lengths", "777", "--batch", "2", "--split", "31", "--remap", "--audit"],
            ),
            (
                "watcher_batch32",
                "run_optimized",
                ["--lengths", "33", "--batch", "32", "--decode-steps", "3", "--audit"],
            ),
            ("watcher_accurate", "run_optimized_positions", ["--pages", "1025", "--batch", "4"]),
        ],
    }[args.phase]
    record = {
        "completed": False,
        "phase": args.phase,
        "commands": [],
        "runtime_sha256": hashlib.sha256((DOC.parents[1] / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
    }
    for name, module, flags in cases:
        env = os.environ.copy()
        for key in ["TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_DPRINT_CORES"]:
            env.pop(key, None)
        if args.phase == "watcher":
            env["TT_METAL_WATCHER"] = "10"
            env["TT_METAL_LOGS_PATH"] = str(DOC / "watcher" / name)
        command = [sys.executable, "-m", MODULE + module, *flags, "--output", str(DOC / (name + ".json"))]
        log = DOC / "logs" / (name + ".log")
        started = time.time()
        print("START", name, flush=True)
        with log.open("w") as f:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        record["commands"].append(
            dict(name=name, argv=command, exit_code=proc.returncode, wall_seconds=time.time() - started, log=str(log))
        )
        (DOC / ("acceptance_" + args.phase + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        print("END", name, proc.returncode, flush=True)
        if proc.returncode:
            print(log.read_text()[-5000:], flush=True)
            raise SystemExit(proc.returncode)
    record["completed"] = True
    (DOC / ("acceptance_" + args.phase + ".json")).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
