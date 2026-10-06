# SPDX-License-Identifier: Apache-2.0
"""Serial experiments; stop on runtime failures so diagnosis precedes retries."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--configs", nargs="*")
    parser.add_argument("--smokes-only", action="store_true")
    args = parser.parse_args()
    if args.wait_pid:
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(5)
    package = "models.autoports.aleph_alpha_kolibri_1_bf16.tests"
    subprocess.run([sys.executable, "-m", package + ".datatype_configs"], check=True)
    root = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"
    configs = args.configs or json.loads((root / "configs/matrix.json").read_text())
    for config in configs:
        config_path = root / "configs" / f"{config}.json"
        for reduced in (True,) if args.smokes_only else (True, False):
            out = root / ("smoke" if reduced else "candidates") / config
            result = out / "result.json"
            if result.exists() and json.loads(result.read_text()).get("status") in (
                "pass",
                "accuracy-fail",
                "smoke-pass",
                "capacity-rejected",
            ):
                continue
            env = dict(os.environ, FULL_ARTIFACT_DIR=str(out))
            env.pop("KOLIBRI_PRECISION_CONFIG", None)
            env["TT_METAL_TRACE_ALLOC_TRACKING"] = "1" if reduced else "0"
            env["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] = "0"
            command = [sys.executable, "-m", package + ".datatype_run", "--config", str(config_path)]
            if reduced:
                command.append("--reduced")
            if config == "baseline_bfp4_lofi" and not reduced:
                command.append("--token-out")
            if config.endswith("_chunk2048"):
                command.append("--capability")
            print("START", config, "smoke" if reduced else "full", flush=True)
            completed = subprocess.run(
                [sys.executable, "-m", package + ".optimized_full_run", "run", *command], env=env
            )
            print("END", config, completed.returncode, flush=True)
            if completed.returncode:
                sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
