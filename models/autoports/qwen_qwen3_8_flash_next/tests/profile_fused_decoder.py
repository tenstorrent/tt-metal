# SPDX-License-Identifier: Apache-2.0
"""Serialize matched functional/fused Tracy runs and render validated windows."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["functional", "fused"], required=True)
    parser.add_argument("--layers", default="0,1,3")
    parser.add_argument("--prefix", default="", help="Preserve previous profiler artifacts under distinct tags")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / "doc/fused_decoder"
    external = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next")
    for layer in map(int, args.layers.split(",")):
        tag = args.prefix + f"profile_{args.impl}_l{layer}"
        profile = external / ("fusion_" + tag)
        if profile.exists():
            raise FileExistsError(f"Preserve existing capture {profile}; choose a new --prefix")
        env = dict(os.environ)
        for name in list(env):
            if name.startswith("TT_METAL_WATCHER"):
                del env[name]
        env.update(
            TT_METAL_TRACE_ALLOC_TRACKING="0",
            TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="8192",
            TT_METAL_PROFILER_DIR=str(profile),
            OMP_NUM_THREADS="8",
        )
        command = [
            sys.executable,
            "-m",
            "tracy",
            "-r",
            "-p",
            "-v",
            "--no-web-server",
            "--disable-device-data-push-to-tracy",
            "-o",
            str(profile),
            "-m",
            "models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_fused_decoder",
            "--impl",
            args.impl,
            "--layer",
            str(layer),
            "--tag",
            tag,
            "--profile",
        ]
        (evidence / (tag + "_command.json")).write_text(
            json.dumps(
                dict(
                    command=command,
                    environment={
                        k: env[k]
                        for k in (
                            "TT_METAL_TRACE_ALLOC_TRACKING",
                            "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT",
                            "TT_METAL_PROFILER_DIR",
                            "OMP_NUM_THREADS",
                        )
                    },
                ),
                indent=2,
            )
            + "\n"
        )
        print("START", tag, flush=True)
        with (evidence / (tag + ".log")).open("w") as handle:
            subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        workload_log = (evidence / (tag + ".log")).read_text()
        for failure in ("TT_FATAL:", "markers were dropped", "Traceback (most recent call last)"):
            if failure in workload_log:
                raise RuntimeError(f"Rejected profiler capture containing {failure}; inspect {tag}.log")
        metadata = (profile / ".logs/tracy_ops_data.csv").read_text()
        captured = re.findall(r"TT_METAL_TRACE_ENQUEUE_PROGRAM: (\d+), (\d+), (\d+)", metadata)
        if not captured or len({(device, trace) for device, trace, _ in captured}) != 1:
            raise RuntimeError("Expected one captured trace with a complete enqueue-program list")
        csvs = sorted(profile.glob("reports/*/ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            raise RuntimeError(f"No enriched ops CSV in {profile}")
        render = [
            sys.executable,
            str(root / "tests/render_fused_perf.py"),
            str(csvs[-1]),
            "--layer",
            str(layer),
            "--expected-decode-programs",
            str(len(captured)),
            "--tag",
            args.prefix + f"{args.impl}_l{layer}",
        ]
        with (evidence / (tag + "_render.log")).open("w") as handle:
            subprocess.run(render, stdout=handle, stderr=subprocess.STDOUT, check=True)
        print("PROFILE_AND_REPORT_PASS", tag, flush=True)


if __name__ == "__main__":
    main()
