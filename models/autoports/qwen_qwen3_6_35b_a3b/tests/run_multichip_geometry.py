# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial, precision-locked local DRAM projection geometry screening."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DOC = Path(__file__).parents[1] / "doc/multichip_decoder"
CASES = {
    "control": None,
    "readers1": {"default": {"readers": 1}},
    "readers2": {"default": {"readers": 2}},
    "block8": {"default": {"block_w": 8}, "shared_down": {"block_w": 4}},
    "block16": {"default": {"block_w": 16}, "shared_down": {"block_w": 4}},
    "input8": {"default": {"input_cores": 8}, "shared_down": {"input_cores": 4}},
    "output4": {"default": {"per_core_n": 4}},
}
SPARSE_CASES = {
    "control": None,
    "dram_block32": {"gu": {"block_w": 32}},
    "dram_block64": {"gu": {"block_w": 64}},
    "dram_gu4": {"gu": {"cores": 4, "block_w": 32, "subblock_w": 2}},
    "dram_down32": {"down": {"cores": 32, "subblock_w": 2}},
    "dram_down16": {"down": {"cores": 16, "subblock_w": 4}},
    "l1": {"input_l1": True},
    "l1_block32": {"input_l1": True, "gu": {"block_w": 32}},
    "l1_block64": {"input_l1": True, "gu": {"block_w": 64}},
    "l1_gu4": {"input_l1": True, "gu": {"cores": 4, "block_w": 32, "subblock_w": 2}},
    "l1_down32": {"input_l1": True, "down": {"cores": 32, "subblock_w": 2}},
    "l1_down16": {"input_l1": True, "down": {"cores": 16, "subblock_w": 4}},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--only", help="Comma-separated subset; retain previous search rows")
    args = parser.parse_args()
    destination = DOC / ("sparse_geometry" if args.sparse else "geometry")
    destination.mkdir(exist_ok=True)
    env = os.environ.copy()
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING"):
        env.pop(key, None)
    results = (
        json.loads((destination / "search.json").read_text())["results"]
        if args.only and (destination / "search.json").exists()
        else []
    )
    for name, geometry in (SPARSE_CASES if args.sparse else CASES).items():
        if args.only and name not in args.only.split(","):
            continue
        if (DOC / "STOP").exists():
            raise SystemExit("Stopped between device jobs")
        owners = subprocess.run(["fuser", *map(str, Path("/dev/tenstorrent").glob("*"))], capture_output=True)
        if owners.returncode == 0:
            raise SystemExit("Target mesh is owned")
        output = destination / f"{name}.json"
        command = [
            "timeout",
            "1800",
            sys.executable,
            "-m",
            "models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder",
            "--layer",
            "3",
            "--length",
            "32",
            "--decode-steps",
            "2",
            "--repeats",
            "100",
            "--prefill-repeats",
            "1",
            "--stress-repeats",
            "2",
            "--baseline-load",
            str(DOC / "baseline_l3_s32.pt"),
            "--output",
            str(output),
        ]
        if geometry:
            command += (
                ["--mesh-build-kwargs", json.dumps({"sparse_geometry": geometry})]
                if args.sparse
                else ["--mesh-projection-geometry", json.dumps(geometry)]
            )
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(output.read_text())
        assert data["passed"] and data["multichip"]["output_replicas_equal"]
        results = [row for row in results if row["name"] != name]
        results.append(
            dict(
                name=name,
                geometry=geometry,
                artifact=str(output),
                decode_ms=data["multichip"]["decode_traced_warmed_ms"],
                prefill_pcc=data["prefill_pcc"],
                min_decode_pcc=min(data["decode_pcc"]),
            )
        )
        (destination / "search.json").write_text(
            json.dumps(
                dict(
                    comparison="clean serial mesh-only timings; saved baseline is a correctness control, not paired latency",
                    results=results,
                ),
                indent=2,
            )
            + "\n"
        )
        print(name, results[-1]["decode_ms"], flush=True)


if __name__ == "__main__":
    main()
