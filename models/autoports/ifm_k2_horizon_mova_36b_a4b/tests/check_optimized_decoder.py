# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized acceptance commands; preserves a timed-out process for triage."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_decoder"
MODULE = "models.autoports.ifm_k2_horizon_mova_36b_a4b.tests."


def run(tag, module, args, watcher=False):
    env = {
        **os.environ,
        "TT_METAL_TRACE_ALLOC_TRACKING": "1",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0",
        "TTNN_CONFIG_OVERRIDES": '{"throw_exception_on_fallback":true}',
    }
    env.pop("TT_METAL_WATCHER", None)
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    if watcher:
        env.update(TT_METAL_WATCHER="10", TT_METAL_LOGS_PATH=str(DOC / "watcher" / tag))
    command = [sys.executable, "-m", MODULE + module, *map(str, args)]
    path = DOC / "logs" / (tag + ".log")
    print("START", tag, flush=True)
    start = time.time()
    with path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            print("TRIAGE_REQUIRED", process.pid, "left alive", flush=True)
            raise
    record = dict(
        command=command,
        exit_code=code,
        seconds=time.time() - start,
        log=str(path),
        watcher=watcher,
        trace_tracking=True,
        program_cache_tracking=True,
        decoder_sha256=hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
    )
    (DOC / (tag + "_execution.json")).write_text(json.dumps(record, indent=2))
    print("DONE", tag, code, flush=True)
    if code:
        raise RuntimeError(f"{tag} failed; inspect {path} before continuing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["correctness", "context", "boundary", "watcher"], required=True)
    args = ap.parse_args()
    if args.group == "correctness":
        for layer in (0, 3):
            run(
                f"final_b2_layer{layer}",
                "validate_optimized_decoder",
                ["--layer", layer, "--batch", 2, "--tag", "final"],
            )
            for batch in (1, 32):
                run(
                    f"watcher_b{batch}_layer{layer}",
                    "validate_optimized_decoder",
                    ["--layer", layer, "--batch", batch, "--lengths", 31, 33, "--tag", "watcher"],
                    watcher=True,
                )
            for batch, page in ((13, 64), (13, 128), (9, 256)):
                run(
                    f"final_b{batch}_page{page}_layer{layer}",
                    "validate_optimized_decoder",
                    ["--layer", layer, "--batch", batch, "--page-size", page, "--lengths", 33, 65, "--tag", "final"],
                )
            run(
                f"final_bf16cache_layer{layer}",
                "validate_optimized_decoder",
                [
                    "--layer",
                    layer,
                    "--batch",
                    2,
                    "--lengths",
                    33,
                    65,
                    "--cache-dtype",
                    "bfloat16",
                    "--tag",
                    "bf16cache",
                ],
            )
    elif args.group == "watcher":
        for layer in (0, 3):
            for batch in (1, 32):
                run(
                    f"watcher_b{batch}_layer{layer}",
                    "validate_optimized_decoder",
                    ["--layer", layer, "--batch", batch, "--lengths", 31, 33, "--tag", "watcher"],
                    watcher=True,
                )
    elif args.group == "boundary":
        for layer in (0, 3):
            run(
                f"final_boundary_layer{layer}",
                "validate_optimized_decoder",
                [
                    "--layer",
                    layer,
                    "--batch",
                    1,
                    "--lengths",
                    1,
                    31,
                    32,
                    33,
                    255,
                    256,
                    257,
                    2047,
                    2048,
                    2049,
                    "--tag",
                    "final",
                ],
            )
    else:
        for layer in (0, 3):
            for capacity in (16352, 16384, 65504, 65536, 262112, 262144, 524256):
                run(
                    f"final_capacity{capacity}_layer{layer}",
                    "optimized_decode_capacity",
                    ["--layer", layer, "--capacity", capacity, *(["--probe-prefill"] if capacity == 65504 else [])],
                )
            run(
                f"final_bf16capacity_layer{layer}",
                "optimized_decode_capacity",
                [
                    "--layer",
                    layer,
                    "--capacity",
                    524288,
                    "--cache-dtype",
                    "bfloat16",
                    "--probe-prefill",
                    "--tag",
                    "final_bf16",
                ],
            )
            run(f"final_long_layer{layer}", "optimized_long_context", ["--layer", layer])
