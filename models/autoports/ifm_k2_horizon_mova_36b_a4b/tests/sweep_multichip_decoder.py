# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized coherent topology and precision-locked local-geometry candidates."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"
MODULE = "models.autoports.ifm_k2_horizon_mova_36b_a4b.tests."


def run(tag, layer, options, source=None, seq=33, baseline=False):
    if os.environ.get("MULTICHIP_SWEEP_RESUME") == "1" and (DOC / f"{tag}_l{layer}_b1_s{seq}.json").exists():
        print("REUSE", tag, layer, seq, flush=True)
        return
    cmd = [
        sys.executable,
        "-m",
        MODULE + "run_multichip_decoder",
        "--layer",
        str(layer),
        "--seq",
        str(seq),
        "--context",
        str(max(1024, seq * 2)),
        "--trace",
        "--iterations",
        "200",
        "--tag",
        tag,
        "--options",
        json.dumps(options),
    ]
    if source:
        cmd += ["--source", str(ROOT / "tests" / source)]
    if baseline:
        cmd += ["--baseline"]
    env = {**os.environ, "TT_METAL_TRACE_ALLOC_TRACKING": "0", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0"}
    for k in ["TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_LOGS_PATH"]:
        env.pop(k, None)
    log = DOC / "logs" / f"{tag}_l{layer}_s{seq}.log"
    print("START", tag, layer, seq, flush=True)
    begin = time.time()
    with log.open("w") as f:
        process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            print("TRIAGE_REQUIRED process left alive", process.pid, flush=True)
            raise
    (DOC / f"{tag}_l{layer}_s{seq}_execution.json").write_text(
        json.dumps(dict(command=cmd, exit_code=code, seconds=time.time() - begin, log=str(log)), indent=2) + "\n"
    )
    if code:
        raise RuntimeError(f"{tag} failed; inspect and reset before further hardware work")
    print("DONE", tag, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["ccl", "sparse", "dense", "prefill", "fusion"], required=True)
    a = ap.parse_args()
    ar = "candidate_multichip_allreduce.py"
    if a.group == "ccl":
        for layer in (0, 3):
            for cores in (8, 16, 20, 40):
                run(f"ccl_cores{cores}", layer, {"ccl_cores": cores}, ar)
            run("ccl_bf16", layer, {"ccl_dtype": "bfloat16"}, ar)
            run("ccl_link1", layer, {"num_links": 1}, ar)
    if a.group == "sparse":
        for name, opts in [
            ("gate6_k40", {"gate_up_grid": [6, 1], "gate_up_subblock": 2, "gate_up_block": 40}),
            ("gate6_k80", {"gate_up_grid": [6, 1], "gate_up_subblock": 2, "gate_up_block": 80}),
            ("gate12_k80", {"gate_up_block": 80}),
            ("down20", {"down_grid": [5, 4], "down_subblock": 4}),
            ("down80", {"down_grid": [8, 10], "down_subblock": 1}),
            ("value4_k40", {"value_grid": [4, 1], "value_subblock": 2, "value_block": 40}),
            ("value4_k80", {"value_grid": [4, 1], "value_subblock": 2, "value_block": 80}),
            ("value8_k80", {"value_block": 80}),
        ]:
            run(name, 3, opts, ar)
    if a.group == "dense":
        for layer in (0, 3):
            for role in ("attn", "output", "up", "down"):
                for readers in (1, 2, 3):
                    run(f"{role}_readers{readers}", layer, {role + "_readers": readers}, ar)
            run("attn_k10", layer, {"dram_attn_block": 10}, ar)
            run("up_k10", layer, {"dram_up_block": 10}, ar)
            for role, blocks in [
                ("attn", (20, 40, 80)),
                ("up", (20, 40, 80)),
                ("output", (8, 16, 32)),
                ("down", (2, 4, 8) if layer == 3 else (12, 24, 48)),
            ]:
                for block in blocks:
                    opts = {"dram_" + role + "_block": block}
                    suffix = ""
                    if layer == 0 and role == "attn" and block >= 40:
                        opts["attn_readers"] = 2 if block == 40 else 3
                        suffix = "_adapted_r" + str(opts["attn_readers"])
                    run(f"{role}_k{block}" + suffix, layer, opts, ar)
    if a.group == "prefill":
        for seq in (32, 256, 2048, 2049):
            for layer in (0, 3):
                run("wall_baseline", layer, {}, seq=seq, baseline=True)
                run("wall_initial", layer, {}, seq=seq)
                run("wall_oneop", layer, {}, ar, seq=seq)
                if layer == 0:
                    run("wall_swiglu", layer, {}, "candidate_multichip_swiglu.py", seq=seq)
    if a.group == "fusion":
        for layer in (0, 3):
            for dtype in ("bfloat16", "bfloat8_b"):
                run(
                    "fused_rs_" + dtype,
                    layer,
                    {"residual_layout": "sharded", "fused_dtype": dtype},
                    "candidate_multichip_fused_rs.py",
                )
            run("agmm_sharded", layer, {"residual_layout": "sharded"}, "candidate_multichip_agmm.py")


if __name__ == "__main__":
    main()
