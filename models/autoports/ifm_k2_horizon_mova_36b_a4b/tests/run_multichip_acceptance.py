# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize stage acceptance commands; preserve exact command/env/exit evidence."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"
MODULE = "models.autoports.ifm_k2_horizon_mova_36b_a4b.tests."


def run(name, module, args=(), extra_env=None, timeout=1800):
    cmd = [sys.executable, "-m", MODULE + module, *map(str, args)]
    env = {**os.environ, "TT_METAL_TRACE_ALLOC_TRACKING": "1", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0"}
    for key in list(env):
        if key.startswith("TT_METAL_WATCHER") or key in (
            "TT_METAL_DEVICE_PROFILER",
            "TT_METAL_LOGS_PATH",
            "TT_METAL_FABRIC_OPT_LEVEL",
        ):
            env.pop(key, None)
    env.update(extra_env or {})
    log = DOC / "logs" / f"{name}.log"
    start = time.time()
    print("START", name, flush=True)
    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "tests" / f"{module}.py", ROOT / "tt/multichip_decoder.py", ROOT / "tt/optimized_decoder.py")
    }
    (DOC / f"{name}_started.json").write_text(
        json.dumps(dict(command=cmd, sources=sources, start_unix=start), indent=2) + "\n"
    )
    with log.open("w") as f:
        process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print("TRIAGE_REQUIRED left alive pid", process.pid, flush=True)
            raise
    result = dict(
        command=cmd,
        sources=sources,
        environment={k: v for k, v in env.items() if k.startswith("TT_METAL_")},
        seconds=time.time() - start,
        exit_code=code,
        log=str(log),
    )
    (DOC / f"{name}_execution.json").write_text(json.dumps(result, indent=2) + "\n")
    if code:
        raise RuntimeError(f"{name} failed; inspect evidence and reset before hardware retry")
    print("DONE", name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--group", required=True, choices=["smoke", "contracts", "capacity", "full", "wall", "watcher", "review"]
    )
    a = ap.parse_args()
    if a.group == "smoke":
        for l in (0, 3):
            run(f"final_smoke_l{l}", "run_multichip_decoder", ["--layer", l, "--trace", "--tag", "final_smoke"])
        for baseline in (True, False):
            run(
                "stack_" + ("baseline" if baseline else "final"),
                "stack_multichip_decoder",
                ["--baseline"] if baseline else [],
            )
    if a.group == "contracts":
        for l in (0, 3):
            run(f"final_boundaries_l{l}", "validate_multichip_decoder", ["--layer", l, "--tag", "final_boundaries"])
            for b, p, lengths, dtype in [
                (2, 32, [31, 33, 65], "bfloat8_b"),
                (9, 64, [1, 63, 65], "bfloat8_b"),
                (13, 128, [127, 129], "bfloat8_b"),
                (32, 256, [255, 257], "bfloat8_b"),
                (2, 32, [31, 33], "bfloat16"),
            ]:
                args = ["--layer", l, "--batch", b, "--page-size", p, "--lengths", *lengths, "--cache-dtype", dtype]
                key = f"l{l}_b{b}_p{p}_{dtype}"
                run("baseline_" + key, "validate_multichip_decoder", [*args, "--baseline", "--tag", "final_reference"])
                run("final_" + key, "validate_multichip_decoder", [*args, "--tag", "final_contract"])
    if a.group == "capacity":
        for l in (0, 3):
            run(f"final_capacity_l{l}", "multichip_capacity", ["--layer", l, "--reserve-stack"])
        for l in (0, 3):
            args = ["--layer", l, "--capacity", 524256]
            run(f"baseline_odd_capacity_l{l}", "multichip_capacity", [*args, "--baseline"])
            run(f"final_odd_capacity_l{l}", "multichip_capacity", [*args, "--reserve-stack"])
        for l in (0, 3):
            args = ["--layer", l, "--cache-dtype", "bfloat16"]
            run(f"baseline_bf16_capacity_l{l}", "multichip_capacity", [*args, "--baseline"])
            run(f"final_bf16_capacity_l{l}", "multichip_capacity", args)
    if a.group == "full":
        for l in (0, 3):
            for baseline in (True, False):
                run(
                    f'full_prefill_{"baseline" if baseline else "final"}_l{l}',
                    "multichip_full_prefill",
                    ["--layer", l, *(["--baseline"] if baseline else [])],
                    timeout=3600,
                )
    if a.group == "wall":
        for s in (32, 256, 2048, 2049):
            for l in (0, 3):
                for baseline in (True, False):
                    tag = "final_baseline" if baseline else "final_wall"
                    run(
                        f"{tag}_l{l}_s{s}",
                        "run_multichip_decoder",
                        [
                            "--layer",
                            l,
                            "--seq",
                            s,
                            "--context",
                            max(1024, s * 2),
                            "--trace",
                            "--iterations",
                            200,
                            "--prefill-iterations",
                            20,
                            "--tag",
                            tag,
                            *(["--baseline"] if baseline else []),
                        ],
                        {"TT_METAL_TRACE_ALLOC_TRACKING": "0"},
                    )
    if a.group == "review":
        for layer in (0, 2):
            run(f"accepted_stack_baseline_l{layer}", "stack_multichip_decoder", ["--dense-layer", layer, "--baseline"])
            run(
                f"accepted_stack_l{layer}",
                "stack_multichip_decoder",
                ["--dense-layer", layer, "--tag", f"accepted_stack_l{layer}"],
            )
        for capacity in (65504, 65536):
            for layer in (0, 3):
                for baseline in (True, False):
                    run(
                        f'middle_{capacity}_l{layer}_{"baseline" if baseline else "tp4"}',
                        "multichip_capacity",
                        ["--layer", layer, "--capacity", capacity, *(["--baseline"] if baseline else [])],
                    )
    if a.group == "watcher":
        for l in (0, 3):
            run(
                f"watcher_os_fresh_l{l}",
                "validate_multichip_decoder",
                [
                    "--layer",
                    l,
                    "--batch",
                    2,
                    "--lengths",
                    31,
                    33,
                    65,
                    "--stress-seconds",
                    12,
                    "--tag",
                    "watcher_os_fresh",
                ],
                {
                    "TT_METAL_WATCHER": "10",
                    "TT_METAL_FABRIC_OPT_LEVEL": "Os",
                    "TT_METAL_CACHE": str(DOC.resolve() / "local/watcher_os_cache"),
                    "TT_METAL_LOGS_PATH": str(DOC.resolve() / f"watcher/os_fresh_l{l}"),
                },
            )


if __name__ == "__main__":
    main()
