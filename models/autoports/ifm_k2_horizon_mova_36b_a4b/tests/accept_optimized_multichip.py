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
DOC = ROOT / "doc/optimized_multichip_decoder"
MODULE = "models.autoports.ifm_k2_horizon_mova_36b_a4b.tests."


def run(name, module, args=(), extra_env=None, timeout=1800):
    cmd = [sys.executable, "-m", MODULE + module, *map(str, args), "--artifact-dir", str(DOC)]
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
        raise RuntimeError(f"{name} failed; inspect evidence before deciding whether hardware recovery is needed")
    print("DONE", name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--group",
        choices=["contracts", "capacity", "full", "watcher", "stack", "chunk", "awkward", "sdpa_boundary"],
        required=True,
    )
    a = ap.parse_args()
    if a.group == "contracts":
        for layer in (0, 3):
            run(
                f"boundaries_l{layer}",
                "validate_multichip_decoder",
                ["--layer", layer, "--tag", "final_boundaries", "--stress-seconds", 2],
            )
            for batch, page, lengths, dtype in [
                (2, 32, [31, 33, 65], "bfloat8_b"),
                (9, 64, [1, 63, 65], "bfloat8_b"),
                (13, 128, [127, 129], "bfloat8_b"),
                (32, 256, [255, 257], "bfloat8_b"),
                (2, 32, [31, 33], "bfloat16"),
            ]:
                run(
                    f"contract_l{layer}_b{batch}_p{page}_{dtype}",
                    "validate_multichip_decoder",
                    [
                        "--layer",
                        layer,
                        "--batch",
                        batch,
                        "--page-size",
                        page,
                        "--lengths",
                        *lengths,
                        "--cache-dtype",
                        dtype,
                        "--tag",
                        "final_contract",
                        "--stress-seconds",
                        2,
                    ],
                )
    elif a.group == "chunk":
        for layer in (0, 3):
            common = ["--layer", layer, "--lengths", 4095, 4096, 4097, "--tag", "final_chunk", "--stress-seconds", 2]
            run(f"chunk_reference_l{layer}", "validate_multichip_decoder", [*common, "--baseline"])
            run(f"chunk_contract_l{layer}", "validate_multichip_decoder", common)
    elif a.group == "awkward":
        for layer in (0, 3):
            for batch in (17, 31):
                common = [
                    "--layer",
                    layer,
                    "--batch",
                    batch,
                    "--page-size",
                    64,
                    "--lengths",
                    1,
                    33,
                    "--tag",
                    "final_awkward",
                    "--stress-seconds",
                    2,
                ]
                run(f"awkward_reference_l{layer}_b{batch}", "validate_multichip_decoder", [*common, "--baseline"])
                run(f"awkward_contract_l{layer}_b{batch}", "validate_multichip_decoder", common)
    elif a.group == "stack":
        for layer in (0, 2):
            run(
                f"stack_l{layer}", "stack_multichip_decoder", ["--dense-layer", layer, "--tag", f"final_stack_l{layer}"]
            )
    elif a.group == "sdpa_boundary":
        for layer in (0, 3):
            for capacity in (2016, 2048, 16352, 16384):
                common = ["--layer", layer, "--capacity", capacity]
                run(f"sdpa_boundary_reference_l{layer}_{capacity}", "multichip_capacity", [*common, "--baseline"])
                run(f"sdpa_boundary_l{layer}_{capacity}", "multichip_capacity", common)
    elif a.group == "capacity":
        for layer in (0, 3):
            for capacity in (65504, 65536, 524256, 524288):
                run(
                    f"capacity_l{layer}_{capacity}",
                    "multichip_capacity",
                    ["--layer", layer, "--capacity", capacity, "--reserve-stack"],
                )
            run(f"capacity_bf16_l{layer}", "multichip_capacity", ["--layer", layer, "--cache-dtype", "bfloat16"])
    elif a.group == "full":
        for layer in (0, 3):
            run(f"full_prefill_l{layer}", "multichip_full_prefill", ["--layer", layer], timeout=1800)
    elif a.group == "watcher":
        for layer in (0, 3):
            run(
                f"watcher_l{layer}",
                "validate_multichip_decoder",
                ["--layer", layer, "--batch", 2, "--lengths", 31, 33, 65, "--stress-seconds", 12, "--tag", "watcher"],
                {
                    "TT_METAL_WATCHER": "10",
                    "TT_METAL_FABRIC_OPT_LEVEL": "Os",
                    "TT_METAL_CACHE": str(ROOT.resolve() / "doc/multichip_decoder/local/watcher_os_cache"),
                    "TT_METAL_LOGS_PATH": str(DOC.resolve() / f"watcher/layer{layer}"),
                },
            )


if __name__ == "__main__":
    main()
