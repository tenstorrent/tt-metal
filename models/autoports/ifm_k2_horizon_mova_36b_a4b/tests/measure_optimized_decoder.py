# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized final wall/profiler evidence, separate from watcher runs."""

import argparse
import ast
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


def command(tag, argv, tracking=False):
    env = {
        **os.environ,
        "TT_METAL_TRACE_ALLOC_TRACKING": "1" if tracking else "0",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0",
    }
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_LOGS_PATH"):
        env.pop(key, None)
    path = DOC / "logs" / (tag + ".log")
    start = time.time()
    print("START", tag, flush=True)
    with path.open("w") as out:
        process = subprocess.Popen([sys.executable, *argv], stdout=out, stderr=subprocess.STDOUT, env=env)
        try:
            code = process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            print("TRIAGE_REQUIRED", process.pid, "left alive", flush=True)
            raise
    (DOC / (tag + "_execution.json")).write_text(
        json.dumps(
            dict(
                command=[sys.executable, *argv],
                exit_code=code,
                seconds=time.time() - start,
                log=str(path),
                trace_tracking=tracking,
            ),
            indent=2,
        )
    )
    if code:
        raise RuntimeError(f"{tag} failed: {path}")
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--group",
        choices=["wall", "profile", "micro", "attention_prefill", "router_precision", "prefill_advice"],
        required=True,
    )
    ap.add_argument("--optimized-only", action="store_true", help="Reuse unchanged fused baseline artifacts")
    args = ap.parse_args()
    sha = hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest()
    if args.group == "wall":
        for layer in (0, 3):
            for seq, iterations in ((32, 100), (256, 300), (2048, 20), (2049, 20)):
                for baseline in (True, False):
                    if baseline and args.optimized_only:
                        continue
                    tag = ("final_baseline" if baseline else "final_default") + f"_s{seq}"
                    command(
                        tag + f"_l{layer}",
                        [
                            "-m",
                            MODULE + "run_optimized_decoder",
                            "--layer",
                            str(layer),
                            "--seq",
                            str(seq),
                            "--context",
                            str(4096 if seq == 2049 else max(1024, seq * 2)),
                            "--trace",
                            "--profile",
                            "--iterations",
                            str(iterations),
                            "--real-inputs",
                            "--tag",
                            tag,
                            *(["--baseline"] if baseline else []),
                        ],
                    )
        # Recheck the strongest legal separate topologies after final routing,
        # precision, layout and attention changes; preserve complete group costs.
        controls = [
            (0, "attention_separate", {"split_attention": True, "attn_readers": 1, "dram_attn_block": 10}, None),
            (0, "mlp_separate", {"packed_dense": False, "dram_up_block": 5}, None),
            (3, "attention_separate", {"split_attention": True, "attn_readers": 2, "dram_attn_block": 10}, None),
            (3, "shared_separate_dram", {"split_shared": True, "router_interleaved": False}, None),
            (3, "shared_separate_interleaved", {"split_shared": True, "router_interleaved": True}, None),
            (
                3,
                "experts_separate24",
                {"split_experts": True, "split_grid": [8, 3], "split_subblock": 1, "fused_gate": True},
                "candidate_optimized_separate_experts.py",
            ),
            (
                3,
                "experts_separate12",
                {"split_experts": True, "split_grid": [6, 2], "split_subblock": 2, "fused_gate": True},
                "candidate_optimized_separate_experts.py",
            ),
        ]
        controls += [
            (0, "attention_hifi2", {"attention_fidelity": "HiFi2"}, None),
            (0, "mlp_hifi2", {"mlp_fidelity": "HiFi2"}, None),
            (3, "attention_hifi2", {"attention_fidelity": "HiFi2"}, None),
            (3, "mlp_hifi2", {"mlp_fidelity": "HiFi2"}, None),
            (3, "experts_hifi2", {"experts_fidelity": "HiFi2"}, None),
            (3, "value_hifi2", {"value_fidelity": "HiFi2"}, None),
        ]
        for reader in (1, 2, 3):
            controls.append(
                (
                    3,
                    f"shared_separate_lofi_r{reader}",
                    dict(split_shared=True, router_interleaved=True, up_readers=reader),
                    "candidate_optimized_separate_router.py",
                )
            )
        controls.append(
            (3, "shared_separate_hifi_r1", dict(split_shared=True, router_interleaved=True, up_readers=1), None)
        )
        controls.append(
            (3, "shared_separate_dram_r1", dict(split_shared=True, router_interleaved=False, up_readers=1), None)
        )
        for layer, name, options, source in controls:
            tag = "final_topology_" + name
            command(
                tag + f"_l{layer}",
                [
                    "-m",
                    MODULE + "run_optimized_decoder",
                    "--layer",
                    str(layer),
                    "--seq",
                    "256",
                    "--context",
                    "1024",
                    "--trace",
                    "--profile",
                    "--iterations",
                    "100",
                    "--real-inputs",
                    "--tag",
                    tag,
                    "--options",
                    json.dumps(options),
                    *(["--source", str(ROOT / "tests" / source)] if source else []),
                ],
            )
        for layer in (0, 3):
            command(
                f"final_capacity_timing_l{layer}",
                [
                    "-m",
                    MODULE + "optimized_decode_capacity",
                    "--layer",
                    str(layer),
                    "--capacity",
                    "524288",
                    "--tag",
                    "final_timing",
                ],
            )
    elif args.group == "attention_prefill":
        for layer in (0, 3):
            for capacity in (65536, 524288):
                for grid in ([8, 8], "full"):
                    tag = "final_prefill_grid_" + ("full" if grid == "full" else "8x8") + f"_c{capacity}"
                    command(
                        tag + f"_l{layer}",
                        [
                            "-m",
                            MODULE + "optimized_decode_capacity",
                            "--layer",
                            str(layer),
                            "--capacity",
                            str(capacity),
                            "--tag",
                            tag,
                            "--probe-prefill",
                            "--probe-prefill-length",
                            "2048",
                            "--source",
                            str(ROOT / "tests/candidate_optimized_attention_compute.py"),
                            "--options",
                            json.dumps(
                                dict(sdpa_fidelity="HiFi2", sdpa_fp32=False, sdpa_k=1024, prefill_sdpa_grid=grid)
                            ),
                        ],
                    )
    elif args.group == "router_precision":
        for dtype in ("bfloat8_b", "bfloat4_b"):
            for variant, seq in ((0, 256), (1, 257), (2, 33), (3, 256)):
                tag = f"final_shared_{dtype}_v{variant}"
                command(
                    tag,
                    [
                        "-m",
                        MODULE + "run_optimized_decoder",
                        "--layer",
                        "3",
                        "--seq",
                        str(seq),
                        "--context",
                        "1024",
                        "--real-inputs",
                        "--input-variant",
                        str(variant),
                        "--trace",
                        "--profile",
                        "--iterations",
                        "100",
                        "--source",
                        str(ROOT / "tests/candidate_optimized_shared_precision.py"),
                        "--tag",
                        tag,
                        "--options",
                        json.dumps(dict(shared_router_dtype=dtype)),
                    ],
                )
    elif args.group == "prefill_advice":
        for layer in (0, 3):
            for seq in (256, 2048):
                for name, options, source in [
                    ("grid11", dict(prefill_grid=[11, 10]), None),
                    ("producer_l1", {}, "candidate_optimized_prefill_l1.py"),
                ]:
                    tag = f"final_prefill_{name}_s{seq}"
                    command(
                        tag + f"_l{layer}",
                        [
                            "-m",
                            MODULE + "run_optimized_decoder",
                            "--layer",
                            str(layer),
                            "--seq",
                            str(seq),
                            "--context",
                            str(4096 if seq == 2049 else max(1024, seq * 2)),
                            "--real-inputs",
                            "--trace",
                            "--profile",
                            "--iterations",
                            "20",
                            "--tag",
                            tag,
                            "--options",
                            json.dumps(options),
                            *(["--source", str(ROOT / "tests" / source)] if source else []),
                        ],
                    )
    elif args.group == "profile":
        for layer in (0, 3):
            raw = DOC / "profile_raw" / f"final_layer{layer}"
            tag = "profile_final"
            command(
                tag + f"_l{layer}",
                [
                    "-m",
                    "tracy",
                    "-r",
                    "-p",
                    "-v",
                    "--no-web-server",
                    "-o",
                    str(raw),
                    "-m",
                    MODULE + "run_optimized_decoder",
                    "--layer",
                    str(layer),
                    "--seq",
                    "256",
                    "--trace",
                    "--profile",
                    "--iterations",
                    "1",
                    "--real-inputs",
                    "--tag",
                    tag,
                ],
            )
            evidence = json.loads((DOC / "candidates" / f"{tag}_layer{layer}.json").read_text())
            assert evidence["source_sha256"] == sha
            raw_csv = sorted(raw.rglob("ops_perf_results*.csv"))[-1]
            command(
                f"perf_report_final_l{layer}",
                ["-m", MODULE + "optimized_perf_report", str(layer), str(raw_csv), "--tag", "final"],
            )
    else:
        tree = ast.parse((ROOT / "tt/optimized_decoder.py").read_text())
        policies = {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ("DENSE_POLICY", "MOE_POLICY")
        }
        for layer, kind, policy in ((0, "dense", policies["DENSE_POLICY"]), (3, "sparse", policies["MOE_POLICY"])):
            options = DOC / "candidates" / f"delivered_policy_layer{layer}.json"
            options.write_text(json.dumps(policy, indent=2))
            common = ["--layer", str(layer), "--options-file", str(options)]
            command(
                f"micro_final_tracked_l{layer}",
                ["-m", MODULE + "optimized_dram_microbench", *common, "--tag", "final_tracked"],
                tracking=True,
            )
            raw = DOC / "profile_raw" / f"micro_{kind}"
            command(
                f"micro_final_profile_l{layer}",
                [
                    "-m",
                    "tracy",
                    "-r",
                    "-p",
                    "-v",
                    "--no-web-server",
                    "-o",
                    str(raw),
                    "-m",
                    MODULE + "optimized_dram_microbench",
                    *common,
                    "--tag",
                    "final_profile",
                    "--profile",
                ],
            )
