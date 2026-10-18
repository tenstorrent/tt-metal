# SPDX-License-Identifier: Apache-2.0
"""Serialize review-remediation controls and native fusion candidates."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "controls",
            "followup_controls",
            "hc_controls",
            "split_controls",
            "coupled_controls",
            "conv_default_controls",
            "conv_default_benchmarks",
            "benchmarks",
        ],
    )
    parser.add_argument("--suffix", default="")
    parser.add_argument("--candidates", default="", help="Optional comma-separated benchmark candidate filter")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / "doc/fused_decoder"
    module = "models.autoports.qwen_qwen3_8_flash_next.tests."
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("TT_METAL_WATCHER"):
            del env[name]
    env.update(
        QWEN38_WEIGHT_SOURCE="real",
        QWEN38_WEIGHT_CACHE="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next",
        OMP_NUM_THREADS="8",
    )
    jobs = []
    if args.mode in ("controls", "followup_controls"):
        controls = (
            [("dedicated_moe", 0, 2, 1), ("split_short_hc", 1, 1, 3)]
            if args.mode == "controls"
            else [("composite_conv", 0, 2, 1), ("composite_norm", 0, 2, 1), ("split_all_hc", 0, 2, 1)]
        )
        for candidate, layer, batch, length in controls:
            tag = "review_control_" + candidate + args.suffix
            command = [
                sys.executable,
                "-m",
                module + "fused_precision_controls",
                candidate,
                "--layers",
                f"layer{layer}",
                "--evidence-dir",
                str(evidence / tag),
            ]
            overrides = dict(
                TT_METAL_TRACE_ALLOC_TRACKING="1", QWEN38_DECODER_BATCHES=str(batch), QWEN38_DECODER_LENGTHS=str(length)
            )
            jobs.append((tag, command, overrides))
    elif args.mode == "hc_controls":
        for candidate, layers, batch, length in [
            ("fp32_norm", "layer0", 2, 1),
            ("sequential_sum", "layer0", 2, 1),
            ("fp32_norm", "layer0 or layer1", 1, 3),
        ]:
            tag = f"review_hc_{candidate}_B{batch}" + args.suffix
            command = [
                sys.executable,
                "-m",
                module + "fused_hc_precision_candidates",
                candidate,
                "--layers",
                layers,
                "--evidence-dir",
                str(evidence / tag),
            ]
            jobs.append(
                (
                    tag,
                    command,
                    dict(
                        TT_METAL_TRACE_ALLOC_TRACKING="1",
                        QWEN38_DECODER_BATCHES=str(batch),
                        QWEN38_DECODER_LENGTHS=str(length),
                    ),
                )
            )
    elif args.mode == "split_controls":
        for candidate in [
            "functional_arithmetic",
            "packed_unscaled_functional",
            "split_unfold_sigmoid",
            "split_weighted_vector_rest",
        ]:
            if args.candidates and candidate not in args.candidates.split(","):
                continue
            tag = "review_hc_" + candidate + args.suffix
            command = [
                sys.executable,
                "-m",
                module + "fused_hc_split_candidates",
                candidate,
                "--layers",
                "layer0 or layer1",
                "--evidence-dir",
                str(evidence / tag),
            ]
            jobs.append(
                (
                    tag,
                    command,
                    dict(TT_METAL_TRACE_ALLOC_TRACKING="1", QWEN38_DECODER_BATCHES="1,2", QWEN38_DECODER_LENGTHS="3"),
                )
            )
    elif args.mode == "coupled_controls":
        for control in ["conv", "norm", "both"]:
            tag = "review_hc_functional_gdn_" + control + args.suffix
            command = [
                sys.executable,
                "-m",
                module + "fused_hc_split_candidates",
                "functional_arithmetic",
                "--gdn-control",
                control,
                "--layers",
                "layer0",
                "--evidence-dir",
                str(evidence / tag),
            ]
            jobs.append(
                (
                    tag,
                    command,
                    dict(TT_METAL_TRACE_ALLOC_TRACKING="1", QWEN38_DECODER_BATCHES="1", QWEN38_DECODER_LENGTHS="3"),
                )
            )
    elif args.mode == "conv_default_controls":
        for candidate in ["current", "split_all"]:
            tag = "review_conv_default_" + candidate + args.suffix
            command = [
                sys.executable,
                "-m",
                module + "fused_hc_split_candidates",
                candidate,
                "--gdn-control",
                "conv_bf16",
                "--layers",
                "layer0 or layer1 or layer3",
                "--evidence-dir",
                str(evidence / tag),
            ]
            jobs.append(
                (
                    tag,
                    command,
                    dict(TT_METAL_TRACE_ALLOC_TRACKING="1", QWEN38_DECODER_BATCHES="1,2", QWEN38_DECODER_LENGTHS="3"),
                )
            )
    elif args.mode == "conv_default_benchmarks":
        for candidate, bf16 in [("default", False), ("default", True), ("hc_split_all", True)]:
            tag = f"review_conv_benchmark_{candidate}_{bf16}" + args.suffix
            command = [
                sys.executable,
                "-m",
                module + "benchmark_fused_decoder",
                "--impl",
                "fused",
                "--layer",
                "0",
                "--candidate",
                candidate,
                "--tag",
                tag,
            ]
            if bf16:
                command.append("--conv-bf16")
            jobs.append((tag, command, dict(TT_METAL_TRACE_ALLOC_TRACKING="0")))
    else:
        for candidate, layer in [
            ("default", 0),
            ("weighted_reduce", 0),
            ("attn_weighted_reduce", 0),
            ("default", 1),
            ("default", 3),
            ("signed_sqrt", 1),
            ("floor_position", 3),
            ("indexed_cache", 3),
            ("indexed_decode_cache", 3),
            ("paged_decode_cache", 3),
        ]:
            if args.candidates and candidate not in args.candidates.split(","):
                continue
            tag = f"review_{candidate}_l{layer}" + args.suffix
            command = [
                sys.executable,
                "-m",
                module + "benchmark_fused_decoder",
                "--impl",
                "fused",
                "--layer",
                str(layer),
                "--candidate",
                candidate,
                "--tag",
                tag,
            ]
            jobs.append((tag, command, dict(TT_METAL_TRACE_ALLOC_TRACKING="0")))
    for tag, command, overrides in jobs:
        active_env = env | overrides
        source = root / "tt/fused_decoder.py"
        record = dict(
            command=command,
            environment={
                k: v for k, v in active_env.items() if k.startswith(("QWEN38_", "TT_METAL_")) or k == "OMP_NUM_THREADS"
            },
            runtime_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            started_unix=time.time(),
        )
        (evidence / (tag + "_source.py.txt")).write_bytes(source.read_bytes())
        target = evidence / (tag + "_command.json")
        target.write_text(json.dumps(record, indent=2) + "\n")
        print("START", tag, flush=True)
        with (evidence / (tag + ".log")).open("w") as handle:
            result = subprocess.run(command, env=active_env, stdout=handle, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, elapsed_seconds=time.time() - record["started_unix"])
        target.write_text(json.dumps(record, indent=2) + "\n")
        print("END", tag, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
