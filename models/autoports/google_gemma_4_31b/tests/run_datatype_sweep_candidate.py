# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one full-model Gemma 4 precision policy on the readiness workload."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from models.autoports.google_gemma_4_31b.tt.generator import build_generator
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.readiness_check.run_prefill_check import _run_one_entry_prefill
from models.common.readiness_check.schema import load_reference
from models.common.readiness_check.teacher_forcing import TokenAccuracy


def _git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _teacher_forcing(generator, reference_path: Path) -> dict:
    acc = TokenAccuracy(reference_path)
    if acc.num_entries != 1:
        raise ValueError("Gemma 4 datatype sweep expects the one-entry AIME24 readiness reference")
    prompt_ids = acc.get_prompt_token_ids(0)
    n_steps = acc.num_gt_tokens(0)
    if n_steps != 100:
        raise ValueError(f"datatype sweep requires exactly 100 generated reference tokens, got {n_steps}")

    timing = {"start": time.perf_counter(), "first": None, "last": None, "callbacks": 0}

    def next_input(step: int, predicted: int) -> int:
        now = time.perf_counter()
        if timing["first"] is None:
            timing["first"] = now
        else:
            timing["last"] = now
        timing["callbacks"] += 1
        return acc.collect_predicted_tokens(predicted, user_idx=0)

    generator.generate(
        prompt_token_ids=prompt_ids,
        max_new_tokens=n_steps,
        next_input=next_input,
        enable_trace=True,
    )
    end = time.perf_counter()
    if timing["callbacks"] != n_steps or acc.num_pred_tokens(0) != n_steps:
        raise RuntimeError("teacher forcing did not cover all 100 reference tokens")
    stats = acc.compute_accuracy(user_idx=0)
    first = timing["first"]
    last = timing["last"] or end
    ttft_s = first - timing["start"]
    decode_elapsed_s = last - first
    stats.update(
        {
            "ttft_ms": ttft_s * 1000.0,
            "decode_tokens": n_steps - 1,
            "decode_elapsed_s": decode_elapsed_s,
            "decode_t/s/u": (n_steps - 1) / decode_elapsed_s,
            "elapsed_s": end - timing["start"],
            "e2e_t/s/u": n_steps / (end - timing["start"]),
        }
    )
    return stats


def run(args: argparse.Namespace) -> None:
    precision_path = args.precision_config.resolve()
    reference_path = args.reference.resolve()
    output_path = args.output.resolve()
    reference = load_reference(reference_path)
    if len(reference.entries) != 1 or reference.entries[0].generated_tokens.numel() != 100:
        raise ValueError("datatype sweep requires the one-entry 100-token readiness reference")

    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    generator = None
    try:
        generator = build_generator(
            model_dir=args.model_dir.resolve(),
            mesh_device=mesh,
            precision_config_path=precision_path,
            tensor_cache_path=args.tensor_cache,
        )
        runtime_summary = generator.model.precision_runtime_summary()
        prefill_stats = None
        if args.run_prefill:
            prefill_stats = _run_one_entry_prefill(
                generator=generator,
                entry=reference.entries[0],
                reference=reference,
            )
            generator.reset()
        teacher_stats = _teacher_forcing(generator, reference_path)
        counters = dict(generator.model.trace_state.counters)
        trace_verified = counters["model_trace_replays"] == teacher_stats["decode_tokens"]
        if not trace_verified:
            raise RuntimeError(
                "teacher-forcing performance is not trace verified: "
                f"replays={counters['model_trace_replays']} decode_tokens={teacher_stats['decode_tokens']}"
            )
        passed = (
            teacher_stats["top1"] >= args.min_top1
            and teacher_stats["top5"] >= args.min_top5
            and teacher_stats["top100"] >= args.min_top100
        )
        result = {
            "config_id": runtime_summary["config_id"],
            "precision_config": str(precision_path),
            "dtype_policy": json.loads(precision_path.read_text(encoding="utf-8")),
            "runtime_policy_summary": runtime_summary,
            "accuracy": {
                "teacher_forcing": teacher_stats,
                "prefill": prefill_stats,
                "thresholds": {
                    "top1": args.min_top1,
                    "top5": args.min_top5,
                    "top100": args.min_top100,
                },
            },
            "performance": {
                "ttft_ms": teacher_stats["ttft_ms"],
                "trace_verified_teacher_forcing_decode_t/s/u": teacher_stats["decode_t/s/u"],
                "teacher_forcing_e2e_t/s/u": teacher_stats["e2e_t/s/u"],
                "measurement_regime": "full-60-layer batch-1 traced teacher forcing; AIME24 149-token prompt; 100 reference tokens",
            },
            "trace_verified": trace_verified,
            "trace_counters": counters,
            "pass": passed,
            "status": "pass" if passed else "fail_accuracy",
            "command": shlex.join(sys.argv),
            "git_commit": _git_revision(),
            "hardware": "4x Blackhole P150b",
            "mesh": "MeshShape(1,4) TP4 FABRIC_1D",
            "reference": str(reference_path),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "config_id": result["config_id"],
                    "top1": teacher_stats["top1"],
                    "top5": teacher_stats["top5"],
                    "top100": teacher_stats["top100"],
                    "ttft_ms": teacher_stats["ttft_ms"],
                    "decode_t/s/u": teacher_stats["decode_t/s/u"],
                    "trace_verified": trace_verified,
                    "status": result["status"],
                },
                indent=2,
            )
        )
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--precision-config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-cache", type=Path, default=Path("/tmp/gemma4_31b_datatype_sweep_cache"))
    parser.add_argument("--run-prefill", action="store_true")
    parser.add_argument("--min-top1", type=float, default=0.90)
    parser.add_argument("--min-top5", type=float, default=0.98)
    parser.add_argument("--min-top100", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
