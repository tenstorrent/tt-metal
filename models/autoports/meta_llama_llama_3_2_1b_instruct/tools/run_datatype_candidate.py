# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run one full-model datatype-sweep candidate and write structured evidence."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.generator import build_generator
from models.common.readiness_check.mesh_device import (
    add_mesh_device_args,
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)
from models.common.readiness_check.run_prefill_check import run_prefill_check
from models.common.readiness_check.teacher_forcing import TokenAccuracy


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _aggregate_accuracy(per_entry: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(s["total"]) for s in per_entry)
    if total == 0:
        return {}
    aggregate = {
        "top1": sum(int(s["matches_top1"]) for s in per_entry) / total,
        "top5": sum(int(s["matches_top5"]) for s in per_entry) / total,
        "top100": sum(int(s["matches_top100"]) for s in per_entry) / total,
        "matches_top1": sum(int(s["matches_top1"]) for s in per_entry),
        "matches_top5": sum(int(s["matches_top5"]) for s in per_entry),
        "matches_top100": sum(int(s["matches_top100"]) for s in per_entry),
        "total": total,
        "k": int(per_entry[0]["k"]),
    }
    total_elapsed_s = sum(float(s.get("elapsed_s", 0.0)) for s in per_entry)
    if total_elapsed_s > 0:
        aggregate["elapsed_s"] = total_elapsed_s
        aggregate["e2e_t/s/u"] = total / total_elapsed_s
    ttft_values = [float(s["ttft_ms"]) for s in per_entry if s.get("ttft_ms") is not None]
    if ttft_values:
        aggregate["ttft_ms"] = sum(ttft_values) / len(ttft_values)
    decode_tokens = sum(float(s.get("decode_tokens", 0.0)) for s in per_entry)
    decode_elapsed_s = sum(float(s.get("decode_elapsed_s", 0.0)) for s in per_entry)
    if decode_elapsed_s > 0:
        aggregate["decode_tokens"] = decode_tokens
        aggregate["decode_elapsed_s"] = decode_elapsed_s
        aggregate["decode_t/s/u"] = decode_tokens / decode_elapsed_s
    return aggregate


def _compute_perf_stats(*, timing: dict[str, Any], end_s: float, token_count: int) -> dict[str, float]:
    start_s = timing["start_s"]
    first_token_s = timing["first_token_s"]
    elapsed_s = max(end_s - start_s, 0.0)
    perf = {
        "elapsed_s": elapsed_s,
        "e2e_t/s/u": (token_count / elapsed_s) if elapsed_s > 0 else 0.0,
    }
    if first_token_s is None:
        return perf
    ttft_s = max(first_token_s - start_s, 0.0)
    perf["ttft_ms"] = ttft_s * 1000.0
    decode_tokens = max(token_count - 1, 0)
    if decode_tokens > 0:
        decode_end_s = timing["last_decode_token_s"] if timing["last_decode_token_s"] is not None else end_s
        decode_elapsed_s = max(decode_end_s - first_token_s, 0.0)
        perf["decode_tokens"] = float(decode_tokens)
        perf["decode_elapsed_s"] = decode_elapsed_s
        perf["decode_t/s/u"] = (decode_tokens / decode_elapsed_s) if decode_elapsed_s > 0 else 0.0
    return perf


def _run_teacher_forcing_with_trace_audit(
    *,
    model_dir: Path,
    reference_path: Path,
    mesh_device,
    precision_config_path: Path | None,
) -> dict[str, Any]:
    generator = build_generator(
        model_dir=model_dir,
        mesh_device=mesh_device,
        precision_config_path=precision_config_path,
    )
    acc = TokenAccuracy(reference_path)
    per_entry: list[dict[str, Any]] = []
    trace_audits: list[dict[str, Any]] = []
    try:
        for entry_idx in range(acc.num_entries):
            if entry_idx > 0:
                generator.reset()
            prompt_ids = acc.get_prompt_token_ids(entry_idx)
            n_steps = acc.num_gt_tokens(entry_idx)
            timing: dict[str, Any] = {
                "start_s": None,
                "first_token_s": None,
                "last_decode_token_s": None,
                "callback_count": 0,
            }

            def next_input(_step: int, predicted: int) -> int:
                now = time.perf_counter()
                if timing["first_token_s"] is None:
                    timing["first_token_s"] = now
                else:
                    timing["last_decode_token_s"] = now
                timing["callback_count"] += 1
                return acc.collect_predicted_tokens(predicted, user_idx=entry_idx)

            timing["start_s"] = time.perf_counter()
            generator.generate(
                prompt_token_ids=prompt_ids,
                max_new_tokens=n_steps,
                next_input=next_input,
                enable_trace=True,
            )
            end_s = time.perf_counter()
            if acc.num_pred_tokens(entry_idx) != n_steps or int(timing["callback_count"]) != n_steps:
                raise RuntimeError(
                    f"entry {entry_idx} teacher forcing produced {acc.num_pred_tokens(entry_idx)}/{n_steps} "
                    f"predictions via {timing['callback_count']} callbacks"
                )

            stats = acc.compute_accuracy(user_idx=entry_idx)
            stats.update(_compute_perf_stats(timing=timing, end_s=end_s, token_count=stats["total"]))
            per_entry.append(stats)
            trace_audits.append(generator.trace_audit())
    finally:
        teardown = getattr(generator, "teardown", None)
        if callable(teardown):
            teardown()

    aggregate = _aggregate_accuracy(per_entry)
    trace_verified = all(
        bool(audit["trace_state"]["model_trace_captured"])
        and int(audit["counters"]["model_trace_captures"]) >= 1
        and (
            int(audit["counters"]["model_trace_captures"]) + int(audit["counters"]["model_trace_replays"])
        )
        >= max(int(entry["total"]) - 1, 0)
        and bool(audit["last_generation"].get("teacher_forcing"))
        for audit, entry in zip(trace_audits, per_entry)
    )
    return {
        "entries": per_entry,
        "aggregate": aggregate,
        "trace_verified": trace_verified,
        "trace_evidence": trace_audits,
    }


def _reference_shape(reference_path: Path) -> dict[str, Any]:
    from models.common.readiness_check.schema import load_reference

    reference = load_reference(reference_path)
    return {
        "hf_model_id": reference.hf_model_id,
        "k": reference.k,
        "entries": len(reference.entries),
        "prompt_lens": [int(entry.prompt_tokens.shape[1]) for entry in reference.entries],
        "generated_lens": [int(entry.generated_tokens.shape[1]) for entry in reference.entries],
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run one Llama-3.2-1B datatype sweep candidate.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--precision-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-top1", type=float, default=0.90)
    parser.add_argument("--min-top5", type=float, default=0.98)
    add_mesh_device_args(parser)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    reference_path = args.reference.resolve()
    precision_config_path = args.precision_config.resolve() if args.precision_config is not None else None

    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    try:
        build_kwargs = {"precision_config_path": precision_config_path}
        prefill_entries = run_prefill_check(
            model_dir=model_dir,
            reference_path=reference_path,
            mesh_device=mesh_device,
            build_kwargs=build_kwargs,
        )
        teacher_forcing = _run_teacher_forcing_with_trace_audit(
            model_dir=model_dir,
            reference_path=reference_path,
            mesh_device=mesh_device,
            precision_config_path=precision_config_path,
        )
    finally:
        close_readiness_mesh_device(mesh_device, args.fabric_config)

    prefill_aggregate = _aggregate_accuracy(prefill_entries)
    tf_aggregate = teacher_forcing["aggregate"]
    passed = (
        bool(teacher_forcing["trace_verified"])
        and float(tf_aggregate["top1"]) >= args.min_top1
        and float(tf_aggregate["top5"]) >= args.min_top5
    )
    result = {
        "config_id": args.config_id,
        "precision_config_path": str(precision_config_path) if precision_config_path is not None else None,
        "reference": {
            "path": str(reference_path),
            **_reference_shape(reference_path),
        },
        "thresholds": {
            "top1": args.min_top1,
            "top5": args.min_top5,
            "top100": "recorded_not_gated",
        },
        "prefill": {
            "entries": prefill_entries,
            "aggregate": prefill_aggregate,
        },
        "teacher_forcing": teacher_forcing,
        "measurement_regime": {
            "selection_metric": "trace-verified teacher-forcing decode t/s/u",
            "workload": "batch-1 AIME24 chat-template reference with 100 generated tokens",
            "prefill": "run_prefill_check-compatible full-sequence prefill accuracy",
            "decode": "generator.generate(..., next_input=..., enable_trace=True)",
        },
        "pass_fail": {
            "passed": passed,
            "reasons": {
                "top1_gate": float(tf_aggregate["top1"]) >= args.min_top1,
                "top5_gate": float(tf_aggregate["top5"]) >= args.min_top5,
                "trace_verified": bool(teacher_forcing["trace_verified"]),
            },
        },
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "repo": {
            "branch": _git_value(["branch", "--show-current"]),
            "commit": _git_value(["rev-parse", "HEAD"]),
        },
        "hardware": {
            "mesh_device": args.mesh_device,
            "fabric_config": args.fabric_config,
            "mesh_shape": [1, 8],
            "platform": platform.platform(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    _main()
