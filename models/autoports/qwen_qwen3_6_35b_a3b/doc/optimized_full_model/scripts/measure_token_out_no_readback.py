# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure serving-style traced token-out decode for the optimized Qwen full model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import MODEL_ID
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import build_generator
from models.common.readiness_check.mesh_device import (
    add_mesh_device_args,
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)

MODEL_DIR = Path("models/autoports/qwen_qwen3_6_35b_a3b")
DEFAULT_PROMPT_FILE = Path("models/common/readiness_check/autoregressive_prompt.txt")
DEFAULT_OUTPUT = MODEL_DIR / "doc/optimized_full_model/artifacts/token_out_no_readback_prompt128_gen128.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--include-readback-baseline", action="store_true")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    add_mesh_device_args(parser)
    args = parser.parse_args()

    prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    prompt_token_ids = [int(token_id) for token_id in tokenizer.encode(prompt_text, add_special_tokens=True)]
    if args.prompt_len > 0:
        if len(prompt_token_ids) >= args.prompt_len:
            prompt_token_ids = prompt_token_ids[: args.prompt_len]
        else:
            eos = tokenizer.eos_token_id
            if eos is None:
                eos = 0
            prompt_token_ids = prompt_token_ids + [int(eos)] * (args.prompt_len - len(prompt_token_ids))

    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    try:
        generator = build_generator(
            model_dir=args.model_dir.resolve(),
            mesh_device=mesh_device,
            model_id=args.hf_model,
            local_files_only=args.local_files_only,
        )
        try:
            runtime_policy_summary = generator.model.describe_precision_policy()
            warmup_metrics = None
            if args.warmup:
                warmup_metrics = generator.measure_token_out_no_readback(
                    prompt_token_ids=prompt_token_ids,
                    max_new_tokens=1,
                    validate_final_token=False,
                )

            readback_baseline = None
            if args.include_readback_baseline:
                baseline_tokens = generator.generate(
                    prompt_token_ids,
                    max_new_tokens=args.max_new_tokens,
                    enable_trace=True,
                )
                baseline_trace = generator._trace
                baseline_timings = dict(generator.last_timings)
                baseline_decode_tokens = max(0, args.max_new_tokens - 1)
                baseline_decode_s = float(baseline_timings.get("decode_s", 0.0))
                baseline_e2e_s = float(baseline_timings.get("e2e_s", 0.0))
                readback_baseline = {
                    "description": "completed full-model traced token-out path with per-token sync/readback",
                    "prompt_len": len(prompt_token_ids),
                    "max_new_tokens": args.max_new_tokens,
                    "decode_tokens": baseline_decode_tokens,
                    "ttft_ms": float(baseline_timings.get("ttft_s", 0.0)) * 1000.0,
                    "decode_s": baseline_decode_s,
                    "decode_t_s_u": (baseline_decode_tokens / baseline_decode_s) if baseline_decode_s > 0 else 0.0,
                    "e2e_t_s_u": (args.max_new_tokens / baseline_e2e_s) if baseline_e2e_s > 0 else 0.0,
                    "trace_present": baseline_trace is not None,
                    "trace_generated_steps": baseline_trace.generated if baseline_trace is not None else 0,
                    "host_boundary_counters": dict(generator.last_trace_counters),
                    "first_token": int(baseline_tokens[0]) if baseline_tokens else None,
                    "final_token": int(baseline_tokens[-1]) if baseline_tokens else None,
                }

            metrics = generator.measure_token_out_no_readback(
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=args.max_new_tokens,
            )
            trace = generator._trace
            timings = dict(generator.last_timings)
            counters = dict(generator.last_trace_counters)
        finally:
            teardown = getattr(generator, "teardown", None)
            if callable(teardown):
                teardown()
    finally:
        close_readiness_mesh_device(mesh_device, args.fabric_config)

    decode_tokens = int(timings.get("decode_tokens", max(args.max_new_tokens - 1, 0)))
    decode_replay_s = float(timings.get("decode_replay_s", timings.get("decode_s", 0.0)))
    decode_including_capture_s = float(timings.get("decode_s_including_capture", decode_replay_s))
    e2e_s = float(timings.get("e2e_s", 0.0))
    payload = {
        "measurement_regime": "serving_style_token_out_no_per_token_readback",
        "prompt_file": str(args.prompt_file),
        "prompt_len": len(prompt_token_ids),
        "max_new_tokens": args.max_new_tokens,
        "decode_tokens": decode_tokens,
        "first_token": metrics["first_token"],
        "final_token": metrics["final_token"],
        "ttft_ms": float(timings.get("ttft_s", 0.0)) * 1000.0,
        "trace_capture_ms": float(timings.get("trace_capture_s", 0.0)) * 1000.0,
        "decode_replay_s": decode_replay_s,
        "decode_replay_t_s_u": (decode_tokens / decode_replay_s) if decode_replay_s > 0 else 0.0,
        "decode_including_capture_s": decode_including_capture_s,
        "decode_including_capture_t_s_u": (decode_tokens / decode_including_capture_s)
        if decode_including_capture_s > 0
        else 0.0,
        "e2e_s": e2e_s,
        "e2e_t_s_u": (args.max_new_tokens / e2e_s) if e2e_s > 0 else 0.0,
        "trace_present": trace is not None,
        "trace_generated_steps": trace.generated if trace is not None else 0,
        "position_end_expected_exclusive": metrics["position_end_expected_exclusive"],
        "host_boundary_counters": counters,
        "raw_timings": timings,
        "runtime_policy_summary": runtime_policy_summary,
        "command": " ".join(sys.argv),
        "env": {
            "QWEN36_PRECISION_CONFIG": os.environ.get("QWEN36_PRECISION_CONFIG"),
            "TT_READINESS_TRACE_REGION_SIZE": os.environ.get("TT_READINESS_TRACE_REGION_SIZE"),
            "TT_METAL_WATCHER_DISABLE_ETH": os.environ.get("TT_METAL_WATCHER_DISABLE_ETH"),
        },
        "warmup_before_measurement": bool(args.warmup),
        "warmup_metrics": warmup_metrics,
        "before_readback_baseline": readback_baseline,
    }
    if readback_baseline is not None:
        payload["matches_readback_final_token"] = metrics["final_token"] == readback_baseline["final_token"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
