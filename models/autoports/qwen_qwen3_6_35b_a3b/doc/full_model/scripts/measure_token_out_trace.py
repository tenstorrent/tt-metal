# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure traced batch-1 token-out generation for the Qwen full model."""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT = MODEL_DIR / "doc/full_model/artifacts/token_out_trace_perf_default_prompt_100.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--max-new-tokens", type=int, default=100)
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

    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    try:
        generator = build_generator(
            model_dir=args.model_dir.resolve(),
            mesh_device=mesh_device,
            model_id=args.hf_model,
            local_files_only=args.local_files_only,
        )
        try:
            tokens = generator.generate(
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=args.max_new_tokens,
                next_input=None,
                enable_trace=True,
            )
            trace = generator._trace
            timings = dict(generator.last_timings)
        finally:
            teardown = getattr(generator, "teardown", None)
            if callable(teardown):
                teardown()
    finally:
        close_readiness_mesh_device(mesh_device, args.fabric_config)

    decode_tokens = int(timings.get("decode_tokens", max(len(tokens) - 1, 0)))
    decode_s = float(timings.get("decode_s", 0.0))
    e2e_s = float(timings.get("e2e_s", 0.0))
    payload = {
        "prompt_file": str(args.prompt_file),
        "prompt_len": len(prompt_token_ids),
        "max_new_tokens": args.max_new_tokens,
        "num_tokens": len(tokens),
        "tokens": [int(token_id) for token_id in tokens],
        "completion": tokenizer.decode(tokens, skip_special_tokens=False),
        "ttft_ms": float(timings.get("ttft_s", 0.0)) * 1000.0,
        "decode_s": decode_s,
        "e2e_s": e2e_s,
        "decode_tokens": decode_tokens,
        "trace_decode_t_s_u": (decode_tokens / decode_s) if decode_s > 0 else 0.0,
        "e2e_t_s_u": (len(tokens) / e2e_s) if e2e_s > 0 else 0.0,
        "trace_present": trace is not None,
        "trace_generated_steps": trace.generated if trace is not None else 0,
        "position_end_expected_exclusive": len(prompt_token_ids) + len(tokens),
        "raw_timings": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
