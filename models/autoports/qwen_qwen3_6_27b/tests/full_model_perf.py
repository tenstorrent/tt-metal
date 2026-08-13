# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synchronized B1 full-model TTFT and canonical split-trace token-out benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from tracy import signpost
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--layer-indices", type=int, nargs="+", default=None)
    parser.add_argument("--profile-only-decode", action="store_true")
    parser.add_argument("--candidate-gather-greedy", action="store_true")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    try:
        generator = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"), mesh_device=mesh,
            max_context=512, batch=1, num_layers=args.num_layers, layer_indices=args.layer_indices,
            force_argmax_greedy=not args.candidate_gather_greedy,
        )
        rendered = generator.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt.read_text().strip()}],
            tokenize=False, add_generation_prompt=True,
        )
        token_ids = generator.tokenizer.encode(rendered, add_special_tokens=False)
        token_ids = token_ids[: args.prompt_tokens]
        tokens = torch.tensor([token_ids], dtype=torch.long)

        generator.reset()
        ttnn.synchronize_device(mesh)
        started = time.perf_counter()
        logits = generator.prefill_forward(
            tokens, page_table=generator._page_table, kv_cache=generator.kv_cache,
            prompt_lens=[len(token_ids)],
        )
        ttnn.synchronize_device(mesh)
        ttft_seconds = time.perf_counter() - started
        first_token = int(torch.argmax(logits[0, 0]).item())

        capture_started = time.perf_counter()
        generator._capture_token_out_trace(first_token, len(token_ids))
        generator._seed_token_out_trace(first_token, len(token_ids))
        ttnn.synchronize_device(mesh)
        capture_seconds = time.perf_counter() - capture_started

        output = [first_token]
        if args.profile_only_decode:
            # Construction, prefill, compilation and trace capture contain far
            # more markers than device profiler buffers can retain. Flush them
            # before the requested steady-state terminal/sampler replay.
            ttnn.ReadDeviceProfiler(mesh)
            signpost("FULL_MODEL_DECODE", "one reduced model + canonical sampler replay")
        started = time.perf_counter()
        for _ in range(args.decode_tokens):
            ttnn.execute_trace(mesh, generator._decode_trace_id, cq_id=0, blocking=False)
            generator.sampling.sample(generator._trace_logits, enable_trace=True, tt_out_tok=generator._trace_token)
            generator.trace_counters["replays"] += 1
            output.append(generator._read_sampled_token())
        ttnn.synchronize_device(mesh)
        decode_seconds = time.perf_counter() - started
        if args.profile_only_decode:
            ttnn.ReadDeviceProfiler(mesh)
            signpost("FULL_MODEL_DECODE_END")

        result = {
            "prompt_tokens": len(token_ids),
            "measured_decode_replays": args.decode_tokens,
            "ttft_seconds": ttft_seconds,
            "ttft_ms": 1000 * ttft_seconds,
            "trace_capture_seconds": capture_seconds,
            "token_out_seconds": decode_seconds,
            "token_out_t_s_u": args.decode_tokens / decode_seconds,
            "trace_counters": dict(generator.trace_counters),
            "canonical_split_sampling": True,
            "model_trace_id": str(generator._decode_trace_id),
            "sampler_trace_live": any(
                slot["id"] is not None for slot in generator.sampling._trace_states.values()
            ),
            "token_ids": output,
            "text": generator.tokenizer.decode(output, skip_special_tokens=False),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key not in ("token_ids", "text")}, indent=2))
    finally:
        if generator is not None:
            generator.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
