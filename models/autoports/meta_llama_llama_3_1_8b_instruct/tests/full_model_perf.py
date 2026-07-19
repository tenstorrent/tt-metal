# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Measure the public batch-1 prompt-128/generate-128 token-out contract."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.generator import _round_up, build_generator
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device


def _sync(mesh_device) -> None:
    ttnn.synchronize_device(mesh_device)


def collect(model_dir: Path, reference: Path, output: Path) -> dict:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    reference_data = torch.load(reference, map_location="cpu", weights_only=False)
    prompt = reference_data["entries"][0]["prompt_tokens"][0, :128].tolist()
    if len(prompt) != 128:
        raise RuntimeError("the performance workload requires exactly 128 prompt tokens")

    mesh_device = open_readiness_mesh_device("P300", "FABRIC_1D_RING")
    try:
        generator = build_generator(model_dir, mesh_device)

        # Compile/capture the exact public device-sampling path before timing.
        generator.generate(prompt, 2, enable_trace=True, sampling_mode="device")
        generator.reset()

        kv_cache = generator._ensure_kv_cache()
        page_table = generator._make_page_table([_round_up(128 + 128, 64)])
        prompt_tensor = torch.tensor([prompt], dtype=torch.long)
        counters_before_ttft = dict(generator.trace_stats)
        _sync(mesh_device)
        start = time.perf_counter()
        first_token = generator.prefill_forward(
            prompt_tensor,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[128],
            sampling_mode="device",
            enable_trace=True,
        )
        first_token_id = int(generator._sampled_tokens_to_torch(first_token)[0])
        _sync(mesh_device)
        ttft_s = time.perf_counter() - start
        ttft_counter_delta = {
            name: generator.trace_stats[name] - counters_before_ttft[name] for name in generator.trace_stats
        }
        generator.reset()

        counters_before = dict(generator.trace_stats)
        _sync(mesh_device)
        start = time.perf_counter()
        generated = generator.generate(
            prompt,
            128,
            enable_trace=True,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
        )
        _sync(mesh_device)
        end_to_end_s = time.perf_counter() - start
        counter_delta = {name: generator.trace_stats[name] - counters_before[name] for name in generator.trace_stats}
        decode_s = end_to_end_s - ttft_s
        result = {
            "mesh": "P300 1x4 FABRIC_1D_RING TP=4",
            "num_layers": generator.model.num_layers,
            "workload": {"prompt_tokens": 128, "generated_tokens": 128, "batch": 1},
            "sampling": {"top_k": 1, "top_p": 0.0, "temperature": 1.0},
            "first_token": first_token_id,
            "output_token_count": len(generated),
            "ttft_ms": ttft_s * 1000,
            "ttft_trace_counter_delta": ttft_counter_delta,
            "end_to_end_ms": end_to_end_s * 1000,
            "token_out_decode_ms": decode_s * 1000,
            "token_out_decode_ms_per_token": decode_s * 1000 / 127,
            "token_out_decode_t_s_u": 127 / decode_s,
            "trace_counter_delta": counter_delta,
            "feedback_contract": (
                "Sampler tt_out_tok writes the persistent decode-token tensor; caller token-ID "
                "readback is output-only and is never copied back as feedback."
            ),
            "public_api_contract": (
                "TTFT calls public prefill_forward with device sampling; end-to-end calls public generate, "
                "which delegates to public prefill_forward/decode_forward and reuses compatible traces."
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    finally:
        close_readiness_mesh_device(mesh_device, "FABRIC_1D_RING")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.model_dir, args.reference, args.output)


if __name__ == "__main__":
    main()
