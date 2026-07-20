# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reduced real-weight full-model profile with terminal and trace signposts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tracy import signpost

import ttnn
from models.autoports.tiiuae_falcon3_10b_base.tt.generator import _first_device_to_torch, build_generator
from models.common.readiness_check.schema import load_reference


def collect(model_dir: Path, reference_path: Path, output: Path, weight_cache_path: str) -> dict:
    reference = load_reference(reference_path)
    prompt = reference.entries[0].prompt_tokens[0].tolist()[:128]
    mesh = None
    generator = None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=256_000_000)
        generator = build_generator(
            model_dir,
            mesh,
            override_num_layers=1,
            max_context_len=256,
            weight_cache_path=weight_cache_path,
        )
        kv_cache = generator._ensure_kv_cache()
        page_table = generator._make_page_table([256])

        warm_logits = generator._prefill_single_local_logits(prompt, page_table, kv_cache)
        generator.model.sample_greedy_split(warm_logits, tt_out_tok=generator._prefill_sampled)
        ttnn.synchronize_device(mesh)
        generator.reset()

        ttnn.ReadDeviceProfiler(mesh)
        signpost(header="PERF_PREFILL")
        prefill_start = time.perf_counter()
        measured_logits = generator._prefill_single_local_logits(prompt, page_table, kv_cache)
        measured_token = generator.model.sample_greedy_split(measured_logits, tt_out_tok=generator._prefill_sampled)
        ttnn.synchronize_device(mesh)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        signpost(header="PERF_PREFILL_END")
        ttnn.ReadDeviceProfiler(mesh)
        prefill_token = int(_first_device_to_torch(measured_token).reshape(-1)[0])

        generator.reset()
        generated = generator.generate(prompt, 2, sampling_mode="device", enable_trace=True)
        position_before = int(_first_device_to_torch(generator._trace_inputs[1]).reshape(-1)[0])
        replays_before = generator.trace_stats["replays"]
        ttnn.ReadDeviceProfiler(mesh)
        signpost(header="PERF_DECODE")
        decode_start = time.perf_counter()
        for _ in range(3):
            generator._replay_split_sampling()
        ttnn.synchronize_device(mesh)
        decode_total_ms = (time.perf_counter() - decode_start) * 1000.0
        signpost(header="PERF_DECODE_END")
        ttnn.ReadDeviceProfiler(mesh)
        position_after = int(_first_device_to_torch(generator._trace_inputs[1]).reshape(-1)[0])
        feedback_token = int(_first_device_to_torch(generator._trace_inputs[0]).reshape(-1)[0])

        result = {
            "mesh": "4x Blackhole p300c, 1x4 FABRIC_1D_RING, TP4",
            "layers_executed": 1,
            "weights": "real Falcon3 checkpoint embedding, layer 0, final norm, untied TP LM head",
            "prompt_tokens": len(prompt),
            "prefill_wall_ms": prefill_ms,
            "prefill_greedy_token": prefill_token,
            "trace_generated_tokens": generated,
            "decode_profile_iterations": 3,
            "decode_wall_total_ms": decode_total_ms,
            "decode_wall_ms_per_token": decode_total_ms / 3,
            "position_before_profile": position_before,
            "position_after_profile": position_after,
            "feedback_token_after_profile": feedback_token,
            "profile_replay_delta": generator.trace_stats["replays"] - replays_before,
            "trace_stats": dict(generator.trace_stats),
        }
        result["passed"] = bool(
            result["layers_executed"] == 1
            and result["prompt_tokens"] == 128
            and result["position_after_profile"] == result["position_before_profile"] + 3
            and result["profile_replay_delta"] == 3
            and result["decode_wall_total_ms"] > 0.0
            and result["prefill_wall_ms"] > 0.0
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    finally:
        if generator is not None:
            generator.teardown()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-cache-path", default="/tmp/falcon3-full-model-cache")
    args = parser.parse_args()
    result = collect(args.model_dir, args.reference, args.output, args.weight_cache_path)
    if not result["passed"]:
        raise SystemExit("reduced full-model profile gate failed")


if __name__ == "__main__":
    main()
