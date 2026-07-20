# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Attribute full-model trace scaling across real Falcon decoder depths.

The sweep deliberately uses the same 128-token prompt, 128 replay window,
32K cache construction, terminal path, and sampler as the official full-model
measurement.  Depth regression therefore isolates the effective decoder-layer
slope from the fixed embedding/final-norm/LM-head/trace intercept.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.tiiuae_falcon3_10b_base.tt.generator import build_generator
from models.common.readiness_check.schema import load_reference

INHERITED_PERF_ARTIFACT = (
    Path(__file__).parents[1] / "doc" / "optimized_multichip_decoder" / "results" / "final_default_batch1_perf.json"
)


def _timed(mesh, function, iterations: int) -> float:
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    ttnn.synchronize_device(mesh)
    return time.perf_counter() - start


def _linear_fit(points: list[dict]) -> dict[str, float]:
    xs = [float(point["layers"]) for point in points]
    ys = [float(point["model_trace_ms_per_token"]) for point in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - y_mean) ** 2 for y in ys)
    return {
        "slope_ms_per_layer": slope,
        "intercept_ms": intercept,
        "r_squared": 1.0 if total == 0.0 else 1.0 - residual / total,
    }


def _measure_depth(
    model_dir: Path,
    prompt: list[int],
    *,
    layers: int,
    iterations: int,
    max_context_len: int,
    weight_cache_path: str,
) -> dict:
    mesh = None
    generator = None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=512_000_000)
        generator = build_generator(
            model_dir,
            mesh,
            override_num_layers=layers,
            max_context_len=max_context_len,
            weight_cache_path=weight_cache_path,
        )
        generated = generator.generate(prompt, 2, sampling_mode="device", enable_trace=True)
        page_table = generator._page_table_host
        kv_cache = generator._kv_cache
        host_inputs = generator._prepare_decode_host_inputs(
            torch.tensor([generated[-1]]),
            torch.tensor([len(prompt)]),
            page_table,
        )

        generator._refresh_trace_state(host_inputs, kv_cache, active_batch=1)
        model_s = _timed(
            mesh,
            lambda: ttnn.execute_trace(mesh, generator._trace_model_id, cq_id=0, blocking=False),
            iterations,
        )
        generator._refresh_trace_state(host_inputs, kv_cache, active_batch=1)
        sampler_s = _timed(
            mesh,
            lambda: ttnn.execute_trace(mesh, generator._trace_sampling_id, cq_id=0, blocking=False),
            iterations,
        )
        generator._refresh_trace_state(host_inputs, kv_cache, active_batch=1)
        pair_s = _timed(mesh, generator._replay_split_sampling, iterations)
        generator._refresh_trace_state(host_inputs, kv_cache, active_batch=1)

        def pair_and_read():
            generator._sampled_to_torch(generator._replay_split_sampling())

        caller_s = _timed(mesh, pair_and_read, iterations)
        return {
            "layers": layers,
            "iterations": iterations,
            "model_trace_ms_per_token": model_s * 1000.0 / iterations,
            "sampling_trace_ms_per_token": sampler_s * 1000.0 / iterations,
            "device_only_pair_ms_per_token": pair_s * 1000.0 / iterations,
            "device_only_pair_t_s_u": iterations / pair_s,
            "caller_visible_pair_ms_per_token": caller_s * 1000.0 / iterations,
            "caller_visible_pair_t_s_u": iterations / caller_s,
            "caller_token_readbacks": generator.trace_stats["caller_token_readbacks"],
            "token_feedback_mode": "sampling-trace output copied directly to model-trace input",
            "max_context_len": generator.model.max_cache_len,
        }
    finally:
        if generator is not None:
            generator.teardown()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def collect(
    model_dir: Path,
    reference_path: Path,
    output: Path,
    *,
    depths: list[int],
    iterations: int,
    max_context_len: int,
    weight_cache_path: str,
) -> dict:
    if len(depths) < 3 or depths != sorted(set(depths)) or depths[0] < 1 or depths[-1] > 40:
        raise ValueError("depths must contain at least three unique ascending values in [1,40]")
    if iterations < 16 or 128 + iterations > 256:
        raise ValueError("iterations must be in [16,128]")
    reference = load_reference(reference_path)
    prompt = reference.entries[0].prompt_tokens[0].tolist()[:128]
    points = [
        _measure_depth(
            model_dir,
            prompt,
            layers=depth,
            iterations=iterations,
            max_context_len=max_context_len,
            weight_cache_path=weight_cache_path,
        )
        for depth in depths
    ]
    fit = _linear_fit(points)
    inherited = json.loads(INHERITED_PERF_ARTIFACT.read_text(encoding="utf-8"))
    inherited_layer_ms = float(inherited["multichip_trace_ms"])
    inherited_projected_stack_ms = inherited_layer_ms * 40
    full_depth = points[-1]
    fitted_full_depth_ms = fit["intercept_ms"] + fit["slope_ms_per_layer"] * depths[-1]
    result = {
        "mesh": "4x Blackhole p300c, 1x4 FABRIC_1D_RING, TP4",
        "workload": {
            "batch": 1,
            "prompt_tokens": len(prompt),
            "replay_tokens": iterations,
            "max_context_len": max_context_len,
            "matches_official_128_128": len(prompt) == iterations == 128 and max_context_len == 32768,
        },
        "depths": depths,
        "points": points,
        "model_trace_linear_fit": fit,
        "like_for_like_effective_decoder_slope_ms": fit["slope_ms_per_layer"],
        "fixed_embedding_terminal_trace_intercept_ms": fit["intercept_ms"],
        "full_depth_fit": {
            "layers": depths[-1],
            "measured_model_trace_ms": full_depth["model_trace_ms_per_token"],
            "fitted_model_trace_ms": fitted_full_depth_ms,
            "fit_residual_ms": full_depth["model_trace_ms_per_token"] - fitted_full_depth_ms,
            "sampling_trace_ms": full_depth["sampling_trace_ms_per_token"],
            "caller_visible_ms": full_depth["caller_visible_pair_ms_per_token"],
        },
        "inherited_decoder_reference": {
            "artifact": str(INHERITED_PERF_ARTIFACT.relative_to(Path(__file__).parents[4])),
            "batch": int(inherited["batch"]),
            "sequence_length": int(inherited["sequence_length"]),
            "real_layer": 20,
            "trace_ms_per_layer": inherited_layer_ms,
            "projected_40_layer_ms": inherited_projected_stack_ms,
            "comparison_class": (
                "provisional non-like-for-like reference: sequence 17 and isolated layer; "
                "the depth-regression slope above is the like-for-like 128--255 position control"
            ),
            "effective_slope_delta_percent": 100.0
            * (fit["slope_ms_per_layer"] - inherited_layer_ms)
            / inherited_layer_ms,
        },
        "policy": {
            "tensor_parallel": 4,
            "topology": "1x4 FABRIC_1D_RING, two links",
            "weight_and_math": "BFP4_B/LoFi decoder and LM-head matmuls",
            "activation_residual_and_ccl": "BF16",
            "kv_cache": "BFP8_B paged, one local KV head per rank",
            "inter_layer_residual": "BF16 TILE L1 width-sharded, replicated values across TP ranks",
        },
    }
    result["passed"] = bool(
        result["workload"]["matches_official_128_128"]
        and result["depths"][-1] == 40
        and all(point["max_context_len"] == 32768 for point in points)
        and all(point["caller_visible_pair_t_s_u"] > 0.0 for point in points)
        and fit["slope_ms_per_layer"] > 0.0
        and fit["r_squared"] >= 0.98
        and abs(result["full_depth_fit"]["fit_residual_ms"]) < 0.5
        and result["full_depth_fit"]["sampling_trace_ms"] < 0.5 * result["full_depth_fit"]["caller_visible_ms"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", default="1,2,4,8,16,40")
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--max-context-len", type=int, default=32768)
    parser.add_argument("--weight-cache-path", default="/tmp/falcon3-full-model-cache")
    args = parser.parse_args()
    depths = [int(value) for value in args.depths.split(",") if value]
    result = collect(
        args.model_dir,
        args.reference,
        args.output,
        depths=depths,
        iterations=args.iterations,
        max_context_len=args.max_context_len,
        weight_cache_path=args.weight_cache_path,
    )
    if not result["passed"]:
        raise SystemExit("full-model depth-scaling gate failed")


if __name__ == "__main__":
    main()
