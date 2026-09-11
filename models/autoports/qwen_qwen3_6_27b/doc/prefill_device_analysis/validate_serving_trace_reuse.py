# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real reduced-adapter A/B: repeated S128 prefill, three decode steps, reset.

Run only after obtaining the coordinating agent's hardware window. The default
loads four layers with 32 allocated slots, one active request and greedy sampling.
Allocation tracking is required; its overhead is present in both timing arms.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.doc.prefill_device_analysis.serving_trace_reuse_runtime import (
    ServingTraceReuseExperiment,
)
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_6_27b.tt.generator_vllm import Qwen36ForCausalLM
from models.common.sampling import SamplingParams


def cache_digests(generator):
    """Read all ranks and all fixed slots; digest logical tensor bytes exactly."""
    result = {}
    for layer in generator.model.layers:
        for name in ("conv", "recurrent", "key", "value"):
            if name not in layer.caches:
                continue
            for rank, shard in enumerate(ttnn.get_device_tensors(layer.caches[name])):
                host = ttnn.to_torch(shard).contiguous()
                digest = hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()
                result[f"layer_{layer.layer_idx}/{name}/rank_{rank}"] = {
                    "shape": list(host.shape),
                    "dtype": str(host.dtype),
                    "sha256": digest,
                }
    return result


def all_rank_tokens(tensor):
    return [ttnn.to_torch(shard).reshape(-1)[:32].to(torch.int64).tolist() for shard in ttnn.get_device_tensors(tensor)]


def request(adapter, generator, prompt, *, correctness, sampling_params):
    page_table = generator.model.allocate_page_table()[:1].clone()
    # Change only prompt contents across A/B/A/B, leaving the closed shape
    # envelope exact. Reusing allocated pages is valid after slot prefill/reset.
    started = time.perf_counter()
    (sampled, logprobs), _ = adapter.prefill_forward(
        torch.tensor([prompt], dtype=torch.long),
        page_table=page_table,
        kv_cache=generator.kv_cache,
        prompt_lens=[128],
        sampling_params=sampling_params,
        empty_slots=[0],
    )
    assert logprobs is None
    prefill_ms = (time.perf_counter() - started) * 1000
    tokens = torch.zeros(32, dtype=torch.int32)
    tokens[0] = sampled[0]
    positions = torch.full((32,), -1, dtype=torch.int32)
    positions[0] = 128
    prompt_tokens = torch.zeros((32, 128), dtype=torch.int64)
    prompt_tokens[0] = torch.tensor(prompt)
    output_tokens = tokens.reshape(32, 1).to(torch.int64)
    token_history = [int(sampled[0])]
    decode_ms, ranks = [], []
    for step in range(3):
        start = time.perf_counter()
        output = adapter.decode_forward(
            tokens,
            positions,
            generator.page_table_host.clone(),
            generator.kv_cache,
            enable_trace=True,
            read_from_device=False,
            sampling_params=sampling_params,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reload_inputs=step == 0,
            reload_page_table=False,
            reload_sampling_params=step == 0,
            reset_sampling_state=step == 0,
        )
        host = adapter.read_decode_output(output)
        host_tokens, _ = adapter.process_decode_output_host(host, is_tokens=True)
        decode_ms.append((time.perf_counter() - start) * 1000)
        token_history.append(int(host_tokens[0]))
        if correctness:
            vectors = all_rank_tokens(output)
            if len({vector[0] for vector in vectors}) != 1:
                raise AssertionError(f"greedy active tokens disagree across ranks at decode step {step}: {vectors}")
            ranks.append(vectors)
        # Keep tokens/positions host-stale on purpose: reload_inputs=False must
        # use the resident token feedback and device-owned position advance.
    result = {
        "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        "token_ids": token_history,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prefill_plus_first_decode_ms": prefill_ms + decode_ms[0],
        "program_cache_entries": generator.mesh_device.num_program_cache_entries(),
    }
    if correctness:
        result["decode_tokens_by_rank"] = ranks
        result["cache_digests"] = cache_digests(generator)
        result["trace_position_by_rank"] = all_rank_tokens(generator._trace_position)
        if any(values[0] != 131 for values in result["trace_position_by_rank"]):
            raise AssertionError("three resident decode steps did not advance position from 128 to 131")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--timed-requests", type=int, default=3)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if not ttnn.TRACE_ALLOC_TRACKING:
        raise ValueError("TT_METAL_TRACE_ALLOC_TRACKING=1 is required at process startup")
    if args.num_layers != 4:
        raise ValueError("This reviewed initial probe is restricted to four real layers")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        raise ValueError("This adapter experiment uses host/serving timing, not device profiling")
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    result = {
        "invocation": sys.argv,
        "scope": "Reduced adapter, four layers, B32 allocated, C1 slot0 S128, three decode steps/request.",
        "measurement": "Host request boundaries including token readback; allocation tracker enabled in both arms.",
        "actual_http_ttft": False,
        "allocation_tracking": True,
        "arms": {},
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    generator = mesh = experiment = None
    prompts = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
        generator = build_generator(Path(__file__).parents[2], mesh, batch=32, num_layers=4, max_context=256)
        adapter = Qwen36ForCausalLM(generator)
        result["precision"] = generator.model.precision_summary()
        prompts = []
        for text in (
            "Explain how stable merge sort works, with an example and time complexity. ",
            "Explain why the sky looks blue and how sunlight scatters in the atmosphere. ",
        ):
            rendered = generator.tokenizer.apply_chat_template(
                [{"role": "user", "content": text * 20}], tokenize=False, add_generation_prompt=True
            )
            ids = generator.tokenizer.encode(rendered, add_special_tokens=False)[:128]
            if len(ids) != 128:
                raise AssertionError("the two shape-control prompts must each contain 128 token IDs")
            prompts.append(ids)
        result["prompt_token_ids"] = prompts
        params = SamplingParams(temperature=1.0, top_k=1, top_p=0.0, seed=None)
        references = None
        for name, enabled in (("original", False), ("reuse", True)):
            generator.reset()
            adapter._decode_ready = False
            experiment = ServingTraceReuseExperiment(adapter, enabled=enabled)
            arm = result["arms"][name] = {"correctness": [], "timing": []}
            for index in range(4):
                row = request(adapter, generator, prompts[index % 2], correctness=True, sampling_params=params)
                if references is not None:
                    reference = references[index]
                    row["comparison"] = {
                        "token_ids_exact": row["token_ids"] == reference["token_ids"],
                        "all_rank_tokens_exact": row["decode_tokens_by_rank"] == reference["decode_tokens_by_rank"],
                        "all_cache_digests_exact": row["cache_digests"] == reference["cache_digests"],
                        "positions_exact": row["trace_position_by_rank"] == reference["trace_position_by_rank"],
                    }
                    if not all(row["comparison"].values()):
                        arm["correctness"].append(row)
                        save()
                        raise AssertionError(f"reuse request {index} differs from forced recapture")
                arm["correctness"].append(row)
                arm["counters"] = dict(experiment.counters)
                arm["events"] = list(experiment.events)
                save()
                print("REUSE_CORRECTNESS", name, index, json.dumps(row.get("comparison", {})), flush=True)
            if references is None:
                references = arm["correctness"]
            arm["validation_reuses"] = experiment.counters["setup_reuses"]
            if enabled and not arm["validation_reuses"]:
                raise AssertionError("reuse guards rejected every request; inspect allocation/program events")
            if not args.correctness_only:
                # Validation readbacks and cache hashing finish before timing.
                # One additional untimed request settles the same run state.
                request(adapter, generator, prompts[0], correctness=False, sampling_params=params)
                for index in range(args.timed_requests):
                    row = request(adapter, generator, prompts[index % 2], correctness=False, sampling_params=params)
                    arm["timing"].append(row)
                    print("REUSE_TIMING", name, index, json.dumps(row), flush=True)
                arm["median_ms"] = {
                    "prefill": statistics.median(row["prefill_ms"] for row in arm["timing"]),
                    "first_decode": statistics.median(row["decode_ms"][0] for row in arm["timing"]),
                    "prefill_plus_first_decode": statistics.median(
                        row["prefill_plus_first_decode_ms"] for row in arm["timing"]
                    ),
                }
            arm["counters"] = dict(experiment.counters)
            arm["events"] = list(experiment.events)
            experiment.close()
            experiment = None
            save()
        result["status"] = "passed_reduced_adapter_correctness"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        if experiment is not None:
            result["last_counters"] = dict(experiment.counters)
            result["last_events"] = list(experiment.events)
        save()
        raise
    finally:
        if experiment is not None:
            experiment.close()
        if generator is not None:
            generator.teardown()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        save()


if __name__ == "__main__":
    main()
