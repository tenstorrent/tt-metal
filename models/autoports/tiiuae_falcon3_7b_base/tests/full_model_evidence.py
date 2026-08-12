# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Collect full-stack capacity, sampler, trace-state, and performance evidence."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tt.generator import _first_device_to_torch, _round_up, build_generator
from models.common.readiness_check.schema import load_reference
from models.common.sampling.tt_sampling import TTSampling


def _timed(mesh_device, function, iterations: int = 1) -> float:
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    ttnn.synchronize_device(mesh_device)
    return time.perf_counter() - start


def _dram_view(mesh_device) -> dict[str, int]:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return {
        "banks": int(view.num_banks),
        "total_bytes_per_device": int(view.num_banks * view.total_bytes_per_bank),
        "allocated_bytes_per_device": int(view.num_banks * view.total_bytes_allocated_per_bank),
        "free_bytes_per_device": int(view.num_banks * view.total_bytes_free_per_bank),
        "largest_contiguous_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
    }


def _phase(name: str) -> None:
    print(f"FULL_MODEL_EVIDENCE_PHASE: {name}", flush=True)


def collect(
    model_dir: Path,
    reference_path: Path,
    output: Path,
    *,
    replay_iterations: int = 128,
    sampler_iterations: int = 20,
    prompt_length: int = 128,
    weight_cache_path: str | Path = "/tmp/falcon3-full-model-cache",
) -> dict:
    reference = load_reference(reference_path)
    reference_prompt = reference.entries[0].prompt_tokens[0].tolist()
    if prompt_length < 1 or prompt_length > len(reference_prompt):
        raise ValueError(f"prompt_length must be in [1,{len(reference_prompt)}]")
    prompt = reference_prompt[:prompt_length]
    mesh_device = None
    generator = None
    legacy_output_buffer = None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    try:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 4),
            trace_region_size=512_000_000,
        )
        generator = build_generator(model_dir, mesh_device, weight_cache_path=weight_cache_path)
        _phase("generator_ready")
        memory_after_weights = _dram_view(mesh_device)
        kv_cache = generator._ensure_kv_cache()
        memory_after_cache = _dram_view(mesh_device)
        mapped_tokens = _round_up(len(prompt) + replay_iterations + 8, generator.model.page_block_size)
        page_table = generator._make_page_table([mapped_tokens])

        cold_logits = None

        def cold_prefill():
            nonlocal cold_logits
            cold_logits = generator._prefill_single_local_logits(prompt, page_table, kv_cache)

        cold_prefill_s = _timed(mesh_device, cold_prefill)
        _phase("cold_prefill_complete")
        generator.reset()
        warm_logits = None

        def warm_prefill():
            nonlocal warm_logits
            warm_logits = generator._prefill_single_local_logits(prompt, page_table, kv_cache)

        warm_prefill_s = _timed(mesh_device, warm_prefill)
        _phase("warm_prefill_complete")

        split_buffer = generator._decode_trace_input_pool[0]
        force_buffer = generator._prefill_sampled
        legacy_output_buffer = generator._replicated_device_tensor(
            torch.zeros((1, 1, 1, 32), dtype=torch.int32), dtype=ttnn.uint32
        )
        split_output = generator.model.sample_greedy_split(warm_logits, tt_out_tok=split_buffer)
        force_output = generator.model.sample_force_argmax(warm_logits, tt_out_tok=force_buffer)
        ttnn.synchronize_device(mesh_device)
        split_token = int(_first_device_to_torch(split_output).reshape(-1)[0])
        force_token = int(_first_device_to_torch(force_output).reshape(-1)[0])
        legacy_args = SimpleNamespace(
            vocab_size=generator.model.vocab_size,
            padded_vocab_size=generator.model.padded_vocab_size,
            max_batch_size=32,
            max_top_k=32,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            sub_core_grids=None,
            sub_core_grid_topk=None,
            start_core=ttnn.CoreCoord(0, 0),
            pad_logits_to_power_of_2=True,
            sampling_dp=1,
            use_topk_logprobs=False,
            model_config={
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": True,
                    "num_links": 2,
                    "chunks_per_sync": 10,
                    "topology": ttnn.Topology.Ring,
                }
            },
        )
        legacy_sampler = TTSampling(
            mesh_device=mesh_device,
            tt_ccl=generator.model.tt_ccl,
            args=legacy_args,
            k=torch.ones(32),
            p=torch.zeros(32),
            temp=torch.ones(32),
        )
        legacy_output, _ = legacy_sampler(warm_logits, tt_out_tok=legacy_output_buffer)
        ttnn.synchronize_device(mesh_device)
        legacy_token = int(_first_device_to_torch(legacy_output).reshape(-1)[0])
        split_sampler_s = _timed(
            mesh_device,
            lambda: generator.model.sample_greedy_split(warm_logits, tt_out_tok=split_buffer),
            sampler_iterations,
        )
        force_sampler_s = _timed(
            mesh_device,
            lambda: generator.model.sample_force_argmax(warm_logits, tt_out_tok=force_buffer),
            sampler_iterations,
        )
        legacy_sampler_s = _timed(
            mesh_device,
            lambda: legacy_sampler(warm_logits, tt_out_tok=legacy_output_buffer),
            sampler_iterations,
        )
        _phase("sampler_comparison_complete")

        generator.reset()
        generated = generator.generate(prompt, 5, enable_trace=True, sampling_mode="device")
        _phase("five_token_trace_complete")
        stats_after_generation = dict(generator.trace_stats)
        position_after_generation = int(_first_device_to_torch(generator._trace_inputs[1]).reshape(-1)[0])
        rotary_after_generation = int(_first_device_to_torch(generator._trace_inputs[2]).reshape(-1)[0])
        feedback_token = int(_first_device_to_torch(generator._trace_inputs[0]).reshape(-1)[0])

        page_copies_before = generator.trace_stats["page_table_host_copies"]
        generator._refresh_persistent_page_table(
            generator._trace_page_table_snapshot.clone(),
            kv_cache,
            active_batch=1,
        )
        unchanged_page_delta = generator.trace_stats["page_table_host_copies"] - page_copies_before

        page_copies_before = generator.trace_stats["page_table_host_copies"]
        generator._refresh_persistent_page_table(page_table, kv_cache, active_batch=1)
        expanded_page_delta = generator.trace_stats["page_table_host_copies"] - page_copies_before

        original_inputs = generator._prepare_decode_host_inputs(
            torch.tensor([generated[-1]]),
            torch.tensor([len(prompt)]),
            page_table,
        )
        generator._refresh_trace_state(original_inputs, kv_cache, active_batch=1)
        changed_page_table = page_table.clone()
        changed_page_table[0, 0], changed_page_table[0, 1] = (
            changed_page_table[0, 1].clone(),
            changed_page_table[0, 0].clone(),
        )
        changed_inputs = generator._prepare_decode_host_inputs(
            torch.tensor([generated[-1]]),
            torch.tensor([len(prompt)]),
            changed_page_table,
        )

        def replay_from(host_inputs):
            generator._refresh_trace_state(host_inputs, kv_cache, active_batch=1)
            sampled = generator._replay_split_sampling()
            ttnn.synchronize_device(mesh_device)
            local_logits = ttnn.to_torch(ttnn.get_device_tensors(generator._trace_logits)[0]).float().clone()
            token = int(_first_device_to_torch(sampled).reshape(-1)[0])
            return local_logits, token

        original_logits, original_page_token = replay_from(original_inputs)
        unchanged_replay_copies_before = generator.trace_stats["page_table_host_copies"]
        repeated_logits, repeated_page_token = replay_from(original_inputs)
        unchanged_replay_delta = generator.trace_stats["page_table_host_copies"] - unchanged_replay_copies_before
        page_copies_before = generator.trace_stats["page_table_host_copies"]
        changed_logits, changed_page_token = replay_from(changed_inputs)
        changed_page_delta = generator.trace_stats["page_table_host_copies"] - page_copies_before
        persistent_changed_table = _first_device_to_torch(generator._trace_inputs[3]).to(torch.int32)
        changed_table_consumed = torch.equal(persistent_changed_table, changed_page_table)
        changed_logits_max_abs_delta = float((changed_logits - original_logits).abs().max())
        generator._refresh_trace_state(original_inputs, kv_cache, active_batch=1)
        restored_logits, restored_page_token = replay_from(original_inputs)
        restored_logits_match = torch.allclose(restored_logits, original_logits, rtol=0.0, atol=0.0)
        repeated_logits_match = torch.allclose(repeated_logits, original_logits, rtol=0.0, atol=0.0)
        _phase("page_table_replay_complete")

        generator._refresh_trace_state(original_inputs, kv_cache, active_batch=1)
        steady_state_copy_counters = (
            "token_host_copies",
            "position_host_copies",
            "rotary_position_host_copies",
            "page_table_host_copies",
            "sampling_param_host_copies",
        )
        steady_state_before = {name: generator.trace_stats[name] for name in steady_state_copy_counters}
        pair_s = _timed(mesh_device, generator._replay_split_sampling, replay_iterations)
        _phase("device_only_pair_complete")
        steady_state_host_copy_deltas = {
            name: generator.trace_stats[name] - steady_state_before[name] for name in steady_state_copy_counters
        }
        pair_position = int(_first_device_to_torch(generator._trace_inputs[1]).reshape(-1)[0])
        pair_rotary = int(_first_device_to_torch(generator._trace_inputs[2]).reshape(-1)[0])
        pair_token = int(_first_device_to_torch(generator._trace_inputs[0]).reshape(-1)[0])

        # Token-out includes the caller-visible token read required by the
        # public generator.  Feedback remains the device-to-device edge from
        # the sampling trace into the next model-trace input; this read is an
        # observation, never the source of the following token.
        generator._refresh_trace_state(original_inputs, kv_cache, active_batch=1)

        def replay_and_read_token():
            sampled = generator._replay_split_sampling()
            generator._sampled_to_torch(sampled)

        caller_visible_s = _timed(mesh_device, replay_and_read_token, replay_iterations)
        _phase("caller_visible_pair_complete")

        generator._refresh_trace_state(original_inputs, kv_cache, active_batch=1)
        teacher_s = _timed(
            mesh_device,
            lambda: (
                generator._copy_forced_tokens(torch.tensor([generated[-1]])),
                generator._replay_split_sampling(),
            ),
            replay_iterations,
        )
        _phase("teacher_forcing_pair_complete")
        generator._refresh_trace_state(original_inputs, kv_cache, active_batch=1)
        model_only_s = _timed(
            mesh_device,
            lambda: ttnn.execute_trace(mesh_device, generator._trace_model_id, cq_id=0, blocking=False),
            replay_iterations,
        )
        sampling_only_s = _timed(
            mesh_device,
            lambda: ttnn.execute_trace(mesh_device, generator._trace_sampling_id, cq_id=0, blocking=False),
            replay_iterations,
        )
        _phase("split_trace_timing_complete")

        result = {
            "mesh": "4x Blackhole p300c, 1x4 FABRIC_1D_RING, TP4, two links",
            "num_layers": generator.model.num_layers,
            "prompt_tokens": len(prompt),
            "workload": {
                "batch": 1,
                "prompt_tokens": len(prompt),
                "generated_tokens": replay_iterations,
                "standard_128_128": len(prompt) == replay_iterations == 128,
            },
            "context_tokens": generator.model.max_cache_len,
            "cache_dtype": "BFP8_B paged, one KV head/rank",
            "memory_after_weights": memory_after_weights,
            "memory_after_full_context_cache": memory_after_cache,
            "samplers": {
                "split_greedy_token": split_token,
                "force_argmax_token": force_token,
                "ttsampling_token": legacy_token,
                "semantic_match": split_token == force_token == legacy_token,
                "iterations": sampler_iterations,
                "split_greedy_ms_per_call": split_sampler_s * 1000 / sampler_iterations,
                "force_argmax_ms_per_call": force_sampler_s * 1000 / sampler_iterations,
                "ttsampling_ms_per_call": legacy_sampler_s * 1000 / sampler_iterations,
                "selected": "Sampling1D exact local argmax plus packed FP32 rank-candidate gather",
                "rejected": (
                    "TTSampling exact greedy full-vocabulary all-gather plus argmax; broader mutable "
                    "request state and explicit request seeds disable SamplingGenerator internal tracing"
                ),
            },
            "trace": {
                "generated_tokens": generated,
                "stats_after_five_tokens": stats_after_generation,
                "expected_position_after_four_decode_replays": len(prompt) + 4,
                "position_after_generation": position_after_generation,
                "rotary_position_after_generation": rotary_after_generation,
                "feedback_token_matches_last_output": feedback_token == generated[-1],
                "unchanged_page_table_host_copy_delta": unchanged_page_delta,
                "expanded_page_table_host_copy_delta": expanded_page_delta,
                "changed_page_table_host_copy_delta": changed_page_delta,
                "changed_page_table_persistent_input_matches": changed_table_consumed,
                "changed_page_table_logits_max_abs_delta": changed_logits_max_abs_delta,
                "changed_page_table_output_token": changed_page_token,
                "original_page_table_output_token": original_page_token,
                "restored_page_table_output_token": restored_page_token,
                "unchanged_replay_page_table_host_copy_delta": unchanged_replay_delta,
                "unchanged_replay_logits_exact_match": repeated_logits_match,
                "unchanged_replay_token_match": repeated_page_token == original_page_token,
                "restored_page_table_logits_exact_match": restored_logits_match,
                "restored_page_table_token_match": restored_page_token == original_page_token,
                "replay_iterations": replay_iterations,
                "position_after_pair_replays": pair_position,
                "rotary_position_after_pair_replays": pair_rotary,
                "final_feedback_token": pair_token,
                "steady_state_host_copy_deltas": steady_state_host_copy_deltas,
            },
            "performance": {
                "cold_ttft_ms": cold_prefill_s * 1000 + split_sampler_s * 1000 / sampler_iterations,
                "warm_ttft_ms": warm_prefill_s * 1000 + split_sampler_s * 1000 / sampler_iterations,
                "trace_pair_elapsed_s": pair_s,
                "device_only_trace_pair_t_s_u": replay_iterations / pair_s,
                "caller_visible_token_out_elapsed_s": caller_visible_s,
                "caller_visible_token_out_t_s_u": replay_iterations / caller_visible_s,
                "teacher_forcing_trace_t_s_u": replay_iterations / teacher_s,
                "model_trace_ms_per_token": model_only_s * 1000 / replay_iterations,
                "sampling_trace_ms_per_token": sampling_only_s * 1000 / replay_iterations,
                "sampling_fraction": sampling_only_s / (model_only_s + sampling_only_s),
            },
        }
        result["passed"] = bool(
            result["num_layers"] == 28
            and result["context_tokens"] == 32768
            and result["workload"]["standard_128_128"]
            and result["samplers"]["semantic_match"]
            and result["trace"]["position_after_generation"]
            == result["trace"]["expected_position_after_four_decode_replays"]
            and result["trace"]["rotary_position_after_generation"]
            == result["trace"]["expected_position_after_four_decode_replays"]
            and result["trace"]["feedback_token_matches_last_output"]
            and result["trace"]["unchanged_page_table_host_copy_delta"] == 0
            and result["trace"]["unchanged_replay_page_table_host_copy_delta"] == 0
            and result["trace"]["changed_page_table_host_copy_delta"] == 1
            and result["trace"]["changed_page_table_persistent_input_matches"]
            and result["trace"]["changed_page_table_logits_max_abs_delta"] > 0.0
            and result["trace"]["unchanged_replay_logits_exact_match"]
            and result["trace"]["unchanged_replay_token_match"]
            and result["trace"]["restored_page_table_logits_exact_match"]
            and result["trace"]["restored_page_table_token_match"]
            and result["trace"]["position_after_pair_replays"] == len(prompt) + replay_iterations
            and result["trace"]["rotary_position_after_pair_replays"] == len(prompt) + replay_iterations
            and all(value == 0 for value in result["trace"]["steady_state_host_copy_deltas"].values())
            and math.isfinite(result["performance"]["device_only_trace_pair_t_s_u"])
            and result["performance"]["device_only_trace_pair_t_s_u"] > 0.0
            and math.isfinite(result["performance"]["caller_visible_token_out_t_s_u"])
            and result["performance"]["caller_visible_token_out_t_s_u"] > 0.0
            and result["performance"]["sampling_fraction"] < 0.5
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    finally:
        if legacy_output_buffer is not None:
            legacy_output_buffer.deallocate(True)
        if generator is not None:
            generator.teardown()
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-iterations", type=int, default=128)
    parser.add_argument("--sampler-iterations", type=int, default=20)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--weight-cache-path", default="/tmp/falcon3-full-model-cache")
    args = parser.parse_args()
    result = collect(
        args.model_dir,
        args.reference,
        args.output,
        replay_iterations=args.replay_iterations,
        sampler_iterations=args.sampler_iterations,
        prompt_length=args.prompt_length,
        weight_cache_path=args.weight_cache_path,
    )
    if not result["passed"]:
        raise SystemExit("full-model evidence gate failed")


if __name__ == "__main__":
    main()
