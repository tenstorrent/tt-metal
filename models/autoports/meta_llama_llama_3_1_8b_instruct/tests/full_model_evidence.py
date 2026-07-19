# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Collect full-model capacity, sampler, trace-state, and warmed performance evidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tracy import signpost

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.generator import (
    _first_device_to_torch,
    _round_up,
    build_generator,
)
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device


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
        "total_bytes": int(view.num_banks * view.total_bytes_per_bank),
        "allocated_bytes": int(view.num_banks * view.total_bytes_allocated_per_bank),
        "free_bytes": int(view.num_banks * view.total_bytes_free_per_bank),
        "largest_contiguous_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
    }


def collect(
    model_dir: Path,
    prompt_file: Path,
    output: Path,
    replay_iterations: int,
    sampler_iterations: int = 20,
    override_num_layers: int | None = None,
) -> dict:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    if not prompt_file.is_file():
        raise FileNotFoundError(f"prompt file does not exist: {prompt_file}")

    mesh_device = open_readiness_mesh_device("P300", "FABRIC_1D_RING")
    try:
        generator = build_generator(model_dir, mesh_device, override_num_layers=override_num_layers)
        memory_after_weights = _dram_view(mesh_device)

        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        rendered = generator.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt = generator.tokenizer.encode(rendered, add_special_tokens=False)
        padded_prefill = max(128, _round_up(len(prompt), 128))
        # Every replay section restarts at the same position, but each section
        # advances as far as ``replay_iterations``.  Map the largest position
        # before any traced write so the evidence never relies on an invalid
        # page-table entry at a page boundary.
        mapped_tokens = max(
            _round_up(len(prompt) + replay_iterations + 8, 64),
            padded_prefill,
        )
        page_table = generator._make_page_table([mapped_tokens])
        kv_cache = generator._ensure_kv_cache()
        memory_after_cache = _dram_view(mesh_device)

        cold_prefill_s = _timed(
            mesh_device,
            lambda: generator._prefill_single_device(prompt, page_table, kv_cache),
        )
        generator.reset()
        warm_logits = None

        def warm_prefill():
            nonlocal warm_logits
            warm_logits = generator._prefill_single_device(prompt, page_table, kv_cache)

        signpost("start FULL_MODEL_REDUCED_PREFILL")
        warm_prefill_s = _timed(mesh_device, warm_prefill)
        signpost("stop FULL_MODEL_REDUCED_PREFILL")
        k, p, temp = generator._ensure_sampling_params()

        split_output = generator.model.sample_greedy_split(warm_logits, k=k, p=p, temp=temp)
        argmax_output = generator.model.sample_force_argmax(warm_logits)
        ttnn.synchronize_device(mesh_device)
        split_token = int(generator._sampled_tokens_to_torch(split_output)[0])
        argmax_token = int(generator._sampled_tokens_to_torch(argmax_output)[0])

        split_sampler_s = _timed(
            mesh_device,
            lambda: generator.model.sample_greedy_split(
                warm_logits,
                k=k,
                p=p,
                temp=temp,
                tt_out_tok=split_output,
            ),
            sampler_iterations,
        )
        force_argmax_s = _timed(
            mesh_device,
            lambda: generator.model.sample_force_argmax(warm_logits, tt_out_tok=argmax_output),
            sampler_iterations,
        )

        generator.reset()
        generated = generator.generate(prompt, 5, enable_trace=True, sampling_mode="device")
        stats_after_generation = dict(generator.trace_stats)
        position_after_generation = int(_first_device_to_torch(generator._trace_inputs[1]).reshape(-1)[0])
        rope_after_generation = int(_first_device_to_torch(generator._trace_inputs[2]).reshape(-1)[0])
        feedback_token = int(generator._sampled_tokens_to_torch(generator._trace_inputs[0])[0])

        captured_page_table = generator._trace_page_table_snapshot.clone()
        page_copies_before = generator.trace_stats["page_table_host_copies"]
        generator._refresh_persistent_page_table(
            captured_page_table,
            kv_cache,
            active_batch=generator._trace_active_batch,
        )
        unchanged_page_copy_delta = generator.trace_stats["page_table_host_copies"] - page_copies_before

        # The public five-token request maps only its own horizon.  Expand the
        # same fixed-shape table once at the request boundary before the
        # 100-replay proof; steady replays must not rebuild it.
        page_copies_before = generator.trace_stats["page_table_host_copies"]
        generator._refresh_persistent_page_table(
            page_table,
            kv_cache,
            active_batch=generator._trace_active_batch,
        )
        expanded_page_copy_delta = generator.trace_stats["page_table_host_copies"] - page_copies_before

        original_inputs = generator._prepare_decode_host_inputs(
            torch.tensor([generated[-1]]),
            torch.tensor([len(prompt)]),
            page_table,
        )
        generator._refresh_trace_state(original_inputs, kv_cache)

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
        page_copies_before = generator.trace_stats["page_table_host_copies"]
        generator._refresh_trace_state(changed_inputs, kv_cache)
        changed_page_copy_delta = generator.trace_stats["page_table_host_copies"] - page_copies_before
        generator._refresh_trace_state(original_inputs, kv_cache)

        signpost("start FULL_MODEL_REDUCED_DECODE")
        pair_s = _timed(mesh_device, generator._replay_split_sampling, replay_iterations)
        signpost("stop FULL_MODEL_REDUCED_DECODE")
        pair_position = int(_first_device_to_torch(generator._trace_inputs[1]).reshape(-1)[0])
        pair_rope = int(_first_device_to_torch(generator._trace_inputs[2]).reshape(-1)[0])
        pair_final_token = int(generator._sampled_tokens_to_torch(generator._trace_inputs[0])[0])

        generator._refresh_trace_state(original_inputs, kv_cache)
        teacher_forcing_s = _timed(
            mesh_device,
            lambda: (
                generator._copy_forced_token(generated[-1]),
                generator._replay_split_sampling(),
            ),
            replay_iterations,
        )

        generator._refresh_trace_state(original_inputs, kv_cache)
        model_only_s = _timed(
            mesh_device,
            lambda: ttnn.execute_trace(
                mesh_device,
                generator._trace_model_id,
                cq_id=0,
                blocking=False,
            ),
            replay_iterations,
        )
        sampling_only_s = _timed(
            mesh_device,
            lambda: ttnn.execute_trace(
                mesh_device,
                generator._trace_sampling_id,
                cq_id=0,
                blocking=False,
            ),
            replay_iterations,
        )

        result = {
            "mesh": "P300 1x4 FABRIC_1D_RING TP=4",
            "num_layers": generator.model.num_layers,
            "prompt_tokens": len(prompt),
            "cache_dtype": "BFP8",
            "memory_after_weights": memory_after_weights,
            "memory_after_full_context_cache": memory_after_cache,
            "cold_prefill_ms": cold_prefill_s * 1000,
            "warm_prefill_ms": warm_prefill_s * 1000,
            "samplers": {
                "split_greedy_token": split_token,
                "force_argmax_token": argmax_token,
                "semantic_match": split_token == argmax_token,
                "iterations": sampler_iterations,
                "split_greedy_ms_per_call": split_sampler_s * 1000 / sampler_iterations,
                "force_argmax_ms_per_call": force_argmax_s * 1000 / sampler_iterations,
                "selected": "Sampling1D exact local argmax plus one packed FP32 rank-candidate gather",
                "rejected": "Sampling1D full-vocabulary all-gather plus argmax",
            },
            "trace": {
                "generated_tokens": generated,
                "trace_stats_after_five_tokens": stats_after_generation,
                "expected_position_after_four_decode_replays": len(prompt) + 4,
                "position_after_generation": position_after_generation,
                "rope_after_generation": rope_after_generation,
                "feedback_token_matches_last_output": feedback_token == generated[-1],
                "unchanged_page_table_host_copy_delta": unchanged_page_copy_delta,
                "expanded_page_table_host_copy_delta": expanded_page_copy_delta,
                "changed_page_table_host_copy_delta": changed_page_copy_delta,
                "replay_iterations": replay_iterations,
                "pair_position_after_replays": pair_position,
                "pair_rope_after_replays": pair_rope,
                "pair_final_token": pair_final_token,
            },
            "performance": {
                "cold_ttft_ms_including_split_sampler": cold_prefill_s * 1000
                + split_sampler_s * 1000 / sampler_iterations,
                "warm_ttft_ms_including_split_sampler": warm_prefill_s * 1000
                + split_sampler_s * 1000 / sampler_iterations,
                "trace_pair_elapsed_s": pair_s,
                "trace_pair_t_s_u": replay_iterations / pair_s,
                "warm_teacher_forcing_t_s_u": replay_iterations / teacher_forcing_s,
                "model_trace_ms_per_token": model_only_s * 1000 / replay_iterations,
                "sampling_trace_ms_per_token": sampling_only_s * 1000 / replay_iterations,
                "sampling_fraction_of_separate_trace_time": sampling_only_s / (model_only_s + sampling_only_s),
            },
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
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-iterations", type=int, default=100)
    parser.add_argument("--sampler-iterations", type=int, default=20)
    parser.add_argument("--override-num-layers", type=int)
    args = parser.parse_args()
    collect(
        args.model_dir,
        args.prompt_file,
        args.output,
        args.replay_iterations,
        args.sampler_iterations,
        args.override_num_layers,
    )


if __name__ == "__main__":
    main()
