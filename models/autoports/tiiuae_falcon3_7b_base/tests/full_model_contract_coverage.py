# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exercise fixed slots, mixed prompts, inactive rows, reset, and host compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tt.generator import _first_device_to_torch, build_generator


def collect(model_dir: Path, output: Path, weight_cache_path: str) -> dict:
    mesh = None
    generator = None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=512_000_000)
        generator = build_generator(
            model_dir,
            mesh,
            override_num_layers=1,
            max_batch_size=32,
            max_context_len=4096,
            weight_cache_path=weight_cache_path,
        )
        page_boundary_table = generator._make_page_table([66])
        page_boundary_position = torch.tensor([64], dtype=torch.int32)
        generator._validate_page_coverage(page_boundary_table, page_boundary_position, 1)

        def coverage_rejected(mutated: torch.Tensor) -> bool:
            try:
                generator._validate_page_coverage(mutated, page_boundary_position, 1)
            except ValueError:
                return True
            return False

        missing_rounded_tail = page_boundary_table.clone()
        missing_rounded_tail[0, 3] = -1
        out_of_range_tail = page_boundary_table.clone()
        out_of_range_tail[0, 3] = generator.num_blocks
        aliased_tail = page_boundary_table.clone()
        aliased_tail[0, 3] = aliased_tail[0, 2]
        rounded_page_validation = {
            "decode_position": 64,
            "logical_pages": 3,
            "mapped_sdpa_read_pages": int((page_boundary_table[0] >= 0).sum()),
            "valid_mapping_accepted": True,
            "missing_rounded_tail_rejected": coverage_rejected(missing_rounded_tail),
            "out_of_range_tail_rejected": coverage_rejected(out_of_range_tail),
            "aliased_tail_rejected": coverage_rejected(aliased_tail),
        }
        prompt_lens = [33, 47]
        prompts = torch.zeros((2, max(prompt_lens)), dtype=torch.long)
        prompts[0, : prompt_lens[0]] = 1
        prompts[1, : prompt_lens[1]] = 2
        page_table = generator._make_page_table([length + 2 for length in prompt_lens])
        kv_cache = generator._ensure_kv_cache()
        generator.set_sampling_params(active_batch=2)

        prefill_sampled = generator.prefill_forward(
            prompts,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            sampling_mode="device",
        )
        prefill_tokens = _first_device_to_torch(prefill_sampled).reshape(-1)[:2].to(torch.long)
        decode_sampled = generator.decode_forward(
            prefill_tokens,
            torch.tensor(prompt_lens),
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_mode="device",
            enable_trace=True,
            active_batch=2,
        )
        decode_tokens = _first_device_to_torch(decode_sampled).reshape(-1)[:2].to(torch.long)
        positions = _first_device_to_torch(generator._trace_inputs[1]).reshape(-1).to(torch.int32)
        rotary_positions = _first_device_to_torch(generator._trace_inputs[2]).reshape(-1).to(torch.int32)
        initial_trace_captures = generator.trace_stats["captures"]
        initial_trace_replays = generator.trace_stats["replays"]
        trace_ids_before_reset = (generator._trace_model_id, generator._trace_sampling_id)
        trace_buffers_before_reset = tuple(id(tensor) for tensor in generator._decode_trace_input_pool)
        releases_before_reset = generator.trace_stats["releases"]
        generator.reset()
        trace_ids_after_reset = (generator._trace_model_id, generator._trace_sampling_id)
        trace_buffers_after_reset = tuple(id(tensor) for tensor in generator._decode_trace_input_pool)
        traces_released_by_reset = trace_ids_after_reset == (None, None)
        trace_buffers_preserved_by_reset = trace_buffers_before_reset == trace_buffers_after_reset
        reset_release_delta = generator.trace_stats["releases"] - releases_before_reset
        first_key_page = ttnn.to_torch(ttnn.get_device_tensors(kv_cache[0][0])[0])[0]
        cache_zero_after_reset = float(first_key_page.float().abs().sum()) == 0.0
        reset_token = _first_device_to_torch(generator._decode_trace_input_pool[0]).reshape(-1).to(torch.int32)
        reset_positions = _first_device_to_torch(generator._decode_trace_input_pool[1]).reshape(-1).to(torch.int32)
        reset_rotary = _first_device_to_torch(generator._decode_trace_input_pool[2]).reshape(-1).to(torch.int32)
        reset_pages = _first_device_to_torch(generator._decode_trace_input_pool[3]).to(torch.int32)
        reset_seeds = _first_device_to_torch(generator.model.sampler._seeds).reshape(-1).to(torch.int64)
        reset_device_state_cleared = bool(
            torch.all(reset_token == 0)
            and torch.all(reset_positions == -1)
            and torch.all(reset_rotary == 0)
            and torch.all(reset_pages == -1)
            and torch.equal(reset_seeds, torch.arange(32, dtype=torch.int64))
        )
        releases_before_next_prefill = generator.trace_stats["releases"]
        next_prefill_sampled = generator.prefill_forward(
            prompts,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            sampling_mode="device",
        )
        next_prefill_tokens = _first_device_to_torch(next_prefill_sampled).reshape(-1)[:2].to(torch.long)
        trace_released_before_allocating_prefill = bool(
            generator._trace_model_id is None and generator._trace_sampling_id is None
        )
        allocating_prefill_release_delta = generator.trace_stats["releases"] - releases_before_next_prefill
        trace_buffers_preserved_across_allocating_prefill = trace_buffers_before_reset == tuple(
            id(tensor) for tensor in generator._decode_trace_input_pool
        )
        host_compat_tokens = generator.generate(
            [1] * 17,
            2,
            sampling_mode="host",
            enable_trace=False,
        )
        generator.reset()
        long_prompt_lens = [2049, 2079]
        long_prompts = torch.zeros((2, max(long_prompt_lens)), dtype=torch.long)
        long_prompts[0, : long_prompt_lens[0]] = 1
        long_prompts[1, : long_prompt_lens[1]] = 2
        long_page_table = generator._make_page_table([length + 1 for length in long_prompt_lens])
        generator.set_sampling_params(active_batch=2)
        long_sampled = generator.prefill_forward(
            long_prompts,
            page_table=long_page_table,
            kv_cache=kv_cache,
            prompt_lens=long_prompt_lens,
            sampling_mode="device",
        )
        long_tokens = _first_device_to_torch(long_sampled).reshape(-1)[:2].to(torch.long)
        generator.reset()
        host_copies_before = generator.trace_stats["token_host_copies"]
        device_copies_before = generator.trace_stats["token_device_copies"]
        public_device_tokens = generator.generate([1] * 17, 3, sampling_mode="device", enable_trace=True)
        public_token_host_copy_delta = generator.trace_stats["token_host_copies"] - host_copies_before
        public_token_device_copy_delta = generator.trace_stats["token_device_copies"] - device_copies_before
        public_feedback_token = int(_first_device_to_torch(generator._trace_inputs[0]).reshape(-1)[0])
        generator.reset()
        stochastic_host_copies_before = generator.trace_stats["token_host_copies"]
        stochastic_tokens = generator.generate(
            [1] * 17,
            3,
            sampling_mode="device",
            enable_trace=True,
            top_k=8,
            top_p=0.9,
            temperature=0.7,
            seed=42,
        )
        stochastic_token_host_copy_delta = generator.trace_stats["token_host_copies"] - stochastic_host_copies_before
        stochastic_feedback_token = int(_first_device_to_torch(generator._trace_inputs[0]).reshape(-1)[0])
        stochastic_trace_ids = (generator._trace_model_id, generator._trace_sampling_id)
        stochastic_captures = generator.trace_stats["captures"]
        stochastic_releases = generator.trace_stats["releases"]
        stochastic_rng_checkpoints = generator.trace_stats["rng_checkpoints"]
        stochastic_rng_restores = generator.trace_stats["rng_restores"]
        generator.reset()
        stochastic_tokens_after_recapture = generator.generate(
            [1] * 17,
            3,
            sampling_mode="device",
            enable_trace=True,
            top_k=8,
            top_p=0.9,
            temperature=0.7,
            seed=42,
        )
        stochastic_trace_recaptured_after_safe_prefill = bool(
            stochastic_trace_ids != (generator._trace_model_id, generator._trace_sampling_id)
            and generator.trace_stats["captures"] == stochastic_captures + 1
            and generator.trace_stats["releases"] == stochastic_releases + 1
        )

        result = {
            "mesh": "4x Blackhole p300c, 1x4 FABRIC_1D_RING, TP4",
            "layers_executed": 1,
            "fixed_slots": 32,
            "active_slots": 2,
            "rounded_sdpa_page_validation": rounded_page_validation,
            "prompt_lens": prompt_lens,
            "non_aligned_prompts": all(length % 32 for length in prompt_lens),
            "prefill_tokens": prefill_tokens.tolist(),
            "decode_tokens": decode_tokens.tolist(),
            "active_positions_after_trace_replay": positions[:2].tolist(),
            "expected_active_positions": [length + 1 for length in prompt_lens],
            "inactive_cache_positions_unchanged": bool(torch.all(positions[2:] == -1)),
            "inactive_rotary_positions_are_nonnegative": bool(torch.all(rotary_positions[2:] >= 0)),
            "page_table_shape": list(page_table.shape),
            "inactive_page_rows_unmapped": bool(torch.all(page_table[2:] == -1)),
            "trace_captures": initial_trace_captures,
            "trace_replays": initial_trace_replays,
            "traces_released_by_reset": traces_released_by_reset,
            "trace_buffers_preserved_by_reset": trace_buffers_preserved_by_reset,
            "reset_release_delta": reset_release_delta,
            "cache_zero_after_reset": cache_zero_after_reset,
            "reset_device_state_cleared": reset_device_state_cleared,
            "next_prefill_tokens": next_prefill_tokens.tolist(),
            "trace_released_before_allocating_prefill": trace_released_before_allocating_prefill,
            "allocating_prefill_release_delta": allocating_prefill_release_delta,
            "trace_buffers_preserved_across_allocating_prefill": trace_buffers_preserved_across_allocating_prefill,
            "host_sampling_compatibility_tokens": host_compat_tokens,
            "chunk_boundary_prompt_lens": long_prompt_lens,
            "chunk_boundary_tokens": long_tokens.tolist(),
            "chunk_boundary_crossed": min(long_prompt_lens) > generator.prefill_chunk_size,
            "chunk_boundary_inactive_page_rows_unmapped": bool(torch.all(long_page_table[2:] == -1)),
            "public_device_tokens": public_device_tokens,
            "public_token_host_copy_delta": public_token_host_copy_delta,
            "public_token_device_copy_delta": public_token_device_copy_delta,
            "public_feedback_matches_last_token": public_feedback_token == public_device_tokens[-1],
            "stochastic_device_tokens": stochastic_tokens,
            "stochastic_token_host_copy_delta": stochastic_token_host_copy_delta,
            "stochastic_feedback_matches_last_token": stochastic_feedback_token == stochastic_tokens[-1],
            "stochastic_tokens_after_recapture": stochastic_tokens_after_recapture,
            "stochastic_same_seed_matches_first_capture": stochastic_tokens_after_recapture == stochastic_tokens,
            "stochastic_trace_recaptured_after_safe_prefill": stochastic_trace_recaptured_after_safe_prefill,
            "stochastic_rng_checkpoints": stochastic_rng_checkpoints,
            "stochastic_rng_restores": stochastic_rng_restores,
        }
        result["passed"] = bool(
            result["rounded_sdpa_page_validation"]["mapped_sdpa_read_pages"] == 4
            and result["rounded_sdpa_page_validation"]["valid_mapping_accepted"]
            and result["rounded_sdpa_page_validation"]["missing_rounded_tail_rejected"]
            and result["rounded_sdpa_page_validation"]["out_of_range_tail_rejected"]
            and result["rounded_sdpa_page_validation"]["aliased_tail_rejected"]
            and result["non_aligned_prompts"]
            and len(result["prefill_tokens"]) == 2
            and len(result["decode_tokens"]) == 2
            and result["active_positions_after_trace_replay"] == result["expected_active_positions"]
            and result["inactive_cache_positions_unchanged"]
            and result["inactive_page_rows_unmapped"]
            and result["trace_captures"] == 1
            and result["trace_replays"] == 1
            and result["traces_released_by_reset"]
            and result["trace_buffers_preserved_by_reset"]
            and result["reset_release_delta"] == 1
            and result["cache_zero_after_reset"]
            and result["reset_device_state_cleared"]
            and len(result["next_prefill_tokens"]) == 2
            and result["trace_released_before_allocating_prefill"]
            and result["allocating_prefill_release_delta"] == 0
            and result["trace_buffers_preserved_across_allocating_prefill"]
            and len(result["host_sampling_compatibility_tokens"]) == 2
            and len(result["chunk_boundary_tokens"]) == 2
            and result["chunk_boundary_crossed"]
            and result["chunk_boundary_inactive_page_rows_unmapped"]
            and len(result["public_device_tokens"]) == 3
            and result["public_token_host_copy_delta"] == 0
            and result["public_token_device_copy_delta"] > 0
            and result["public_feedback_matches_last_token"]
            and len(result["stochastic_device_tokens"]) == 3
            and result["stochastic_token_host_copy_delta"] == 0
            and result["stochastic_feedback_matches_last_token"]
            and result["stochastic_same_seed_matches_first_capture"]
            and result["stochastic_trace_recaptured_after_safe_prefill"]
            and result["stochastic_rng_checkpoints"] == result["stochastic_rng_restores"] == 1
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-cache-path", default="/tmp/falcon3-full-model-cache")
    args = parser.parse_args()
    result = collect(args.model_dir, args.output, args.weight_cache_path)
    if not result["passed"]:
        raise SystemExit("full-model serving contract coverage failed")


if __name__ == "__main__":
    main()
