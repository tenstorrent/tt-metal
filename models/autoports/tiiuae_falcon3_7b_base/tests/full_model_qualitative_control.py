# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Isolate the haiku loop across host, traced, feedback, and RNG paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from models.autoports.tiiuae_falcon3_7b_base.tests.full_model_qualitative import BASE_COMPLETION_PROMPTS
from models.autoports.tiiuae_falcon3_7b_base.tt.generator import _first_device_to_torch, build_generator
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _trace_state(generator) -> dict:
    return {
        "token_input": int(_first_device_to_torch(generator._trace_inputs[0]).reshape(-1)[0]),
        "current_position": int(_first_device_to_torch(generator._trace_inputs[1]).reshape(-1)[0]),
        "rotary_position": int(_first_device_to_torch(generator._trace_inputs[2]).reshape(-1)[0]),
        "trace_ids": [int(generator._trace_model_id), int(generator._trace_sampling_id)],
    }


def collect(
    model_dir: Path,
    model_path: Path,
    output: Path,
    *,
    max_new_tokens: int,
    weight_cache_path: str,
) -> dict:
    if not 64 <= max_new_tokens <= 128:
        raise ValueError("max_new_tokens must be in [64,128]")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompt_text = BASE_COMPLETION_PROMPTS[0].completion_stem
    prompt = tokenizer.encode(prompt_text, add_special_tokens=True)
    mesh = open_readiness_mesh_device("P300X2", "FABRIC_1D_RING")
    generator = None
    try:
        generator = build_generator(
            model_dir,
            mesh,
            model_path=model_path,
            weight_cache_path=weight_cache_path,
        )

        host_tokens = generator.generate(
            prompt,
            max_new_tokens,
            sampling_mode="host",
            enable_trace=False,
        )

        generator.reset()
        stats_before_free = dict(generator.trace_stats)
        traced_free_tokens = generator.generate(
            prompt,
            max_new_tokens,
            sampling_mode="device",
            enable_trace=True,
        )
        free_stats = {key: generator.trace_stats[key] - stats_before_free[key] for key in generator.trace_stats}
        free_state = _trace_state(generator)

        releases_before_reset = generator.trace_stats["releases"]
        generator.reset()
        trace_ids_after_reset = [generator._trace_model_id, generator._trace_sampling_id]
        reset_release_delta = generator.trace_stats["releases"] - releases_before_reset
        input_pool_ids_before_reprefill = [id(tensor) for tensor in generator._decode_trace_input_pool]
        stats_before_recap = dict(generator.trace_stats)
        traced_recap_tokens = generator.generate(
            prompt,
            max_new_tokens,
            sampling_mode="device",
            enable_trace=True,
        )
        recap_stats = {key: generator.trace_stats[key] - stats_before_recap[key] for key in generator.trace_stats}
        recap_state = _trace_state(generator)
        input_pool_ids_after_reprefill = [id(tensor) for tensor in generator._decode_trace_input_pool]

        generator.reset()
        teacher_tokens = generator.generate(
            prompt,
            max_new_tokens,
            sampling_mode="device",
            enable_trace=True,
            next_input=lambda step, _predicted: host_tokens[step],
        )
        teacher_state = _trace_state(generator)

        generator.reset()
        sampled_tokens = generator.generate(
            prompt,
            max_new_tokens,
            sampling_mode="device",
            enable_trace=True,
            top_k=8,
            top_p=0.9,
            temperature=0.7,
            seed=42,
        )
        sampled_state = _trace_state(generator)

        expected_position = len(prompt) + max_new_tokens - 1
        result = {
            "prompt_text": prompt_text,
            "prompt_tokens": len(prompt),
            "max_new_tokens": max_new_tokens,
            "host_eager_tokens": host_tokens,
            "traced_free_tokens": traced_free_tokens,
            "traced_safe_recapture_tokens": traced_recap_tokens,
            "traced_teacher_forced_predictions": teacher_tokens,
            "seeded_sampled_tokens": sampled_tokens,
            "host_vs_traced_first_divergence": _first_divergence(host_tokens, traced_free_tokens),
            "host_vs_safe_recapture_first_divergence": _first_divergence(host_tokens, traced_recap_tokens),
            "host_vs_teacher_first_divergence": _first_divergence(host_tokens, teacher_tokens),
            "host_traced_exact_match": host_tokens == traced_free_tokens,
            "host_safe_recapture_exact_match": host_tokens == traced_recap_tokens,
            "host_teacher_forced_exact_match": host_tokens == teacher_tokens,
            "free_trace_stats": free_stats,
            "safe_recapture_trace_stats": recap_stats,
            "free_trace_state": free_state,
            "trace_ids_released_by_reset": trace_ids_after_reset == [None, None],
            "reset_release_delta": reset_release_delta,
            "trace_ids_recaptured_before_next_decode": all(
                trace_id is not None for trace_id in recap_state["trace_ids"]
            ),
            "persistent_input_pool_preserved": input_pool_ids_before_reprefill == input_pool_ids_after_reprefill,
            "safe_recapture_trace_state": recap_state,
            "teacher_trace_state": teacher_state,
            "sampled_trace_state": sampled_state,
            "expected_final_position": expected_position,
            "host_text": tokenizer.decode(host_tokens, skip_special_tokens=False),
            "traced_text": tokenizer.decode(traced_free_tokens, skip_special_tokens=False),
            "sampled_text": tokenizer.decode(sampled_tokens, skip_special_tokens=False),
        }
        result["passed"] = bool(
            len(host_tokens)
            == len(traced_free_tokens)
            == len(traced_recap_tokens)
            == len(teacher_tokens)
            == max_new_tokens
            and result["host_traced_exact_match"]
            and result["host_safe_recapture_exact_match"]
            and result["host_teacher_forced_exact_match"]
            and result["free_trace_state"]["token_input"] == traced_free_tokens[-1]
            and result["safe_recapture_trace_state"]["token_input"] == traced_recap_tokens[-1]
            and result["free_trace_state"]["current_position"] == expected_position
            and result["free_trace_state"]["rotary_position"] == expected_position
            and result["safe_recapture_trace_state"]["current_position"] == expected_position
            and result["safe_recapture_trace_state"]["rotary_position"] == expected_position
            and result["teacher_trace_state"]["current_position"] == expected_position
            and result["teacher_trace_state"]["rotary_position"] == expected_position
            and result["free_trace_stats"]["token_host_copies"] == 0
            and result["safe_recapture_trace_stats"]["token_host_copies"] == 0
            and result["trace_ids_released_by_reset"]
            and result["reset_release_delta"] == 1
            and result["trace_ids_recaptured_before_next_decode"]
            and result["persistent_input_pool_preserved"]
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output.parent / "haiku_host_greedy.txt").write_text(result["host_text"] + "\n", encoding="utf-8")
        (output.parent / "haiku_traced_greedy.txt").write_text(result["traced_text"] + "\n", encoding="utf-8")
        (output.parent / "haiku_seeded_sampled.txt").write_text(result["sampled_text"] + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D_RING")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--weight-cache-path", default="/tmp/falcon3-full-model-cache")
    args = parser.parse_args()
    result = collect(
        args.model_dir,
        args.model_path,
        args.output,
        max_new_tokens=args.max_new_tokens,
        weight_cache_path=args.weight_cache_path,
    )
    if not result["passed"]:
        raise SystemExit("haiku greedy-path control failed")


if __name__ == "__main__":
    main()
