# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run the shared qualitative prompt suite through one HF and TT instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.generator import build_generator
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device


def _read_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def _hf_generate(model, tokenizer, prompt_token_ids: list[int], max_new_tokens: int) -> list[int]:
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        eos = tokenizer.eos_token_id
        pad_id = eos[0] if isinstance(eos, (list, tuple)) else eos
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
        )
    return output[0, input_ids.shape[1] :].tolist()


def _first_divergence(lhs: list[int], rhs: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return index
    return None if len(lhs) == len(rhs) else min(len(lhs), len(rhs))


def collect(model_dir: Path, hf_model: Path, prompts_file: Path, output_dir: Path, max_new_tokens: int) -> dict:
    prompts = _read_prompts(prompts_file)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("the exact Llama Instruct tokenizer must provide a chat template")
    hf = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True).eval()
    mesh_device = open_readiness_mesh_device("P300", "FABRIC_1D_RING")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        generator = build_generator(model_dir, mesh_device, model_path=hf_model)
        entries = []
        for prompt_id, prompt in enumerate(prompts):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_tokens = tokenizer.encode(rendered, add_special_tokens=False)
            hf_tokens = _hf_generate(hf, tokenizer, prompt_tokens, max_new_tokens)

            stats_before = dict(generator.trace_stats)
            tt_tokens = generator.generate(
                prompt_tokens,
                max_new_tokens,
                enable_trace=True,
                sampling_mode="device",
                top_k=1,
                top_p=0.0,
                temperature=1.0,
                stop_on_eos=True,
            )
            stats_delta = {key: generator.trace_stats[key] - stats_before[key] for key in generator.trace_stats}
            compatible_repeat = None
            if prompt_id == 0:
                trace_ids = (generator._trace_model_id, generator._trace_sampling_id)
                generator.reset()
                repeat_before = dict(generator.trace_stats)
                repeat_tokens = generator.generate(
                    prompt_tokens,
                    max_new_tokens,
                    enable_trace=True,
                    sampling_mode="device",
                    top_k=1,
                    top_p=0.0,
                    temperature=1.0,
                    stop_on_eos=True,
                )
                repeat_delta = {key: generator.trace_stats[key] - repeat_before[key] for key in generator.trace_stats}
                if repeat_tokens != tt_tokens:
                    raise AssertionError("compatible cross-reset request changed TT tokens")
                if trace_ids != (generator._trace_model_id, generator._trace_sampling_id):
                    raise AssertionError("compatible cross-reset request replaced decode traces")
                if repeat_delta["captures"] != 0 or repeat_delta["prefill_replays"] != 1:
                    raise AssertionError(f"compatible request did not reuse traces: {repeat_delta}")
                compatible_repeat = {
                    "token_ids_exact": True,
                    "decode_trace_ids_reused": True,
                    "trace_counter_delta": repeat_delta,
                }
            prompt_dir = output_dir / f"prompt_{prompt_id:02d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            hf_text = tokenizer.decode(hf_tokens, skip_special_tokens=False)
            tt_text = tokenizer.decode(tt_tokens, skip_special_tokens=False)
            (prompt_dir / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")
            (prompt_dir / "hf_completion.txt").write_text(hf_text, encoding="utf-8")
            (prompt_dir / "tt_completion.txt").write_text(tt_text, encoding="utf-8")
            meta = {
                "hf_model_id": str(hf_model),
                "prompt_file": str(prompts_file),
                "prompt_id": prompt_id,
                "prompt_text": prompt,
                "rendered_prompt": rendered,
                "prompt_token_ids": prompt_tokens,
                "chat_template": True,
                "max_new_tokens": max_new_tokens,
                "hf": {"token_ids": hf_tokens, "num_tokens": len(hf_tokens)},
                "tt": {"token_ids": tt_tokens, "num_tokens": len(tt_tokens)},
                "first_divergence": _first_divergence(hf_tokens, tt_tokens),
                "matching_tokens": sum(left == right for left, right in zip(hf_tokens, tt_tokens)),
                "trace_counter_delta": stats_delta,
                "compatible_cross_reset_repeat": compatible_repeat,
            }
            (prompt_dir / "autoregressive_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            entries.append(meta)

            trace_ids = (generator._trace_model_id, generator._trace_sampling_id)
            captures = generator.trace_stats["captures"]
            generator.reset()
            if trace_ids != (generator._trace_model_id, generator._trace_sampling_id):
                raise AssertionError("reset released serving traces")
            if captures != generator.trace_stats["captures"]:
                raise AssertionError("reset unexpectedly changed the capture count")

        if entries[0]["trace_counter_delta"]["captures"] != 1:
            raise AssertionError("the first prompt must capture exactly one decode trace pair")
        if entries[0]["compatible_cross_reset_repeat"] is None:
            raise AssertionError("missing compatible cross-reset trace-reuse evidence")

        summary = {
            "hf_model_id": str(hf_model),
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template_present": True,
            "prompt_mode": "chat",
            "rendering_method": "apply_chat_template(tokenize=False, add_generation_prompt=True) then encode(add_special_tokens=False)",
            "prompt_source": str(prompts_file),
            "generation": {"max_new_tokens": max_new_tokens, "greedy": True},
            "mesh": "P300 1x4 FABRIC_1D_RING TP=4",
            "num_prompts": len(entries),
            "entries": entries,
        }
        (output_dir / "qualitative_prompt_format.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        teardown = locals().get("generator")
        if teardown is not None:
            teardown.teardown()
        close_readiness_mesh_device(mesh_device, "FABRIC_1D_RING")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()
    collect(args.model_dir, args.hf_model, args.prompts_file, args.output_dir, args.max_new_tokens)


if __name__ == "__main__":
    main()
