# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run the shared qualitative suite as native Falcon3 base completions.

Falcon3-7B-Base's exact tokenizer has no chat template.  The repository's
shared qualitative prompts are phrased as instructions for serving checks,
so this runner maps each one to a semantically equivalent completion stem.
The mapping is deliberately exact: if the shared prompt file changes, this
runner fails instead of accidentally judging a base model as an instruction
model.

Each prompt gets the same artifact shape as ``run_autoregressive``.  This
makes all TT completions discoverable by ``check_degenerate_output.py
--scope autoregressive`` while retaining the rendered prompt, prompt tokens,
paired HF/TT output tokens, decoded texts, trace counters, and human-review
inputs.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.autoports.tiiuae_falcon3_7b_base.tt.generator import build_generator
from models.common.readiness_check.mesh_device import (
    add_mesh_device_args,
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)

DEFAULT_SHARED_PROMPTS = Path("models/common/readiness_check/vllm_prompts.txt")
DEFAULT_OUTPUT_DIR = Path("models/autoports/tiiuae_falcon3_7b_base/doc/full_model/results/qualitative_suite")
DEFAULT_MAX_NEW_TOKENS = 100


@dataclass(frozen=True)
class CompletionPrompt:
    prompt_id: str
    shared_instruction: str
    completion_stem: str


# Keep this keyed by the exact shared prompts.  These are not a new model-local
# suite: they are completion-form renderings of every entry in the common
# readiness suite, in the common file's order.
BASE_COMPLETION_PROMPTS = (
    CompletionPrompt(
        "machine_learning_haiku",
        "Write a haiku about machine learning.",
        "Machine learning\n\nA haiku:\n",
    ),
    CompletionPrompt(
        "supervised_vs_unsupervised",
        "Explain the difference between supervised and unsupervised learning in simple terms.",
        "In simple terms, the difference between supervised and unsupervised learning is that",
    ),
    CompletionPrompt(
        "inventor_story",
        "Complete this story: Once upon a time, in a faraway kingdom, there lived a curious young inventor who discovered",
        "Once upon a time, in a faraway kingdom, there lived a curious young inventor who discovered",
    ),
    CompletionPrompt(
        "thermodynamics",
        "What are the three laws of thermodynamics?",
        "The three laws of thermodynamics are:\n\n1.",
    ),
    CompletionPrompt(
        "english_to_french",
        'Translate the following to French: "Hello, how are you today?"',
        'English: "Hello, how are you today?"\nFrench: "',
    ),
    CompletionPrompt(
        "fibonacci_python",
        "Write a Python function to calculate the Fibonacci sequence.",
        "Here is a Python function that calculates the Fibonacci sequence:\n\n```python\n",
    ),
)

REVIEW_DIMENSIONS = (
    "coherent_and_on_topic_relative_to_hf_control",
    "mechanical_repetition_or_token_doubling",
    "wrong_language_drift",
    "prompt_echo",
    "control_or_special_token_leakage",
    "early_divergence_followed_by_quality_collapse",
    "cross_request_leakage",
)


def _read_shared_prompts(path: Path) -> list[str]:
    prompts = [item.strip() for item in path.read_text(encoding="utf-8").split("\n\n") if item.strip()]
    expected = [item.shared_instruction for item in BASE_COMPLETION_PROMPTS]
    if prompts != expected:
        raise ValueError(
            "shared qualitative prompts no longer match the reviewed base-completion mapping; "
            f"expected {expected!r}, got {prompts!r}"
        )
    return prompts


def _hf_generate(
    model,
    tokenizer,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
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
    return output[0, input_ids.shape[1] :].cpu().tolist()


def _first_divergence(lhs: Sequence[int], rhs: Sequence[int]) -> int | None:
    for index, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return index
    return None if len(lhs) == len(rhs) else min(len(lhs), len(rhs))


def _stats_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: int(after[key] - before.get(key, 0)) for key in after}


def collect(
    *,
    model_dir: Path,
    hf_model: str,
    prompts_file: Path,
    output_dir: Path,
    max_new_tokens: int,
    mesh_device_label: str,
    fabric_config: str | None,
    weight_cache_path: Path | None = None,
) -> dict[str, Any]:
    if not 64 <= max_new_tokens <= 128:
        raise ValueError("qualitative generations must request 64-128 new tokens")
    _read_shared_prompts(prompts_file)

    local_files_only = Path(hf_model).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Falcon3-7B-Base unexpectedly has a chat template; revisit the prompt contract")

    prompt_tokens = [
        tokenizer.encode(item.completion_stem, add_special_tokens=True) for item in BASE_COMPLETION_PROMPTS
    ]
    hf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_model_instance = AutoModelForCausalLM.from_pretrained(
        hf_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    ).eval()
    hf_model_instance.to(hf_device)
    hf_outputs = [
        _hf_generate(hf_model_instance, tokenizer, tokens, max_new_tokens, hf_device) for tokens in prompt_tokens
    ]
    del hf_model_instance
    gc.collect()
    if hf_device.type == "cuda":
        torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_device = open_readiness_mesh_device(mesh_device_label, fabric_config)
    generator = None
    entries: list[dict[str, Any]] = []
    try:
        build_kwargs: dict[str, Any] = {}
        if Path(hf_model).is_dir():
            build_kwargs["model_path"] = hf_model
        if weight_cache_path is not None:
            build_kwargs["weight_cache_path"] = weight_cache_path
        generator = build_generator(model_dir=model_dir, mesh_device=mesh_device, **build_kwargs)

        for index, (spec, tokens, hf_tokens) in enumerate(
            zip(BASE_COMPLETION_PROMPTS, prompt_tokens, hf_outputs, strict=True)
        ):
            stats_before = dict(generator.trace_stats)
            tt_tokens = generator.generate(
                prompt_token_ids=tokens,
                max_new_tokens=max_new_tokens,
                enable_trace=True,
                sampling_mode="device",
                top_k=1,
                top_p=0.0,
                temperature=1.0,
                stop_on_eos=True,
            )
            stats = _stats_delta(generator.trace_stats, stats_before)
            hf_text = tokenizer.decode(hf_tokens, skip_special_tokens=False)
            tt_text = tokenizer.decode(tt_tokens, skip_special_tokens=False)
            compared = min(len(hf_tokens), len(tt_tokens))
            matching = sum(left == right for left, right in zip(hf_tokens, tt_tokens))
            divergence = _first_divergence(hf_tokens, tt_tokens)

            prompt_dir = output_dir / f"prompt_{index:02d}_{spec.prompt_id}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / "rendered_prompt.txt").write_text(spec.completion_stem, encoding="utf-8")
            (prompt_dir / "hf_completion.txt").write_text(hf_text, encoding="utf-8")
            (prompt_dir / "tt_completion.txt").write_text(tt_text, encoding="utf-8")

            meta = {
                "hf_model_id": hf_model,
                "prompt_file": str(prompts_file),
                "prompt_id": spec.prompt_id,
                "shared_instruction": spec.shared_instruction,
                "prompt_mode": "completion",
                "rendering_method": "exact reviewed base-completion transform; tokenizer.encode(add_special_tokens=True)",
                "prompt_text": spec.completion_stem,
                "prompt_token_ids": tokens,
                "max_new_tokens": max_new_tokens,
                "generation": {
                    "greedy": True,
                    "top_k": 1,
                    "top_p": 0.0,
                    "temperature": 1.0,
                    "stop_on_eos": True,
                },
                "hf": {"token_ids": hf_tokens, "num_tokens": len(hf_tokens)},
                "tt": {"token_ids": tt_tokens, "num_tokens": len(tt_tokens)},
                "comparison": {
                    "first_divergence": divergence,
                    "matching_tokens": matching,
                    "compared_tokens": compared,
                    "exact_match": hf_tokens == tt_tokens,
                },
                "trace_counter_delta": stats,
                "verdict_inputs": {
                    "review_dimensions": list(REVIEW_DIMENSIONS),
                    "hf_completion_file": "hf_completion.txt",
                    "tt_completion_file": "tt_completion.txt",
                    "status": "human_review_required",
                },
            }
            (prompt_dir / "autoregressive_meta.json").write_text(
                json.dumps(meta, indent=2) + "\n",
                encoding="utf-8",
            )
            entries.append(meta)
            generator.reset()

        revision = Path(hf_model).resolve().name if Path(hf_model).is_dir() else None
        summary = {
            "hf_model_id": hf_model,
            "revision": revision,
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template_present": False,
            "prompt_mode": "completion",
            "rendering_method": "exact reviewed base-completion transform; tokenizer.encode(add_special_tokens=True)",
            "shared_prompt_source": str(prompts_file),
            "shared_prompt_count": len(BASE_COMPLETION_PROMPTS),
            "generation": {
                "max_new_tokens": max_new_tokens,
                "greedy": True,
                "stop_on_eos": True,
            },
            "mesh": f"{mesh_device_label} {fabric_config or 'fabric disabled'}",
            "entries": entries,
            "degeneracy_check": {
                "scope": "autoregressive",
                "artifact_root": str(output_dir),
                "command": (
                    "python models/common/readiness_check/check_degenerate_output.py "
                    f"{output_dir} --scope autoregressive --missing-artifacts critical "
                    f"--json {output_dir / 'degenerate_output.json'}"
                ),
            },
        }
        (output_dir / "qualitative_prompt_format.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "prompt_verdict_inputs.json").write_text(
            json.dumps(
                [
                    {
                        "prompt_id": item["prompt_id"],
                        "completion_prompt": item["prompt_text"],
                        "hf_completion": tokenizer.decode(item["hf"]["token_ids"], skip_special_tokens=False),
                        "tt_completion": tokenizer.decode(item["tt"]["token_ids"], skip_special_tokens=False),
                        "comparison": item["comparison"],
                        "review_dimensions": list(REVIEW_DIMENSIONS),
                        "verdict": "human_review_required",
                    }
                    for item in entries
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh_device, fabric_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--hf-model", required=True, help="Exact HF id or local snapshot used by TT weights.")
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_SHARED_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--weight-cache-path", type=Path)
    add_mesh_device_args(parser)
    args = parser.parse_args()
    collect(
        model_dir=args.model_dir.resolve(),
        hf_model=args.hf_model,
        prompts_file=args.prompts_file.resolve(),
        output_dir=args.output_dir.resolve(),
        max_new_tokens=args.max_new_tokens,
        mesh_device_label=args.mesh_device,
        fabric_config=args.fabric_config,
        weight_cache_path=args.weight_cache_path,
    )


if __name__ == "__main__":
    main()
