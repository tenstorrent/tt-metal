# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run the shared qualitative suite through exact HF and the TT full model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.generator import build_generator
from models.common.readiness_check.generate import _chat_or_plain_prompt_tokens
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device


def _prompts(path: Path) -> list[str]:
    prompts = [item.strip() for item in path.read_text(encoding="utf-8").split("\n\n") if item.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def _write_completion(
    root: Path,
    *,
    prompt_index: int,
    prompt: str,
    rendered_prompt: str,
    prompt_ids: list[int],
    hf_ids: list[int],
    tt_ids: list[int],
    tokenizer,
    max_new_tokens: int,
    tt_stats: dict,
) -> None:
    prompt_dir = root / f"prompt_{prompt_index:02d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    hf_text = tokenizer.decode(hf_ids, skip_special_tokens=False)
    tt_text = tokenizer.decode(tt_ids, skip_special_tokens=False)
    (prompt_dir / "rendered_prompt.txt").write_text(rendered_prompt, encoding="utf-8")
    (prompt_dir / "hf_completion.txt").write_text(hf_text, encoding="utf-8")
    (prompt_dir / "tt_completion.txt").write_text(tt_text, encoding="utf-8")
    (prompt_dir / "autoregressive_meta.json").write_text(
        json.dumps(
            {
                "prompt_id": f"shared_{prompt_index:02d}",
                "prompt_source": "models/common/readiness_check/vllm_prompts.txt",
                "prompt_text": prompt,
                "rendered_prompt": rendered_prompt,
                "prompt_token_ids": prompt_ids,
                "chat_template": True,
                "fix_mistral_regex": True,
                "max_new_tokens": max_new_tokens,
                "generation": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 0.0,
                    "stop_on_eos": True,
                },
                "hf": {"token_ids": hf_ids, "num_tokens": len(hf_ids)},
                "tt": {"token_ids": tt_ids, "num_tokens": len(tt_ids)},
                "tt_generate_stats": tt_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    prompts_path = args.prompts.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _prompts(prompts_path)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, fix_mistral_regex=True)
    rendered_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    prompt_ids = [_chat_or_plain_prompt_tokens(tokenizer, prompt, chat_template=True) for prompt in prompts]

    hf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True).eval().to(hf_device)
    hf_outputs: list[list[int]] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for index, ids in enumerate(prompt_ids, 1):
        input_ids = torch.tensor([ids], dtype=torch.long, device=hf_device)
        with torch.no_grad():
            output = hf_model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
            )
        continuation = output[0, input_ids.shape[1] :].cpu().tolist()
        hf_outputs.append(continuation)
        print(f"QUALITATIVE_HF prompt={index} prompt_tokens={len(ids)} output_tokens={len(continuation)}")
    del hf_model
    gc.collect()
    if hf_device.type == "cuda":
        torch.cuda.empty_cache()

    ttnn.CONFIG.throw_exception_on_fallback = True
    print("FULL_MODEL_RUNTIME_FALLBACK_POLICY throw_exception_on_fallback=true")
    mesh_device = open_readiness_mesh_device("P300_QUAD", "FABRIC_1D", 200000000)
    tt_outputs: list[list[int]] = []
    tt_stats: list[dict] = []
    try:
        generator = build_generator(
            model_dir=snapshot,
            mesh_device=mesh_device,
            snapshot_path=snapshot,
        )
        try:
            for index, ids in enumerate(prompt_ids, 1):
                output = generator.generate(
                    prompt_token_ids=ids,
                    max_new_tokens=args.max_new_tokens,
                    next_input=None,
                    enable_trace=True,
                    sampling_mode="device",
                    stop_on_eos=True,
                )
                tt_outputs.append(output)
                tt_stats.append(dict(generator.last_generate_stats))
                print(f"QUALITATIVE_TT prompt={index} prompt_tokens={len(ids)} output_tokens={len(output)}")

            repeated = generator.generate(
                prompt_token_ids=prompt_ids[0],
                max_new_tokens=args.max_new_tokens,
                next_input=None,
                enable_trace=True,
                sampling_mode="device",
                stop_on_eos=True,
            )
            deterministic = repeated == tt_outputs[0]
            if not deterministic:
                raise AssertionError("greedy repeat of shared prompt 1 was not deterministic")

            benchmark_ids = [token for ids in prompt_ids for token in ids][:128]
            if len(benchmark_ids) < 128:
                raise AssertionError("shared suite did not provide 128 aggregate benchmark tokens")
            benchmark_output = generator.generate(
                prompt_token_ids=benchmark_ids,
                max_new_tokens=128,
                next_input=None,
                enable_trace=True,
                sampling_mode="device",
                stop_on_eos=False,
            )
            if len(benchmark_output) != 128:
                raise AssertionError("128/128 primary workload did not emit exactly 128 tokens")
            primary_stats = dict(generator.last_generate_stats)
        finally:
            generator.teardown()
    finally:
        close_readiness_mesh_device(mesh_device, "FABRIC_1D")

    for index, (prompt, rendered, ids, hf_ids, tt_ids, stats) in enumerate(
        zip(prompts, rendered_prompts, prompt_ids, hf_outputs, tt_outputs, tt_stats),
        1,
    ):
        _write_completion(
            output_dir,
            prompt_index=index,
            prompt=prompt,
            rendered_prompt=rendered,
            prompt_ids=ids,
            hf_ids=hf_ids,
            tt_ids=tt_ids,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            tt_stats=stats,
        )

    prompt_format = {
        "hf_model": "mistralai/Mistral-Small-24B-Instruct-2501",
        "hf_snapshot": snapshot.name,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_present": bool(tokenizer.chat_template),
        "prompt_mode": "chat",
        "rendering_method": "tokenizer.apply_chat_template(single user turn, add_generation_prompt=True)",
        "fix_mistral_regex": True,
        "prompt_source": str(prompts_path),
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "hf_control": "same snapshot/tokenizer/rendered token IDs, greedy",
        "tt_runtime_fallback_policy": "throw_exception_on_fallback=true",
    }
    (output_dir / "qualitative_prompt_format.json").write_text(json.dumps(prompt_format, indent=2), encoding="utf-8")
    (output_dir / "suite_summary.json").write_text(
        json.dumps(
            {
                "prompt_ids": [f"shared_{index:02d}" for index in range(1, len(prompts) + 1)],
                "hf_output_lengths": [len(output) for output in hf_outputs],
                "tt_output_lengths": [len(output) for output in tt_outputs],
                "greedy_prompt_1_repeat_deterministic": deterministic,
                "primary_128_128": {
                    "prompt_tokens": 128,
                    "generated_tokens": len(benchmark_output),
                    "tt_generate_stats": primary_stats,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"QUALITATIVE_SUITE_PASS prompts={len(prompts)} repeat_deterministic={deterministic}")
    print(f"PRIMARY_128_128 {primary_stats}")


if __name__ == "__main__":
    main()
