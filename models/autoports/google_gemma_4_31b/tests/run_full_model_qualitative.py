# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run prompt-correct HF/TT qualitative controls for the Stage 06 model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.autoports.google_gemma_4_31b.tt.generator import build_generator
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.readiness_check.schema import load_reference


def _load_prompts(path: Path) -> list[str]:
    prompts = [entry.strip() for entry in path.read_text(encoding="utf-8").split("\n\n") if entry.strip()]
    if not prompts:
        raise ValueError(f"prompt source is empty: {path}")
    return prompts


def _generate_hf(model, tokenizer, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    source = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        generated = model.generate(
            source,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    return generated[0, source.shape[1] :].cpu().tolist()


def run(args: argparse.Namespace) -> None:
    prompts = _load_prompts(args.prompt_source)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True, trust_remote_code=True)
    if tokenizer.chat_template:
        raise RuntimeError("google/gemma-4-31B unexpectedly acquired a chat template; update prompt rendering")

    rendered = []
    for prompt_id, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        rendered.append({"id": prompt_id, "prompt": prompt, "prompt_token_ids": token_ids})

    hf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        local_files_only=True,
        trust_remote_code=True,
    ).eval()
    hf_model.to(hf_device)
    outputs = []
    for entry in rendered:
        tokens = _generate_hf(hf_model, tokenizer, entry["prompt_token_ids"], args.max_new_tokens)
        outputs.append(
            {
                "id": entry["id"],
                "prompt": entry["prompt"],
                "hf_token_ids": tokens,
                "hf_greedy_completion": tokenizer.decode(tokens, skip_special_tokens=False),
            }
        )
    del hf_model
    gc.collect()
    if hf_device.type == "cuda":
        torch.cuda.empty_cache()

    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    generator = None
    benchmark = None
    try:
        generator = build_generator(model_dir=args.model_dir, mesh_device=mesh)
        for output, entry in zip(outputs, rendered):
            tokens = generator.generate(
                prompt_token_ids=entry["prompt_token_ids"],
                max_new_tokens=args.max_new_tokens,
                enable_trace=True,
                stop_on_eos=True,
            )
            output["tt_token_ids"] = tokens
            output["tt_greedy_completion"] = tokenizer.decode(tokens, skip_special_tokens=False)
            output["greedy_completion"] = output["tt_greedy_completion"]
        if args.benchmark_reference is not None:
            reference = load_reference(args.benchmark_reference)
            benchmark_prompt = reference.entries[0].prompt_tokens.reshape(-1).tolist()
            benchmark = generator.benchmark_token_out_no_readback(
                benchmark_prompt, max_new_tokens=args.benchmark_tokens
            )
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "hf_model": str(args.hf_model),
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template_present": False,
        "prompt_mode": "completion",
        "rendering_method": "tokenizer.encode(prompt, add_special_tokens=True)",
        "prompt_source_path": str(args.prompt_source),
        "max_new_tokens": args.max_new_tokens,
        "generation": {"do_sample": False, "num_beams": 1},
    }
    (args.output_dir / "qualitative_prompt_format.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "qualitative_rendered_prompts.json").write_text(
        json.dumps(rendered, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "vllm_qualitative_outputs.json").write_text(
        json.dumps(outputs, indent=2) + "\n", encoding="utf-8"
    )
    if benchmark is not None:
        (args.output_dir.parent / "token_out_no_readback.json").write_text(
            json.dumps(benchmark, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"prompt_count": len(outputs), "output_dir": str(args.output_dir)}))


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--prompt-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--benchmark-reference", type=Path)
    parser.add_argument("--benchmark-tokens", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
