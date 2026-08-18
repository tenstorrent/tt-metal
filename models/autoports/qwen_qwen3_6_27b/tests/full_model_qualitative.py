# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matched HF/TT chat-template run over the shared readiness prompt suite."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import default_snapshot, MODEL_ID, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("models/common/readiness_check/vllm_prompts.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()
    snapshot = default_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    rendered = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        for prompt in prompts
    ]
    prompt_ids = [tokenizer.encode(text, add_special_tokens=False) for text in rendered]

    hf = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True).eval()
    hf_outputs = []
    with torch.no_grad():
        for ids in prompt_ids:
            input_ids = torch.tensor([ids], dtype=torch.long)
            output = hf.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated = output[0, len(ids) :].tolist()
            hf_outputs.append({"token_ids": generated, "text": tokenizer.decode(generated, skip_special_tokens=False)})
    del hf
    gc.collect()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    try:
        generator = build_generator(model_dir=Path("models/autoports/qwen_qwen3_6_27b"), mesh_device=mesh, max_context=512, batch=1)
        tt_outputs = []
        for ids in prompt_ids:
            generated = generator.generate(ids, args.max_new_tokens)
            tt_outputs.append({"token_ids": generated, "text": tokenizer.decode(generated, skip_special_tokens=False)})
    finally:
        if generator is not None:
            generator.teardown()
        ttnn.close_mesh_device(mesh)

    result = {
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "tokenizer_class": type(tokenizer).__name__, "chat_template_present": bool(tokenizer.chat_template),
        "rendering_method": "apply_chat_template(add_generation_prompt=True)",
        "prompt_source": str(args.prompts), "generation": {"greedy": True, "max_new_tokens": args.max_new_tokens},
        "cases": [
            {"id": index, "prompt": prompt, "rendered_prompt": text, "prompt_token_ids": ids, "hf": hf_out, "tt": tt_out}
            for index, (prompt, text, ids, hf_out, tt_out) in enumerate(zip(prompts, rendered, prompt_ids, hf_outputs, tt_outputs))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
