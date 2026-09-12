# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture checkpoint decoder inputs on CPU; no TT hardware or full model construction."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from models.autoports.qwen_qwen3_8_27b.tests.reference import load_config, load_layer_weights, make_reference


def record(snapshot, output, length):
    torch.set_num_threads(8)
    config = load_config(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    prompts = [
        "Explain how a computer stores numbers and adds them. Include a simple example. ",
        "Write a short Python function to sort a list and explain its running time. ",
        "Describe the water cycle and why rainfall varies across mountain ranges. ",
    ]
    ids = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt * (length // 10 + 1)}], tokenize=False, add_generation_prompt=True
        )
        ids.append(tokenizer(rendered, add_special_tokens=False)["input_ids"][:length])
    tokens = torch.tensor(ids)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.language_model.embed_tokens.weight"
    with safe_open(snapshot / index[name], framework="pt", device="cpu") as f:
        x = torch.nn.functional.embedding(tokens, f.get_tensor(name))
    output.mkdir(parents=True, exist_ok=True)
    torch.save(x, output / "layer0.pt")
    rope = Qwen3_5TextRotaryEmbedding(config)
    cos, sin = rope(x, torch.arange(length)[None].expand(len(prompts), -1))
    with torch.no_grad():
        for layer in range(3):
            reference = make_reference(config, layer, load_layer_weights(snapshot, layer))
            x = reference(x, position_embeddings=(cos, sin), past_key_values=DynamicCache(config=config))
            print("CAPTURE_LAYER", layer, flush=True)
            del reference
    torch.save(x, output / "layer3.pt")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "snapshot": str(snapshot),
                "length": length,
                "prompts": prompts,
                "token_ids": ids,
                "description": "Real HF embedding and preceding decoder outputs; CPU reference only.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=514)
    a = parser.parse_args()
    record(a.snapshot, a.output, a.length)
