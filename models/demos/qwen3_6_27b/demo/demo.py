#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Demo: Qwen3.6-27B single-chip inference (N-layer subset for memory).

Run:
    cd /home/tt-admin/ssinghal/tt-metal
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python models/demos/qwen3_6_27b/demo/demo.py "Hello, world" --num-layers 16
"""
import argparse
import sys

import torch

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
from transformers import AutoTokenizer
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_config, load_qwen36_tensors
from models.demos.qwen3_6_27b.tests.ttnn.test_decoder_layer_e2e import _layer_keys
from models.demos.qwen3_6_27b.tt.model import TtQwen36Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, default="Hello", nargs="?")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=16,
        help="Number of decoder layers to load (full=64; memory-bound on single chip)",
    )
    parser.add_argument("--num-tokens", type=int, default=5)
    args = parser.parse_args()

    print(f"[demo] loading config + tokenizer...")
    cfg_dict = load_qwen36_config()
    hf_cfg = Qwen3NextConfig(**cfg_dict["text_config"])
    hf_cfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

    print(f"[demo] tokenizing: {args.prompt!r}")
    input_ids = tok(args.prompt, return_tensors="pt").input_ids
    print(f"[demo] tokens: {input_ids[0].tolist()}")

    print(f"[demo] loading {args.num_layers} layers of weights from safetensors...")
    keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]
    for i in range(args.num_layers):
        keys.extend(_layer_keys(i, hf_cfg.layer_types[i]))
    weights = load_qwen36_tensors(keys)
    print(f"[demo] loaded {len(weights)} tensors")

    print(f"[demo] opening BH device 0...")
    device = ttnn.open_device(device_id=0)
    try:
        print(f"[demo] constructing TtQwen36Model({args.num_layers} layers)...")
        tt_model = TtQwen36Model(device, weights, hf_cfg, num_layers=args.num_layers)
        rot = Qwen3NextRotaryEmbedding(hf_cfg)

        print(f"[demo] generating {args.num_tokens} tokens (greedy)...")
        generated = list(input_ids[0].tolist())
        for step in range(args.num_tokens):
            current_ids = torch.tensor([generated])
            T = current_ids.shape[1]
            cos, sin = rot(torch.zeros(1, T, hf_cfg.hidden_size), torch.arange(T).unsqueeze(0))
            mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)
            logits = tt_model(current_ids, cos=cos, sin=sin, attention_mask=mask)
            next_id = logits[0, -1].argmax().item()
            generated.append(next_id)
            print(f"  step {step}: +{next_id!r} ({tok.decode([next_id])!r})")

        print()
        print(f"[demo] final: {tok.decode(generated)!r}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
