# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness test with bfloat8_b weights to isolate quantization effects."""

import sys
import time

import torch
import ttnn

from transformers import AutoTokenizer

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator
from models.demos.qwen36_27b.tt.deltanet import USE_FUSED_KERNEL, USE_FULL_FUSED_KERNEL

PROMPTS = [
    ("factual", "What is the capital of France?", 32),
    ("reasoning", "Explain why the sky is blue in one sentence.", 48),
]


def main():
    print(f"[Config] Fused DeltaNet kernel: {USE_FUSED_KERNEL}")
    print(f"[Config] Full fused kernel: {USE_FULL_FUSED_KERNEL}")

    config = Qwen36ModelConfig()
    config.weights_dtype = ttnn.bfloat8_b
    print(f"[Config] Weight dtype: bfloat8_b (overridden from bfloat4_b)")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model_path = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

    print("[Weights] Loading...")
    state_dict = load_state_dict(config, model_path=model_path)

    device = ttnn.open_device(device_id=0)
    try:
        print("[Model] Building (bfloat8_b weights)...")
        t0 = time.time()
        model = TtQwen36Model(device, state_dict, config)
        generator = Qwen36Generator(model, config, tokenizer=tokenizer)
        del state_dict
        print(f"[Model] Built in {time.time() - t0:.1f}s")

        for name, prompt, max_tokens in PROMPTS:
            print(f"\n{'='*60}")
            print(f"  Test: {name} — {prompt}")
            print(f"{'='*60}")
            generator.reset()
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
            )
            t0 = time.perf_counter()
            generated = generator.generate(input_ids, max_new_tokens=max_tokens)
            elapsed = time.perf_counter() - t0
            output_text = tokenizer.decode(generated, skip_special_tokens=True)
            print(f"  {len(generated)} tokens in {elapsed:.2f}s ({len(generated)/elapsed:.1f} t/s)")
            print(f"  Output: {output_text}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
