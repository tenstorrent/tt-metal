# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness test with partial-fused kernel (recurrence only, conv/norm on CPU)."""

import os
os.environ["DISABLE_FULL_FUSED_KERNEL"] = "1"

import sys
import time
import torch
import ttnn
from transformers import AutoTokenizer
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.load_weights import load_state_dict
from models.demos.qwen3_coder_next.tt.model import TtQwen3CoderNextModel
from models.demos.qwen3_coder_next.tt.generator import Qwen3CoderNextGenerator
from models.demos.qwen3_coder_next.tt.deltanet import USE_FUSED_KERNEL, USE_FULL_FUSED_KERNEL


def main():
    print(f"[Config] Fused DeltaNet kernel: {USE_FUSED_KERNEL}")
    print(f"[Config] Full fused kernel: {USE_FULL_FUSED_KERNEL}")

    config = Qwen3CoderNextConfig()
    config.weights_dtype = ttnn.bfloat8_b
    print(f"[Config] Weight dtype: bfloat8_b")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model_path = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    state_dict = load_state_dict(config, model_path=model_path)

    device = ttnn.open_device(device_id=0)
    try:
        print("[Model] Building...")
        t0 = time.time()
        model = TtQwen3CoderNextModel(device, state_dict, config)
        generator = Qwen3CoderNextGenerator(model, config, tokenizer=tokenizer)
        del state_dict
        print(f"[Model] Built in {time.time() - t0:.1f}s")

        prompts = [
            ("factual", "What is the capital of France?", 32),
            ("reasoning", "Explain why the sky is blue in one sentence.", 48),
        ]

        for name, prompt, max_tokens in prompts:
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
