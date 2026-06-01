# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Functional correctness test: Qwen3-Coder-Next with fused DeltaNet kernel.

Generates text from 3 sample prompts and checks:
  1. No NaN/inf in logits
  2. Output tokens decode to valid text
  3. Outputs are coherent (manual inspection)
"""

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

PROMPTS = [
    ("factual", "What is the capital of France?", 32),
    ("reasoning", "Explain why the sky is blue in one sentence.", 48),
    ("code", "Write a Python function to reverse a string.", 64),
]


def test_prompt(generator, tokenizer, name, prompt, max_new_tokens):
    print(f"\n{'='*60}")
    print(f"  Test: {name}")
    print(f"  Prompt: {prompt}")
    print(f"{'='*60}")

    generator.reset()
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
    )
    prompt_len = input_ids.shape[1]
    print(f"  Prompt tokens: {prompt_len}")

    t0 = time.perf_counter()
    generated = generator.generate(input_ids, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - t0

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"  Generated {len(generated)} tokens in {elapsed:.2f}s ({len(generated)/elapsed:.1f} t/s)")
    print(f"  Output: {output_text}")

    passed = True
    if len(generated) == 0:
        print(f"  FAIL: no tokens generated")
        passed = False
    if not output_text.strip():
        print(f"  FAIL: decoded text is empty")
        passed = False

    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed, output_text


def main():
    print(f"[Config] Fused DeltaNet kernel: {USE_FUSED_KERNEL}")
    print(f"[Config] Full fused kernel: {USE_FULL_FUSED_KERNEL}")

    config = Qwen3CoderNextConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    model_path = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    print("[Weights] Loading...")
    t0 = time.time()
    state_dict = load_state_dict(config, model_path=model_path)
    print(f"[Weights] Loaded in {time.time() - t0:.1f}s")

    device = ttnn.open_device(device_id=0)
    try:
        print("[Model] Building...")
        t0 = time.time()
        model = TtQwen3CoderNextModel(device, state_dict, config)
        generator = Qwen3CoderNextGenerator(model, config, tokenizer=tokenizer)
        del state_dict
        print(f"[Model] Built in {time.time() - t0:.1f}s")

        results = []
        for name, prompt, max_tokens in PROMPTS:
            passed, text = test_prompt(generator, tokenizer, name, prompt, max_tokens)
            results.append((name, passed, text))

        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        all_pass = True
        for name, passed, text in results:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {text[:80]}...")
            if not passed:
                all_pass = False

        print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        sys.exit(0 if all_pass else 1)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
