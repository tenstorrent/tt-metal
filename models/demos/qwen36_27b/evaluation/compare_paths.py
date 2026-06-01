# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare full_fused vs old fused decode for first few tokens."""

import torch
import ttnn
from transformers import AutoTokenizer
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator
import models.demos.qwen36_27b.tt.deltanet as dn

MODEL_PATH = "/tmp/qwen36_model/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
NUM_TOKENS = 10


def run_generation(device, state_dict, config, tokenizer, prompt, use_full_fused):
    """Run prefill + N decode tokens with specified kernel path."""
    dn.USE_FULL_FUSED_KERNEL = use_full_fused
    dn.USE_FUSED_KERNEL = True

    model = TtQwen36Model(device, state_dict, config)
    gen = Qwen36Generator(model, config, tokenizer=tokenizer)
    gen.reset()

    ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long)
    logits = gen.prefill(ids)
    l = ttnn.to_torch(logits).float().reshape(-1)
    tok = torch.argmax(l[:config.vocab_size]).item()
    tokens = [tok]

    for i in range(NUM_TOKENS - 1):
        _, next_t = gen.decode_one_token(torch.tensor([[tok]], dtype=torch.long))
        tok = next_t.item()
        tokens.append(tok)

    # Cleanup model weights
    del model, gen
    return tokens


def main():
    config = Qwen36ModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    state_dict = load_state_dict(config, model_path=MODEL_PATH)
    device = ttnn.open_device(device_id=0)

    prompt = "Explain the theory of relativity in simple terms."

    try:
        print(f"=== Old fused kernel ===", flush=True)
        tokens_old = run_generation(device, state_dict, config, tokenizer, prompt, use_full_fused=False)
        text_old = tokenizer.decode(tokens_old, skip_special_tokens=True)
        print(f"Tokens: {tokens_old}", flush=True)
        print(f"Text: {text_old[:200]}", flush=True)

        print(f"\n=== Full fused kernel ===", flush=True)
        tokens_fused = run_generation(device, state_dict, config, tokenizer, prompt, use_full_fused=True)
        text_fused = tokenizer.decode(tokens_fused, skip_special_tokens=True)
        print(f"Tokens: {tokens_fused}", flush=True)
        print(f"Text: {text_fused[:200]}", flush=True)

        print(f"\n=== CPU only (no fused kernel) ===", flush=True)
        tokens_cpu = run_generation(device, state_dict, config, tokenizer, prompt, use_full_fused=False)
        dn.USE_FUSED_KERNEL = False
        # Can't easily test CPU path without reloading - reuse old fused result
        print(f"(same as old fused for this test)")

        print(f"\n=== Comparison ===", flush=True)
        match_count = sum(a == b for a, b in zip(tokens_old, tokens_fused))
        print(f"Token match: {match_count}/{len(tokens_old)}", flush=True)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
