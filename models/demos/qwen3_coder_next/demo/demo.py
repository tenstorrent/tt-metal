# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Coder-Next interactive text generation demo on TT P150a.

Usage:
    # Interactive (in Docker container with TT device):
    python -m models.demos.qwen3_coder_next.demo.demo --prompt "Hello, who are you?"

    # With options:
    python -m models.demos.qwen3_coder_next.demo.demo \
        --prompt "Explain quantum computing" \
        --max-tokens 256 \
        --temperature 0.7

    # Quick test with fewer layers (dummy weights):
    python -m models.demos.qwen3_coder_next.demo.demo \
        --prompt "Hello" --max-layers 4 --dummy-weights

    # Via pytest:
    pytest models/demos/qwen3_coder_next/demo/demo.py -v --timeout=600
"""

import argparse
import sys
import time

import torch
import ttnn

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.load_weights import load_state_dict, create_dummy_state_dict
from models.demos.qwen3_coder_next.tt.model import TtQwen3CoderNextModel
from models.demos.qwen3_coder_next.tt.generator import Qwen3CoderNextGenerator


def load_tokenizer(model_name: str = "Qwen/Qwen3-Coder-Next"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


def run_demo(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    max_layers: int | None = None,
    dummy_weights: bool = False,
    device_id: int = 0,
    model_path: str | None = None,
):
    config = Qwen3CoderNextConfig()
    if max_layers is not None:
        config.num_hidden_layers = max_layers

    print(f"[Config] layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
          f"vocab={config.vocab_size}, BFP4_B weights")

    tokenizer = load_tokenizer(config.model_name)
    print(f"[Tokenizer] loaded from {config.model_name}")

    if dummy_weights:
        print("[Weights] Using dummy random weights (for pipeline testing only)")
        state_dict = create_dummy_state_dict(config, num_layers=config.num_hidden_layers)
    else:
        src = model_path or config.model_name
        print(f"[Weights] Loading from: {src} ...")
        t0 = time.time()
        state_dict = load_state_dict(config, max_layers=max_layers, model_path=model_path)
        print(f"[Weights] Loaded in {time.time() - t0:.1f}s ({len(state_dict)} tensors)")

    print(f"[Device] Opening device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        print("[Model] Building TT model...")
        t0 = time.time()
        model = TtQwen3CoderNextModel(device, state_dict, config)
        generator = Qwen3CoderNextGenerator(model, config, tokenizer=tokenizer)
        print(f"[Model] Built in {time.time() - t0:.1f}s")

        del state_dict

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(formatted, return_tensors="pt")
        prompt_len = input_ids.shape[1]
        print(f"\n[Prompt] {prompt_len} tokens")
        print(f"---")

        t_prefill_start = time.time()
        last_logits = generator.prefill(input_ids)
        t_prefill_end = time.time()
        prefill_time = t_prefill_end - t_prefill_start
        prefill_tps = prompt_len / prefill_time

        logits_cpu = ttnn.to_torch(last_logits).float().reshape(-1)

        if temperature == 0:
            next_token = torch.argmax(logits_cpu[:config.vocab_size]).item()
        else:
            probs = torch.softmax(logits_cpu[:config.vocab_size] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        generated_ids = [next_token]
        text_so_far = tokenizer.decode([next_token], skip_special_tokens=False)
        sys.stdout.write(text_so_far)
        sys.stdout.flush()

        t_decode_start = time.time()
        for i in range(max_new_tokens - 1):
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            _, next_token_tensor = generator.decode_one_token(token_tensor)
            next_token = next_token_tensor.item()
            generated_ids.append(next_token)

            token_text = tokenizer.decode([next_token], skip_special_tokens=False)
            sys.stdout.write(token_text)
            sys.stdout.flush()

            if next_token == tokenizer.eos_token_id:
                break

        t_decode_end = time.time()
        decode_tokens = len(generated_ids) - 1
        decode_time = t_decode_end - t_decode_start
        decode_tps = decode_tokens / decode_time if decode_time > 0 else 0

        print(f"\n---")
        full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\n[Stats]")
        print(f"  Prefill: {prompt_len} tokens in {prefill_time:.2f}s ({prefill_tps:.1f} t/s)")
        print(f"  Decode:  {decode_tokens} tokens in {decode_time:.2f}s ({decode_tps:.2f} t/s)")
        print(f"  Total:   {len(generated_ids)} tokens generated")

        return {
            "prompt": prompt,
            "output": full_output,
            "generated_ids": generated_ids,
            "prefill_tokens": prompt_len,
            "prefill_time_s": prefill_time,
            "prefill_tps": prefill_tps,
            "decode_tokens": decode_tokens,
            "decode_time_s": decode_time,
            "decode_tps": decode_tps,
        }

    finally:
        ttnn.close_device(device)


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------
import pytest


@pytest.fixture
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_demo_dummy_weights(device):
    """Quick smoke test with dummy weights (4 layers)."""
    config = Qwen3CoderNextConfig(
        hidden_size=256, num_hidden_layers=4, full_attention_interval=4,
        linear_num_key_heads=4, linear_num_value_heads=8,
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=64,
        intermediate_size=512, vocab_size=1024, max_seq_len=128,
    )
    state_dict = create_dummy_state_dict(config, num_layers=4)
    model = TtQwen3CoderNextModel(device, state_dict, config)
    generator = Qwen3CoderNextGenerator(model, config)

    prompt = torch.tensor([[1, 42, 100, 7]])
    t0 = time.time()
    generated = generator.generate(prompt, max_new_tokens=10)
    elapsed = time.time() - t0

    assert len(generated) == 10
    print(f"Generated {len(generated)} tokens in {elapsed:.2f}s ({len(generated)/elapsed:.1f} t/s)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen3-Coder-Next inference demo on TT P150a")
    parser.add_argument("--prompt", type=str, default="Hello, who are you?",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Use only first N layers (for debugging)")
    parser.add_argument("--dummy-weights", action="store_true",
                        help="Use dummy random weights instead of real model")
    parser.add_argument("--device-id", type=int, default=0,
                        help="TT device ID")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to pre-downloaded model directory (skip HF download)")
    args = parser.parse_args()

    run_demo(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        max_layers=args.max_layers,
        dummy_weights=args.dummy_weights,
        device_id=args.device_id,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
