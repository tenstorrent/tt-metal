# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Performance benchmark: Qwen3-Coder-Next with fused DeltaNet kernel.

Measures TTFT, decode throughput (tokens/s), and end-to-end latency
across prompts of varying lengths.
"""

import json
import time
import sys

import torch
import ttnn

from transformers import AutoTokenizer

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.load_weights import load_state_dict
from models.demos.qwen3_coder_next.tt.model import TtQwen3CoderNextModel
from models.demos.qwen3_coder_next.tt.generator import Qwen3CoderNextGenerator
from models.demos.qwen3_coder_next.tt.deltanet import USE_FUSED_KERNEL

SAMPLE_PROMPTS = [
    {
        "name": "short",
        "prompt": "What is the capital of France?",
        "max_new_tokens": 32,
    },
    {
        "name": "medium",
        "prompt": (
            "Explain the difference between supervised and unsupervised "
            "machine learning. Give one example of each approach and describe "
            "when you would choose one over the other."
        ),
        "max_new_tokens": 64,
    },
    {
        "name": "long",
        "prompt": (
            "You are a senior software engineer reviewing a pull request. "
            "The PR adds a new caching layer to a web application. The cache "
            "uses an LRU eviction policy with a configurable TTL. The implementation "
            "stores entries in a hash map with a doubly-linked list for ordering. "
            "Write a detailed code review covering correctness, performance, "
            "thread safety, and potential edge cases. Be specific about what "
            "could go wrong and suggest improvements."
        ),
        "max_new_tokens": 128,
    },
    {
        "name": "code",
        "prompt": "Write a Python function that implements binary search on a sorted list. Include type hints and docstring.",
        "max_new_tokens": 96,
    },
]


def measure_generation(generator, tokenizer, prompt, max_new_tokens, warmup=False):
    """Run a single generation and return timing metrics."""
    generator.reset()

    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
    )
    prompt_len = input_ids.shape[1]

    # --- Prefill (TTFT) ---
    t_prefill_start = time.perf_counter()
    last_logits = generator.prefill(input_ids)
    t_prefill_end = time.perf_counter()

    logits_cpu = ttnn.to_torch(last_logits).float().reshape(-1)
    next_token = torch.argmax(logits_cpu[: generator.config.vocab_size]).item()

    ttft = t_prefill_end - t_prefill_start

    # --- Decode ---
    generated = [next_token]
    decode_times = []

    for _ in range(max_new_tokens - 1):
        token_tensor = torch.tensor([[next_token]], dtype=torch.long)

        t_step_start = time.perf_counter()
        _, next_token_tensor = generator.decode_one_token(token_tensor)
        t_step_end = time.perf_counter()

        decode_times.append(t_step_end - t_step_start)
        next_token = next_token_tensor.item()
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    total_decode_time = sum(decode_times)
    num_decoded = len(generated)
    decode_tps = num_decoded / total_decode_time if total_decode_time > 0 else 0

    # Per-token stats
    if decode_times:
        avg_step = sum(decode_times) / len(decode_times) * 1000
        min_step = min(decode_times) * 1000
        max_step = max(decode_times) * 1000
        p50_step = sorted(decode_times)[len(decode_times) // 2] * 1000
    else:
        avg_step = min_step = max_step = p50_step = 0

    total_time = ttft + total_decode_time
    output_text = tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": num_decoded,
        "ttft_ms": ttft * 1000,
        "decode_throughput_tps": decode_tps,
        "total_time_s": total_time,
        "prefill_tps": prompt_len / ttft if ttft > 0 else 0,
        "decode_avg_ms": avg_step,
        "decode_min_ms": min_step,
        "decode_max_ms": max_step,
        "decode_p50_ms": p50_step,
        "output_text": output_text,
    }


def main():
    max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else len(SAMPLE_PROMPTS)
    warmup_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"[Config] Fused DeltaNet kernel: {'ENABLED' if USE_FUSED_KERNEL else 'DISABLED'}")
    print(f"[Config] Samples: {max_samples}, Warmup runs: {warmup_runs}")

    config = Qwen3CoderNextConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    print("[Weights] Loading...")
    t0 = time.time()
    state_dict = load_state_dict(config)
    print(f"[Weights] Loaded in {time.time() - t0:.1f}s")

    device = ttnn.open_device(device_id=0)
    try:
        print("[Model] Building...")
        t0 = time.time()
        model = TtQwen3CoderNextModel(device, state_dict, config)
        generator = Qwen3CoderNextGenerator(model, config, tokenizer=tokenizer)
        del state_dict
        print(f"[Model] Built in {time.time() - t0:.1f}s")

        samples = SAMPLE_PROMPTS[:max_samples]

        # Warmup
        for w in range(warmup_runs):
            print(f"\n[Warmup {w+1}/{warmup_runs}] Running short prompt...")
            _ = measure_generation(
                generator, tokenizer, "Hello", max_new_tokens=8, warmup=True
            )
            print(f"[Warmup {w+1}/{warmup_runs}] Done")

        # Benchmark
        results = []
        print(f"\n{'='*70}")
        print(f"  Performance Benchmark — Qwen3-Coder-Next (fused={'YES' if USE_FUSED_KERNEL else 'NO'})")
        print(f"{'='*70}")

        for i, sample in enumerate(samples):
            print(f"\n--- [{i+1}/{len(samples)}] {sample['name']} ---")
            print(f"  Prompt: {sample['prompt'][:80]}...")
            print(f"  Max new tokens: {sample['max_new_tokens']}")

            metrics = measure_generation(
                generator, tokenizer, sample["prompt"], sample["max_new_tokens"]
            )
            metrics["name"] = sample["name"]
            results.append(metrics)

            print(f"  Prompt tokens:       {metrics['prompt_tokens']}")
            print(f"  Generated tokens:    {metrics['generated_tokens']}")
            print(f"  TTFT:                {metrics['ttft_ms']:.0f} ms")
            print(f"  Prefill throughput:  {metrics['prefill_tps']:.1f} t/s")
            print(f"  Decode throughput:   {metrics['decode_throughput_tps']:.2f} t/s")
            print(f"  Decode latency:      avg={metrics['decode_avg_ms']:.0f} ms  "
                  f"p50={metrics['decode_p50_ms']:.0f} ms  "
                  f"min={metrics['decode_min_ms']:.0f} ms  "
                  f"max={metrics['decode_max_ms']:.0f} ms")
            print(f"  Total time:          {metrics['total_time_s']:.1f} s")
            print(f"  Output: {metrics['output_text'][:120]}...")

        # Summary table
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Name':<10} {'Prompt':>6} {'Gen':>5} {'TTFT(ms)':>9} "
              f"{'Prefill':>9} {'Decode':>8} {'Total(s)':>9}")
        print(f"  {'':_<10} {'tok':_>6} {'tok':_>5} {'':_>9} "
              f"{'t/s':_>9} {'t/s':_>8} {'':_>9}")

        total_gen_tokens = 0
        total_decode_time = 0
        for r in results:
            print(f"  {r['name']:<10} {r['prompt_tokens']:>6} {r['generated_tokens']:>5} "
                  f"{r['ttft_ms']:>8.0f}  {r['prefill_tps']:>8.1f} "
                  f"{r['decode_throughput_tps']:>7.2f}  {r['total_time_s']:>8.1f}")
            total_gen_tokens += r["generated_tokens"]
            total_decode_time += r["total_time_s"] - r["ttft_ms"] / 1000

        if total_decode_time > 0:
            avg_decode_tps = total_gen_tokens / total_decode_time
        else:
            avg_decode_tps = 0
        avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)

        print(f"\n  Avg TTFT:            {avg_ttft:.0f} ms")
        print(f"  Avg decode:          {avg_decode_tps:.2f} t/s")
        print(f"  Fused kernel:        {'YES' if USE_FUSED_KERNEL else 'NO'}")
        print(f"{'='*70}")

        output = {
            "fused_kernel": USE_FUSED_KERNEL,
            "avg_ttft_ms": avg_ttft,
            "avg_decode_tps": avg_decode_tps,
            "results": [
                {k: v for k, v in r.items() if k != "output_text"}
                for r in results
            ],
        }
        with open("/tmp/eval_perf_result.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to /tmp/eval_perf_result.json")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
