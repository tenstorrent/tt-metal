# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Combined evaluation: correctness + profiling + latency benchmark.
Loads model once, runs all three evaluations.

Usage:
  python -u run_full_eval.py [--skip-correctness] [--skip-profile] [--skip-bench]
"""

import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
import ttnn

from transformers import AutoTokenizer

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator
from models.demos.qwen36_27b.tt.deltanet import USE_FUSED_KERNEL, USE_FULL_FUSED_KERNEL


class TimingCollector:
    def __init__(self, device):
        self.records = defaultdict(list)
        self.device = device

    @contextmanager
    def section(self, name):
        ttnn.synchronize_device(self.device)
        t0 = time.perf_counter()
        yield
        ttnn.synchronize_device(self.device)
        self.records[name].append(time.perf_counter() - t0)

    def report(self, title="Timing Report"):
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
        total = 0
        rows = []
        for name, times in sorted(self.records.items()):
            avg_ms = sum(times) / len(times) * 1000
            total_ms = sum(times) * 1000
            count = len(times)
            total += total_ms
            rows.append((name, count, avg_ms, total_ms))

        for name, count, avg_ms, total_ms in sorted(rows, key=lambda r: -r[3]):
            pct = total_ms / total * 100 if total > 0 else 0
            print(f"  {name:<50} {avg_ms:>8.3f} ms × {count:>4} = {total_ms:>9.1f} ms ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<50} {'':>8}          {' ':>4}   {total:>9.1f} ms")
        print(f"{'='*80}")
        return total


# ── Part 1: Correctness ──────────────────────────────────────────────

def run_correctness(generator, tokenizer, config):
    print("\n" + "=" * 80)
    print("  PART 1: CORRECTNESS TEST")
    print("=" * 80)

    prompts = [
        "What is the capital of France?",
        "Write a haiku about autumn.",
        "2 + 3 = ",
    ]

    for prompt in prompts:
        generator.reset()
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
        )
        generated = generator.generate(input_ids, max_new_tokens=24)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\n  [Prompt] {prompt}")
        print(f"  [Output] {text}")

    print("\n  Correctness test complete — verify outputs above are coherent.")


# ── Part 2: Profile ──────────────────────────────────────────────────

def profiled_decode_step(model, generator, token_tensor, tc):
    config = model.config

    with tc.section("00_embed"):
        hidden = model.embed(token_tensor)
        cos, sin = model.get_rope(generator.position)

    kv_caches = dict(generator.kv_caches)

    for layer_idx, layer in enumerate(model.layers):
        layer_type = config.layer_types[layer_idx]
        kv_cache = kv_caches.get(layer_idx) if layer_type == "full_attention" else None

        with tc.section("01_input_layernorm"):
            normed = layer.input_layernorm(hidden)

        if layer_type == "linear_attention":
            with tc.section("02a_deltanet_layer"):
                mixer_out = layer.token_mixer(normed, generator.deltanet_state, mode="decode")
        else:
            with tc.section("02b_attention_layer"):
                mixer_out, new_kv = layer.token_mixer(normed, cos, sin, kv_cache, mode="decode")
                if new_kv is not None:
                    kv_caches[layer_idx] = new_kv

        with tc.section("03_residual_add_1"):
            hidden = ttnn.add(hidden, mixer_out)

        with tc.section("04_post_attn_layernorm"):
            normed2 = layer.post_attention_layernorm(hidden)

        with tc.section("05_mlp"):
            mlp_out = layer.mlp(normed2)

        with tc.section("06_residual_add_2"):
            hidden = ttnn.add(hidden, mlp_out)

    with tc.section("07_final_norm"):
        hidden = model.rms_norm(hidden, model.final_norm_weight)

    with tc.section("08_lm_head"):
        logits = ttnn.linear(hidden, model.lm_head_w)

    generator.kv_caches = kv_caches
    generator.position += 1

    logits_cpu = ttnn.to_torch(logits).float().reshape(-1)
    next_token = torch.argmax(logits_cpu[:config.vocab_size]).item()
    return logits, next_token


def run_profile(model, generator, tokenizer, config, device, num_decode=10, warmup=3):
    print("\n" + "=" * 80)
    print("  PART 2: PROFILER — DECODE BOTTLENECK ANALYSIS")
    print("=" * 80)

    prompt = "What is the meaning of life?"
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
    )
    generator.reset()

    print("  [Prefill]...")
    logits = generator.prefill(input_ids)
    logits_cpu = ttnn.to_torch(logits).float().reshape(-1)
    next_token = torch.argmax(logits_cpu[:config.vocab_size]).item()

    print(f"  [Warmup] {warmup} steps...")
    for _ in range(warmup):
        token_tensor = torch.tensor([[next_token]], dtype=torch.long)
        _, next_token_tensor = generator.decode_one_token(token_tensor)
        next_token = next_token_tensor.item()

    print(f"  [Profile] {num_decode} decode steps...")
    tc = TimingCollector(device)
    for i in range(num_decode):
        token_tensor = torch.tensor([[next_token]], dtype=torch.long)
        _, next_token = profiled_decode_step(model, generator, token_tensor, tc)

    n_dn = sum(1 for t in config.layer_types if t == "linear_attention")
    n_attn = sum(1 for t in config.layer_types if t == "full_attention")
    total_ms = tc.report(
        f"Decode Profile ({num_decode} tokens, {n_dn} DeltaNet + {n_attn} Attention)"
    )
    avg_ms = total_ms / num_decode
    print(f"\n  Average per-token latency: {avg_ms:.1f} ms ({1000/avg_ms:.2f} t/s)")


# ── Part 3: Latency Benchmark ────────────────────────────────────────

def run_benchmark(generator, tokenizer, config, num_warmup=1):
    print("\n" + "=" * 80)
    print("  PART 3: LATENCY BENCHMARK")
    print("=" * 80)

    samples = [
        ("short", "What is the capital of France?", 32),
        ("medium",
         "Explain the difference between supervised and unsupervised "
         "machine learning. Give one example of each approach.",
         64),
        ("code",
         "Write a Python function that implements binary search on a sorted list.",
         64),
    ]

    # Warmup
    for w in range(num_warmup):
        print(f"  [Warmup {w+1}]...")
        generator.reset()
        input_ids = torch.tensor(
            [tokenizer.encode("Hello", add_special_tokens=False)], dtype=torch.long
        )
        generator.generate(input_ids, max_new_tokens=8)

    results = []
    for name, prompt, max_tokens in samples:
        generator.reset()
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
        )
        prompt_len = input_ids.shape[1]

        # Prefill
        t_prefill_start = time.perf_counter()
        last_logits = generator.prefill(input_ids)
        t_prefill_end = time.perf_counter()

        logits_cpu = ttnn.to_torch(last_logits).float().reshape(-1)
        next_token = torch.argmax(logits_cpu[:config.vocab_size]).item()
        ttft = t_prefill_end - t_prefill_start

        # Decode
        generated = [next_token]
        decode_times = []
        for _ in range(max_tokens - 1):
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            t0 = time.perf_counter()
            _, next_token_tensor = generator.decode_one_token(token_tensor)
            t1 = time.perf_counter()
            decode_times.append(t1 - t0)
            next_token = next_token_tensor.item()
            generated.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break

        total_decode = sum(decode_times)
        num_gen = len(generated)
        tps = num_gen / total_decode if total_decode > 0 else 0
        avg_ms = sum(decode_times) / len(decode_times) * 1000 if decode_times else 0
        p50_ms = sorted(decode_times)[len(decode_times) // 2] * 1000 if decode_times else 0
        text = tokenizer.decode(generated, skip_special_tokens=True)

        results.append({
            "name": name,
            "prompt_tokens": prompt_len,
            "generated": num_gen,
            "ttft_ms": ttft * 1000,
            "tps": tps,
            "avg_ms": avg_ms,
            "p50_ms": p50_ms,
            "text": text,
        })

        print(f"\n  [{name}] {prompt_len} prompt → {num_gen} gen tokens")
        print(f"    TTFT: {ttft*1000:.0f} ms | Decode: {tps:.2f} t/s | avg={avg_ms:.0f} ms p50={p50_ms:.0f} ms")
        print(f"    Output: {text[:100]}...")

    # Summary
    print(f"\n  {'─'*70}")
    print(f"  {'Name':<10} {'Prompt':>6} {'Gen':>5} {'TTFT(ms)':>9} {'t/s':>8} {'avg(ms)':>8} {'p50(ms)':>8}")
    for r in results:
        print(f"  {r['name']:<10} {r['prompt_tokens']:>6} {r['generated']:>5} "
              f"{r['ttft_ms']:>8.0f}  {r['tps']:>7.2f}  {r['avg_ms']:>7.0f}  {r['p50_ms']:>7.0f}")

    total_gen = sum(r["generated"] for r in results)
    total_decode_s = sum(r["generated"] / r["tps"] for r in results if r["tps"] > 0)
    overall_tps = total_gen / total_decode_s if total_decode_s > 0 else 0
    print(f"\n  Overall decode throughput: {overall_tps:.2f} t/s")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    skip_correctness = "--skip-correctness" in sys.argv
    skip_profile = "--skip-profile" in sys.argv
    skip_bench = "--skip-bench" in sys.argv

    print(f"USE_FUSED_KERNEL={USE_FUSED_KERNEL}, USE_FULL_FUSED_KERNEL={USE_FULL_FUSED_KERNEL}")

    config = Qwen36ModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    print(f"weights_dtype={config.weights_dtype}")

    print("[Weights] Loading...")
    state_dict = load_state_dict(config)
    print("[Weights] Loaded")

    device = ttnn.open_device(device_id=0)
    try:
        print("[Model] Building...")
        t0 = time.time()
        model = TtQwen36Model(device, state_dict, config)
        generator = Qwen36Generator(model, config, tokenizer=tokenizer)
        del state_dict
        print(f"[Model] Built in {time.time() - t0:.1f}s")

        if not skip_correctness:
            run_correctness(generator, tokenizer, config)

        if not skip_profile:
            run_profile(model, generator, tokenizer, config, device, num_decode=5, warmup=2)

        if not skip_bench:
            run_benchmark(generator, tokenizer, config, num_warmup=1)

        print("\n" + "=" * 80)
        print("  ALL EVALUATIONS COMPLETE")
        print("=" * 80)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
