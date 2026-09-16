#!/usr/bin/env python3
"""
Tenstorrent Performance and Latency Benchmarking Harness for Qwen2.5-Coder
Measures prefill throughput (tokens/sec) and decode step latency (ms/token).
"""

import os
import sys
import time

# Ensure parent package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from common import Qwen2_5Config
from ttnn.ttnn_qwen_model import TTNNQwenForCausalLM

def benchmark_qwen_performance():
    print("=" * 75)
    print("⚡ RUNNING TENSTORRENT TTNN QWEN2.5 PERFORMANCE & LATENCY BENCHMARK")
    print("=" * 75)

    config = Qwen2_5Config(
        vocab_size=32000,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=4,
        num_attention_heads=14,
        num_key_value_heads=2
    )

    model = TTNNQwenForCausalLM(config)
    model.eval()

    seq_len = 128
    input_ids = torch.randint(0, 32000, (1, seq_len))

    # 1. Warm-up
    print("🔥 Warming up execution pipeline...")
    for _ in range(3):
        _ = model(input_ids)

    # 2. Prefill Benchmark
    print("⏱️  Benchmarking Prefill Phase (Seq Len = 128)...")
    iters = 10
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(input_ids)
    elapsed = time.perf_counter() - start
    prefill_throughput = (iters * seq_len) / elapsed
    print(f"    🚀 Prefill Throughput: {prefill_throughput:.1f} tokens/sec ({elapsed/iters*1000:.2f} ms/batch)")

    # 3. Decode Step Latency
    print("⏱️  Benchmarking Decode Auto-Regressive Step...")
    decode_tokens = 20
    start_decode = time.perf_counter()
    _ = model.generate(input_ids, max_new_tokens=decode_tokens)
    elapsed_decode = time.perf_counter() - start_decode
    decode_step_ms = (elapsed_decode / decode_tokens) * 1000
    print(f"    🚀 Decode Step Latency: {decode_step_ms:.2f} ms/token ({decode_tokens/elapsed_decode:.1f} tokens/sec)")

    print("\n" + "=" * 75)
    print("🎉 PERFORMANCE BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    benchmark_qwen_performance()
