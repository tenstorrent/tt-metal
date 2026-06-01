# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Profiler for Qwen3-Coder-Next decode and prefill paths.

Produces:
  1. Per-component timing breakdown (manual instrumentation)
  2. cProfile dump for host-side function-level analysis
"""

import cProfile
import pstats
import time
import sys
import io
from collections import defaultdict
from contextlib import contextmanager

import torch
import ttnn

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.load_weights import load_state_dict
from models.demos.qwen3_coder_next.tt.model import TtQwen3CoderNextModel
from models.demos.qwen3_coder_next.tt.generator import Qwen3CoderNextGenerator


class TimingCollector:
    def __init__(self):
        self.records = defaultdict(list)

    @contextmanager
    def section(self, name):
        t0 = time.perf_counter()
        yield
        self.records[name].append(time.perf_counter() - t0)

    def report(self, title="Timing Report"):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        total = 0
        rows = []
        for name, times in sorted(self.records.items()):
            avg_ms = sum(times) / len(times) * 1000
            total_ms = sum(times) * 1000
            total += total_ms
            rows.append((name, len(times), avg_ms, total_ms))

        for name, count, avg_ms, total_ms in sorted(rows, key=lambda r: -r[3]):
            pct = total_ms / total * 100 if total > 0 else 0
            print(f"  {name:<40} {avg_ms:>8.2f} ms/call × {count:>4} = {total_ms:>10.1f} ms ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<40} {'':>8}          {' ':>4}   {total:>10.1f} ms")
        print(f"{'='*70}")
        return total


def profile_decode(generator, tokenizer, num_tokens=20, warmup=3):
    """Profile the decode path with per-component timing."""
    tc = TimingCollector()

    prompt = "What is the meaning of life?"
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
    )

    generator.reset()
    last_logits = generator.prefill(input_ids)
    logits_cpu = ttnn.to_torch(last_logits).float().reshape(-1)
    next_token = torch.argmax(logits_cpu[:generator.config.vocab_size]).item()

    # Warmup decode
    for _ in range(warmup):
        token_tensor = torch.tensor([[next_token]], dtype=torch.long)
        _, next_token_tensor = generator.decode_one_token(token_tensor)
        next_token = next_token_tensor.item()

    # Profiled decode
    for i in range(num_tokens):
        token_tensor = torch.tensor([[next_token]], dtype=torch.long)

        with tc.section("decode_total"):
            with tc.section("01_embed"):
                hidden = generator.model.embed(token_tensor)
            with tc.section("02_rope"):
                cos, sin = generator.model.get_rope(generator.position)

            kv_caches = dict(generator.kv_caches)

            for layer_idx, layer in enumerate(generator.model.layers):
                layer_type = generator.model.config.layer_types[layer_idx]
                kv_cache = kv_caches.get(layer_idx) if layer_type == "full_attention" else None

                with tc.section(f"03_layernorm_input"):
                    normed = layer.input_layernorm(hidden)

                if layer_type == "linear_attention":
                    with tc.section("04_deltanet"):
                        mixer_out = layer.token_mixer(normed, generator.deltanet_state, mode="decode")
                    new_kv = None
                else:
                    with tc.section("05_attention"):
                        mixer_out, new_kv = layer.token_mixer(normed, cos, sin, kv_cache)

                with tc.section("06_residual_1"):
                    hidden = ttnn.add(hidden, mixer_out)

                with tc.section("07_layernorm_post"):
                    normed2 = layer.post_attention_layernorm(hidden)

                with tc.section("08_mlp"):
                    mlp_out = layer.mlp(normed2)

                with tc.section("09_residual_2"):
                    hidden = ttnn.add(hidden, mlp_out)

                if new_kv is not None:
                    kv_caches[layer_idx] = new_kv

            with tc.section("10_final_norm"):
                hidden = generator.model.rms_norm(hidden, generator.model.final_norm_weight)
            with tc.section("11_lm_head"):
                logits = ttnn.linear(hidden, generator.model.lm_head_w)

        generator.kv_caches = kv_caches
        generator.position += 1

        logits_cpu = ttnn.to_torch(logits).float().reshape(-1)
        next_token = torch.argmax(logits_cpu[:generator.config.vocab_size]).item()

    tc.report(f"Decode Breakdown ({num_tokens} tokens, {generator.model.config.num_hidden_layers} layers)")
    return tc


def profile_prefill(generator, tokenizer, prompt="Explain quantum computing in simple terms."):
    """Profile the prefill path."""
    tc = TimingCollector()

    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
    )
    S = input_ids.shape[1]

    generator.reset()

    with tc.section("prefill_total"):
        with tc.section("01_embed"):
            hidden = generator.model.embed(input_ids)
        with tc.section("02_rope"):
            position_ids = torch.arange(S)
            cos, sin = generator.model.get_rope(position_ids)

        kv_caches = {}

        for layer_idx, layer in enumerate(generator.model.layers):
            layer_type = generator.model.config.layer_types[layer_idx]
            kv_cache = kv_caches.get(layer_idx) if layer_type == "full_attention" else None

            with tc.section("03_layernorm_input"):
                normed = layer.input_layernorm(hidden)

            if layer_type == "linear_attention":
                with tc.section("04_deltanet_prefill"):
                    mixer_out = layer.token_mixer(normed, generator.deltanet_state, mode="prefill")
                new_kv = None
            else:
                with tc.section("05_attention_prefill"):
                    mixer_out, new_kv = layer.token_mixer(normed, cos, sin, kv_cache)

            with tc.section("06_residual_1"):
                hidden = ttnn.add(hidden, mixer_out)

            with tc.section("07_layernorm_post"):
                normed2 = layer.post_attention_layernorm(hidden)

            with tc.section("08_mlp_prefill"):
                mlp_out = layer.mlp(normed2)

            with tc.section("09_residual_2"):
                hidden = ttnn.add(hidden, mlp_out)

            if new_kv is not None:
                kv_caches[layer_idx] = new_kv

        with tc.section("10_slice_last"):
            last_hidden = ttnn.to_torch(hidden)[:, :, -1:, :]
            hidden = ttnn.from_torch(
                last_hidden, dtype=generator.model.dtype,
                layout=ttnn.TILE_LAYOUT, device=generator.model.device,
            )
        with tc.section("11_final_norm"):
            hidden = generator.model.rms_norm(hidden, generator.model.final_norm_weight)
        with tc.section("12_lm_head"):
            logits = ttnn.linear(hidden, generator.model.lm_head_w)

    generator.kv_caches = kv_caches
    generator.deltanet_state = generator.deltanet_state
    generator.position = S

    tc.report(f"Prefill Breakdown (S={S} tokens, {generator.model.config.num_hidden_layers} layers)")
    return tc


def profile_cprofile(generator, tokenizer, num_decode=10):
    """Run cProfile on a full generation to see host function hotspots."""
    prompt = "Hello"
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
    )

    def run():
        generator.reset()
        logits = generator.prefill(input_ids)
        logits_cpu = ttnn.to_torch(logits).float().reshape(-1)
        next_token = torch.argmax(logits_cpu[:generator.config.vocab_size]).item()
        for _ in range(num_decode):
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            _, next_token_tensor = generator.decode_one_token(token_tensor)
            next_token = next_token_tensor.item()

    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()

    print(f"\n{'='*70}")
    print(f"  cProfile Top 30 (prefill + {num_decode} decode steps)")
    print(f"{'='*70}")
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())

    print(f"\n{'='*70}")
    print(f"  cProfile Top 30 by tottime")
    print(f"{'='*70}")
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")
    stats2.print_stats(30)
    print(stream2.getvalue())


def main():
    num_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    num_decode = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    config = Qwen3CoderNextConfig(num_hidden_layers=num_layers)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    print(f"[Config] Layers: {num_layers}, Decode tokens: {num_decode}")
    print("[Weights] Loading...")
    if num_layers < 64:
        from models.demos.qwen3_coder_next.tt.load_weights import create_dummy_state_dict
        state_dict = create_dummy_state_dict(config, num_layers=num_layers)
    else:
        state_dict = load_state_dict(
            config,
            model_path="/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        )
    print("[Weights] Loaded")

    device = ttnn.open_device(device_id=0)
    try:
        print("[Model] Building...")
        t0 = time.time()
        model = TtQwen3CoderNextModel(device, state_dict, config)
        generator = Qwen3CoderNextGenerator(model, config, tokenizer=tokenizer)
        del state_dict
        print(f"[Model] Built in {time.time() - t0:.1f}s")

        print("\n[1/3] Profiling decode path...")
        profile_decode(generator, tokenizer, num_tokens=num_decode, warmup=3)

        print("\n[2/3] Profiling prefill path...")
        profile_prefill(generator, tokenizer, "Explain the theory of relativity in simple terms for a high school student.")

        print("\n[3/3] Running cProfile...")
        profile_cprofile(generator, tokenizer, num_decode=num_decode)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
