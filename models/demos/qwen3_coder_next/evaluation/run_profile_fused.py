# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Profiler for the B2+ fused decode path.

Breaks down per-token decode latency into:
  - DeltaNet layers (fused kernel + out_proj)
  - Attention layers (projections + CPU RoPE/SDPA + o_proj)
  - MLP layers (gate/up/down)
  - LayerNorm (input + post-attention per layer)
  - Residual adds
  - Embedding + final norm + lm_head
  - ttnn.synchronize overhead

Usage:
  python run_profile_fused.py [num_decode_tokens] [warmup_tokens]
  Default: 10 decode tokens, 3 warmup
"""

import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
import ttnn

from transformers import AutoTokenizer

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.load_weights import load_state_dict
from models.demos.qwen3_coder_next.tt.model import TtQwen3CoderNextModel
from models.demos.qwen3_coder_next.tt.generator import Qwen3CoderNextGenerator


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
            per_call = avg_ms
            print(f"  {name:<50} {per_call:>8.3f} ms × {count:>4} = {total_ms:>9.1f} ms ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<50} {'':>8}          {' ':>4}   {total:>9.1f} ms")
        print(f"{'='*80}")
        return total


def profiled_decode_step(model, generator, token_tensor, tc):
    """One decode step with fine-grained per-layer-type profiling."""
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


def profiled_attention_breakdown(model, generator, token_tensor, tc):
    """One decode step with attention internals broken out separately."""
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
            # DeltaNet: break into projections + kernel + out_proj
            dn = layer.token_mixer
            with tc.section("02a_dn_projections"):
                qkv = ttnn.linear(normed, dn.in_proj_qkv_w)
                z = ttnn.linear(normed, dn.in_proj_z_w)
                b_proj = ttnn.linear(normed, dn.in_proj_b_w)
                a_proj = ttnn.linear(normed, dn.in_proj_a_w)

            with tc.section("02b_dn_fused_kernel"):
                from models.demos.qwen3_coder_next.tt.deltanet import _deltanet_decode_full_op
                conv_state = generator.deltanet_state.get_conv_state(layer_idx)
                state = generator.deltanet_state.get_recurrent_state(layer_idx)
                if state.dtype != ttnn.bfloat16:
                    state = ttnn.typecast(state, ttnn.bfloat16)

                output_tt, new_state, new_conv_state = _deltanet_decode_full_op(
                    qkv, z, b_proj, a_proj,
                    conv_state, state,
                    dn.conv1d_weight_tt, dn.A_log_bf16,
                    dn.dt_bias_bf16, dn.norm_weight,
                    num_heads=dn.num_v_heads, num_k_heads=dn.num_k_heads,
                    k_head_dim=dn.head_k_dim, v_head_dim=dn.head_v_dim,
                    conv_dim=dn.key_dim * 2 + dn.value_dim,
                    conv_kernel_size=dn.conv_kernel_size,
                    head_expand_ratio=dn.head_expand_ratio,
                )

            with tc.section("02c_dn_state_update"):
                generator.deltanet_state.set_recurrent_state(layer_idx, ttnn.typecast(new_state, ttnn.float32))
                generator.deltanet_state.set_conv_state(layer_idx, new_conv_state)

            with tc.section("02d_dn_out_proj"):
                mixer_out = ttnn.linear(output_tt, dn.out_proj_w)

        else:
            # Attention: break into projections, CPU ops, o_proj
            attn = layer.token_mixer
            with tc.section("03a_attn_projections"):
                q_proj = ttnn.linear(normed, attn.q_proj_w)
                k_proj = ttnn.linear(normed, attn.k_proj_w)
                v_proj = ttnn.linear(normed, attn.v_proj_w)

            with tc.section("03b_attn_reshape_norm"):
                q_2d = ttnn.reshape(q_proj, [1, 1, attn.num_heads, attn.head_dim * 2])
                query = ttnn.slice(q_2d, [0, 0, 0, 0], [1, 1, attn.num_heads, attn.head_dim])
                gate = ttnn.slice(q_2d, [0, 0, 0, attn.head_dim], [1, 1, attn.num_heads, attn.head_dim * 2])
                key = ttnn.reshape(k_proj, [1, 1, attn.num_kv_heads, attn.head_dim])
                value = ttnn.reshape(v_proj, [1, 1, attn.num_kv_heads, attn.head_dim])
                query = ttnn.rms_norm(query, epsilon=1e-6, weight=attn.q_norm_w_tt)
                key = ttnn.rms_norm(key, epsilon=1e-6, weight=attn.k_norm_w_tt)

            with tc.section("03c_attn_to_cpu"):
                cos_cpu = cos if isinstance(cos, torch.Tensor) else ttnn.to_torch(cos)
                sin_cpu = sin if isinstance(sin, torch.Tensor) else ttnn.to_torch(sin)
                query_cpu = ttnn.to_torch(query).reshape(1, attn.num_heads, 1, attn.head_dim)
                key_cpu = ttnn.to_torch(key).reshape(1, attn.num_kv_heads, 1, attn.head_dim)
                value_cpu = ttnn.to_torch(value).reshape(1, attn.num_kv_heads, 1, attn.head_dim)
                gate_cpu = ttnn.to_torch(gate).reshape(1, 1, attn.num_heads * attn.head_dim)

            with tc.section("03d_attn_cpu_rope_sdpa"):
                query_cpu, key_cpu = attn._apply_partial_rotary(query_cpu, key_cpu, cos_cpu, sin_cpu)
                if kv_cache is not None:
                    k_cache, v_cache = kv_cache
                    key_cpu = torch.cat([k_cache, key_cpu], dim=2)
                    value_cpu = torch.cat([v_cache, value_cpu], dim=2)
                new_kv = (key_cpu.detach(), value_cpu.detach())
                kv_caches[layer_idx] = new_kv

                if attn.num_kv_groups > 1:
                    key_exp = key_cpu.repeat_interleave(attn.num_kv_groups, dim=1)
                    value_exp = value_cpu.repeat_interleave(attn.num_kv_groups, dim=1)
                else:
                    key_exp = key_cpu
                    value_exp = value_cpu

                attn_weights = torch.matmul(query_cpu.float(), key_exp.float().transpose(-1, -2)) * attn.scaling
                seq_len = key_exp.shape[2]
                causal_mask = torch.triu(torch.full((1, seq_len), float("-inf")), diagonal=seq_len)
                attn_weights = attn_weights + causal_mask
                attn_weights = torch.softmax(attn_weights, dim=-1).to(query_cpu.dtype)
                attn_output = torch.matmul(attn_weights, value_exp.float()).to(query_cpu.dtype)
                attn_output = attn_output.transpose(1, 2).reshape(1, 1, -1)
                attn_output = attn_output * torch.sigmoid(gate_cpu.float()).to(attn_output.dtype)

            with tc.section("03e_attn_from_cpu_oproj"):
                attn_output_tt = ttnn.from_torch(
                    attn_output.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=attn.device,
                )
                mixer_out = ttnn.linear(attn_output_tt, attn.o_proj_w)

        with tc.section("04_residual_add_1"):
            hidden = ttnn.add(hidden, mixer_out)

        with tc.section("05_post_attn_layernorm"):
            normed2 = layer.post_attention_layernorm(hidden)

        with tc.section("06_mlp"):
            mlp_out = layer.mlp(normed2)

        with tc.section("07_residual_add_2"):
            hidden = ttnn.add(hidden, mlp_out)

    with tc.section("08_final_norm"):
        hidden = model.rms_norm(hidden, model.final_norm_weight)

    with tc.section("09_lm_head"):
        logits = ttnn.linear(hidden, model.lm_head_w)

    generator.kv_caches = kv_caches
    generator.position += 1

    logits_cpu = ttnn.to_torch(logits).float().reshape(-1)
    next_token = torch.argmax(logits_cpu[:config.vocab_size]).item()
    return logits, next_token


def main():
    num_decode = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    detailed = "--detailed" in sys.argv

    print(f"[Config] Decode tokens: {num_decode}, Warmup: {warmup}, Detailed: {detailed}")

    config = Qwen3CoderNextConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    from models.demos.qwen3_coder_next.tt.deltanet import USE_FUSED_KERNEL, USE_FULL_FUSED_KERNEL
    print(f"[Config] USE_FUSED_KERNEL={USE_FUSED_KERNEL}, USE_FULL_FUSED_KERNEL={USE_FULL_FUSED_KERNEL}")
    print(f"[Config] weights_dtype={config.weights_dtype}")

    print("[Weights] Loading...")
    state_dict = load_state_dict(config)
    print("[Weights] Loaded")

    device = ttnn.open_device(device_id=0)
    try:
        print("[Model] Building...")
        t0 = time.time()
        model = TtQwen3CoderNextModel(device, state_dict, config)
        generator = Qwen3CoderNextGenerator(model, config, tokenizer=tokenizer)
        del state_dict
        print(f"[Model] Built in {time.time() - t0:.1f}s")

        prompt = "What is the meaning of life?"
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
        )
        generator.reset()

        print("[Prefill] Running...")
        logits = generator.prefill(input_ids)
        logits_cpu = ttnn.to_torch(logits).float().reshape(-1)
        next_token = torch.argmax(logits_cpu[:config.vocab_size]).item()
        print(f"[Prefill] Done, first token: {tokenizer.decode([next_token])!r}")

        # Warmup decode
        print(f"[Warmup] Running {warmup} decode steps...")
        for i in range(warmup):
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            _, next_token_tensor = generator.decode_one_token(token_tensor)
            next_token = next_token_tensor.item()
        print(f"[Warmup] Done")

        # Profiled decode (high level)
        print(f"\n[Profile] Running {num_decode} decode steps (high-level)...")
        tc_high = TimingCollector(device)
        for i in range(num_decode):
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            _, next_token = profiled_decode_step(model, generator, token_tensor, tc_high)
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{num_decode}] done")

        n_dn = sum(1 for t in config.layer_types if t == "linear_attention")
        n_attn = sum(1 for t in config.layer_types if t == "full_attention")
        total_ms = tc_high.report(
            f"High-Level Decode Profile ({num_decode} tokens, {n_dn} DeltaNet + {n_attn} Attention layers)"
        )
        avg_ms = total_ms / num_decode
        print(f"\n  Average per-token latency: {avg_ms:.1f} ms ({1000/avg_ms:.2f} t/s)")

        # Profiled decode (detailed per-sublayer)
        if detailed:
            print(f"\n[Profile] Running {num_decode} decode steps (detailed breakdown)...")
            tc_detail = TimingCollector(device)
            for i in range(num_decode):
                token_tensor = torch.tensor([[next_token]], dtype=torch.long)
                _, next_token = profiled_attention_breakdown(model, generator, token_tensor, tc_detail)
                if (i + 1) % 5 == 0:
                    print(f"  [{i+1}/{num_decode}] done")

            total_ms = tc_detail.report(
                f"Detailed Decode Profile ({num_decode} tokens)"
            )
            avg_ms = total_ms / num_decode
            print(f"\n  Average per-token latency: {avg_ms:.1f} ms ({1000/avg_ms:.2f} t/s)")

            # Summary by category
            records = tc_detail.records
            groups = {
                "DeltaNet (projections)": ["02a_dn_projections"],
                "DeltaNet (fused kernel)": ["02b_dn_fused_kernel"],
                "DeltaNet (state update)": ["02c_dn_state_update"],
                "DeltaNet (out_proj)": ["02d_dn_out_proj"],
                "Attention (projections)": ["03a_attn_projections"],
                "Attention (reshape+norm)": ["03b_attn_reshape_norm"],
                "Attention (to_cpu)": ["03c_attn_to_cpu"],
                "Attention (CPU RoPE+SDPA)": ["03d_attn_cpu_rope_sdpa"],
                "Attention (from_cpu+o_proj)": ["03e_attn_from_cpu_oproj"],
                "MLP": ["06_mlp"],
                "LayerNorm": ["01_input_layernorm", "05_post_attn_layernorm"],
                "Residual": ["04_residual_add_1", "07_residual_add_2"],
                "Embed+Head": ["00_embed", "08_final_norm", "09_lm_head"],
            }
            print(f"\n{'='*80}")
            print(f"  Summary by Category ({num_decode} tokens)")
            print(f"{'='*80}")
            grand_total = sum(sum(t) for t in records.values()) * 1000
            for group_name, keys in groups.items():
                group_total = sum(sum(records.get(k, [])) for k in keys) * 1000
                pct = group_total / grand_total * 100 if grand_total > 0 else 0
                per_tok = group_total / num_decode
                print(f"  {group_name:<35} {per_tok:>8.1f} ms/tok  {group_total:>9.1f} ms ({pct:>5.1f}%)")
            print(f"  {'TOTAL':<35} {grand_total/num_decode:>8.1f} ms/tok  {grand_total:>9.1f} ms")
            print(f"{'='*80}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
