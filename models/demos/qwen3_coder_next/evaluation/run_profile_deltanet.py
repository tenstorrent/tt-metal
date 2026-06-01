# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fine-grained profiling of DeltaNet _decode_step internals."""

import time
import sys
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
            print(f"  {name:<45} {avg_ms:>8.3f} ms × {count:>4} = {total_ms:>8.1f} ms ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<45} {'':>8}          {' ':>4}   {total:>8.1f} ms")
        print(f"{'='*70}")


def profile_deltanet_decode_step(dn_layer, hidden_states, deltanet_state, tc):
    """Instrumented version of _decode_step."""
    B = 1

    with tc.section("A1_linear_qkv"):
        qkv = ttnn.linear(hidden_states, dn_layer.in_proj_qkv_w)
    with tc.section("A2_linear_z"):
        z = ttnn.linear(hidden_states, dn_layer.in_proj_z_w)
    with tc.section("A3_linear_b"):
        b_proj = ttnn.linear(hidden_states, dn_layer.in_proj_b_w)
    with tc.section("A4_linear_a"):
        a_proj = ttnn.linear(hidden_states, dn_layer.in_proj_a_w)

    with tc.section("B1_to_torch_qkv"):
        qkv_cpu = ttnn.to_torch(qkv).flatten()

    with tc.section("B2_conv1d"):
        conv_state = deltanet_state.get_conv_state(dn_layer.layer_idx)
        if conv_state is not None:
            conv_state_np = conv_state.squeeze(0)
            conv_state_np = torch.roll(conv_state_np, shifts=-1, dims=-1)
            conv_state_np[:, -1] = qkv_cpu
            deltanet_state.set_conv_state(dn_layer.layer_idx, conv_state_np.unsqueeze(0))
            qkv_conv = (conv_state_np * dn_layer.conv1d_weight).sum(dim=-1)
            qkv_conv = torch.nn.functional.silu(qkv_conv)
        else:
            deltanet_state.set_conv_state(
                dn_layer.layer_idx,
                qkv_cpu.unsqueeze(0).unsqueeze(-1).expand(-1, -1, dn_layer.conv_kernel_size).clone()
            )
            qkv_conv = torch.nn.functional.silu(qkv_cpu)

    with tc.section("B3_split_reshape"):
        query_t, key_t, value_t = torch.split(
            qkv_conv, [dn_layer.key_dim, dn_layer.key_dim, dn_layer.value_dim], dim=-1
        )
        query_t = query_t.reshape(B, dn_layer.num_k_heads, dn_layer.head_k_dim)
        key_t = key_t.reshape(B, dn_layer.num_k_heads, dn_layer.head_k_dim)
        value_t = value_t.reshape(B, dn_layer.num_v_heads, dn_layer.head_v_dim)

    with tc.section("C1_to_torch_b"):
        b_cpu = ttnn.to_torch(b_proj).flatten()[:dn_layer.num_v_heads]
    with tc.section("C2_to_torch_a"):
        a_cpu = ttnn.to_torch(a_proj).flatten()[:dn_layer.num_v_heads]

    with tc.section("C3_decay_beta"):
        beta = torch.sigmoid(b_cpu.float())
        g = -dn_layer.A_log_cpu.exp() * torch.nn.functional.softplus(a_cpu.float() + dn_layer.dt_bias_cpu)

    with tc.section("D1_head_expand"):
        if dn_layer.head_expand_ratio > 1:
            query_t = query_t.repeat_interleave(dn_layer.head_expand_ratio, dim=1)
            key_t = key_t.repeat_interleave(dn_layer.head_expand_ratio, dim=1)

    with tc.section("D2_l2norm_scale"):
        query_t = dn_layer._l2norm_cpu(query_t, dim=-1).float()
        key_t = dn_layer._l2norm_cpu(key_t, dim=-1).float()
        value_t = value_t.float()
        scale = dn_layer.head_k_dim**-0.5
        query_t = query_t * scale

    with tc.section("E1_to_torch_state"):
        state_cpu = ttnn.to_torch(deltanet_state.get_recurrent_state(dn_layer.layer_idx))

    with tc.section("E2_recurrence"):
        q_t = query_t[0]
        k_t = key_t[0]
        v_t = value_t[0]
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta.unsqueeze(-1)

        S = state_cpu[0]
        S = S * g_t
        kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)

    with tc.section("E3_set_state"):
        deltanet_state.set_recurrent_state(
            dn_layer.layer_idx,
            ttnn.from_torch(S.unsqueeze(0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dn_layer.device),
        )

    with tc.section("F1_to_torch_z"):
        z_cpu = ttnn.to_torch(z).flatten()[:dn_layer.num_v_heads * dn_layer.head_v_dim]

    with tc.section("F2_norm_gate"):
        z_cpu = z_cpu.reshape(dn_layer.num_v_heads, dn_layer.head_v_dim).float()
        variance = output_t.pow(2).mean(-1, keepdim=True)
        out_normed = output_t * torch.rsqrt(variance + 1e-6)
        out_normed = dn_layer.norm_weight_cpu * out_normed
        out_gated = out_normed * torch.nn.functional.silu(z_cpu)

    with tc.section("F3_from_torch_output"):
        out_gated = out_gated.reshape(1, 1, 1, -1).to(torch.bfloat16)
        out_tt = ttnn.from_torch(out_gated, dtype=dn_layer.dtype, layout=ttnn.TILE_LAYOUT, device=dn_layer.device)

    with tc.section("G1_linear_out"):
        output = ttnn.linear(out_tt, dn_layer.out_proj_w)

    return output


def main():
    num_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    num_decode = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    warmup = 3

    config = Qwen3CoderNextConfig(num_hidden_layers=num_layers)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    print(f"[Config] Layers: {num_layers}, Decode tokens: {num_decode}")
    print("[Weights] Loading...")
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

        # Prefill + warmup
        prompt = "What is the meaning of life?"
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
        )
        generator.reset()
        logits = generator.prefill(input_ids)
        logits_cpu = ttnn.to_torch(logits).float().reshape(-1)
        next_token = torch.argmax(logits_cpu[:config.vocab_size]).item()

        for _ in range(warmup):
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            _, next_token_tensor = generator.decode_one_token(token_tensor)
            next_token = next_token_tensor.item()

        # Profiled decode
        tc = TimingCollector()
        for i in range(num_decode):
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            hidden = model.embed(token_tensor)
            cos, sin = model.get_rope(generator.position)
            kv_caches = dict(generator.kv_caches)

            for layer_idx, layer in enumerate(model.layers):
                layer_type = config.layer_types[layer_idx]
                kv_cache = kv_caches.get(layer_idx) if layer_type == "full_attention" else None

                normed = layer.input_layernorm(hidden)

                if layer_type == "linear_attention":
                    mixer_out = profile_deltanet_decode_step(
                        layer.token_mixer, normed, generator.deltanet_state, tc
                    )
                    new_kv = None
                else:
                    mixer_out, new_kv = layer.token_mixer(normed, cos, sin, kv_cache)

                hidden = ttnn.add(hidden, mixer_out)
                normed2 = layer.post_attention_layernorm(hidden)
                mlp_out = layer.mlp(normed2)
                hidden = ttnn.add(hidden, mlp_out)

                if new_kv is not None:
                    kv_caches[layer_idx] = new_kv

            hidden = model.rms_norm(hidden, model.final_norm_weight)
            logits = ttnn.linear(hidden, model.lm_head_w)

            generator.kv_caches = kv_caches
            generator.position += 1
            logits_cpu = ttnn.to_torch(logits).float().reshape(-1)
            next_token = torch.argmax(logits_cpu[:config.vocab_size]).item()

        tc.report(f"DeltaNet Decode Step Internals ({num_decode} tokens × {sum(1 for t in config.layer_types if t == 'linear_attention')} DeltaNet layers)")

        # Summary groups
        records = tc.records
        groups = {
            "Device linears (A1-A4, G1)": ["A1_linear_qkv", "A2_linear_z", "A3_linear_b", "A4_linear_a", "G1_linear_out"],
            "to_torch (B1, C1-C2, E1, F1)": ["B1_to_torch_qkv", "C1_to_torch_b", "C2_to_torch_a", "E1_to_torch_state", "F1_to_torch_z"],
            "from_torch (E3, F3)": ["E3_set_state", "F3_from_torch_output"],
            "CPU compute (B2-B3, C3, D1-D2, E2, F2)": ["B2_conv1d", "B3_split_reshape", "C3_decay_beta", "D1_head_expand", "D2_l2norm_scale", "E2_recurrence", "F2_norm_gate"],
        }
        print(f"\n{'='*70}")
        print(f"  Summary by Category")
        print(f"{'='*70}")
        grand_total = sum(sum(t) for t in records.values()) * 1000
        for group_name, keys in groups.items():
            group_total = sum(sum(records[k]) for k in keys if k in records) * 1000
            pct = group_total / grand_total * 100 if grand_total > 0 else 0
            print(f"  {group_name:<45} {group_total:>8.1f} ms ({pct:>5.1f}%)")
        print(f"  {'GRAND TOTAL':<45} {grand_total:>8.1f} ms")
        print(f"{'='*70}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
