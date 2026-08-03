# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 hooks for advisor-challenger's fixed timing harness."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
TEMPLATE = ROOT / ".agents/skills/advisor-challenger/scripts/harness_template.py"
spec = importlib.util.spec_from_file_location("advisor_challenger_harness_template", TEMPLATE)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def _config():
    layer_types = ["sliding_attention"] * 30
    for index in (5, 11, 17, 23, 29):
        layer_types[index] = "full_attention"
    return SimpleNamespace(
        hidden_size=2816,
        intermediate_size=2112,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        num_hidden_layers=30,
        layer_types=layer_types,
        sliding_window=1024,
        rms_norm_eps=1e-6,
        num_experts=128,
        top_k_experts=8,
        moe_intermediate_size=704,
        enable_moe_block=True,
        hidden_size_per_layer_input=0,
        hidden_activation="gelu_pytorch_tanh",
        attention_k_eq_v=True,
    )


def _decode_state(device, policy):
    import ttnn
    from models.autoports.google_gemma_4_26b_a4b_it.tests.synthetic_weights import synthetic_layer_state_dict
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

    kind = os.environ["CHALLENGER_LAYER_KIND"]
    layer_idx = {"sliding_attention": 0, "full_attention": 5}[kind]
    cfg = _config()
    assert cfg.layer_types[layer_idx] == kind
    dtype = {"BFLOAT16": ttnn.bfloat16, "BFLOAT8_B": ttnn.bfloat8_b}
    fidelity = {
        "HiFi4": ttnn.MathFidelity.HiFi4,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "LoFi": ttnn.MathFidelity.LoFi,
    }
    for role, block_w in policy["dram_in0_block_w_by_role"].items():
        key = f"GEMMA4_OPT_DRAM_BLOCK_W_{role.upper()}"
        if block_w is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(block_w)
    decoder = OptimizedDecoder.from_state_dict(
        synthetic_layer_state_dict(layer_idx),
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=device,
        weight_dtype=dtype[policy["weight_dtype"]],
        activation_dtype=dtype[policy["activation_dtype"]],
        attention_weight_dtype=dtype[policy["attention_weight_dtype_by_layer_kind"][kind]],
        mlp_weight_dtype=dtype[policy["mlp_weight_dtype"]],
        mlp_down_weight_dtype=dtype[policy["mlp_down_weight_dtype"]],
        prefill_expert_weight_dtype=dtype[policy["prefill_expert_weight_dtype"]],
        expert_weight_dtype=dtype[policy["expert_weight_dtype"]],
        attention_math_fidelity=fidelity[policy["attention_math_fidelity"]],
        full_attention_math_fidelity=fidelity[policy["full_attention_math_fidelity"]],
        mlp_math_fidelity=fidelity[policy["mlp_math_fidelity"]],
        expert_gate_math_fidelity=fidelity[policy["expert_gate_math_fidelity"]],
        expert_math_fidelity=fidelity[policy["expert_math_fidelity"]],
        packed_dense_gate_up=policy["packed_dense_gate_up"],
        dram_in0_block_w=policy["dram_in0_block_w"],
        dram_sharded_roles=tuple(policy["dram_sharded_roles_by_layer_kind"][kind]),
        expert_decode_input_l1=policy["expert_decode_input_l1"],
        expert_gate_in0_block_w=policy["expert_gate_in0_block_w"],
        expert_down_in0_block_w=policy["expert_down_in0_block_w"],
        expert_gate_per_core_n=policy["expert_gate_per_core_n"],
        expert_down_per_core_n=policy["expert_down_per_core_n"],
    )

    batch = harness.BATCH
    current_pos = 1024
    generator = torch.Generator().manual_seed(20260803 + layer_idx)
    hidden = torch.randn(batch, 1, 2816, generator=generator, dtype=torch.bfloat16)
    rotary_width = 256 if kind == "sliding_attention" else 512
    cos = torch.randn(batch, 1, rotary_width, generator=generator, dtype=torch.bfloat16)
    sin = torch.randn(batch, 1, rotary_width, generator=generator, dtype=torch.bfloat16)
    if kind == "sliding_attention":
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        block_size, num_heads, head_dim = 64, 8, 256
    else:
        cos, sin = cos.transpose(0, 1).unsqueeze(0), sin.transpose(0, 1).unsqueeze(0)
        block_size, num_heads, head_dim = 128, 2, 512
    blocks = (current_pos + 1 + block_size - 1) // block_size

    def as_tt(value, *, out_dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.as_tensor(
            value,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            dtype=out_dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    cache_shape = (batch * blocks, num_heads, block_size, head_dim)
    kwargs = {
        "position_cos": as_tt(cos),
        "position_sin": as_tt(sin),
        "current_pos": as_tt(
            torch.full((batch,), current_pos, dtype=torch.int32),
            out_dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": as_tt(
            torch.arange(batch * blocks, dtype=torch.int32).view(batch, blocks),
            out_dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "kv_cache": (
            as_tt(torch.zeros(cache_shape, dtype=torch.bfloat16)),
            as_tt(torch.zeros(cache_shape, dtype=torch.bfloat16)),
        ),
    }
    return decoder, as_tt(hidden.transpose(0, 1).unsqueeze(0)), kwargs


def build(device, policy):
    return _decode_state(device, policy)


def decode(state):
    decoder, hidden, kwargs = state
    if os.environ.get("CHALLENGER_SIGNPOST_EAGER") == "1" and not getattr(decode, "profiled", False):
        from tracy import signpost

        decode.profiled = True
        signpost(header="PERF_DECODE")
        output = decoder.decode_forward(hidden_states=hidden, cache_position_modulo=None, **kwargs)
        import ttnn

        ttnn.synchronize_device(decoder.mesh_device)
        signpost(header="PERF_DECODE_END")
        return output
    return decoder.decode_forward(hidden_states=hidden, cache_position_modulo=None, **kwargs)


harness.build = build
harness.decode = decode

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="incumbent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy")
    args = parser.parse_args()
    default_policy = ROOT / f"models/autoports/{harness.MODEL_DIR}/doc/advisor_challenger/incumbent.json"
    if args.label == "incumbent" and not args.policy:
        raise SystemExit("--policy is required for the incumbent")
    harness.measure(args.label, args.out, args.policy or str(default_policy))
