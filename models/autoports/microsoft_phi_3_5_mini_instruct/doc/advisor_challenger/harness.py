#!/usr/bin/env python3
"""Phi-3.5 hooks for the mandated advisor-challenger timing template."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[5]
TEMPLATE = ROOT / ".agents/skills/advisor-challenger/scripts/harness_template.py"
spec = importlib.util.spec_from_file_location("advisor_challenger_harness_template", TEMPLATE)
template = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(template)


def _config():
    from types import SimpleNamespace

    return SimpleNamespace(
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=96,
        num_hidden_layers=32,
        max_position_embeddings=64,
        original_max_position_embeddings=4096,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        attention_bias=False,
        rope_scaling={"short_factor": [1.0] * 48, "long_factor": [1.0] * 48},
    )


def _state_dict(seed=23):
    generator = torch.Generator().manual_seed(seed)
    shapes = {
        "self_attn.qkv_proj.weight": (9216, 3072),
        "self_attn.o_proj.weight": (3072, 3072),
        "mlp.gate_up_proj.weight": (16384, 3072),
        "mlp.down_proj.weight": (3072, 8192),
        "input_layernorm.weight": (3072,),
        "post_attention_layernorm.weight": (3072,),
    }
    return {name: torch.randn(shape, generator=generator) * 0.01 for name, shape in shapes.items()}


def build(device, policy):
    import ttnn
    from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder

    batch, context = template.BATCH, 32
    cfg = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _state_dict(), hf_config=cfg, layer_idx=0, mesh_device=device,
        max_position_embeddings=64, batch=batch, **policy
    )
    kv_cache = OptimizedDecoder.allocate_paged_kv_cache(
        hf_config=cfg, mesh_device=device, max_batch_size=batch, max_seq_len=64, block_size=32
    )
    page_table = ttnn.Tensor(torch.arange(batch * 2, dtype=torch.int32).reshape(batch, 2), ttnn.int32).to(device)
    current_host = torch.full((batch,), context, dtype=torch.int32)
    current_pos = ttnn.Tensor(current_host, ttnn.int32).to(device)
    position_ids = ttnn.Tensor(current_host.to(torch.uint32), ttnn.uint32).to(device)
    hidden = ttnn.Tensor(
        torch.randn(1, 1, batch, cfg.hidden_size, generator=torch.Generator().manual_seed(29)).to(torch.bfloat16),
        ttnn.bfloat16,
    ).to(ttnn.TILE_LAYOUT).to(device)
    return decoder, hidden, {
        "current_pos": current_pos, "position_ids": position_ids, "page_table": page_table,
        "kv_cache": kv_cache, "rope_sequence_length": context + 1,
    }


def decode(state):
    decoder, hidden, kwargs = state
    return decoder.decode_forward(hidden, **kwargs)


template.build = build
template.decode = decode


if __name__ == "__main__":
    template.measure = template.measure
    args = template.argparse.ArgumentParser()
    args.add_argument("--label", default="incumbent")
    args.add_argument("--out", required=True)
    args.add_argument("--policy", default=None)
    parsed = args.parse_args()
    default_policy = f"models/autoports/{template.MODEL_DIR}/doc/advisor_challenger/incumbent.json"
    if parsed.label == "incumbent" and not parsed.policy:
        raise SystemExit("--policy is required for the incumbent run")
    template.measure(parsed.label, parsed.out, parsed.policy or default_policy)
