# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Layer-only HuggingFace reference for Qwen3-Coder-30B-A3B-Instruct.

Loading the full 30.5B causal LM to test one decoder layer wastes ~57GB of host
RAM and several minutes, so this reads just the tensors for a single layer
straight out of the safetensors shards and populates one ``Qwen3MoeDecoderLayer``.

Checkpoint-vs-module weight layout
----------------------------------
The checkpoint stores experts as 3 separate tensors per expert::

    model.layers.L.mlp.experts.E.gate_proj.weight   [moe_inter, hidden]
    model.layers.L.mlp.experts.E.up_proj.weight     [moe_inter, hidden]
    model.layers.L.mlp.experts.E.down_proj.weight   [hidden, moe_inter]

``Qwen3MoeExperts`` instead holds them batched and gate/up fused::

    gate_up_proj  [num_experts, 2 * moe_inter, hidden]   # [gate ; up] along dim 0
    down_proj     [num_experts, hidden, moe_inter]

The fusion order matters: ``Qwen3MoeExperts.forward`` does
``linear(x, gate_up_proj[e]).chunk(2, dim=-1)``, so the FIRST half is gate and
the second is up. Concatenating in the other order silently swaps them and the
layer still runs -- it just produces wrong numbers.
"""

from __future__ import annotations

import json
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding

HF_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def load_config(hf_model: str = HF_MODEL):
    return AutoConfig.from_pretrained(hf_model)


def layer_state_dict(layer_idx: int = 0, hf_model: str = HF_MODEL) -> dict[str, torch.Tensor]:
    """Read only ``model.layers.<layer_idx>.*`` from the shards that hold them."""
    index = json.load(open(hf_hub_download(hf_model, "model.safetensors.index.json")))["weight_map"]
    prefix = f"model.layers.{layer_idx}."

    per_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in index.items():
        if name.startswith(prefix):
            per_shard[shard].append(name)
    if not per_shard:
        raise KeyError(f"no tensors found for layer {layer_idx}")

    out: dict[str, torch.Tensor] = {}
    for shard, names in per_shard.items():
        path = hf_hub_download(hf_model, shard)
        with safe_open(path, framework="pt") as f:
            for name in names:
                out[name[len(prefix) :]] = f.get_tensor(name)
    return out


def build_reference_layer(layer_idx: int = 0, hf_model: str = HF_MODEL):
    """Return ``(layer, config)`` with real checkpoint weights, in eval mode."""
    config = load_config(hf_model)
    sd = layer_state_dict(layer_idx, hf_model)

    with torch.device("meta"):
        layer = Qwen3MoeDecoderLayer(config, layer_idx)
    layer.to_empty(device="cpu")

    direct = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate.weight",
    ]
    params = dict(layer.named_parameters())
    for key in direct:
        params[key].data.copy_(sd[key])

    # Fuse + stack the experts. gate first, then up -- see module docstring.
    n_experts = config.num_experts
    gate_up = torch.stack(
        [
            torch.cat([sd[f"mlp.experts.{e}.gate_proj.weight"], sd[f"mlp.experts.{e}.up_proj.weight"]], dim=0)
            for e in range(n_experts)
        ]
    )
    down = torch.stack([sd[f"mlp.experts.{e}.down_proj.weight"] for e in range(n_experts)])
    params["mlp.experts.gate_up_proj"].data.copy_(gate_up)
    params["mlp.experts.down_proj"].data.copy_(down)

    return layer.eval(), config


def rotary_embeddings(config, seq_len: int, device="cpu"):
    """Return the ``(cos, sin)`` pair the decoder layer expects."""
    rope = Qwen3MoeRotaryEmbedding(config=config, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy = torch.zeros(1, seq_len, config.hidden_size, dtype=torch.float32, device=device)
    return rope(dummy, position_ids)


def weight_stats(sd: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Per-tensor name/shape/dtype/mean/std, for deterministic synthetic weights."""
    stats = {}
    for name, t in sd.items():
        f = t.float()
        stats[name] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "mean": f.mean().item(),
            "std": f.std().item(),
        }
    return stats


if __name__ == "__main__":
    torch.manual_seed(0)

    layer, config = build_reference_layer(0)
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"layer 0 built: {n_params/1e9:.2f}B params, dtype={next(layer.parameters()).dtype}")

    seq_len = 32
    hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.float32) * 0.02
    cos, sin = rotary_embeddings(config, seq_len)

    with torch.no_grad():
        out = layer(hidden, position_embeddings=(cos, sin), attention_mask=None)
    out = out[0] if isinstance(out, tuple) else out

    print(f"forward OK: {tuple(hidden.shape)} -> {tuple(out.shape)}")
    print(f"  out mean={out.mean():.6f} std={out.std():.6f} finite={torch.isfinite(out).all().item()}")
