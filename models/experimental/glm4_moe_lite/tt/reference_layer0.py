# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMLP, Glm4MoeRMSNorm, Glm4MoeRotaryEmbedding

from models.demos.glm4_moe_lite.tt.weights import LazyStateDict, load_glm_lazy_state_dict


@dataclass(frozen=True)
class Layer0RefOutputs:
    input_ids: torch.Tensor  # [B, S]
    x_embed: torch.Tensor  # [B, S, D]
    x_attn_out: torch.Tensor  # [B, S, D] (post-attn residual)
    x_mlp_out: torch.Tensor  # [B, S, D] (post-mlp residual / layer output)


def _load_config_json(snapshot_dir: Path) -> dict:
    config_path = Path(snapshot_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    return json.loads(config_path.read_text())


def _build_minimal_config(snapshot_dir: Path):
    """
    Build a minimal config object sufficient for using HF attention/MLP/RMSNorm modules.

    We cannot rely on `AutoConfig` inside the vLLM dev container because its Transformers
    version may not yet recognize `model_type=glm4_moe_lite`.
    """
    c = _load_config_json(snapshot_dir)

    qk_nope_head_dim = int(c["qk_nope_head_dim"])
    qk_rope_head_dim = int(c["qk_rope_head_dim"])
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # DeepseekV3Attention expects `rope_parameters` (dict-like) for scaling adjustments.
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": float(c.get("rope_theta", 10000.0)),
        "partial_rotary_factor": float(c.get("partial_rotary_factor", 1.0)),
    }

    # Glm4MoeRotaryEmbedding expects `rope_scaling` and reads `rope_theta` + `partial_rotary_factor`
    # directly from the config. The GLM-4.7-Flash config sets rope_scaling to null.
    return SimpleNamespace(
        # Common model dims
        hidden_size=int(c["hidden_size"]),
        intermediate_size=int(c["intermediate_size"]),
        num_hidden_layers=int(c["num_hidden_layers"]),
        vocab_size=int(c["vocab_size"]),
        hidden_act=str(c.get("hidden_act", "silu")),
        rms_norm_eps=float(c.get("rms_norm_eps", 1e-5)),
        max_position_embeddings=int(c.get("max_position_embeddings", 2048)),
        # Attention dims
        num_attention_heads=int(c["num_attention_heads"]),
        num_key_value_heads=int(c.get("num_key_value_heads", c["num_attention_heads"])),
        q_lora_rank=int(c["q_lora_rank"]),
        kv_lora_rank=int(c["kv_lora_rank"]),
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_head_dim,
        v_head_dim=int(c["v_head_dim"]),
        # Rotary
        rope_theta=float(c["rope_theta"]),
        partial_rotary_factor=float(c.get("partial_rotary_factor", 1.0)),
        head_dim=qk_rope_head_dim,  # RoPE applies to the rotary slice only (64)
        rope_interleave=bool(c.get("rope_interleave", True)),
        rope_scaling=c.get("rope_scaling", None),
        rope_parameters=rope_parameters,
        # Attention behavior
        attention_bias=bool(c.get("attention_bias", False)),
        attention_dropout=float(c.get("attention_dropout", 0.0)),
        _attn_implementation="eager",
    )


class _Layer0Ref(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_layernorm = Glm4MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DeepseekV3Attention(config, layer_idx=0)
        self.post_attention_layernorm = Glm4MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Glm4MoeMLP(config)


def _build_causal_mask(batch: int, seq_len: int, *, device: torch.device) -> torch.Tensor:
    # Shape expected by eager_attention_forward: [B, 1, S, S]
    # 0 where allowed; -inf where masked.
    mask = torch.full((batch, 1, seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1)
    return mask


def _load_layer0_state(state: LazyStateDict) -> dict[str, torch.Tensor]:
    prefix = "model.layers.0."
    out: dict[str, torch.Tensor] = {}
    for full_key in state.keys():
        if not full_key.startswith(prefix):
            continue
        out[full_key[len(prefix) :]] = state[full_key]
    return out


@torch.no_grad()
def run_layer0_reference_from_input_ids(
    snapshot_dir: Path,
    input_ids: torch.Tensor,
    *,
    device: torch.device = torch.device("cpu"),
) -> Layer0RefOutputs:
    """CPU reference for GLM-4.7-Flash layer 0 given explicit input ids."""
    snapshot_dir = Path(snapshot_dir)

    config = _build_minimal_config(snapshot_dir)

    if input_ids.ndim != 2:
        raise ValueError(f"expected input_ids [B,S], got shape={tuple(input_ids.shape)}")
    input_ids = input_ids.to(device=device, dtype=torch.long)
    batch, seq_len = input_ids.shape

    state = load_glm_lazy_state_dict(snapshot_dir, num_layers=getattr(config, "num_hidden_layers", None))

    embed_w = state["model.embed_tokens.weight"].to(device=device)
    x_embed = torch.nn.functional.embedding(input_ids, embed_w)

    layer0 = _Layer0Ref(config).to(device=device)
    layer0.load_state_dict(_load_layer0_state(state), strict=True)
    layer0.eval()

    rotary = Glm4MoeRotaryEmbedding(config=config).to(device=device)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    cos, sin = rotary(x_embed, position_ids)

    attn_mask = _build_causal_mask(batch, seq_len, device=device)

    # Manually run to capture intermediates.
    residual = x_embed
    x = layer0.input_layernorm(x_embed)
    attn_out, _ = layer0.self_attn(
        hidden_states=x,
        attention_mask=attn_mask,
        position_embeddings=(cos, sin),
        past_key_values=None,
        cache_position=None,
    )
    x_attn_out = residual + attn_out

    residual = x_attn_out
    x = layer0.post_attention_layernorm(x_attn_out)
    x = layer0.mlp(x)
    x_mlp_out = residual + x

    return Layer0RefOutputs(
        input_ids=input_ids.cpu(),
        x_embed=x_embed.cpu(),
        x_attn_out=x_attn_out.cpu(),
        x_mlp_out=x_mlp_out.cpu(),
    )


@torch.no_grad()
def run_layer0_reference(
    snapshot_dir: Path, prompt: str, *, device: torch.device = torch.device("cpu")
) -> Layer0RefOutputs:
    """
    CPU reference for GLM-4.7-Flash layer 0 (embedding + decoder layer 0).

    Notes:
    - Uses HuggingFace Transformers modules for the layer math (DeepseekV3Attention + GLM MLP).
    - Loads only the weights needed for layer 0 from safetensors via LazyStateDict.
    - This is intended as a correctness oracle for TT implementations.
    """
    snapshot_dir = Path(snapshot_dir)
    tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True, use_fast=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    return run_layer0_reference_from_input_ids(snapshot_dir, enc["input_ids"], device=device)
