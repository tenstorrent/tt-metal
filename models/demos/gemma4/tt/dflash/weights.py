# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load all real Gemma4-31B DFlash drafter weights (z-lab/gemma-4-31B-it-DFlash)
onto ttnn tensors, TP-sharded per models/demos/gemma4/docs/dflash_design.md's
"shard, don't replicate" decision.

Reuses this repo's existing, generic loaders where the drafter's architecture
matches them exactly (GQA attention with sliding-window support; RMSNorm), and
only adds new code where it genuinely doesn't (the SiLU/SwiGLU MLP -- Gemma4's
own SharedMLP is GeGLU, the wrong activation for this Qwen3-style drafter).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import ttnn
from models.demos.gemma4.tt.attention.weights import AttentionWeights, load_attention_weights
from models.demos.gemma4.tt.dflash.config import Gemma4DFlashDrafterConfig
from models.demos.gemma4.tt.dflash.mlp import DFlashMLPWeights, load_dflash_mlp_weights
from models.demos.gemma4.tt.dflash.weight_mapping import (
    DEFAULT_DFLASH_MODEL,
    fc_state_dict,
    layer_attention_state_dict,
    layer_mlp_state_dict,
    layer_norm_state_dict,
    load_dflash_flat_state_dict,
    top_level_norm_state_dict,
)
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.utils.general_utils import get_cache_file_name


@dataclass(frozen=True)
class DFlashLayerWeights:
    attn: AttentionWeights
    mlp: DFlashMLPWeights
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm


@dataclass(frozen=True)
class Gemma4DFlashWeights:
    fc: ttnn.Tensor  # replicated [1,1,32256,5376] -- feeds every layer's k_ctx/v_ctx projection
    hidden_norm: RMSNorm
    norm: RMSNorm
    layers: list[DFlashLayerWeights]


def _attn_config(config: Gemma4DFlashDrafterConfig) -> SimpleNamespace:
    """Lightweight config exposing exactly what load_attention_weights reads --
    it does not need a full Gemma4TextConfig, and the drafter has no K=V tying
    or MoE (unlike the target's own attention config)."""
    return SimpleNamespace(
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        hidden_size=config.hidden_size,
        use_kv_tying=False,
        enable_moe_block=False,
    )


def _norm_config(config: Gemma4DFlashDrafterConfig) -> SimpleNamespace:
    """RMSNorm only reads .rms_norm_eps from this."""
    return SimpleNamespace(rms_norm_eps=config.rms_norm_eps)


def load_gemma4_dflash_weights(
    mesh_device,
    config: Gemma4DFlashDrafterConfig,
    mesh_config,
    model_path: str = DEFAULT_DFLASH_MODEL,
    attn_dtype=ttnn.bfloat16,
    mlp_dtype=ttnn.bfloat16,
    tensor_cache_path=None,
) -> Gemma4DFlashWeights:
    flat = load_dflash_flat_state_dict(model_path)
    attn_cfg = _attn_config(config)
    norm_cfg = _norm_config(config)

    def cache(name):
        return get_cache_file_name(tensor_cache_path, name)

    # fc: small (5376x32256), replicated on every device -- its output feeds
    # every layer's k_proj/v_proj (column-parallel, which need a full-width
    # input), so it must not be TP-sharded itself.
    fc_w = fc_state_dict(flat)["weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)  # [1,1,32256,5376]
    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    fc_tt = ttnn.as_tensor(
        fc_w,
        device=mesh_device,
        dtype=attn_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate,
        cache_file_name=cache("fc.weight"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    hidden_norm = RMSNorm(
        mesh_device,
        norm_cfg,
        top_level_norm_state_dict(flat, "hidden_norm"),
        tensor_cache_path=cache("hidden_norm"),
        mesh_config=mesh_config,
    )
    norm = RMSNorm(
        mesh_device,
        norm_cfg,
        top_level_norm_state_dict(flat, "norm"),
        tensor_cache_path=cache("norm"),
        mesh_config=mesh_config,
    )

    layers = []
    for i in range(config.num_hidden_layers):
        attn_w = load_attention_weights(
            mesh_device,
            attn_cfg,
            layer_attention_state_dict(flat, i),
            mesh_config,
            weight_dtype=attn_dtype,
            tensor_cache_path=cache(f"layers.{i}.self_attn"),
        )
        mlp_w = load_dflash_mlp_weights(
            mesh_device,
            config,
            layer_mlp_state_dict(flat, i),
            mesh_config,
            weight_dtype=mlp_dtype,
            tensor_cache_path=cache(f"layers.{i}.mlp"),
        )
        ln1 = RMSNorm(
            mesh_device,
            norm_cfg,
            layer_norm_state_dict(flat, i, "input_layernorm"),
            tensor_cache_path=cache(f"layers.{i}.input_layernorm"),
            mesh_config=mesh_config,
        )
        ln2 = RMSNorm(
            mesh_device,
            norm_cfg,
            layer_norm_state_dict(flat, i, "post_attention_layernorm"),
            tensor_cache_path=cache(f"layers.{i}.post_attention_layernorm"),
            mesh_config=mesh_config,
        )
        layers.append(DFlashLayerWeights(attn=attn_w, mlp=mlp_w, input_layernorm=ln1, post_attention_layernorm=ln2))

    return Gemma4DFlashWeights(fc=fc_tt, hidden_norm=hidden_norm, norm=norm, layers=layers)
