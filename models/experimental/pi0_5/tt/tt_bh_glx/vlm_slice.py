# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-chip VLM block slice (one Gemma-2B block per chip).

Each VLMBlockSlice owns:
  - its own RoPE cos/sin tables on its 1x1 submesh
  - per-layer TTNN weights uploaded to its submesh
  - one GemmaBlockTTNN instance

Forward returns (hidden_out, (K, V)) — KV is left on this chip for later
layer-paired migration to the denoise stage.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.tt.ttnn_common import (
    get_ln_weight_memory_config,
    tensor_1d_to_2d_ttnn,
)
from models.experimental.pi0_5.tt.ttnn_gemma import GemmaBlockTTNN, precompute_freqs_cis_meta_format


def _load_block_weights_to_submesh(
    weights: Dict[str, torch.Tensor],
    layer_idx: int,
    submesh,
) -> Dict[str, "ttnn.Tensor"]:
    """Convert one layer's torch weights to TTNN on the given submesh.

    Mirrors Pi0_5PaliGemmaBackboneTTNN._get_vlm_block_weights_ttnn but takes
    the destination submesh explicitly instead of using a backbone-owned device.
    Behaviors copied from that helper:
      - Q/K/V projections fused via on-device concat into self_attn.wqkv (bf8_b).
      - Norm weights get pre-added +1.0 (Gemma offset) on host before upload.
      - o_proj + MLP weights upload as bf8_b; norm γ as bf16 (small + precision-sensitive).
    """
    prefix = f"model.layers.{layer_idx}."
    block_weights: Dict[str, "ttnn.Tensor"] = {}

    q_key = f"{prefix}self_attn.q_proj.weight"
    k_key = f"{prefix}self_attn.k_proj.weight"
    v_key = f"{prefix}self_attn.v_proj.weight"
    if q_key in weights and k_key in weights and v_key in weights:
        wq = ttnn.from_torch(
            weights[q_key].T.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=submesh
        )
        wk = ttnn.from_torch(
            weights[k_key].T.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=submesh
        )
        wv = ttnn.from_torch(
            weights[v_key].T.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=submesh
        )
        block_weights["self_attn.wqkv"] = ttnn.concat([wq, wk, wv], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(wq)
        ttnn.deallocate(wk)
        ttnn.deallocate(wv)

    for key, value in weights.items():
        if not key.startswith(prefix):
            continue
        new_key = key[len(prefix) :]
        if new_key in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            continue
        is_norm = "layernorm" in new_key or "norm" in new_key
        if "weight" in new_key and not is_norm:
            value = value.T
        elif is_norm:
            value = value + 1.0

        if len(value.shape) == 1:
            mc = get_ln_weight_memory_config() if is_norm else ttnn.DRAM_MEMORY_CONFIG
            block_weights[new_key] = tensor_1d_to_2d_ttnn(value, submesh, dtype=ttnn.bfloat16, memory_config=mc)
        else:
            weight_dtype = ttnn.bfloat16 if is_norm else ttnn.bfloat8_b
            block_weights[new_key] = ttnn.from_torch(value, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=submesh)

    return block_weights


class VLMBlockSlice:
    """One Gemma-2B transformer block on a single-chip submesh."""

    def __init__(
        self,
        vlm_config: GemmaConfig,
        vlm_weights: Dict[str, torch.Tensor],
        submesh,
        layer_idx: int,
        max_seq_len: int,
    ):
        self.config = vlm_config
        self.submesh = submesh
        self.layer_idx = layer_idx
        self.cos_meta, self.sin_meta = precompute_freqs_cis_meta_format(vlm_config.head_dim, max_seq_len, submesh)
        block_w = _load_block_weights_to_submesh(vlm_weights, layer_idx, submesh)
        self.block = GemmaBlockTTNN(vlm_config, block_w, layer_idx, submesh, self.cos_meta, self.sin_meta)

    def forward(
        self,
        hidden: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
    ) -> Tuple["ttnn.Tensor", Tuple["ttnn.Tensor", "ttnn.Tensor"]]:
        # cos_override / sin_override: position-aware RoPE tables shaped
        # [1, 1, prefix_padded, head_dim] for the upstream-openpi compat path
        # (PI0_UPSTREAM_MASKS=1). When None, the block falls back to its own
        # sequential cos_meta / sin_meta.
        hidden, new_kv = self.block.forward(
            hidden,
            cos_override,
            sin_override,
            attention_mask,
            position_ids,
            None,  # no past KV — this IS the prefill
            True,  # use_cache=True so we get (K, V) back
        )
        return hidden, new_kv
