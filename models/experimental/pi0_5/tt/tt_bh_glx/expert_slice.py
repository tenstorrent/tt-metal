# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AdaRMS Gemma-300M expert blocks on a mesh submesh.

A chunk owns a contiguous range of expert layers + their weights, fused adaRMS
modulation Dense, and RoPE tables. In the 1×8 pipeline a single chunk covers all
18 layers, replicated across the mesh. The block forward path mirrors the
single-chip Pi0_5PaliGemmaBackboneTTNN.forward_expert iteration body.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.tt.ttnn_common import get_ln_weight_memory_config, tensor_1d_to_2d_ttnn
from models.experimental.pi0_5.tt.ttnn_gemma import AdaRMSGemmaBlockTTNN, precompute_freqs_cis_meta_format


def _load_block_weights_to_submesh(
    weights: Dict[str, torch.Tensor],
    layer_idx: int,
    submesh,
) -> Dict[str, "ttnn.Tensor"]:
    """Convert one layer's torch weights to TTNN on the given submesh.

    Shared by VLM and expert blocks — both use the same fused-QKV / +1 norm /
    bf8 weight convention (mirrors Pi0_5PaliGemmaBackboneTTNN._get_vlm_block_weights_ttnn
    but takes the destination submesh explicitly). The expert additionally fuses
    the AdaRMS modulation Dense via _inject_adarms_weights_to_submesh below.
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


def _inject_adarms_weights_to_submesh(
    block_weights: Dict[str, "ttnn.Tensor"],
    expert_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    submesh,
) -> None:
    """Concatenate pre-attn + pre-FFW modulation Denses into one (6*W, W) fused linear.

    Same fusion as Pi0_5PaliGemmaBackboneTTNN._inject_adarms_weights but uploads
    to the provided submesh.
    """
    prefix = f"model.layers.{layer_idx}."
    names = ("input_layernorm.dense", "post_attention_layernorm.dense")
    w_keys = [f"{prefix}{n}.weight" for n in names]
    b_keys = [f"{prefix}{n}.bias" for n in names]
    for wk in w_keys:
        if wk not in expert_weights:
            raise KeyError(f"PI0.5 expects adaRMS weight '{wk}' in the action_expert checkpoint.")
    fused_w = torch.cat([expert_weights[wk] for wk in w_keys], dim=0).contiguous()
    block_weights["adarms_mod.weight"] = ttnn.from_torch(
        fused_w.T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh
    )
    biases = [expert_weights[bk] for bk in b_keys if bk in expert_weights]
    if biases:
        assert len(biases) == 2, "expected biases for both modulation Denses or neither"
        fused_b = torch.cat(biases, dim=0).contiguous()
        block_weights["adarms_mod.bias"] = tensor_1d_to_2d_ttnn(fused_b, submesh, dtype=ttnn.bfloat16)


class ExpertChunkSlice:
    """A contiguous range of AdaRMS expert blocks on a single-chip submesh."""

    def __init__(
        self,
        expert_config: GemmaConfig,
        expert_weights: Dict[str, torch.Tensor],
        submesh,
        layer_range: Tuple[int, int],
        max_seq_len: int,
    ):
        self.config = expert_config
        self.submesh = submesh
        self.layer_range = layer_range
        lo, hi = layer_range
        self.cos_meta, self.sin_meta = precompute_freqs_cis_meta_format(expert_config.head_dim, max_seq_len, submesh)
        self.blocks: List[AdaRMSGemmaBlockTTNN] = []
        for i in range(lo, hi):
            block_w = _load_block_weights_to_submesh(expert_weights, i, submesh)
            _inject_adarms_weights_to_submesh(block_w, expert_weights, i, submesh)
            self.blocks.append(AdaRMSGemmaBlockTTNN(expert_config, block_w, i, submesh, self.cos_meta, self.sin_meta))

    def forward(
        self,
        hidden: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        prefix_kv_for_chunk: List[Tuple["ttnn.Tensor", "ttnn.Tensor"]],
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
        precomputed_mods: Optional[List[Tuple["ttnn.Tensor", ...]]] = None,
    ) -> "ttnn.Tensor":
        """Run this chip's expert blocks. prefix_kv_for_chunk has one (K, V) per local layer.

        cos_override / sin_override: per-chip position-aware suffix RoPE tables
        for the upstream-openpi compat path (PI0_UPSTREAM_MASKS=1). When None,
        each block uses its own sequential cos_meta / sin_meta.

        precomputed_mods (optional): list of (sa1, ta, ga, sf1, tf, gf) tuples,
        one per local layer. When provided, each block bypasses its per-step
        mod-Dense matmul + split — see AdaRMSGemmaBlockTTNN.forward's fast path.
        Used by the 1×8 pipeline's TIER A precompute (pipeline_1x8.py); callers
        that don't precompute pass None and fall through to the on-device mod-Dense.
        """
        if len(prefix_kv_for_chunk) != len(self.blocks):
            raise RuntimeError(f"expected {len(self.blocks)} prefix KV tuples, got {len(prefix_kv_for_chunk)}")
        if precomputed_mods is not None and len(precomputed_mods) != len(self.blocks):
            raise RuntimeError(f"expected {len(self.blocks)} precomputed mod tuples, got {len(precomputed_mods)}")
        for idx, (block, past_kv) in enumerate(zip(self.blocks, prefix_kv_for_chunk)):
            pm = precomputed_mods[idx] if precomputed_mods is not None else None
            hidden, _ = block.forward(
                hidden,
                cos_override,
                sin_override,
                adarms_cond,
                attention_mask,
                position_ids,
                past_kv,
                False,  # use_cache=False on expert path
                precomputed_mod=pm,
            )
        return hidden
