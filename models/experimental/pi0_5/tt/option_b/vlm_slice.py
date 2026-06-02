# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Submesh-aware VLM transformer slice for Option B.

Wraps a contiguous range of VLM (Gemma-2B PaliGemma) transformer layers onto a
single 4x2 submesh of a Blackhole Galaxy. The slice is the building block used
by stages 1 and 2:

    stage 1: layers 0..9   (Pi0_5SubmeshVLMSlice(submesh_1, layer_range=(0,9)))
    stage 2: layers 9..18  (Pi0_5SubmeshVLMSlice(submesh_2, layer_range=(9,18),
                            holds_vlm_final_norm=True))

This first cut uses REPLICATED weights across the 8 chips of each submesh — it
proves the construction path on real submeshes and gives a working forward()
that can be wired into the pipeline. TP=8 sharding is a follow-up (see
OPTION_B_STATUS.md task #8) and only changes the mesh_mapper + adds an
all_reduce after row-parallel matmuls.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn
from models.experimental.pi0_5.tt.ttnn_gemma import (
    GemmaBlockTTNN,
    precompute_freqs_cis_meta_format,
    rms_norm_ttnn,
)

from .tp_block import Pi0_5SubmeshTPGemmaBlock


def _upload_replicated(
    t: torch.Tensor,
    submesh,
    dtype,
    layout=ttnn.TILE_LAYOUT,
    memory_config=None,
) -> "ttnn.Tensor":
    """Upload a torch tensor replicated across every chip in `submesh`.

    Default is DRAM — full 18-layer replicated weights overflow L1 on a
    single chip (~180 MB total, vs 180 MB L1 cap with no activation
    headroom). The TP-sharded paths (`tp_block.py`, `tp_expert_block.py`,
    `suffix_slice.py`) explicitly request `memory_config=L1_MEMORY_CONFIG`
    because their per-chip footprint is small.
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    mapper = ttnn.replicate_tensor_to_mesh_mapper(submesh)
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=submesh,
        mesh_mapper=mapper,
        memory_config=memory_config,
    )


def _load_vlm_block_weights(
    full_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    submesh,
) -> Dict[str, "ttnn.Tensor"]:
    """Upload one VLM layer's weights onto `submesh` (replicated).

    Mirrors Pi0_5PaliGemmaBackboneTTNN._get_vlm_block_weights_ttnn but routes
    every upload through `replicate_tensor_to_mesh_mapper(submesh)` so the
    resulting block_weights dict is consumable by GemmaBlockTTNN on a
    MeshDevice.
    """
    prefix = f"model.layers.{layer_idx}."
    block_weights: Dict[str, "ttnn.Tensor"] = {}

    q_key = f"{prefix}self_attn.q_proj.weight"
    k_key = f"{prefix}self_attn.k_proj.weight"
    v_key = f"{prefix}self_attn.v_proj.weight"

    if q_key in full_weights and k_key in full_weights and v_key in full_weights:
        wq = _upload_replicated(full_weights[q_key].T.contiguous(), submesh, ttnn.bfloat8_b)
        wk = _upload_replicated(full_weights[k_key].T.contiguous(), submesh, ttnn.bfloat8_b)
        wv = _upload_replicated(full_weights[v_key].T.contiguous(), submesh, ttnn.bfloat8_b)
        block_weights["self_attn.wqkv"] = ttnn.concat([wq, wk, wv], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(wq)
        ttnn.deallocate(wk)
        ttnn.deallocate(wv)

    for key, value in full_weights.items():
        if not key.startswith(prefix):
            continue
        new_key = key[len(prefix) :]
        if new_key in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            continue

        is_norm = "layernorm" in new_key or "norm" in new_key
        if "weight" in new_key and not is_norm:
            value = value.T
        if is_norm:
            value = value + 1.0  # Gemma-style +1 offset, pre-baked

        if len(value.shape) == 1:
            block_weights[new_key] = tensor_1d_to_2d_ttnn(value, submesh, dtype=ttnn.bfloat16)
        else:
            weight_dtype = ttnn.bfloat16 if is_norm else ttnn.bfloat8_b
            block_weights[new_key] = _upload_replicated(value.contiguous(), submesh, weight_dtype)
    return block_weights


class Pi0_5SubmeshVLMSlice:
    """A contiguous slice of VLM transformer layers on a 4x2 submesh.

    Args:
        config:            full PaliGemma config (we only use vlm_config and
                           max_seq_len).
        weights:           the full weights dict (we slice it by layer index).
        submesh:           the 4x2 MeshDevice this stage owns.
        layer_range:       half-open (lo, hi).
        holds_embed_tokens: if True, also upload model.embed_tokens.weight (or
                           lm_head.weight if tied) and expose
                           embed_language_tokens().
        holds_vlm_final_norm: if True, also upload model.norm.weight and apply
                              it at the tail of forward().

    Notes:
        - Weights are replicated across the 8 chips (TP=8 sharding is a
          follow-up). Per-chip memory is therefore the same as on a single
          chip — fine for bring-up, will overflow once we load all 9 layers
          for stage 1/2 from real weights. We add sharding before loading the
          real checkpoint.
        - GemmaBlockTTNN already handles MeshDevice via device.compute_with_storage_grid_size().
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        submesh,
        layer_range: Tuple[int, int],
        holds_embed_tokens: bool = False,
        holds_vlm_final_norm: bool = False,
        tp_shard: bool = False,
    ) -> None:
        if not (0 <= layer_range[0] < layer_range[1] <= config.vlm_config.depth):
            raise ValueError(
                f"layer_range {layer_range} out of bounds for " f"vlm_config.depth={config.vlm_config.depth}"
            )

        self.config = config
        self.submesh = submesh
        self.layer_lo, self.layer_hi = layer_range
        self.num_layers = self.layer_hi - self.layer_lo
        self.holds_embed_tokens = holds_embed_tokens
        self.holds_vlm_final_norm = holds_vlm_final_norm
        self.tp_shard = tp_shard

        lang = weights["vlm_language"]

        # Embedding table (only on the stage that owns it; for Option B this is
        # stage 0 — VLM stages 1 and 2 leave this None).
        self.vlm_embed_tokens: Optional["ttnn.Tensor"] = None
        if holds_embed_tokens:
            embed = lang.get("model.embed_tokens.weight") or lang.get("lm_head.weight")
            if embed is None:
                raise KeyError(
                    "holds_embed_tokens=True but neither model.embed_tokens.weight "
                    "nor lm_head.weight is in weights['vlm_language']"
                )
            self.vlm_embed_tokens = _upload_replicated(
                embed.contiguous(),
                submesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        # Final RMSNorm (only on the stage that owns it; stage 2 for Option B).
        self.vlm_norm: Optional["ttnn.Tensor"] = None
        if holds_vlm_final_norm:
            self.vlm_norm = tensor_1d_to_2d_ttnn(lang["model.norm.weight"] + 1.0, submesh, dtype=ttnn.bfloat16)

        # RoPE tables — built once on this submesh.
        self.cos_meta, self.sin_meta = precompute_freqs_cis_meta_format(
            config.vlm_config.head_dim,
            config.max_seq_len,
            submesh,
        )

        # Per-layer blocks.
        self.vlm_blocks: List = []
        for i in range(self.layer_lo, self.layer_hi):
            if tp_shard:
                self.vlm_blocks.append(
                    Pi0_5SubmeshTPGemmaBlock(
                        config.vlm_config,
                        lang,  # raw layer dict; block extracts its own keys
                        i,
                        submesh,
                        self.cos_meta,
                        self.sin_meta,
                        tp_size=8,
                    )
                )
            else:
                block_weights = _load_vlm_block_weights(lang, i, submesh)
                self.vlm_blocks.append(
                    GemmaBlockTTNN(
                        config.vlm_config,
                        block_weights,
                        i,
                        submesh,
                        self.cos_meta,
                        self.sin_meta,
                    )
                )

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def embed_language_tokens(self, token_ids: "ttnn.Tensor") -> "ttnn.Tensor":
        if self.vlm_embed_tokens is None:
            raise RuntimeError("embed_language_tokens called on a slice without holds_embed_tokens=True")
        return ttnn.embedding(token_ids, self.vlm_embed_tokens)

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_values: Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = None,
        use_cache: bool = False,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
    ) -> Tuple["ttnn.Tensor", Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]]:
        """Forward through layers [layer_lo, layer_hi).

        past_key_values is indexed by GLOBAL layer index — slicing happens here
        so callers can pass the full 18-entry list and we only consume our
        slice. The returned new_cache is also keyed by global index (only the
        slice's entries are populated).
        """
        new_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = (
            [None] * self.config.vlm_config.depth if use_cache else None
        )

        for local_i, block in enumerate(self.vlm_blocks):
            global_i = self.layer_lo + local_i
            if self.tp_shard:
                hidden_states, new_kv = block.forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                )
            else:
                past_kv = past_key_values[global_i] if past_key_values is not None else None
                hidden_states, new_kv = block.forward(
                    hidden_states,
                    cos_override,
                    sin_override,
                    attention_mask,
                    position_ids,
                    past_kv,
                    use_cache,
                )
            if use_cache and new_kv is not None:
                new_cache[global_i] = new_kv

        if self.holds_vlm_final_norm and self.vlm_norm is not None:
            hidden_states = rms_norm_ttnn(
                hidden_states,
                self.vlm_norm,
                self.config.vlm_config.rms_norm_eps,
            )

        return hidden_states, new_cache

    # ------------------------------------------------------------------ #
    # KV migration emitter (used by stage 2)                              #
    # ------------------------------------------------------------------ #

    def get_kv_cache_for_slice(
        self,
        new_cache: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]],
    ) -> List[Tuple[int, Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """Return (global_layer_idx, (K, V)) tuples for this slice's layers only.

        kv_migration.KVMigration consumes this to ship per-layer KV to the
        expert/denoise stage after prefill completes.
        """
        out = []
        for local_i in range(self.num_layers):
            global_i = self.layer_lo + local_i
            kv = new_cache[global_i] if new_cache is not None else None
            if kv is not None:
                out.append((global_i, kv))
        return out
