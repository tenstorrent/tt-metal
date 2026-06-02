# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Submesh-aware action-expert (Gemma-300M w/ adaRMS) slice for Option B stage 3.

Mirrors `Pi0_5SubmeshVLMSlice` but for the expert path:
  - per-layer AdaRMSGemmaBlockTTNN
  - fused 6*W modulation Dense per block (concat of input_layernorm.dense and
    post_attention_layernorm.dense — the `_inject_adarms_weights` pattern from
    Pi0_5PaliGemmaBackboneTTNN)
  - expert final RMSNorm modulation Dense (model.norm.dense)
  - expert RoPE tables

Suffix MLP + 10-step denoise loop live in `stage_3_expert.py` — not here — so
this class is reusable if the denoise schedule ever changes.

Weights are REPLICATED across the 8 chips. TP=8 sharding is a follow-up.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn
from models.experimental.pi0_5.tt.ttnn_gemma import (
    AdaRMSGemmaBlockTTNN,
    ada_rms_norm_no_gate_ttnn,
    ada_rms_norm_no_gate_precomputed_ttnn,
    precompute_freqs_cis_meta_format,
)

from .vlm_slice import _upload_replicated
from .tp_expert_block import Pi0_5SubmeshTPAdaRMSBlock


def _load_expert_block_weights(
    full_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    submesh,
) -> Dict[str, "ttnn.Tensor"]:
    """Upload one expert layer's weights onto `submesh` (replicated).

    Includes the fused adaRMS modulation weight (concat of pre-attn and
    pre-ffw modulation Denses along output dim, [6*W, W]).
    """
    prefix = f"model.layers.{layer_idx}."
    block_weights: Dict[str, "ttnn.Tensor"] = {}

    # Fused QKV (col-concat of q/k/v after .T)
    q_key, k_key, v_key = (
        f"{prefix}self_attn.q_proj.weight",
        f"{prefix}self_attn.k_proj.weight",
        f"{prefix}self_attn.v_proj.weight",
    )
    if q_key in full_weights and k_key in full_weights and v_key in full_weights:
        wq = _upload_replicated(full_weights[q_key].T.contiguous(), submesh, ttnn.bfloat8_b)
        wk = _upload_replicated(full_weights[k_key].T.contiguous(), submesh, ttnn.bfloat8_b)
        wv = _upload_replicated(full_weights[v_key].T.contiguous(), submesh, ttnn.bfloat8_b)
        block_weights["self_attn.wqkv"] = ttnn.concat([wq, wk, wv], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(wq)
        ttnn.deallocate(wk)
        ttnn.deallocate(wv)

    # o_proj, mlp.{gate,up,down}_proj, plus any norm γ in keys we skip below.
    for key, value in full_weights.items():
        if not key.startswith(prefix):
            continue
        new_key = key[len(prefix) :]

        # Skip individual Q/K/V (already fused) and adaRMS Denses (handled below).
        if new_key in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "input_layernorm.dense.weight",
            "input_layernorm.dense.bias",
            "post_attention_layernorm.dense.weight",
            "post_attention_layernorm.dense.bias",
        ):
            continue
        # pi0.5 expert has no plain *_layernorm.weight (only the adaRMS Denses).
        if new_key in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            continue

        is_norm = "norm" in new_key  # rare for the expert (only handled keys above), but keep guard
        if "weight" in new_key and not is_norm:
            value = value.T

        if len(value.shape) == 1:
            block_weights[new_key] = tensor_1d_to_2d_ttnn(value, submesh, dtype=ttnn.bfloat16)
        else:
            block_weights[new_key] = _upload_replicated(
                value.contiguous(),
                submesh,
                ttnn.bfloat16 if is_norm else ttnn.bfloat8_b,
            )

    # Fused adaRMS modulation weight = concat([pre_attn.dense, pre_ffw.dense], dim=0).
    w_keys = [
        f"{prefix}input_layernorm.dense.weight",
        f"{prefix}post_attention_layernorm.dense.weight",
    ]
    for wk in w_keys:
        if wk not in full_weights:
            raise KeyError(f"expert layer {layer_idx} missing adaRMS weight '{wk}'")
    fused_w = torch.cat([full_weights[wk] for wk in w_keys], dim=0).contiguous()
    block_weights["adarms_mod.weight"] = _upload_replicated(
        fused_w.T.contiguous(),  # AdaRMSGemmaBlockTTNN expects column-stored weight
        submesh,
        ttnn.bfloat16,
    )

    b_keys = [
        f"{prefix}input_layernorm.dense.bias",
        f"{prefix}post_attention_layernorm.dense.bias",
    ]
    biases = [full_weights[bk] for bk in b_keys if bk in full_weights]
    if biases:
        assert len(biases) == 2, "expected both adaRMS biases or neither"
        fused_b = torch.cat(biases, dim=0).contiguous()
        block_weights["adarms_mod.bias"] = tensor_1d_to_2d_ttnn(fused_b, submesh, dtype=ttnn.bfloat16)

    return block_weights


class Pi0_5SubmeshExpertSlice:
    """Action-expert layers + final adaRMS-norm Dense on a 4x2 submesh."""

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        submesh,
        expert_layer_range: Tuple[int, int] = (0, 18),
        tp_shard: bool = False,
    ) -> None:
        if not (0 <= expert_layer_range[0] < expert_layer_range[1] <= config.expert_config.depth):
            raise ValueError(
                f"expert_layer_range {expert_layer_range} out of bounds for "
                f"expert_config.depth={config.expert_config.depth}"
            )

        self.config = config
        self.submesh = submesh
        self.layer_lo, self.layer_hi = expert_layer_range
        self.num_layers = self.layer_hi - self.layer_lo
        self.tp_shard = tp_shard

        ae = weights["action_expert"]

        # Expert RoPE — same precompute as VLM but with expert head_dim.
        self.cos_meta, self.sin_meta = precompute_freqs_cis_meta_format(
            config.expert_config.head_dim,
            config.max_seq_len,
            submesh,
        )

        # Per-layer blocks.
        self.expert_blocks: List = []
        for i in range(self.layer_lo, self.layer_hi):
            if tp_shard:
                self.expert_blocks.append(
                    Pi0_5SubmeshTPAdaRMSBlock(
                        config.expert_config,
                        ae,
                        i,
                        submesh,
                        self.cos_meta,
                        self.sin_meta,
                        tp_size=8,
                    )
                )
            else:
                block_weights = _load_expert_block_weights(ae, i, submesh)
                self.expert_blocks.append(
                    AdaRMSGemmaBlockTTNN(
                        config.expert_config,
                        block_weights,
                        i,
                        submesh,
                        self.cos_meta,
                        self.sin_meta,
                    )
                )

        # Final norm modulation Dense (model.norm.dense). Only meaningful when
        # this slice owns the tail — for Option B stage 3, that's always.
        if "model.norm.dense.weight" not in ae:
            raise KeyError("expert checkpoint missing 'model.norm.dense.weight'")
        self.final_norm_mod_weight = _upload_replicated(
            ae["model.norm.dense.weight"].T.contiguous(),
            submesh,
            ttnn.bfloat16,
        )
        self.final_norm_mod_bias = None
        if "model.norm.dense.bias" in ae:
            self.final_norm_mod_bias = tensor_1d_to_2d_ttnn(ae["model.norm.dense.bias"], submesh, dtype=ttnn.bfloat16)

        device_grid = submesh.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        prefix_kv_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = None,
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        precomputed_block_mods: Optional[List[Tuple["ttnn.Tensor", ...]]] = None,
        precomputed_final_mod: Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = None,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
        keep_padded: bool = False,
    ) -> "ttnn.Tensor":
        """Run all expert layers in the slice. Returns the post-final-norm hidden.

        prefix_kv_cache is indexed by GLOBAL layer index over the full expert
        depth (typically same as VLM depth, 18). The denoise loop in stage_3
        passes the migrated VLM KV here.
        """
        for local_i, block in enumerate(self.expert_blocks):
            global_i = self.layer_lo + local_i
            past_kv = prefix_kv_cache[global_i] if prefix_kv_cache is not None else None
            if self.tp_shard:
                hidden_states = block.forward(
                    hidden_states,
                    adarms_cond,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                )
            else:
                block_mod = precomputed_block_mods[local_i] if precomputed_block_mods is not None else None
                hidden_states, _new_kv = block.forward(
                    hidden_states,
                    cos_override,
                    sin_override,
                    adarms_cond,
                    attention_mask,
                    position_ids,
                    past_kv,
                    use_cache=False,
                    precomputed_mod=block_mod,
                    keep_padded=keep_padded,
                )

        if precomputed_final_mod is not None:
            sf1, tf = precomputed_final_mod
            hidden_states = ada_rms_norm_no_gate_precomputed_ttnn(
                hidden_states, sf1, tf, self.config.expert_config.rms_norm_eps
            )
        else:
            hidden_states = ada_rms_norm_no_gate_ttnn(
                hidden_states,
                adarms_cond,
                self.final_norm_mod_weight,
                self.final_norm_mod_bias,
                self.config.expert_config.rms_norm_eps,
                self.core_grid,
            )
        return hidden_states
