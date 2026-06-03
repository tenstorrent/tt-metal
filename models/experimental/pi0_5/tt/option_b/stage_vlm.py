# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 / 2 — VLM transformer half-stack on a 4×2 submesh."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig

from .stages import StageSpec
from .vlm_slice import Pi0_5SubmeshVLMSlice


class StageVLM:
    """A contiguous slice of VLM transformer layers placed on one 4×2 submesh.

    Used for stage 1 (layers 0-8) and stage 2 (layers 9-17 + final norm).

    Construction is two-phase: `__init__` only records arguments; `initialize`
    actually uploads weights and builds blocks. This lets the pipeline
    orchestrator decide ordering across stages (e.g. open all submeshes,
    then initialize each in parallel).

    TP=8 sharding plan (for follow-up; see OPTION_B_STATUS.md task #8):
    - attn.q (col-parallel):  shard output dim 2048 / 8 = 256
    - attn.k/v (col-parallel): shard output dim 256 / 8 = 32 each
    - attn.o (row-parallel):  shard input dim 2048 / 8 = 256, all-reduce after
    - mlp.gate/up (col-parallel): shard 16384 / 8 = 2048
    - mlp.down (row-parallel): shard 16384 / 8 = 2048, all-reduce after
    """

    def __init__(
        self,
        spec: StageSpec,
        submesh,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        tp_shard: bool = False,
    ) -> None:
        if spec.vlm_layer_range[1] <= spec.vlm_layer_range[0]:
            raise ValueError("StageVLM requires non-empty vlm_layer_range")
        self.spec = spec
        self.submesh = submesh
        self.config = config
        self.weights = weights
        self.tp_shard = tp_shard
        self.layer_lo, self.layer_hi = spec.vlm_layer_range
        self.num_layers = self.layer_hi - self.layer_lo
        self.slice: Optional[Pi0_5SubmeshVLMSlice] = None
        self.last_kv_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = None

    def initialize(self) -> None:
        """Upload weights for this stage's layer range onto the submesh."""
        if self.slice is not None:
            return  # idempotent
        self.slice = Pi0_5SubmeshVLMSlice(
            config=self.config,
            weights=self.weights,
            submesh=self.submesh,
            layer_range=(self.layer_lo, self.layer_hi),
            holds_embed_tokens=self.spec.holds_embed_tokens,
            holds_vlm_final_norm=self.spec.holds_vlm_final_norm,
            tp_shard=self.tp_shard,
        )

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_values: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = None,
        use_cache: bool = True,
    ) -> "ttnn.Tensor":
        """Run the layer slice and stash KV cache for the migration emitter."""
        if self.slice is None:
            raise RuntimeError("StageVLM.forward called before initialize()")
        hidden_states, new_cache = self.slice.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        self.last_kv_cache = new_cache
        return hidden_states

    def get_kv_cache(self) -> List[Tuple[int, Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """KV migration emitter (stage 2) reads this after forward().

        Returns (global_layer_idx, (K, V)) tuples for this stage's layers.
        """
        if self.slice is None or self.last_kv_cache is None:
            return []
        return self.slice.get_kv_cache_for_slice(self.last_kv_cache)
