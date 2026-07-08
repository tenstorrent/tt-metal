# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Denoise stage driver: 6-chip AdaRMS expert chain.

18 expert layers striped 3-per-chip across the 6-chip denoise submesh. The
prefix KV cache is migrated layer-paired (prefill chip i → denoise chip i // 3)
before the first step, then reused across denoise iterations. Each iteration
chains suffix_hidden through the 6 chips via host bounce.

This v1 driver exposes only the expert-chain step (suffix + final norm +
action_out_proj + Euler integration live in higher-level pipeline code).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig

from . import stages
from .expert_slice import ExpertChunkSlice


class StageDenoise:
    def __init__(self, config: Pi0_5ModelConfig, weights: Dict[str, dict], mesh_handles, transport=None):
        chips = mesh_handles.denoise_per_chip
        if len(chips) != stages.DENOISE_NUM_CHIPS:
            raise RuntimeError(f"denoise stage requires {stages.DENOISE_NUM_CHIPS} chips, got {len(chips)}")
        if config.expert_config.depth != stages.EXPERT_TOTAL_LAYERS:
            raise RuntimeError(f"expert depth {config.expert_config.depth} != expected {stages.EXPERT_TOTAL_LAYERS}")

        self.config = config
        self.chips = chips
        n_per = stages.EXPERT_LAYERS_PER_CHIP
        ew = weights["action_expert"]
        self.chunks: List[ExpertChunkSlice] = [
            ExpertChunkSlice(
                config.expert_config,
                ew,
                chips[c],
                layer_range=(c * n_per, (c + 1) * n_per),
                max_seq_len=config.max_seq_len,
            )
            for c in range(stages.DENOISE_NUM_CHIPS)
        ]
        if transport is None:
            from .transport import SocketTransport

            transport = SocketTransport()
        self.transport = transport

    def run_expert_chain(
        self,
        suffix_hidden_chip0: "ttnn.Tensor",
        adarms_conds: List["ttnn.Tensor"],
        prefix_kv_per_chip: List[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]],
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        per_chip_attn_mask: Optional[List["ttnn.Tensor"]] = None,
        per_chip_cos: Optional[List["ttnn.Tensor"]] = None,
        per_chip_sin: Optional[List["ttnn.Tensor"]] = None,
    ) -> "ttnn.Tensor":
        """Chain `suffix_hidden` through 6 chips, 3 expert layers per chip.

        adarms_conds: per-chip adarms_cond tensor (already on that chip).
        prefix_kv_per_chip: kv_migration.migrate_layer_paired() output.

        Upstream-openpi compat (PI0_UPSTREAM_MASKS=1): per_chip_attn_mask,
        per_chip_cos, per_chip_sin each are length-6 lists holding the expert
        cross-attention mask and prefix-offset-aware suffix RoPE tables, one
        ttnn.Tensor per denoise chip.

        Returns the final hidden on chips[5].
        """
        if len(adarms_conds) != stages.DENOISE_NUM_CHIPS:
            raise RuntimeError(f"need {stages.DENOISE_NUM_CHIPS} adarms_cond tensors")
        if len(prefix_kv_per_chip) != stages.DENOISE_NUM_CHIPS:
            raise RuntimeError(f"need {stages.DENOISE_NUM_CHIPS} prefix-KV groups")

        hidden = suffix_hidden_chip0
        for i, chunk in enumerate(self.chunks):
            m_i = per_chip_attn_mask[i] if per_chip_attn_mask is not None else attention_mask
            c_i = per_chip_cos[i] if per_chip_cos is not None else None
            s_i = per_chip_sin[i] if per_chip_sin is not None else None
            hidden = chunk.forward(
                hidden,
                adarms_conds[i],
                prefix_kv_per_chip[i],
                m_i,
                position_ids,
                c_i,
                s_i,
            )
            if i < len(self.chunks) - 1:
                hidden = self.transport.send(hidden, self.chips[i + 1])
        return hidden
