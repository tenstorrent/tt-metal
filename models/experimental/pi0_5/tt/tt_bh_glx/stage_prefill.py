# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill stage: 18-chip VLM-prefill driver.

One Gemma-2B layer per chip. Activations chain via host bounce; per-layer K/V
stays on its owning chip for later migration to the denoise stage. The final
VLM RMS norm runs on the last prefill chip (chip 17).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.tt.ttnn_common import get_ln_weight_memory_config, tensor_1d_to_2d_ttnn
from models.experimental.pi0_5.tt.ttnn_gemma import rms_norm_ttnn

from . import stages
from .vlm_slice import VLMBlockSlice


class StagePrefill:
    def __init__(self, config: Pi0_5ModelConfig, weights: Dict[str, dict], mesh_handles, transport=None):
        chips = mesh_handles.prefill_per_chip
        if len(chips) != stages.PREFILL_NUM_CHIPS:
            raise RuntimeError(f"prefill stage requires {stages.PREFILL_NUM_CHIPS} chips, got {len(chips)}")
        if config.vlm_config.depth != stages.VLM_TOTAL_LAYERS:
            raise RuntimeError(f"VLM depth {config.vlm_config.depth} != expected {stages.VLM_TOTAL_LAYERS}")

        vw = weights["vlm_language"]
        self.config = config
        self.chips = chips
        self.slices = [
            VLMBlockSlice(config.vlm_config, vw, chips[i], layer_idx=i, max_seq_len=config.max_seq_len)
            for i in range(stages.VLM_TOTAL_LAYERS)
        ]

        # Final VLM RMS norm lives on the last prefill chip (where the final
        # hidden lands). +1.0 Gemma offset pre-added on host (matches the
        # single-chip backbone init at ttnn_paligemma.py:82).
        last_chip = chips[-1]
        self.vlm_norm = tensor_1d_to_2d_ttnn(
            vw["model.norm.weight"] + 1.0,
            last_chip,
            dtype=ttnn.bfloat16,
            memory_config=get_ln_weight_memory_config(),
        )
        self.vlm_norm_eps = config.vlm_config.rms_norm_eps
        if transport is None:
            from .transport import SocketTransport

            transport = SocketTransport()
        self.transport = transport

    def run(
        self,
        prefix_embs_on_chip0: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        per_chip_attn_mask: Optional[List["ttnn.Tensor"]] = None,
        per_chip_cos: Optional[List["ttnn.Tensor"]] = None,
        per_chip_sin: Optional[List["ttnn.Tensor"]] = None,
    ) -> Tuple["ttnn.Tensor", List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """prefix_embs_on_chip0 must be a ttnn.Tensor on prefill_per_chip[0].

        Returns (final_hidden_on_chip17, [(K_chip_i, V_chip_i)]_i=0..17).
        Per-layer KV stays on the chip that produced it.

        Upstream-openpi compat (PI0_UPSTREAM_MASKS=1): pass per_chip_attn_mask,
        per_chip_cos, per_chip_sin — each a list of 18 ttnn.Tensors (one per
        prefill chip) holding the prefix attention mask and position-aware RoPE
        tables. When None, the legacy "no mask, sequential RoPE" path runs.
        """
        hidden = prefix_embs_on_chip0
        per_layer_kv: List[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = []
        for i, sl in enumerate(self.slices):
            m_i = per_chip_attn_mask[i] if per_chip_attn_mask is not None else attention_mask
            c_i = per_chip_cos[i] if per_chip_cos is not None else None
            s_i = per_chip_sin[i] if per_chip_sin is not None else None
            hidden, (k, v) = sl.forward(hidden, m_i, position_ids, c_i, s_i)
            per_layer_kv.append((k, v))
            if i < len(self.slices) - 1:
                hidden = self.transport.send(hidden, self.chips[i + 1])

        # Final VLM RMS norm on the last chip.
        hidden = rms_norm_ttnn(hidden, self.vlm_norm, self.vlm_norm_eps)
        return hidden, per_layer_kv
