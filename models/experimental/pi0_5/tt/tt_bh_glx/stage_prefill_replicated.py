# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Replicated VLM prefill: 18 Gemma-2B blocks with weights replicated across the 1×8 mesh.

Each chip holds the FULL prefill weights and runs the entire prefill independently —
no TP=8 sharding, no all_reduce, no block-sharded LN-to-matmul handoff. Output KV
cache is replicated across all 8 chips by construction.

This mirrors the single-chip Pi0_5PaliGemmaBackboneTTNN.forward_vlm path but builds
each block on the parent 1×8 MeshDevice (weights replicated implicitly via
ttnn.from_torch without a mesh_mapper — see vlm_slice._load_block_weights_to_submesh).

Drop-in for StagePrefillTP4: same .run() signature and output shape.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.tt.ttnn_common import get_ln_weight_memory_config, tensor_1d_to_2d_ttnn
from models.experimental.pi0_5.tt.ttnn_gemma import rms_norm_ttnn

from .vlm_slice import VLMBlockSlice


class StagePrefillReplicated:
    """18-layer VLM prefill, weights replicated across the 1×8 mesh.

    Each chip runs the full single-chip prefill code path (GemmaBlockTTNN via
    VLMBlockSlice). Per-layer KV cache is replicated across all 8 chips on
    completion.
    """

    def __init__(self, config: Pi0_5ModelConfig, weights: Dict[str, dict], mesh_device):
        self.config = config
        self.mesh_device = mesh_device

        vw = weights["vlm_language"]
        vlm_cfg = config.vlm_config

        self.blocks: List[VLMBlockSlice] = []
        for i in range(vlm_cfg.depth):
            self.blocks.append(VLMBlockSlice(vlm_cfg, vw, mesh_device, layer_idx=i, max_seq_len=config.max_seq_len))

        # Final VLM RMS norm. +1.0 Gemma offset pre-applied on host so the kernel
        # only sees a "pure" γ. Replicated across the mesh via tensor_1d_to_2d_ttnn
        # (no mesh_mapper → implicit replicate, like the per-block norm γ).
        self.vlm_norm = tensor_1d_to_2d_ttnn(
            vw["model.norm.weight"] + 1.0,
            mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=get_ln_weight_memory_config(),
        )

    def run(
        self,
        prefix_embs: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        per_chip_attn_mask: Optional[List[ttnn.Tensor]] = None,
        per_chip_cos: Optional[List[ttnn.Tensor]] = None,
        per_chip_sin: Optional[List[ttnn.Tensor]] = None,
    ) -> Tuple[ttnn.Tensor, List[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """Identical signature to StagePrefillTP4.run(). prefix_embs is replicated
        on the mesh; returns (final_hidden, [(K_i, V_i)]_i=0..17) all replicated."""
        hidden = prefix_embs
        per_layer_kv: List[Tuple[ttnn.Tensor, ttnn.Tensor]] = []
        for i, slice_ in enumerate(self.blocks):
            m_i = per_chip_attn_mask[i] if per_chip_attn_mask is not None else attention_mask
            c_i = per_chip_cos[i] if per_chip_cos is not None else None
            s_i = per_chip_sin[i] if per_chip_sin is not None else None
            hidden, new_kv = slice_.forward(hidden, m_i, position_ids, c_i, s_i)
            per_layer_kv.append(new_kv)

        hidden = rms_norm_ttnn(hidden, self.vlm_norm, self.config.vlm_config.rms_norm_eps)
        return hidden, per_layer_kv
