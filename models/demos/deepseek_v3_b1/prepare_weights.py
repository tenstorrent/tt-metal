# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for DeepSeek weight preparation.

Implementation is split under :mod:`models.demos.deepseek_v3_b1.weights`:

- **Layer 1 (generic cache):** :mod:`models.demos.deepseek_v3_b1.tensor_cache` — content-addressed
  **artifacts**, fingerprinting, storage (no HF key naming or DeepSeek preprocess).
- **Layer 2 (model adapter):** :mod:`models.demos.deepseek_v3_b1.weights` — **SourceSelection** keys,
  **preprocess**, **ArtifactTarget** specs (**catalog**), **pack/fuse** (**fusion_runtime**),
  and **assemble** dataclasses (**types**) wired through **adapter**.

Keep importing from ``prepare_weights`` for stable call sites; new code may import from
``weights.adapter`` and ``weights.types`` directly.
"""

from __future__ import annotations

from models.demos.deepseek_v3_b1.blitz_decode_weights import OverlappedTensor
from models.demos.deepseek_v3_b1.weights.adapter import (
    create_gate_bias_tensor,
    create_gate_indices_tensor,
    get_layer_raw_tensors,
    prepare_attention_weights,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    prepare_mtp_weights,
    prepare_routed_expert_weights,
    prepare_shared_expert_weights,
)
from models.demos.deepseek_v3_b1.weights.catalog import CURRENT_TRANSFORM_VERSION, MOE_SENDER_GRID_SIZE
from models.demos.deepseek_v3_b1.weights.preprocessing import deinterleave_q_b_proj, split_kv_b_proj
from models.demos.deepseek_v3_b1.weights.types import (
    _MTP_LAYER_IDX,
    NUM_ROUTED_EXPERTS,
    AttentionWeights,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    DeepSeekV3MTPWeights,
    DenseRoutedExpertWeights,
    MoERoutedExpertWeights,
    SharedExpertWeights,
)

__all__ = [
    "CURRENT_TRANSFORM_VERSION",
    "MOE_SENDER_GRID_SIZE",
    "NUM_ROUTED_EXPERTS",
    "OverlappedTensor",
    "_MTP_LAYER_IDX",
    "AttentionWeights",
    "SharedExpertWeights",
    "DenseRoutedExpertWeights",
    "MoERoutedExpertWeights",
    "DeepSeekV3DenseLayerWeights",
    "DeepSeekV3MoELayerWeights",
    "DeepSeekV3EmbeddingLayerWeights",
    "DeepSeekV3LMHeadWeights",
    "DeepSeekV3MTPWeights",
    "create_gate_bias_tensor",
    "create_gate_indices_tensor",
    "deinterleave_q_b_proj",
    "get_layer_raw_tensors",
    "split_kv_b_proj",
    "prepare_attention_weights",
    "prepare_dense_layer_weights",
    "prepare_embedding_weights",
    "prepare_lm_head_weights",
    "prepare_moe_layer_weights",
    "prepare_mtp_weights",
    "prepare_routed_expert_weights",
    "prepare_shared_expert_weights",
]
