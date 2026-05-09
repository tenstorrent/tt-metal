# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Weight conversion utilities for the hybrid GLM-4.7-Flash.

Re-exports the agentic layer_weights module which provides:
- _linear_weight_tt: HF [out,in] -> TT [1,1,in,out]
- _per_head_weight_tt: [H,in,out] -> [1,H,in,out]
- _vector_weight_tt: [E] -> [1,1,1,E] ROW_MAJOR
- _experts_weight_tt: [E,in,out] -> [D,1,E_local,in,out] with ShardTensorToMesh
- MoELayerTTWeights / DecoderLayerTTWeights dataclasses
- convert_decoder_layer_weights: full layer conversion from HF state dict
- _prepare_fused_kv_branch_weights: pre-sharded weights for GLMKVCacheBranch
"""

from models.demos.glm4_moe_lite.tt.layer_weights import (
    DecoderLayerTTWeights,
    MoELayerTTWeights,
    _experts_weight_tt,
    _linear_weight_tt,
    _maybe_dram_shard_linear_weight,
    _per_head_weight_tt,
    _prepare_fused_kv_branch_weights,
    _vector_weight_tt,
    convert_decoder_layer_weights,
)

__all__ = [
    "MoELayerTTWeights",
    "DecoderLayerTTWeights",
    "convert_decoder_layer_weights",
    "_linear_weight_tt",
    "_per_head_weight_tt",
    "_vector_weight_tt",
    "_experts_weight_tt",
    "_prepare_fused_kv_branch_weights",
    "_maybe_dram_shard_linear_weight",
]
