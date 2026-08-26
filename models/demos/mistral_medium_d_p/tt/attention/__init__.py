# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 prefill attention module (dense causal GQA, full-rotary YaRN)."""

import ttnn

from models.demos.mistral_medium_d_p.config import MeshConfig

from .config import AttentionConfig, ProgramConfig
from .kv_cache import MistralKVCache, allocate_kv_cache, write_kv_chunk
from .prefill import attention_forward
from .weights import AttentionWeights, load_attention_weights

__all__ = [
    "Attention",
    "AttentionConfig",
    "ProgramConfig",
    "AttentionWeights",
    "MistralKVCache",
    "allocate_kv_cache",
    "write_kv_chunk",
]


class Attention:
    """Mistral-Medium-3.5 prefill attention layer.

    Builds config + weights and dispatches the chunked-prefill forward. Every layer is identical
    (dense causal, full rotary) — unlike gpt-oss there is no per-layer sliding/full selection, so
    this class takes no ``layer_types``. No decode path in this prefill bring-up.
    """

    def __init__(
        self,
        mesh_device,
        config: AttentionConfig,
        state_dict,
        ccl_manager,
        mesh_config: MeshConfig,
        program_config: ProgramConfig,
        layer_idx,
        transformation_mats=None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        self.mesh_config = mesh_config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.program_config = program_config
        self.layer_idx = layer_idx
        self.transformation_mats = transformation_mats
        self.config = config

        self.weights = load_attention_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = config.scaling

    def __call__(
        self,
        hidden_states,
        rope_mats,
        position_idx=None,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=False,
    ):
        transformation_mat = self.transformation_mats["prefill"] if self.transformation_mats else None
        return attention_forward(
            hidden_states=hidden_states,
            rope_mats=rope_mats,
            user_id=user_id,
            weights=self.weights,
            kv_cache=kv_cache,
            config=self.config,
            mesh_config=self.mesh_config,
            mesh_device=self.mesh_device,
            program_config=self.program_config,
            transformation_mat=transformation_mat,
            position_idx=position_idx,
            ccl_manager=self.ccl_manager,
            batch_size=batch_size,
            layer_idx=self.layer_idx,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )
