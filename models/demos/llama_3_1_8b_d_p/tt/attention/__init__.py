# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B attention block: QKV projection, head split, RoPE, ring SDPA, output projection."""

from .config import AttentionConfig, LlamaAttentionProgramConfig, ProgramConfig
from .dense_sp import dense_sp_attention, dense_sp_attention_nocache
from .kv_cache import LlamaKVCache, allocate_kv_caches, cache_capacity, write_kv_chunk
from .prefill import attention_forward
from .weights import load_attention_weights

__all__ = [
    "Attention",
    "AttentionConfig",
    "ProgramConfig",
    "LlamaAttentionProgramConfig",
    "LlamaKVCache",
    "allocate_kv_caches",
    "cache_capacity",
    "write_kv_chunk",
    "dense_sp_attention",
    "dense_sp_attention_nocache",
    "attention_forward",
    "load_attention_weights",
]


class Attention:
    """GQA attention for prefill: QKV proj, head split, RoPE, ring SDPA over the KV cache, o_proj.

    One class, one path — there is no decode here, and no per-layer type dispatch: all 32 Llama
    layers are identical, unlike the donor's dense/sparse split.
    """

    def __init__(
        self,
        mesh_device,
        config: AttentionConfig,
        state_dict,
        ccl_manager,
        mesh_config,
        program_config: ProgramConfig,
        global_layer_idx: int,
        local_layer_idx: int | None = None,
        transformation_mats=None,
        weight_dtype=None,
        tensor_cache_path=None,
    ):
        """
        Args:
            global_layer_idx: index into the checkpoint — selects this layer's weights.
            local_layer_idx: index into the KV cache's layer packing. None => equal to
                `global_layer_idx` (single rank). They differ only under pipeline parallelism, where
                a rank holds a slice of the layers but its cache is packed from 0.
            transformation_mats: the rope op's transformation matrices, built once per model.
        """
        self.mesh_device = mesh_device
        self.config = config
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.program_config = program_config
        self.layer_idx = global_layer_idx
        self.cache_layer_idx = global_layer_idx if local_layer_idx is None else local_layer_idx
        self.transformation_mats = transformation_mats
        self.n_kv_local = config.kv_heads_per_chip(mesh_config.tp)  # 2 at TP=4
        self.n_q_local = config.q_heads_per_chip(mesh_config.tp)  # 8 at TP=4
        self.weights = load_attention_weights(
            mesh_device,
            config,
            state_dict,
            mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

    def __call__(
        self,
        x,
        *,
        rope_mats,
        kv_cache=None,
        slot_idx=0,
        cached_len=0,
        logical_n=None,
        indexed_rope=False,
        write_chunk=True,
    ):
        """See :func:`~.prefill.attention_forward` for the argument contract."""
        return attention_forward(
            x,
            weights=self.weights,
            config=self.config,
            program_config=self.program_config,
            mesh_config=self.mesh_config,
            ccl_manager=self.ccl_manager,
            mesh_device=self.mesh_device,
            rope_mats=rope_mats,
            transformation_mats=self.transformation_mats,
            kv_cache=kv_cache,
            slot_idx=slot_idx,
            layer_idx=self.cache_layer_idx,
            cached_len=cached_len,
            logical_n=logical_n,
            indexed_rope=indexed_rope,
            write_chunk=write_chunk,
        )
