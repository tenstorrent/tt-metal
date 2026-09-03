# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B prefill attention module.

HF anchor: ``transformers.models.llama.modeling_llama.LlamaAttention``.
Template: ``models/demos/gpt_oss_d_p/tt/attention/__init__.py:28`` (the class), ``:38``
(``__init__``), ``:87`` (``load_attention_weights``), ``:103`` (``__call__``), ``:133`` (the dispatch
to ``attention_forward``).

**Deletions vs the template** (``03_OUTLINE.md`` §3.13): ``layer_types`` / ``is_sliding`` (``:47``,
``:78-84``), the per-layer ``dataclasses.replace`` (``:84``) — Llama's layers are all identical, so
there is no per-layer config copy and no layer-type logic — and ``position_idx`` (``:107``), which is
unused in prefill.

Everything after ``state_dict`` is keyword-only (``03_OUTLINE.md`` §1 convention 1).
"""

from __future__ import annotations

import ttnn

from .config import AttentionConfig, ProgramConfig
from .dense_sp import dense_sp_attention
from .kv_cache import (
    LLAMA_HEAD_DIM,
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    LlamaKVCache,
    allocate_kv_cache,
    write_kv_chunk,
)
from .prefill import attention_forward
from .weights import AttentionWeights, load_attention_weights

__all__ = [
    "Attention",
    "AttentionConfig",
    "ProgramConfig",
    "AttentionWeights",
    "LlamaKVCache",
    "allocate_kv_cache",
    "write_kv_chunk",
    "attention_forward",
    "load_attention_weights",
    "dense_sp_attention",
    "LLAMA_HEAD_DIM",
    "NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK",
    "attention_config_from_hf",
]


def attention_config_from_hf(hf_config, *, max_seq_len, sequence_parallel=False) -> AttentionConfig:
    """Build an :class:`AttentionConfig` from a ``LlamaHFConfig``.

    The one place the model dimensions cross from ``tt/model_config.py`` into the attention package,
    so no attention file reads ``hf_config`` directly and none of them can reach past the object
    with ``getattr(..., default)`` (``DEC-009``, Appendix F.2).
    """
    return AttentionConfig(
        hidden_size=hf_config.hidden_size,
        num_heads=hf_config.num_attention_heads,
        num_kv_heads=hf_config.num_key_value_heads,
        head_dim=hf_config.head_dim,
        max_seq_len=max_seq_len,
        rms_norm_eps=hf_config.rms_norm_eps,
        sequence_parallel=sequence_parallel,
    )


class Attention:
    """Llama-3.1-8B prefill attention layer: build config + weights, dispatch the forward."""

    def __init__(
        self,
        mesh_device,
        config: AttentionConfig,
        state_dict,
        *,
        mesh_config,
        ccl_manager,
        program_config: ProgramConfig,
        layer_idx,
        transformation_mats=None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        scatter_output=False,
        meta_swizzle=True,
    ):
        """
        Args:
            mesh_device: the ttnn mesh device.
            config: :class:`~.config.AttentionConfig`. Used as-is — Llama has no per-layer variant,
                so this object is safe to share across all 32 layers.
            state_dict: the already-stripped ``self_attn.*`` sub-dict in HF layout. ``{}`` means
                cache-only mode.
            mesh_config: ``MeshConfig``.
            ccl_manager: ``CCLManager``; unused at TP=1 and SP=1.
            program_config: :class:`~.config.ProgramConfig`.
            layer_idx: this layer's index (the per-layer KV-cache slot).
            transformation_mats: ``{"prefill": tensor}`` from ``tt/rope.build_transformation_mat``.
            weight_dtype: on-device weight dtype (default ``bfloat8_b``).
            tensor_cache_path: directory for the tilized weight cache, or ``None``.
            scatter_output: residual scheme (``DEC-018``); see
                :func:`~.prefill.attention_forward`.
            meta_swizzle: apply the Q/K ``reverse_permute`` at load (``DEC-033``). Only the
                ``G-ATTN`` negative control passes ``False``.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.program_config = program_config
        self.layer_idx = layer_idx
        self.transformation_mats = transformation_mats
        self.config = config
        self.scatter_output = scatter_output

        # DEC-012 / Appendix F.8: fail at BUILD time if the SDPA program grid would break the SP
        # ring path, rather than passing every single-card gate and failing two phases later in P8.
        program_config.assert_sdpa_grid_fits(mesh_device)

        # Built once per layer, not per forward (DEC-031).
        self.compute_kernel_config = program_config.get_compute_kernel_config(mesh_device)

        self.weights = load_attention_weights(
            mesh_device,
            config,
            state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
            meta_swizzle=meta_swizzle,
        )

        # Convenience mirrors, as in the template (gpt_oss __init__.py:97-101).
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = config.scaling

    def __call__(
        self,
        hidden_states,
        rope_mats,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=False,
    ):
        """Prefill attention forward. See :func:`~.prefill.attention_forward` for the shapes."""
        transformation_mat = self.transformation_mats["prefill"] if self.transformation_mats else None

        return attention_forward(
            hidden_states=hidden_states,
            rope_mats=rope_mats,
            weights=self.weights,
            kv_cache=kv_cache,
            config=self.config,
            mesh_config=self.mesh_config,
            mesh_device=self.mesh_device,
            program_config=self.program_config,
            transformation_mat=transformation_mat,
            ccl_manager=self.ccl_manager,
            user_id=user_id,
            batch_size=batch_size,
            layer_idx=self.layer_idx,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
            scatter_output=self.scatter_output,
            compute_kernel_config=self.compute_kernel_config,
        )
