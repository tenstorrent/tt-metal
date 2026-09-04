# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B prefill attention: `class Attention` = config + weights + a forward dispatch.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaAttention`.
**Template:** `models/demos/gpt_oss_d_p/tt/attention/__init__.py:28`, with the `is_sliding` /
`layer_types` selection at `:77-81` and the per-layer `dataclasses.replace` at `:84`
**deleted** — every Llama layer is full-causal (`bringup_log/00_MODEL_CARD.md` §3), so one
`AttentionConfig` is shared across all 32 layers unmodified and there is no per-layer copy to keep
in sync.
"""

import ttnn

from .config import AttentionConfig, ProgramConfig
from .dense_sp import dense_sp_attention, sp_bootstrap_attention, sp_ring_compute_kernel_config, sp_ring_program_config
from .kv_cache import LlamaKVCache, allocate_kv_cache, write_kv_chunk
from .prefill import attention_forward, select_attention_core
from .weights import AttentionWeights, load_attention_weights

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionWeights",
    "LlamaKVCache",
    "ProgramConfig",
    "allocate_kv_cache",
    "attention_forward",
    # P8's sequence-parallel surface. `select_attention_core` is exported because the gates assert
    # which of the three cores ran rather than inferring it (`DEC-075`).
    "dense_sp_attention",
    "load_attention_weights",
    "select_attention_core",
    "sp_bootstrap_attention",
    "sp_ring_compute_kernel_config",
    "sp_ring_program_config",
    "write_kv_chunk",
]


class Attention:
    """One prefill attention layer: builds the weights, then dispatches to `attention_forward`."""

    def __init__(
        self,
        mesh_device,
        config: AttentionConfig,
        state_dict,
        *,
        ccl_manager=None,
        mesh_config,
        program_config: ProgramConfig,
        layer_idx,
        transformation_mats=None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """
        Args:
            mesh_device: the open mesh.
            config: `AttentionConfig`. This module takes the **config**, not the raw `hf` dict, and
                that is a deliberate exception to the package's `(mesh_device, hf, state_dict)`
                convention (`bringup_log/03_OUTLINE.md` §5): the config *is* the model-specific
                normalisation, and rebuilding it per layer would put config parsing in 32 places.
            state_dict: this layer's attention keys (`q/k/v/o_proj.weight`), or `{}` for cache-only.
            ccl_manager: the model's `CCLManager`. Required only when `tp > 1`.
            mesh_config: the model's `MeshConfig`.
            program_config: `ProgramConfig`. Its pinned SDPA grid is validated **here**, at
                construction, so a derived grid fails at build time rather than at SP > 1 in P8
                (`BRINGUP_RECIPE.md:1411-1419`).
            layer_idx: this layer's index, used for the per-layer KV-cache write.
            transformation_mats: `{"prefill": tensor}` — the `[1,1,32,32]` Meta RoPE transformation
                matrix from `tt/rope.py::build_transformation_mat`. `None` skips RoPE entirely,
                which only `G-ATTN`'s rotation invariant uses.
            weight_dtype: on-device weight dtype, `bfloat8_b` (`DEC-022`).
            tensor_cache_path: where `ttnn.as_tensor` persists / reloads the tilized weights.
        """
        self.mesh_device = mesh_device
        self.config = config
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.program_config = program_config
        self.layer_idx = layer_idx
        self.transformation_mats = transformation_mats

        # Fail at construction, not two phases later. See `ProgramConfig.validate_grid`.
        program_config.validate_grid(mesh_device)
        if mesh_config.tp > 1 and ccl_manager is None:
            raise ValueError(f"Attention at tp={mesh_config.tp} needs a ccl_manager for its TP all-reduce")
        assert config.num_kv_heads % mesh_config.tp == 0, (
            f"num_kv_heads ({config.num_kv_heads}) must be divisible by tp ({mesh_config.tp}); the packed KV "
            f"cache holds exactly one KV head per chip (bringup_log/00_MODEL_CARD.md section 4.1)"
        )

        self.weights = load_attention_weights(
            mesh_device,
            config,
            state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

    def __call__(
        self,
        hidden_states,
        rope_mats,
        *,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=False,
    ):
        """`[1, 1, B*S, hidden]` -> `[1, 1, B*S, hidden]`. See `prefill.attention_forward`."""
        transformation_mat = self.transformation_mats["prefill"] if self.transformation_mats else None
        return attention_forward(
            hidden_states,
            rope_mats,
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
        )
