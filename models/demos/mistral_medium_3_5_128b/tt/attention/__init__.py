# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The attention block, assembled. Ported from ``gpt_oss_d_p/tt/attention/__init__.py``.

``Attention`` owns the weights and the per-block config; the dataflow lives in
:mod:`~.prefill`, the SDPA cores in :mod:`~.dense_sp`, and the KV cache layout in
:mod:`~.kv_cache`. Splitting it this way is what lets the decoder test suite exercise the ring op
and the cache read/write path at op level without standing up a whole block.
"""

import ttnn

from models.demos.mistral_medium_3_5_128b.tt.attention.config import AttentionConfig, ProgramConfig
from models.demos.mistral_medium_3_5_128b.tt.attention.prefill import attention_forward
from models.demos.mistral_medium_3_5_128b.tt.attention.weights import AttentionWeights, load_attention_weights
from models.demos.mistral_medium_3_5_128b.tt.rope import build_transformation_mat

__all__ = ["Attention", "AttentionConfig", "AttentionWeights", "ProgramConfig"]


class Attention:
    """GQA attention: 96 Q heads / 8 KV heads / head_dim 128, full causal, full rotary, no biases."""

    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        ccl_manager,
        mesh_config,
        program_config,
        layer_idx: int,
        transformation_mat=None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """
        Args:
            mesh_device: the open mesh.
            config: an :class:`~.config.AttentionConfig`.
            state_dict: ``{q,k,v,o}_proj.weight``. Empty dict => cache-only load.
            ccl_manager: :class:`~...tt.ccl.CCLManager`; required whenever ``mesh_config.tp > 1``.
            mesh_config: :class:`~...tt.config.MeshConfig`.
            program_config: :class:`~.config.ProgramConfig`.
            layer_idx: this layer's index, used to pick the KV cache slot.
            transformation_mat: shared RoPE transformation matrix. Built here if None, but the
                composed model passes one in so all 88 layers share the single allocation.
            weight_dtype: on-device weight dtype (bfloat8_b per the spec).
            tensor_cache_path: directory for the tilized-weight cache, or None.
        """
        assert mesh_config.tp == 1 or ccl_manager is not None, "the o_proj all-reduce needs a CCLManager"

        self.mesh_device = mesh_device
        self.config = config
        self.mesh_config = mesh_config
        self.program_config = program_config
        self.ccl_manager = ccl_manager
        self.layer_idx = layer_idx
        self.transformation_mat = (
            build_transformation_mat(mesh_device) if transformation_mat is None else transformation_mat
        )
        self.weights = load_attention_weights(
            mesh_device,
            config,
            state_dict,
            mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

    def __call__(self, hidden_states, rope_mats, kv_cache=None, user_id: int = 0, cached_len: int = 0):
        """One chunk through the block. See :func:`~.prefill.attention_forward` for the arguments."""
        return attention_forward(
            hidden_states,
            rope_mats,
            weights=self.weights,
            kv_cache=kv_cache,
            config=self.config,
            mesh_config=self.mesh_config,
            mesh_device=self.mesh_device,
            program_config=self.program_config,
            transformation_mat=self.transformation_mat,
            ccl_manager=self.ccl_manager,
            user_id=user_id,
            layer_idx=self.layer_idx,
            cached_len=cached_len,
        )
