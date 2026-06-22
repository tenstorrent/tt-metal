# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Pixtral vision RMSNorm (explicit weight key, ttnn.rms_norm).

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstral2_small.devstral_utils.pixtral_seq_chunk import (
    vision_rms_norm_block_shard_eligible,
    vision_rms_norm_block_shard_memcfg,
    vision_rms_norm_block_shard_program_config,
    vision_rms_norm_gamma_weight,
    vision_rms_norm_memcfg,
    vision_rms_norm_prepare_block_shard_input,
)


def _resolve_weight_state_dict_key(weight_key: str | None, state_dict_prefix: str | None) -> str:
    if weight_key is not None:
        return weight_key
    if state_dict_prefix is None:
        raise ValueError(
            "Provide exactly one of `weight_key` (full key) or `state_dict_prefix` (path without `.weight`)."
        )
    p = state_dict_prefix
    if p.endswith(".weight"):
        return p
    if p.endswith("."):
        return f"{p}weight"
    return f"{p}.weight"


class TtPixtralRMSNorm(LightweightModule):
    # Pixtral RMSNorm (eps=1e-5).

    def __init__(
        self,
        mesh_device,
        state_dict: dict,
        *,
        eps: float = 1e-5,
        weight_key: str | None = None,
        state_dict_prefix: str | None = None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.variance_epsilon = eps

        key = _resolve_weight_state_dict_key(weight_key, state_dict_prefix)
        if key not in state_dict:
            raise KeyError(f"TtPixtralRMSNorm: missing weight key {key!r} in state_dict.")
        pytorch_gamma = state_dict[key]
        if pytorch_gamma.dim() != 1:
            raise ValueError(f"Expected 1D gamma at {key!r}, got shape {tuple(pytorch_gamma.shape)}.")

        self.weight = vision_rms_norm_gamma_weight(
            pytorch_gamma,
            mesh_device,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED),
            dtype=dtype,
        )

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        seq_len = int(hidden_states.shape[-2])
        feature_dim = int(hidden_states.shape[-1])
        grid_x, grid_y = 8, 8

        if vision_rms_norm_block_shard_eligible(seq_len, feature_dim, grid_x, grid_y):
            norm_mem_cfg = vision_rms_norm_block_shard_memcfg(seq_len, feature_dim, grid_x, grid_y)
            program_config = vision_rms_norm_block_shard_program_config(seq_len, feature_dim, grid_x, grid_y)
            hidden_states = vision_rms_norm_prepare_block_shard_input(
                hidden_states, seq_len, feature_dim, grid_x, grid_y
            )
        else:
            norm_mem_cfg = vision_rms_norm_memcfg(seq_len, feature_dim)
            program_config = None
            if hidden_states.memory_config().buffer_type != norm_mem_cfg.buffer_type:
                hidden_states = ttnn.to_memory_config(hidden_states, norm_mem_cfg)

        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.variance_epsilon,
            weight=self.weight,
            program_config=program_config,
            memory_config=norm_mem_cfg,
        )


__all__ = ["TtPixtralRMSNorm"]
