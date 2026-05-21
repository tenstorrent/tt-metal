# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Pixtral vision RMSNorm (explicit weight key, ttnn.rms_norm).

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import pad_by_zero
from models.experimental.devstarl2_small.devstral_utils.pixtral_seq_chunk import vision_rms_norm_memcfg


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

        self.weight = pad_by_zero(
            pytorch_gamma,
            mesh_device,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED),
            dtype,
        )[0]

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        norm_mem_cfg = vision_rms_norm_memcfg(
            int(hidden_states.shape[-2]),
            int(hidden_states.shape[-1]),
        )
        if hidden_states.memory_config().buffer_type != norm_mem_cfg.buffer_type:
            hidden_states = ttnn.to_memory_config(hidden_states, norm_mem_cfg)
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.variance_epsilon,
            weight=self.weight,
            memory_config=norm_mem_cfg,
        )


__all__ = ["TtPixtralRMSNorm"]
