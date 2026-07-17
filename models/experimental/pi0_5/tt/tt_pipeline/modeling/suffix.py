# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN suffix embedding for the pi0.5 action expert (streamed-denoise port).

VENDORED from ``tt_symbiote.models.pi05.modeling_pi05_suffix`` with config import rewired
to the target ``common/configs.py``. The pi0.5 suffix embeds the noisy action chunk
(``embed_actions``) and the flow-matching timestep (``embed_adarms_cond``:
sincos -> Linear -> silu -> Linear -> silu) and projects the expert output back
(``project_output``). ZERO tt_symbiote imports.
"""
from __future__ import annotations

from typing import Optional, Tuple

import ttnn

from .._module import DeviceArch, StatelessTTNNModule, run_on_devices
from .common import create_sinusoidal_pos_embedding
from .gemma import _linear_weight_to_tt

from models.experimental.pi0_5.common.configs import SuffixConfig
from models.experimental.pi0_5.tt.tile_config import from_torch_pi05

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

__all__ = ["TTNNPi05SuffixEmbedding"]

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _bias_to_tt(b: Optional["ttnn.Tensor"]) -> Optional[ttnn.Tensor]:
    if b is None:
        return None
    return from_torch_pi05(b.reshape(1, -1).contiguous(), dtype=ttnn.bfloat16)


class TTNNPi05SuffixEmbedding(StatelessTTNNModule):
    """pi0.5 suffix embedding (action + timestep) for the action expert."""

    @classmethod
    def from_torch(cls, suffix, config: SuffixConfig) -> "TTNNPi05SuffixEmbedding":
        assert config.pi05, "TTNNPi05SuffixEmbedding requires config.pi05=True"
        new = cls()
        new._bypass_tensor_wrapping = True
        new._fallback_torch_layer = suffix
        new._config = config
        new._action_in_w = suffix.action_in_weight
        new._action_in_b = suffix.action_in_bias
        new._action_out_w = suffix.action_out_weight
        new._action_out_b = suffix.action_out_bias
        new._time_mlp_in_w = suffix.time_mlp_in_weight
        new._time_mlp_in_b = suffix.time_mlp_in_bias
        new._time_mlp_out_w = suffix.time_mlp_out_weight
        new._time_mlp_out_b = suffix.time_mlp_out_bias
        new._expert_width = config.expert_width
        return new

    def preprocess_weights_impl(self):
        self.tt_action_in_w = _linear_weight_to_tt(self._action_in_w)
        self.tt_action_in_b = _bias_to_tt(self._action_in_b)
        self.tt_action_out_w = _linear_weight_to_tt(self._action_out_w)
        self.tt_action_out_b = _bias_to_tt(self._action_out_b)
        self.tt_time_mlp_in_w = _linear_weight_to_tt(self._time_mlp_in_w)
        self.tt_time_mlp_in_b = _bias_to_tt(self._time_mlp_in_b)
        self.tt_time_mlp_out_w = _linear_weight_to_tt(self._time_mlp_out_w)
        self.tt_time_mlp_out_b = _bias_to_tt(self._time_mlp_out_b)

    def move_weights_to_device_impl(self):
        self.tt_action_in_w = ttnn.to_device(self.tt_action_in_w, self.device, memory_config=_DRAM)
        self.tt_action_out_w = ttnn.to_device(self.tt_action_out_w, self.device, memory_config=_DRAM)
        self.tt_time_mlp_in_w = ttnn.to_device(self.tt_time_mlp_in_w, self.device, memory_config=_DRAM)
        self.tt_time_mlp_out_w = ttnn.to_device(self.tt_time_mlp_out_w, self.device, memory_config=_DRAM)
        if self.tt_action_in_b is not None:
            self.tt_action_in_b = ttnn.to_device(self.tt_action_in_b, self.device, memory_config=_DRAM)
        if self.tt_action_out_b is not None:
            self.tt_action_out_b = ttnn.to_device(self.tt_action_out_b, self.device, memory_config=_DRAM)
        if self.tt_time_mlp_in_b is not None:
            self.tt_time_mlp_in_b = ttnn.to_device(self.tt_time_mlp_in_b, self.device, memory_config=_DRAM)
        if self.tt_time_mlp_out_b is not None:
            self.tt_time_mlp_out_b = ttnn.to_device(self.tt_time_mlp_out_b, self.device, memory_config=_DRAM)

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def embed_actions(self, noisy_actions: ttnn.Tensor) -> ttnn.Tensor:
        """(B, action_horizon, action_dim) -> (B, action_horizon, expert_width)."""
        return ttnn.linear(noisy_actions, self.tt_action_in_w, bias=self.tt_action_in_b, memory_config=_L1)

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def embed_adarms_cond(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        """sincos(t) -> time_mlp_in -> silu -> time_mlp_out -> silu -> (B, expert_width)."""
        sincos = create_sinusoidal_pos_embedding(
            timestep, self._expert_width, self.device, min_period=4e-3, max_period=4.0
        )
        x = ttnn.linear(sincos, self.tt_time_mlp_in_w, bias=self.tt_time_mlp_in_b, memory_config=_L1)
        ttnn.deallocate(sincos)
        x = ttnn.silu(x, memory_config=_L1)
        x = ttnn.linear(x, self.tt_time_mlp_out_w, bias=self.tt_time_mlp_out_b, memory_config=_L1)
        return ttnn.silu(x, memory_config=_L1)

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def project_output(self, expert_output: ttnn.Tensor) -> ttnn.Tensor:
        """(B, action_horizon, expert_width) -> (B, action_horizon, action_dim)."""
        return ttnn.linear(expert_output, self.tt_action_out_w, bias=self.tt_action_out_b, memory_config=_L1)

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def embed_suffix(self, noisy_actions: ttnn.Tensor, timestep: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        suffix_embs = self.embed_actions(noisy_actions)
        adarms_cond = self.embed_adarms_cond(timestep)
        return suffix_embs, adarms_cond

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def forward(self, noisy_actions: ttnn.Tensor, timestep: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        return self.embed_suffix(noisy_actions, timestep)
