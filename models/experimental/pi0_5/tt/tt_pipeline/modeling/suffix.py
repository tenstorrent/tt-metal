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
from .bs import matmul_pcfg
from .common import create_sinusoidal_pos_embedding
from .gemma import _linear_weight_to_tt

from models.experimental.pi0_5.common.configs import SuffixConfig
from models.experimental.pi0_5.tt.tile_config import TILE_WIDTH, from_torch_pi05

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

__all__ = ["TTNNPi05SuffixEmbedding"]

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _bias_to_tt(b: Optional["ttnn.Tensor"]) -> Optional[ttnn.Tensor]:
    if b is None:
        return None
    return from_torch_pi05(b.reshape(1, -1).contiguous(), dtype=ttnn.bfloat16)


def _linear_kwargs(x: ttnn.Tensor, w: ttnn.Tensor) -> dict:
    """Extra ``ttnn.linear`` kwargs needed for a tiny-tile activation.

    Two non-tile-aware paths have to be avoided at a tiny tile:
      * With NO program_config/core_grid, matmul picks the generic MatmulMultiCore factory, which
        rejects a tiny outer tile outright ("non-optimized program config does not support tiny
        tile").
      * Passing only ``core_grid`` routes to the 1D-systolic AUTO config generator, which computes
        ``m_tiles = (batch * M) / ttnn::TILE_SIZE`` against the global 32 -- so M=16 fails its
        "must be a multiple of tile size" check (and would give m_tiles == 0).
    So build the 1D-mcast program config explicitly with tile-aware m_tiles; the matmul FACTORY is
    tile-aware, only the auto-generator is not. Returns {} at the standard tile so the 32x32 path is
    byte-identical.
    """
    tile_h = int(x.get_tile().tile_shape[0])
    if tile_h == 32:
        return {}
    grid = x.device().compute_with_storage_grid_size()
    m_tiles = max(1, (int(x.padded_shape[-2]) + tile_h - 1) // tile_h)
    k_tiles = max(1, int(x.padded_shape[-1]) // TILE_WIDTH)
    n_tiles = max(1, int(w.padded_shape[-1]) // TILE_WIDTH)
    pc = matmul_pcfg(m_tiles, k_tiles, n_tiles, grid.x, grid.y)
    return {"program_config": pc} if pc is not None else {}


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
        return ttnn.linear(
            noisy_actions, self.tt_action_in_w, bias=self.tt_action_in_b, memory_config=_L1,
            **_linear_kwargs(noisy_actions, self.tt_action_in_w),
        )

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def embed_adarms_cond(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        """sincos(t) -> time_mlp_in -> silu -> time_mlp_out -> silu -> (B, expert_width)."""
        sincos = create_sinusoidal_pos_embedding(
            timestep, self._expert_width, self.device, min_period=4e-3, max_period=4.0
        )
        x = ttnn.linear(
            sincos, self.tt_time_mlp_in_w, bias=self.tt_time_mlp_in_b, memory_config=_L1,
            **_linear_kwargs(sincos, self.tt_time_mlp_in_w),
        )
        ttnn.deallocate(sincos)
        x = ttnn.silu(x, memory_config=_L1)
        x = ttnn.linear(
            x, self.tt_time_mlp_out_w, bias=self.tt_time_mlp_out_b, memory_config=_L1,
            **_linear_kwargs(x, self.tt_time_mlp_out_w),
        )
        return ttnn.silu(x, memory_config=_L1)

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def project_output(self, expert_output: ttnn.Tensor) -> ttnn.Tensor:
        """(B, action_horizon, expert_width) -> (B, action_horizon, action_dim)."""
        return ttnn.linear(
            expert_output, self.tt_action_out_w, bias=self.tt_action_out_b, memory_config=_L1,
            **_linear_kwargs(expert_output, self.tt_action_out_w),
        )

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def embed_suffix(self, noisy_actions: ttnn.Tensor, timestep: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        suffix_embs = self.embed_actions(noisy_actions)
        adarms_cond = self.embed_adarms_cond(timestep)
        return suffix_embs, adarms_cond

    @run_on_devices(DeviceArch.P150, DeviceArch.BHGLX)
    def forward(self, noisy_actions: ttnn.Tensor, timestep: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        return self.embed_suffix(noisy_actions, timestep)
