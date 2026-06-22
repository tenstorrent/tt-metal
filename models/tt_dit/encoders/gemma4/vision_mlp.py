# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 vision MLP — plain gated MLP (gate * up → down) used inside the
vision encoder layer alongside the attention.

Mirrors ``Gemma4VisionMLP`` (inherits ``Gemma3MLP``):

    out = down_proj( gelu_tanh(gate_proj(x)) * up_proj(x) )

The vision config uses ``intermediate_size=4304`` which is NOT tile-aligned
(4304 / 32 = 134.5). We pad to the next tile-aligned multiple per TP factor at
weight load time and zero-fill the trailing channels.
"""

from __future__ import annotations

import torch

import ttnn

from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module
from ...parallel.config import DiTParallelConfig

TILE = ttnn.TILE_SIZE


def _padded_intermediate(intermediate_size: int, tp_factor: int) -> int:
    """Round ``intermediate_size`` up so each TP rank gets a tile-aligned slice."""
    per_dev = (intermediate_size + tp_factor - 1) // tp_factor
    per_dev_aligned = ((per_dev + TILE - 1) // TILE) * TILE
    return per_dev_aligned * tp_factor


class Gemma4VisionMLP(Module):
    """Vision gated MLP with intermediate-dim padding."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        assert hidden_size % TILE == 0

        tp_factor = parallel_config.tensor_parallel.factor
        intermediate_padded = _padded_intermediate(intermediate_size, tp_factor)
        assert (intermediate_padded // tp_factor) % TILE == 0

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_padded = intermediate_padded
        self.parallel_config = parallel_config
        self.mesh_device = mesh_device

        col_kwargs = dict(
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.gate_proj = ColParallelLinear(hidden_size, intermediate_padded, activation_fn="gelu_tanh", **col_kwargs)
        self.up_proj = ColParallelLinear(hidden_size, intermediate_padded, **col_kwargs)
        self.down_proj = RowParallelLinear(
            intermediate_padded,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Pad gate/up output dim and down input dim with zeros."""
        if self.intermediate_padded == self.intermediate_size:
            return
        pad_n = self.intermediate_padded - self.intermediate_size

        for name in ("gate_proj", "up_proj"):
            w = state.get(f"{name}.weight")
            if w is None:
                continue
            # HF weight shape: [intermediate, hidden] → pad dim 0 with zeros.
            state[f"{name}.weight"] = torch.nn.functional.pad(w, (0, 0, 0, pad_n))

        w = state.get("down_proj.weight")
        if w is not None:
            # HF weight shape: [hidden, intermediate] → pad dim 1 with zeros.
            state["down_proj.weight"] = torch.nn.functional.pad(w, (0, pad_n))

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[B, S, hidden_size] → [B, S, hidden_size]."""
        gate = self.gate_proj(x, parallel_config=self.parallel_config)
        up = self.up_proj(x, parallel_config=self.parallel_config)
        gated = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self.down_proj(gated)
        ttnn.deallocate(gated)
        return out
