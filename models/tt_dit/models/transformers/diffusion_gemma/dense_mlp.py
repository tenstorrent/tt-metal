# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dense gated MLP used inside each DiffusionGemma encoder/decoder layer alongside
the MoE block.

Reference: ``DiffusionGemmaText4MLP`` (mirrors Gemma3MLP):
    out = down_proj( gelu_tanh(gate_proj(x)) * up_proj(x) )

Tile alignment note: ``intermediate_size = 2112`` only divides cleanly into 32-tile
multiples for TP ∈ {1, 2}. Higher TP factors need a padding/replication wrapper.
"""

from __future__ import annotations

import ttnn

from ....layers.linear import ColParallelLinear, RowParallelLinear
from ....layers.module import Module
from ....parallel.config import DiTParallelConfig

TILE = ttnn.TILE_SIZE


class DiffusionGemmaDenseMLP(Module):
    """Plain gated MLP — no surrounding norms (caller applies them)."""

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

        tp_factor = parallel_config.tensor_parallel.factor
        assert (intermediate_size // tp_factor) % TILE == 0, (
            f"intermediate_size ({intermediate_size}) / tp_factor ({tp_factor}) " f"must be tile-aligned ({TILE})."
        )
        assert hidden_size % TILE == 0

        self.parallel_config = parallel_config
        self.mesh_device = mesh_device

        col_kwargs = dict(
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.gate_proj = ColParallelLinear(hidden_size, intermediate_size, activation_fn="gelu_tanh", **col_kwargs)
        self.up_proj = ColParallelLinear(hidden_size, intermediate_size, **col_kwargs)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Replicated [B, S, hidden_size] → replicated [B, S, hidden_size]."""
        gate = self.gate_proj(x, parallel_config=self.parallel_config)
        up = self.up_proj(x, parallel_config=self.parallel_config)
        gated = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self.down_proj(gated)
        ttnn.deallocate(gated)
        return out
