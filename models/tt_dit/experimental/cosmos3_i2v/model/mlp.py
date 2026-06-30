# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native Cosmos3VLTextMLP (SwiGLU) with tensor parallelism.

Mirrors the reference `Cosmos3VLTextMLP`:

    down_proj(silu(gate_proj(x)) * up_proj(x))

Implementation collapses `gate_proj` + `up_proj` into a single fused
`ColParallelLinear` with `activation_fn="swiglu"`. The fused linear's
matmul produces `[up_part | gate_part]` along the last dim, and the
activation kernel does `t * silu(gate)` where `t` = first half (= up),
`gate` = second half (= gate). To get that layout, the prepare-state
hook concatenates `up_proj.weight` first, then `gate_proj.weight`.

Sharding:
  - `fused_gate_up`: ColParallelLinear, sharded on `out_features` (TP).
    Each chip holds `intermediate_size / tp` of the fused output.
    After the in-kernel swiglu activation, the per-chip output is
    `[1, 1, N, intermediate_size / tp]`.
  - `down_proj`: RowParallelLinear, sharded on `in_features` (TP).
    Each chip computes a partial-sum `[1, 1, N, hidden_size]`; the kernel
    reduce-scatters to `[1, 1, N, hidden_size / tp]`. We then all-gather
    on the TP axis to restore the replicated layout the tt-symbiote-
    wrapped caller expects.

Input/output contract (preserved across tp factors): replicated
`[1, 1, N, hidden_size]` ttnn tensors on both ends.

Single-device convenience: `parallel_config=None` defaults to the
degenerate (tp=1, sp=1) config — same pattern as `Cosmos3JointAttention`.
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.linear import ColParallelLinear, RowParallelLinear
from ....layers.module import Module
from ....parallel.config import DiTParallelConfig, ParallelFactor


def _default_parallel_config() -> DiTParallelConfig:
    return DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        tensor_parallel=ParallelFactor(1, 1),
        sequence_parallel=ParallelFactor(1, 0),
    )


class Cosmos3VLTextMLP(Module):
    """SwiGLU MLP for the Cosmos3 MoT decoder layer."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig | None = None,
        ccl_manager=None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        if parallel_config is None:
            parallel_config = _default_parallel_config()

        tp_factor = parallel_config.tensor_parallel.factor
        tp_axis = parallel_config.tensor_parallel.mesh_axis

        if intermediate_size % tp_factor != 0:
            msg = f"intermediate_size ({intermediate_size}) must be divisible by tp_factor ({tp_factor})"
            raise ValueError(msg)
        if tp_factor > 1 and ccl_manager is None:
            msg = "ccl_manager is required when tp_factor > 1"
            raise ValueError(msg)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        col_kw = {
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": tp_axis,
            "ccl_manager": ccl_manager,
            "dtype": dtype,
        }
        row_kw = {
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": tp_axis,
            "ccl_manager": ccl_manager,
            "dtype": dtype,
        }

        # Fused gate+up: ColParallelLinear doubles out_features internally for swiglu and
        # applies `t * silu(gate)` after the matmul. We pass intermediate_size; the layer
        # allocates a 2*intermediate_size weight slot.
        self.fused_gate_up = ColParallelLinear(
            hidden_size,
            intermediate_size,
            activation_fn="swiglu",
            **col_kw,
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, **row_kw)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Combine `gate_proj.weight` and `up_proj.weight` into the fused layout.

        Reference state keys (Cosmos3VLTextMLP from
        `reference/transformer_cosmos3.py`):
          - gate_proj.weight  : [intermediate_size, hidden_size]
          - up_proj.weight    : [intermediate_size, hidden_size]
          - down_proj.weight  : [hidden_size, intermediate_size]

        ColParallelLinear's `activation_fn="swiglu"` consumes a single
        fused weight where the doubled-out dim is `[up | gate]` (up first
        because the chunk+activate kernel does `first * silu(second)`).
        """
        gate_weight = state.pop("gate_proj.weight", None)
        up_weight = state.pop("up_proj.weight", None)
        if gate_weight is None or up_weight is None:
            return  # Let the parent load loop report the missing keys.
        # Concatenate on the OUT-features dim (dim 0 of the torch [out, in] layout).
        # Order is up-first, gate-second to satisfy ColParallelLinear's swiglu kernel.
        state["fused_gate_up.weight"] = torch.cat([up_weight, gate_weight], dim=0)

    def _tp_factor(self) -> int:
        return self.parallel_config.tensor_parallel.factor

    def _tp_axis(self) -> int:
        return self.parallel_config.tensor_parallel.mesh_axis

    def forward(self, x_11NH: ttnn.Tensor) -> ttnn.Tensor:
        """Replicated `[1, 1, N, hidden_size]` → replicated `[1, 1, N, hidden_size]`."""
        # Fused gate+up + swiglu: per chip output is [1, 1, N, intermediate_size / tp].
        h = self.fused_gate_up(x_11NH)
        # RowParallelLinear: partial-sum matmul + reduce-scatter on TP axis.
        # Result per chip: [1, 1, N, hidden_size / tp].
        out_fractured = self.down_proj(h)
        ttnn.deallocate(h)

        if self._tp_factor() <= 1:
            return out_fractured

        # All-gather on TP axis → replicated [1, 1, N, hidden_size] for the tt-symbiote caller.
        # NOTE: `out_fractured` is a persistent ping-pong buffer owned by ccl_manager — do NOT
        # `ttnn.deallocate` it (see the matching note in attention.py).
        out_replicated = self.ccl_manager.all_gather_persistent_buffer(out_fractured, dim=3, mesh_axis=self._tp_axis())
        return out_replicated
