# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-27B SwiGLU MLP implementation for TTNN.

Qwen3_5MLP uses a gated MLP with SiLU activation:
    forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
"""

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearIReplicatedWColSharded


class TTNNQwen35MLP(TTNNModule):
    """TTNN-accelerated SwiGLU MLP for Qwen3.5-27B.

    Contains three TTNNLinear children (gate_proj, up_proj, down_proj).
    Weight preprocessing and device movement are handled recursively
    by the TTNNModule base class through children.
    """

    @property
    def _is_distributed(self):
        """Check if running in distributed mode with CCL manager."""
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if in distributed mode."""
        if not self._is_distributed:
            return tensor
        gathered = ttnn.all_gather(
            tensor,
            dim=-1,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
        )
        ttnn.synchronize_device(self.device)
        return gathered

    @classmethod
    def from_torch(cls, mlp, distributed=True):
        """Create TTNNQwen35MLP from PyTorch Qwen3_5MLP.

        Args:
            mlp: PyTorch Qwen3_5MLP layer with gate_proj, up_proj, down_proj.
            distributed: Use col-sharded weights for multi-device (default True for T3K).

        Returns:
            TTNNQwen35MLP instance.
        """
        new_mlp = cls()
        new_mlp._fallback_torch_layer = mlp

        # Choose linear class: col-sharded for distributed, replicated for single device
        LinearCls = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear

        # Create linear children for each projection
        new_mlp.gate_proj = LinearCls.from_torch(mlp.gate_proj)
        new_mlp.up_proj = LinearCls.from_torch(mlp.up_proj)
        new_mlp.down_proj = LinearCls.from_torch(mlp.down_proj)

        return new_mlp

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass: SwiGLU MLP.

        Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

        Args:
            x: Input tensor [batch, seq_len, hidden_size].

        Returns:
            Output tensor [batch, seq_len, hidden_size].
        """
        # Gate and up projections
        gate_out = self.gate_proj(x)
        gate_out = self._maybe_all_gather(gate_out)
        up_out = self.up_proj(x)
        up_out = self._maybe_all_gather(up_out)

        # SiLU activation on gate output
        gate_activated = ttnn.silu(gate_out)
        ttnn.deallocate(gate_out)

        # Element-wise multiply gate and up
        intermediate = ttnn.multiply(gate_activated, up_out)
        ttnn.deallocate(gate_activated)
        ttnn.deallocate(up_out)

        # Down projection (col-sharded: each device has hidden_size/num_devices)
        # Do NOT all-gather here — decoder layer handles col-sharded residual add
        output = self.down_proj(intermediate)
        ttnn.deallocate(intermediate)

        return output
