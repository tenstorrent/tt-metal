# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 Text MLP implementation for TTNN."""

import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIColShardedWAllReduced,
    TTNNLinearIReplicatedWColSharded,
)


class TTNNGemma4TextMLP(TTNNModule):
    """TTNN implementation of the Gemma 4 31B-it dense MLP.

    Architecture (fused gate-up):
        gate_up = fused_gate_up_proj(x)       # [B, S, 5376] -> [B, S, 43008]
        gate = gate_up[:, :, :21504]           # slice
        up = gate_up[:, :, 21504:]             # slice
        output = down_proj(gelu(gate) * up)    # [B, S, 21504] -> [B, S, 5376]
    """

    @classmethod
    def from_torch(cls, torch_mlp):
        """Create a TTNNGemma4TextMLP from a PyTorch Gemma4TextMLP layer."""
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_mlp

        # Store intermediate_size for slice boundary in forward()
        tt_module.intermediate_size = torch_mlp.gate_proj.out_features  # 21504

        # Fused gate-up projection: concatenate gate_proj and up_proj weights
        # into a single [2*intermediate_size, hidden_size] weight matrix.
        # Single matmul + all_reduce replaces two separate matmul + all_reduce chains.
        gate_weight = torch_mlp.gate_proj.weight.data.clone()  # [21504, 5376]
        up_weight = torch_mlp.up_proj.weight.data.clone()  # [21504, 5376]
        fused_weight = torch.cat([gate_weight, up_weight], dim=0)  # [43008, 5376]

        fused_linear = torch.nn.Linear(
            torch_mlp.gate_proj.in_features,  # 5376
            2 * tt_module.intermediate_size,  # 43008
            bias=False,
        )
        fused_linear.weight.data = fused_weight

        tt_module.fused_gate_up_proj = TTNNLinearIColShardedWAllReduced.from_torch(fused_linear)

        # Keep individual proj references as None (fused into fused_gate_up_proj)
        tt_module.gate_proj = None
        tt_module.up_proj = None

        # Down projection unchanged
        tt_module.down_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.down_proj)

        return tt_module

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hidden_states.dtype != ttnn.bfloat16:
            hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Fused gate + up projection (single matmul + all_reduce)
        gate_up = self.fused_gate_up_proj(hidden_states)

        # Slice into gate and up halves
        gate = ttnn.slice(gate_up, [0, 0, 0], [batch_size, seq_len, self.intermediate_size])
        up = ttnn.slice(gate_up, [0, 0, self.intermediate_size], [batch_size, seq_len, 2 * self.intermediate_size])
        ttnn.deallocate(gate_up)

        # GeLU activation on gate path
        gate = ttnn.gelu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Element-wise multiply gate and up
        gate_up_mul = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Down projection
        output = self.down_proj(gate_up_mul)
        ttnn.deallocate(gate_up_mul)

        return output
