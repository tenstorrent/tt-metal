# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearLLamaIColShardedWAllReducedFusedGateUp,
    TTNNLinearLLamaIReplicatedWColSharded,
)


class TTNNDotsOCRMLP(TTNNModule):
    """SwiGLU MLP with fused gate+up projection.

    Original (two separate matmuls + two CCL all-reduces):
        gate = gate_proj(x)              # matmul + reduce_scatter + all_gather
        up   = up_proj(x)                # matmul + reduce_scatter + all_gather
        out  = down_proj(silu(gate) * up)

    Fused (one matmul + one CCL all-reduce, mirrors gemma4_mlp.py):
        gate_up = fused_gate_up_proj(x)  # matmul (2x out) + reduce_scatter + all_gather
        gate    = gate_up[..., :I]
        up      = gate_up[..., I:]
        out     = down_proj(silu(gate) * up)

    Fusing halves the number of CCL launches in MLP. Each all-reduce on N300 has
    ~120 us of latency overhead independent of payload size, so combining the
    two avoids that overhead. The matmul itself becomes 2x larger but the FLOPs
    are identical to the two unfused matmuls; bandwidth-bound decode therefore
    breaks roughly even on raw matmul time and saves the CCL latency outright.
    """

    @classmethod
    def from_torch(cls, torch_mlp):
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_mlp

        # Fuse gate_proj + up_proj into a single column-sharded all-reduced linear.
        tt_module.intermediate_size = torch_mlp.gate_proj.out_features

        tt_module.fused_gate_up_proj = TTNNLinearLLamaIColShardedWAllReducedFusedGateUp.from_two_torch(
            torch_mlp.gate_proj, torch_mlp.up_proj
        )
        # Keep individual proj refs for compatibility with anything that
        # introspects the module; they are no longer used in forward.
        tt_module.gate_proj = None
        tt_module.up_proj = None

        tt_module.down_proj = TTNNLinearLLamaIReplicatedWColSharded.from_torch(torch_mlp.down_proj)
        return tt_module

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        I = self.intermediate_size

        # Single matmul + single all-reduce produces concatenated [gate | up].
        gate_up = self.fused_gate_up_proj(hidden_states)

        # Slice into gate / up halves along the last dim. ttnn.slice on a TILE
        # tensor is a metadata + small copy op, far cheaper than a second CCL.
        gate = ttnn.slice(gate_up, [0, 0, 0], [batch_size, seq_len, I])
        up = ttnn.slice(gate_up, [0, 0, I], [batch_size, seq_len, 2 * I])
        ttnn.deallocate(gate_up)

        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_up_mul = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        output = self.down_proj(gate_up_mul)
        ttnn.deallocate(gate_up_mul)

        return output
