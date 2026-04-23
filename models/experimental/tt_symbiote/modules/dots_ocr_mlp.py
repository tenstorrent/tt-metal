# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIColShardedWAllReduced,
    TTNNLinearIReplicatedWColSharded,
)


class TTNNDotsOCRMLP(TTNNModule):
    @classmethod
    def from_torch(cls, torch_mlp):
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_mlp
        tt_module.gate_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.gate_proj)
        tt_module.up_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.up_proj)
        tt_module.down_proj = TTNNLinearIColShardedWAllReduced.from_torch(torch_mlp.down_proj)
        return tt_module

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)

        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_up = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        output = self.down_proj(gate_up)
        ttnn.deallocate(gate_up)

        return output
