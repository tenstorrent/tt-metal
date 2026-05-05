# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 Text MLP implementation for TTNN."""

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearGemma4IColShardedWRowSharded,
)


def gelu_pytorch_tanh(x: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.gelu(x, fast_and_approximate_mode=True)


class TTNNGemma4TextMLP(TTNNModule):
    """TTNN implementation of the Gemma 4 31B-it dense MLP.

    Architecture (separate gate and up projections, col-sharded activations):
        gate = gate_proj(x)                   # [B, S, 672/dev] -> RS -> [B, S, 2688/dev]
        up = up_proj(x)                       # [B, S, 672/dev] -> RS -> [B, S, 2688/dev]
        output = down_proj(gelu(gate) * up)   # [B, S, 2688/dev] -> RS -> [B, S, 672/dev]

    Uses TTNNLinearIColShardedWRowSharded (matmul + reduce_scatter) for all three
    projections. No all_gather is needed because all intermediate ops (gelu, multiply)
    are element-wise and operate correctly on per-device col-shards.
    """

    @classmethod
    def from_torch(cls, torch_mlp):
        """Create a TTNNGemma4TextMLP from a PyTorch Gemma4TextMLP layer."""
        tt_module = cls()
        tt_module._fallback_torch_layer = torch_mlp

        tt_module.intermediate_size = torch_mlp.gate_proj.out_features  # 21504

        tt_module.gate_proj = TTNNLinearGemma4IColShardedWRowSharded.from_torch(torch_mlp.gate_proj)
        tt_module.up_proj = TTNNLinearGemma4IColShardedWRowSharded.from_torch(torch_mlp.up_proj)

        tt_module.fused_gate_up_proj = None

        tt_module.down_proj = TTNNLinearGemma4IColShardedWRowSharded.from_torch(torch_mlp.down_proj)

        return tt_module

    def move_weights_to_device_impl(self):
        """Move weights to device and enable fp32 accumulation on projections."""
        super().move_weights_to_device_impl()
        linear_compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.gate_proj.compute_kernel_config = linear_compute_config
        self.up_proj.compute_kernel_config = linear_compute_config
        self.down_proj.compute_kernel_config = linear_compute_config

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hidden_states.dtype != ttnn.bfloat16:
            hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)

        gate = gelu_pytorch_tanh(gate)

        gate_up_mul = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        output = self.down_proj(gate_up_mul)
        ttnn.deallocate(gate_up_mul)

        return output
