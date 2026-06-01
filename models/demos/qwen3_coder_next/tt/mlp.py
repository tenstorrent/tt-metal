# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP for Qwen3-Coder-Next (shared expert)."""

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMLP(LightweightModule):
    def __init__(self, device, state_dict, layer_idx, config, dtype=ttnn.bfloat16, weights_dtype=None):
        super().__init__()
        self.device = device
        if weights_dtype is None:
            weights_dtype = getattr(config, "weights_dtype", ttnn.bfloat8_b)
        prefix = f"model.layers.{layer_idx}.mlp"

        gate_w = state_dict[f"{prefix}.gate_proj.weight"].T.contiguous()
        up_w = state_dict[f"{prefix}.up_proj.weight"].T.contiguous()
        down_w = state_dict[f"{prefix}.down_proj.weight"].T.contiguous()

        self.gate_proj_w = ttnn.from_torch(
            gate_w.unsqueeze(0).unsqueeze(0),
            dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.up_proj_w = ttnn.from_torch(
            up_w.unsqueeze(0).unsqueeze(0),
            dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.down_proj_w = ttnn.from_torch(
            down_w.unsqueeze(0).unsqueeze(0),
            dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )

    def forward(self, x):
        gate = ttnn.linear(x, self.gate_proj_w)
        gate = ttnn.silu(gate)
        up = ttnn.linear(x, self.up_proj_w)
        hidden = ttnn.mul(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        output = ttnn.linear(hidden, self.down_proj_w)
        ttnn.deallocate(hidden)
        return output
