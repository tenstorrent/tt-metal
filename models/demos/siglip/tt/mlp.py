# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtSiglipMLP(LightweightModule):
    def __init__(self, mesh_device, hidden_size, intermediate_size, state_dict, state_dict_prefix, dtype):
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Load weights from state dict
        prefix = f"{state_dict_prefix}." if state_dict_prefix else ""
        fc1_weight = state_dict[f"{prefix}fc1.weight"].transpose(-1, -2)
        fc2_weight = state_dict[f"{prefix}fc2.weight"].transpose(-1, -2)

        # Convert to TT tensors
        self.fc1_weight = ttnn.from_torch(
            fc1_weight,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.fc2_weight = ttnn.from_torch(
            fc2_weight,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def gelu_tanh_activation(self, x):
        """
        based on https://github.com/google/gemma_pytorch/blob/main/gemma/siglip_vision/siglip_vision_model.py#L93
        """

        sqrt_2_over_pi = 0.7978845608
        coeff = 0.044715

        x_cubed = ttnn.pow(x, 3.0)
        inner = ttnn.add(x, ttnn.mul(x_cubed, coeff))
        scaled = ttnn.mul(inner, sqrt_2_over_pi)
        tanh_result = ttnn.tanh(scaled)
        one_plus_tanh = ttnn.add(tanh_result, 1.0)

        result = ttnn.mul(x, one_plus_tanh)
        result = ttnn.mul(result, 0.5)

        return result

    def forward(self, x):
        fc1_out = ttnn.linear(
            x,
            self.fc1_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        gelu_out = self.gelu_tanh_activation(fc1_out)
        ttnn.deallocate(fc1_out)

        fc2_out = ttnn.linear(
            gelu_out,
            self.fc2_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gelu_out)

        return fc2_out
