# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Shared/Dense MLP with GeGLU activation.

Each decoder layer has BOTH a shared MLP and routed MoE experts.
Architecture: down_proj(GELU(gate_proj(x)) * up_proj(x))
intermediate_size = 2112, no bias.

HF weight shapes:
  gate_proj.weight: [intermediate_size, hidden_size] = [2112, 2816]
  up_proj.weight:   [intermediate_size, hidden_size] = [2112, 2816]
  down_proj.weight: [hidden_size, intermediate_size] = [2816, 2112]
"""

import ttnn
from models.demos.gemma4.utils.general_utils import get_cache_file_name


class SharedMLP:
    def __init__(self, mesh_device, hf_config, state_dict, mesh_config, dtype=ttnn.bfloat8_b, tensor_cache_path=None):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        if state_dict:
            # gate_proj.weight: [intermediate_size, hidden_size] -> transpose for matmul
            gate_proj_weight = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            up_proj_weight = state_dict["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            gate_proj_weight = None
            up_proj_weight = None
            down_proj_weight = None

        self.gate_proj = ttnn.as_tensor(
            gate_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "gate_proj.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj = ttnn.as_tensor(
            up_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "up_proj.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device, TILE_LAYOUT

        Returns:
            output: [1, 1, seq_len, hidden_size] on device
        """
        # gate = GELU(x @ gate_proj)
        gate = ttnn.linear(hidden_states, self.gate_proj)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True)

        # up = x @ up_proj
        up = ttnn.linear(hidden_states, self.up_proj)

        # hidden = gate * up
        hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        # output = hidden @ down_proj
        output = ttnn.linear(hidden, self.down_proj)
        hidden.deallocate(True)

        return output
