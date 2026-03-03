# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
SwiGLU MLP implementation for Qwen3-TTS.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class MLP(LightweightModule):
    """
    SwiGLU MLP for Qwen3-TTS.

    Architecture: down_proj(silu(gate_proj(x)) * up_proj(x))

    This is a simplified implementation for single device (N150/N300).
    """

    def __init__(
        self,
        device,
        hidden_size: int,
        intermediate_size: int,
        state_dict: dict,
        layer_prefix: str,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        def get_cache_name(name):
            if weight_cache_path is None:
                return None
            return weight_cache_path / f"{layer_prefix}_{name}".replace(".", "_")

        # Load and transpose weights (for matmul: x @ W.T -> x @ W where W is transposed)
        gate_proj_weight = torch.transpose(state_dict[f"{layer_prefix}.mlp.gate_proj.weight"], -2, -1)
        up_proj_weight = torch.transpose(state_dict[f"{layer_prefix}.mlp.up_proj.weight"], -2, -1)
        down_proj_weight = torch.transpose(state_dict[f"{layer_prefix}.mlp.down_proj.weight"], -2, -1)

        # Make weights 4D for TTNN [1, 1, in_features, out_features]
        gate_proj_weight = gate_proj_weight.unsqueeze(0).unsqueeze(0)
        up_proj_weight = up_proj_weight.unsqueeze(0).unsqueeze(0)
        down_proj_weight = down_proj_weight.unsqueeze(0).unsqueeze(0)

        self.gate_proj = ttnn.as_tensor(
            gate_proj_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("gate_proj"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.up_proj = ttnn.as_tensor(
            up_proj_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("up_proj"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("down_proj"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply SwiGLU MLP.

        Args:
            x: Input tensor of shape [batch, 1, seq_len, hidden_size]

        Returns:
            Output tensor of same shape
        """
        seq_len = x.shape[-2]

        # Reshape for large sequences to fit on device
        if seq_len >= 1024:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # Gate projection with SiLU activation
        gate_out = ttnn.linear(
            x,
            self.gate_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Up projection
        up_out = ttnn.linear(
            x,
            self.up_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # SwiGLU: silu(gate) * up
        hidden = ttnn.mul(
            gate_out,
            up_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(gate_out)
        ttnn.deallocate(up_out)

        # Down projection
        output = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(hidden)

        # Reshape back if needed
        if seq_len >= 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        return output
