# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SwiGLU MLP for Molmo2 Text Model.

Implements SwiGLU activation:
    output = ff_out(silu(gate) * up)

Where gate and up are computed from a fused ff_proj:
    ff_proj = [gate; up]  (concatenated along output dim)

Dimensions:
    - hidden_dim: 4096
    - intermediate_dim: 11008 (per gate/up branch)
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TextMLP(LightweightModule):
    """
    SwiGLU MLP for Molmo2 text model.

    Architecture:
        - ff_proj: fused gate+up projection [hidden_dim -> 2*intermediate_dim]
        - ff_out: down projection [intermediate_dim -> hidden_dim]
        - Output: ff_out(silu(gate) * up)
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        layer_num: int,
        hidden_dim: int = 4096,
        intermediate_dim: int = 12288,
        weight_cache_path=None,
        state_dict_prefix: str = "model.transformer.blocks",
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize TextMLP.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            layer_num: Layer number (0-35)
            hidden_dim: Hidden dimension (4096)
            intermediate_dim: Intermediate dimension per branch (11008)
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dtype = dtype

        # Layer prefix
        prefix = f"{state_dict_prefix}.{layer_num}.mlp"

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{prefix}.{name}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

        # Load fused ff_proj: [hidden_dim -> 2*intermediate_dim]
        ff_proj = state_dict[f"{prefix}.ff_proj.weight"]

        # Split into gate and up projections
        # ff_proj shape: [2*intermediate_dim, hidden_dim]
        gate_proj = ff_proj[:intermediate_dim, :]
        up_proj = ff_proj[intermediate_dim:, :]

        # Transpose for TTNN linear
        gate_proj_t = torch.transpose(gate_proj, -2, -1).unsqueeze(0).unsqueeze(0)
        up_proj_t = torch.transpose(up_proj, -2, -1).unsqueeze(0).unsqueeze(0)

        self.gate_proj = ttnn.as_tensor(
            gate_proj_t,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("gate_proj.weight"),
        )

        self.up_proj = ttnn.as_tensor(
            up_proj_t,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("up_proj.weight"),
        )

        # Load ff_out (down projection)
        ff_out = state_dict[f"{prefix}.ff_out.weight"]
        ff_out_t = torch.transpose(ff_out, -2, -1).unsqueeze(0).unsqueeze(0)

        self.down_proj = ttnn.as_tensor(
            ff_out_t,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("down_proj.weight"),
        )

        # Compute kernel config
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through SwiGLU MLP.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]

        Returns:
            Output tensor of shape [1, 1, seq_len, hidden_dim]
        """
        # Gate projection with SiLU activation
        gate = ttnn.linear(
            x,
            self.gate_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.silu(gate)

        # Up projection
        up = ttnn.linear(
            x,
            self.up_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Element-wise multiply: silu(gate) * up
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Down projection
        output = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)

        return output
