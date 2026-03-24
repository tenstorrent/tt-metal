# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
MLP implementation for Molmo2 Vision Transformer.

The Molmo2 ViT uses a standard 2-layer MLP with GELU activation:
    output = fc2(gelu(fc1(x)))

Both fc1 and fc2 have bias terms.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class VisionMLP(LightweightModule):
    """
    MLP block for Molmo2 Vision Transformer.

    Architecture:
        - fc1: Linear(hidden_dim -> intermediate_dim) with bias
        - GELU activation
        - fc2: Linear(intermediate_dim -> hidden_dim) with bias
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix: str,
        hidden_dim: int,
        intermediate_dim: int,
        weight_cache_path=None,
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize VisionMLP.

        Args:
            mesh_device: TTNN mesh device
            state_dict: Model state dict containing weights
            state_dict_prefix: Prefix for weight keys (e.g., "image_vit.transformer.resblocks.0.mlp")
            hidden_dim: Input/output dimension (1152 for Molmo2 ViT)
            intermediate_dim: Hidden dimension of MLP (typically 4x hidden_dim)
            weight_cache_path: Path to cache weights
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dtype = dtype

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

        # Load w1: hidden_dim -> intermediate_dim
        w1_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w1.weight"], -2, -1)
        w1_bias = state_dict[f"{state_dict_prefix}.w1.bias"]

        self.w1_weight = ttnn.as_tensor(
            w1_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w1.weight"),
        )

        self.w1_bias = ttnn.as_tensor(
            w1_bias,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w1.bias"),
        )

        # Load w2: intermediate_dim -> hidden_dim
        w2_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w2.weight"], -2, -1)
        w2_bias = state_dict[f"{state_dict_prefix}.w2.bias"]

        self.w2_weight = ttnn.as_tensor(
            w2_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w2.weight"),
        )

        self.w2_bias = ttnn.as_tensor(
            w2_bias,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w2.bias"),
        )

        # Compute kernel configs
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        matmul_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]

        Returns:
            Output tensor of shape [1, 1, seq_len, hidden_dim]
        """
        seq_len = x.shape[-2]

        # Only chunk when seq_len % 1024 == 0; else reshape volume mismatch (e.g. video ViT).
        chunk_mlp = seq_len > 1024 and (seq_len % 1024 == 0)
        if chunk_mlp:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # w1 with QuickGELU activation (gelu_pytorch_tanh in HF)
        # TTNN uses "gelu" which is the fast approximation
        hidden = ttnn.linear(
            x,
            self.w1_weight,
            bias=self.w1_bias,
            activation="gelu",  # TTNN's gelu is the fast approximation matching pytorch_tanh
            compute_kernel_config=self.compute_kernel_config,
            memory_config=matmul_output_memory_config,
        )

        # w2
        output = ttnn.linear(
            hidden,
            self.w2_weight,
            bias=self.w2_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=matmul_output_memory_config,
        )
        ttnn.deallocate(hidden)

        if chunk_mlp:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        return output
