# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SwiGLU Image Projector for Molmo2 Vision Adapter.

Projects pooled image features from the ViT hidden dimension (1152)
to the language model hidden dimension (4096) using a SwiGLU MLP:
    output = w2(silu(w1(x)) * w3(x))

Dimensions:
    - input_dim: 1152 (adapter hidden size)
    - intermediate_dim: 12288
    - output_dim: 4096 (text model hidden size)

All linear layers have bias=False.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class ImageProjector(LightweightModule):
    """
    SwiGLU projector for mapping vision features to language model space.

    Architecture:
        - w1: Linear(input_dim, intermediate_dim, bias=False) - gate projection
        - w3: Linear(input_dim, intermediate_dim, bias=False) - up projection
        - w2: Linear(intermediate_dim, output_dim, bias=False) - down projection
        - Output: w2(silu(w1(x)) * w3(x))
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        input_dim: int = 1152,
        intermediate_dim: int = 12288,
        output_dim: int = 4096,
        weight_cache_path=None,
        state_dict_prefix: str = "model.vision_backbone.image_projector",
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize ImageProjector.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            input_dim: Input dimension (1152)
            intermediate_dim: Hidden dimension (12288)
            output_dim: Output dimension (4096)
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.dtype = dtype

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

        # Load w1 (gate): input_dim -> intermediate_dim (no bias)
        w1_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w1.weight"], -2, -1)

        self.w1_weight = ttnn.as_tensor(
            w1_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w1.weight"),
        )

        # Load w3 (up): input_dim -> intermediate_dim (no bias)
        w3_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w3.weight"], -2, -1)

        self.w3_weight = ttnn.as_tensor(
            w3_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w3.weight"),
        )

        # Load w2 (down): intermediate_dim -> output_dim (no bias)
        w2_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w2.weight"], -2, -1)

        self.w2_weight = ttnn.as_tensor(
            w2_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w2.weight"),
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
        Forward pass through SwiGLU projector.

        Args:
            x: Input tensor of shape [1, 1, num_tokens, input_dim]

        Returns:
            Output tensor of shape [1, 1, num_tokens, output_dim]
        """
        seq_len = x.shape[-2]

        # Reshape for long sequences (only when divisible)
        if seq_len > 1024 and seq_len % 1024 == 0:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # w1 (gate projection) with SiLU activation
        gate = ttnn.linear(
            x,
            self.w1_weight,
            activation="silu",
            compute_kernel_config=self.compute_kernel_config,
            memory_config=matmul_output_memory_config,
        )

        # w3 (up projection)
        up = ttnn.linear(
            x,
            self.w3_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=matmul_output_memory_config,
        )

        # Element-wise multiply: silu(w1(x)) * w3(x)
        hidden = ttnn.mul(gate, up, memory_config=matmul_output_memory_config)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # w2 (down projection)
        output = ttnn.linear(
            hidden,
            self.w2_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=matmul_output_memory_config,
        )
        ttnn.deallocate(hidden)

        # Reshape back if needed
        if seq_len > 1024 and seq_len % 1024 == 0:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        return output
