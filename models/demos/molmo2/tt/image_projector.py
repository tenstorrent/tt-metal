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

Supports TP=8 tensor parallelism:
    - w1, w3 (gate, up): column-parallel (12288/8 = 1536 per device)
    - w2 (down): row-parallel with all_reduce
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

    Supports TP=8:
        - w1, w3: column-parallel (1536 per device)
        - w2: row-parallel with all_reduce
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
        dtype=ttnn.bfloat16,  # Changed from bfloat8_b for better precision in SwiGLU
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

        # TP=8 configuration
        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.num_devices = mesh_device.get_num_devices() if self.is_mesh_device else 1

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        # Mesh mappers for TP
        if self.is_mesh_device:
            # Column-parallel for w1, w3 (shard output dimension = intermediate_dim)
            col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
            # Row-parallel for w2 (shard input dimension = intermediate_dim)
            row_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
        else:
            col_mesh_mapper = None
            row_mesh_mapper = None

        # Load w1 (gate): input_dim -> intermediate_dim (column-parallel, no bias)
        w1_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w1.weight"], -2, -1)

        # w1 shape: [1, 1, input_dim, intermediate_dim]
        # ShardTensorToMesh(dim=3) shards intermediate_dim across devices
        self.w1_weight = ttnn.as_tensor(
            w1_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=col_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w1.weight.tp8") if self.is_mesh_device else cache_name("w1.weight"),
        )

        # Load w3 (up): input_dim -> intermediate_dim (column-parallel, no bias)
        w3_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w3.weight"], -2, -1)

        self.w3_weight = ttnn.as_tensor(
            w3_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=col_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w3.weight.tp8") if self.is_mesh_device else cache_name("w3.weight"),
        )

        # Load w2 (down): intermediate_dim -> output_dim (row-parallel, no bias)
        w2_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w2.weight"], -2, -1)

        # w2 shape: [1, 1, intermediate_dim, output_dim]
        # ShardTensorToMesh(dim=2) shards intermediate_dim (input) across devices
        self.w2_weight = ttnn.as_tensor(
            w2_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=row_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w2.weight.tp8") if self.is_mesh_device else cache_name("w2.weight"),
        )

        # Compute kernel configs
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through SwiGLU projector.

        With TP=8:
            - w1, w3 are column-parallel (each device has intermediate_dim/8)
            - w2 is row-parallel (each device computes partial sum)
            - all_reduce combines partial sums after w2

        Args:
            x: Input tensor of shape [1, 1, num_tokens, input_dim]

        Returns:
            Output tensor of shape [1, 1, num_tokens, output_dim]
        """
        from loguru import logger

        def _get_stats(tensor, name):
            """Get tensor stats for debugging."""
            try:
                t = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0]
                return f"{name}: shape={list(t.shape)}, mean={t.mean():.4f}, std={t.std():.4f}, min={t.min():.4f}, max={t.max():.4f}"
            except:
                return f"{name}: stats unavailable"

        seq_len = x.shape[-2]
        logger.debug(f"ImageProjector input: shape={list(x.shape)}")
        logger.debug(_get_stats(x, "input"))

        # Reshape for long sequences (only when divisible)
        if seq_len > 1024 and seq_len % 1024 == 0:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # w1 (gate projection) with SiLU activation - column-parallel
        gate = ttnn.linear(
            x,
            self.w1_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.debug(_get_stats(gate, "gate (w1(x))"))
        gate = ttnn.silu(gate)
        logger.debug(_get_stats(gate, "gate (silu(w1(x)))"))

        # w3 (up projection) - column-parallel
        up = ttnn.linear(
            x,
            self.w3_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.debug(_get_stats(up, "up (w3(x))"))

        # Check weight stats
        logger.debug(_get_stats(self.w1_weight, "w1_weight"))
        logger.debug(_get_stats(self.w3_weight, "w3_weight"))

        # Element-wise multiply: silu(w1(x)) * w3(x)
        # Both gate and up are local intermediate dims, so multiply is local
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(_get_stats(hidden, "hidden (gate*up)"))
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # w2 (down projection) - row-parallel
        output = ttnn.linear(
            hidden,
            self.w2_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.debug(_get_stats(output, "output (w2(hidden)) BEFORE all_reduce"))
        ttnn.deallocate(hidden)

        # TP=8: All-reduce to combine partial results from all devices
        if self.is_mesh_device and self.num_devices > 1:
            output = ttnn.all_reduce(
                output,
                cluster_axis=1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logger.debug(_get_stats(output, "output AFTER all_reduce"))

        # Reshape back if needed
        if seq_len > 1024 and seq_len % 1024 == 0:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        logger.debug(_get_stats(output, "final output"))
        return output
