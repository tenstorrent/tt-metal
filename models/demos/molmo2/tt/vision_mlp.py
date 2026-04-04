# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
MLP implementation for Molmo2 Vision Transformer.

The Molmo2 ViT uses a standard 2-layer MLP with GELU activation:
    output = fc2(gelu(fc1(x)))

Both fc1 and fc2 have bias terms.

Supports TP=8 tensor parallelism:
    - w1 (up): column-parallel, sharded across devices
    - w2 (down): row-parallel with all_reduce
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

    Supports TP=8:
        - w1 (up): column-parallel (4304/8 = 538 per device)
        - w2 (down): row-parallel with all_reduce
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix: str,
        hidden_dim: int,
        intermediate_dim: int,
        weight_cache_path=None,
        dtype=ttnn.bfloat16,  # Changed from bfloat8_b for better precision
        use_tensor_parallel: bool = False,  # ViT uses data parallelism by default
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
            use_tensor_parallel: If True, use TP=8 (shard weights). If False, replicate weights
                                 for data parallelism. Default False for ViT (uses DP for frames).
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dtype = dtype
        self.use_tensor_parallel = use_tensor_parallel

        # Device configuration
        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.num_devices = mesh_device.get_num_devices() if self.is_mesh_device else 1

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        # Mesh mappers: TP uses sharding, DP uses replication
        if self.is_mesh_device and use_tensor_parallel:
            # TP=8: Column-parallel for w1 (shard output dimension)
            col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
            # TP=8: Row-parallel for w2 (shard input dimension)
            row_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
            bias_col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
            bias_replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        elif self.is_mesh_device:
            # Data Parallel: Replicate all weights
            col_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            row_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            bias_col_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            bias_replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        else:
            col_mesh_mapper = None
            row_mesh_mapper = None
            bias_col_mapper = None
            bias_replicate_mapper = None

        # Load w1: hidden_dim -> intermediate_dim (column-parallel)
        w1_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w1.weight"], -2, -1)
        w1_bias = state_dict[f"{state_dict_prefix}.w1.bias"]

        # w1 shape: [1, 1, hidden_dim, intermediate_dim]
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

        # w1 bias is sharded for column-parallel
        self.w1_bias = ttnn.as_tensor(
            w1_bias,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=bias_col_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w1.bias.tp8") if self.is_mesh_device else cache_name("w1.bias"),
        )

        # Load w2: intermediate_dim -> hidden_dim (row-parallel)
        w2_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w2.weight"], -2, -1)
        w2_bias = state_dict[f"{state_dict_prefix}.w2.bias"]

        # w2 shape: [1, 1, intermediate_dim, hidden_dim]
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

        # w2 bias is replicated (added after all_reduce)
        self.w2_bias = ttnn.as_tensor(
            w2_bias,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=bias_replicate_mapper,
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

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through MLP.

        With TP=8:
            - w1 is column-parallel (each device has intermediate_dim/8)
            - w2 is row-parallel (each device has partial sum)
            - all_reduce combines partial sums after w2

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]

        Returns:
            Output tensor of shape [1, 1, seq_len, hidden_dim]
        """
        seq_len = x.shape[-2]

        # Reshape for long sequences to fit matmul on device (only when divisible)
        if seq_len > 1024 and seq_len % 1024 == 0:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # w1: HF ViT uses F.gelu(..., approximate="tanh") (gelu_pytorch_tanh). TTNN linear
        # activation="gelu" matches that tanh-approx GELU for Molmo2 PCC vs HF.
        hidden = ttnn.linear(
            x,
            self.w1_weight,
            bias=self.w1_bias,
            activation="gelu",
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # w2 (row-parallel: each device computes partial output)
        output = ttnn.linear(
            hidden,
            self.w2_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)

        # TP=8: All-reduce to combine partial results from all devices
        # Only needed when using tensor parallelism (not data parallelism)
        if self.use_tensor_parallel and self.is_mesh_device and self.num_devices > 1:
            output = ttnn.all_reduce(
                output,
                cluster_axis=1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Add bias after all_reduce (bias is replicated)
        output = output + self.w2_bias

        # Reshape back if needed
        if seq_len > 1024 and seq_len % 1024 == 0:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        return output
