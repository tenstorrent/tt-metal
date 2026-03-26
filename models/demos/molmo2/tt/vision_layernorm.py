# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm implementation for Molmo2 Vision Transformer.

The Molmo2 ViT uses standard LayerNorm (not RMSNorm) with both weight and bias.
"""

import math

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32
# Block shard: split hidden dim across X cores, sequence across Y cores (32 cores total).
BLOCK_SHARD_CORES_X = 4
BLOCK_SHARD_CORES_Y = 8


class VisionLayerNorm(LightweightModule):
    """
    LayerNorm for Molmo2 Vision Transformer blocks.

    Uses standard LayerNorm with weight (gamma) and bias (beta).
    """

    def __init__(
        self,
        device,
        dim: int,
        state_dict,
        state_dict_prefix: str,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        eps: float = 1e-6,
        num_tokens_for_l1_shard: int | None = None,
    ):
        """
        Initialize VisionLayerNorm.

        Args:
            device: TTNN device or mesh device
            dim: Hidden dimension (1152 for Molmo2 ViT)
            state_dict: Model state dict containing weights
            state_dict_prefix: Prefix for weight keys (e.g., "image_vit.transformer.resblocks.0.ln_1")
            weight_cache_path: Path to cache weights
            weight_memory_config: Memory config for weights
            weight_dtype: Data type for weights
            eps: Epsilon for numerical stability
            num_tokens_for_l1_shard: Logical sequence length (last dim -2) used to size block shards
                for L1 sharded layernorm. Defaults to 729 (27×27 ViT patches). Set higher for video /
                multi-crop so shard height covers your padded token count (multiple of 32 per row of cores).
        """
        super().__init__()
        self.device = device
        self.eps = eps
        self.dim = dim

        # Prepare weight and bias tensors (TILE-high shard for sharded LN gamma/beta)
        torch_weight = state_dict[f"{state_dict_prefix}.weight"].unsqueeze(0).view(1, 1, dim).expand([1, TILE, dim])
        torch_bias = state_dict[f"{state_dict_prefix}.bias"].unsqueeze(0).view(1, 1, dim).expand([1, TILE, dim])

        if weight_cache_path is None:
            cache_name = lambda *_: None
        else:
            cache_name = lambda suffix: weight_cache_path / (state_dict_prefix + f"{suffix}")

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name(".weight"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.bias = ttnn.as_tensor(
            torch_bias,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name(".bias"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        assert (
            dim % BLOCK_SHARD_CORES_X == 0
        ), f"hidden dim ({dim}) must be divisible by block shard X ({BLOCK_SHARD_CORES_X})"
        shard_w = dim // BLOCK_SHARD_CORES_X
        assert shard_w % TILE == 0, f"per-core shard width ({shard_w}) must be tile-aligned"

        seq = num_tokens_for_l1_shard if num_tokens_for_l1_shard is not None else 729
        rows_per_core = math.ceil(seq / BLOCK_SHARD_CORES_Y)
        shard_h = math.ceil(rows_per_core / TILE) * TILE

        self.sharded_input_config = ttnn.create_sharded_memory_config(
            shape=(shard_h, shard_w),
            core_grid=ttnn.CoreGrid(x=BLOCK_SHARD_CORES_X, y=BLOCK_SHARD_CORES_Y),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.sharded_output_config = self.sharded_input_config

        # Compute kernel config for high fidelity
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def forward(self, x: ttnn.Tensor, in_sharded: bool = False, out_sharded: bool = False) -> ttnn.Tensor:
        """
        Apply LayerNorm to input tensor.

        Args:
            x: Input tensor of shape [1, 1, seq_len, dim]
            in_sharded: Whether input is sharded
            out_sharded: Whether to return sharded output

        Returns:
            Normalized tensor of same shape
        """
        if in_sharded:
            # Derive block_h / block_w from the actual shard shape (matches layernorm.cpp default).
            x = ttnn.layer_norm(
                x,
                epsilon=self.eps,
                weight=self.weight,
                bias=self.bias,
                memory_config=x.memory_config(),
                compute_kernel_config=self.compute_kernel_config,
            )
            if out_sharded:
                return x
            x_interleaved = ttnn.sharded_to_interleaved(x)
            x.deallocate(True)
            return x_interleaved
        else:
            assert not out_sharded, "Non-sharded version of LayerNorm cannot output a sharded tensor"
            x = ttnn.layer_norm(
                x,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
                compute_kernel_config=self.compute_kernel_config,
            )
            return x
