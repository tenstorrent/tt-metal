# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Attention implementation for Molmo2 Vision Transformer.

The Molmo2 ViT uses standard multi-head attention with:
- Separate wq, wk, wv, wo linear projections (all with bias)
- Bidirectional attention (no causal mask)
- No rotary embeddings (positional info via learned pos embeddings)
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class VisionAttention(LightweightModule):
    """
    Multi-head attention for Molmo2 Vision Transformer.

    Architecture:
        - Separate Q, K, V projections with bias
        - Scaled dot-product attention (bidirectional)
        - Output projection with bias

    Supports TP=8 tensor parallelism:
        - Weights sharded by heads across devices (16 heads / 8 devices = 2 heads/device)
        - Column-parallel for QKV projection
        - Row-parallel for output projection with all_reduce
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix: str,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        weight_cache_path=None,
        dtype=ttnn.bfloat16,  # Changed from bfloat8_b for better vision precision
        use_tensor_parallel: bool = False,  # ViT uses data parallelism by default
    ):
        """
        Initialize VisionAttention.

        Args:
            mesh_device: TTNN mesh device
            state_dict: Model state dict containing weights
            state_dict_prefix: Prefix for weight keys (e.g., "image_vit.transformer.resblocks.0.attn")
            hidden_dim: Model hidden dimension (1152 for Molmo2 ViT)
            num_heads: Number of attention heads (16 for Molmo2 ViT)
            head_dim: Dimension per head (72 for Molmo2 ViT)
            weight_cache_path: Path to cache weights
            dtype: Data type for weights
            use_tensor_parallel: If True, use TP=8 (shard weights). If False, replicate weights
                                 for data parallelism. Default False for ViT (uses DP for frames).
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.tile_size = 32
        self.use_tensor_parallel = use_tensor_parallel

        # Device configuration
        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.num_devices = mesh_device.get_num_devices() if self.is_mesh_device else 1

        # For TP: local heads per device. For DP: all heads on each device.
        if use_tensor_parallel and self.is_mesh_device:
            assert (
                num_heads % self.num_devices == 0
            ), f"num_heads {num_heads} must be divisible by num_devices {self.num_devices}"
            self.n_local_heads = num_heads // self.num_devices  # 16/8 = 2 heads per device
        else:
            self.n_local_heads = num_heads  # All heads on each device (data parallel)

        # Pad head_dim to tile boundary if needed
        self.padded_head_dim = math.ceil(head_dim / self.tile_size) * self.tile_size

        # Scale factor for attention
        self.scale = head_dim**-0.5

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        # Mesh mappers: TP uses sharding, DP uses replication
        if self.is_mesh_device and use_tensor_parallel:
            # TP=8: Column-parallel for QKV (shard output dimension)
            col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
            # TP=8: Row-parallel for output projection (shard input dimension)
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

        # Load separate Q, K, V, O weights and combine Q, K, V for efficient computation
        # We'll create a fused QKV weight for efficiency
        wq = state_dict[f"{state_dict_prefix}.wq.weight"]  # [hidden_dim, hidden_dim]
        wk = state_dict[f"{state_dict_prefix}.wk.weight"]  # [hidden_dim, hidden_dim]
        wv = state_dict[f"{state_dict_prefix}.wv.weight"]  # [hidden_dim, hidden_dim]

        bq = state_dict[f"{state_dict_prefix}.wq.bias"]  # [hidden_dim]
        bk = state_dict[f"{state_dict_prefix}.wk.bias"]  # [hidden_dim]
        bv = state_dict[f"{state_dict_prefix}.wv.bias"]  # [hidden_dim]

        # Handle head_dim padding if needed
        if self.head_dim != self.padded_head_dim:
            # Reshape and pad weights
            def pad_weight(w):
                # w: [num_heads * head_dim, hidden_dim]
                w = w.reshape(self.num_heads, self.head_dim, -1)
                w = torch.nn.functional.pad(w, (0, 0, 0, self.padded_head_dim - self.head_dim))
                return w.reshape(self.num_heads * self.padded_head_dim, -1)

            def pad_bias(b):
                # b: [num_heads * head_dim]
                b = b.reshape(self.num_heads, self.head_dim)
                b = torch.nn.functional.pad(b, (0, self.padded_head_dim - self.head_dim))
                return b.reshape(-1)

            wq = pad_weight(wq)
            wk = pad_weight(wk)
            wv = pad_weight(wv)
            bq = pad_bias(bq)
            bk = pad_bias(bk)
            bv = pad_bias(bv)

        # Transpose weights for linear: [hidden_dim, qkv_dim] -> [qkv_dim, hidden_dim]
        # Then concatenate Q, K, V
        wq_t = torch.transpose(wq, -2, -1)
        wk_t = torch.transpose(wk, -2, -1)
        wv_t = torch.transpose(wv, -2, -1)

        # Fused QKV: [hidden_dim, 3 * num_heads * padded_head_dim]
        wqkv = torch.cat([wq_t, wk_t, wv_t], dim=-1)
        bqkv = torch.cat([bq, bk, bv], dim=-1)

        # For TP: weights are sharded by heads (dim=3 = last dimension after unsqueeze)
        # wqkv shape: [1, 1, hidden_dim, 3 * num_heads * padded_head_dim]
        # ShardTensorToMesh(dim=3) shards last dim across devices
        self.wqkv = ttnn.as_tensor(
            wqkv.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=col_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wqkv.weight.tp8") if self.is_mesh_device else cache_name("wqkv.weight"),
        )

        # Bias for column-parallel is also sharded
        self.bqkv = ttnn.as_tensor(
            bqkv,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=bias_col_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wqkv.bias.tp8") if self.is_mesh_device else cache_name("wqkv.bias"),
        )

        # Output projection
        wo = state_dict[f"{state_dict_prefix}.wo.weight"]  # [hidden_dim, hidden_dim]
        bo = state_dict[f"{state_dict_prefix}.wo.bias"]  # [hidden_dim]

        # Handle padding for output projection input dimension
        if self.head_dim != self.padded_head_dim:
            # Pad input dimension (from concat_heads output)
            wo_reshaped = wo.reshape(-1, self.num_heads, self.head_dim)
            wo_padded = torch.nn.functional.pad(wo_reshaped, (0, self.padded_head_dim - self.head_dim))
            wo = wo_padded.reshape(-1, self.num_heads * self.padded_head_dim)

        wo_t = torch.transpose(wo, -2, -1)

        # For TP: output projection is row-parallel (shard input dim = dim 2)
        # wo shape: [1, 1, num_heads * padded_head_dim, hidden_dim]
        # ShardTensorToMesh(dim=2) shards the input dimension
        self.wo = ttnn.as_tensor(
            wo_t.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=row_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo.weight.tp8") if self.is_mesh_device else cache_name("wo.weight"),
        )

        # Bias for row-parallel is replicated (added after all_reduce)
        self.bo = ttnn.as_tensor(
            bo,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=bias_replicate_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo.bias"),
        )

        # Compute kernel configs
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Max chunk size for SDPA - keep small to fit in L1
        self.max_chunk_size = 128

    def _get_sdpa_program_config(self, seq_len: int) -> ttnn.SDPAProgramConfig:
        """Get SDPA program config with chunking for memory efficiency."""
        # Get device grid size
        if hasattr(self.mesh_device, "get_devices"):
            device = self.mesh_device.get_devices()[0]
        else:
            device = self.mesh_device
        grid_size = device.compute_with_storage_grid_size()

        # Pad seq_len to tile boundary
        padded_seq_len = ((seq_len + 31) // 32) * 32
        chunk_size = min(padded_seq_len, self.max_chunk_size)

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=chunk_size,
            k_chunk_size=chunk_size,
            exp_approx_mode=False,
        )

    # Class-level counter for SDPA calls (for debugging)
    _sdpa_call_count = 0
    _current_request = 0

    @classmethod
    def reset_counters(cls):
        """Reset SDPA call counters. Call between requests to prevent memory growth."""
        cls._sdpa_call_count = 0
        cls._current_request = 0

    @classmethod
    def increment_request(cls):
        """Increment request counter for tracking."""
        cls._current_request += 1

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through attention.

        With TP=8, each device processes n_local_heads (2 heads).
        After output projection, all_reduce combines partial results.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]

        Returns:
            Output tensor of shape [1, 1, seq_len, hidden_dim]
        """
        from loguru import logger

        seq_len = x.shape[-2]

        # Track SDPA calls for debugging
        VisionAttention._sdpa_call_count += 1
        call_num = VisionAttention._sdpa_call_count

        # Log every 25 calls (once per ViT layer set)
        if call_num % 25 == 1 or call_num <= 3:
            logger.debug(
                f"VisionAttention SDPA call #{call_num} (request #{VisionAttention._current_request}): seq_len={seq_len}, n_local_heads={self.n_local_heads}"
            )

        # Reshape for long sequences (only when divisible for memory optimization)
        if seq_len > 2048 and seq_len % 2048 == 0:
            x = ttnn.reshape(x, [1, seq_len // 2048, 2048, -1])

        # QKV projection (column-parallel: each device computes its local heads)
        qkv = ttnn.linear(
            x,
            self.wqkv,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Add bias (sharded for TP)
        qkv = qkv + self.bqkv

        # Reshape back if needed
        if seq_len > 2048 and seq_len % 2048 == 0:
            qkv = ttnn.reshape(qkv, [1, 1, seq_len, -1])

        ttnn.deallocate(x)

        # Split Q, K, V and reshape to LOCAL heads (n_local_heads per device)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.n_local_heads,  # TP: use local heads per device
            num_kv_heads=self.n_local_heads,  # Full attention, not GQA
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(qkv)

        # Convert to bfloat16 for SDPA (changed from bfloat8_b for better precision)
        q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        k = ttnn.typecast(k, dtype=ttnn.bfloat16)
        v = ttnn.typecast(v, dtype=ttnn.bfloat16)

        # Scaled dot-product attention (bidirectional - no causal mask)
        # Use SDPAProgramConfig with chunking to avoid memory issues on repeated calls
        sdpa_config = self._get_sdpa_program_config(seq_len)

        # Log SDPA config for first few calls
        if call_num <= 3:
            logger.debug(f"  SDPA config: q_chunk={sdpa_config.q_chunk_size}, k_chunk={sdpa_config.k_chunk_size}")

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            program_config=sdpa_config,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Reshape attention output: [1, n_local_heads, seq_len, head_dim] -> [1, 1, seq_len, local_hidden_dim]
        attn_output = ttnn.reshape(attn_output, [1, self.n_local_heads, -1, self.padded_head_dim])

        # Concatenate LOCAL heads
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Output projection (row-parallel: each device has partial weights)
        if seq_len > 1024 and seq_len % 1024 == 0:
            attn_output = ttnn.reshape(attn_output, [1, seq_len // 1024, 1024, -1])

        output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(attn_output)

        # TP=8: All-reduce to combine partial results from all devices
        # Only needed when using tensor parallelism (not data parallelism)
        if self.use_tensor_parallel and self.is_mesh_device and self.num_devices > 1:
            output = ttnn.all_reduce(
                output,
                cluster_axis=1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Add output bias (replicated, added after all_reduce)
        output = output + self.bo

        if seq_len > 1024 and seq_len % 1024 == 0:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        return output
