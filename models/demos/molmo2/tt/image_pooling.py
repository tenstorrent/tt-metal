# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Image Pooling for Molmo2 Vision Adapter.

This module implements 2D image pooling using cross-attention, where:
- Query: mean of gathered patch features from pooled_patches_idx neighborhoods
- Keys/Values: gathered patch features

The pooled_patches_idx tensor maps each output visual token to a neighborhood
of K source ViT patches. This is computed by the image processor on CPU.

Dimensions:
    - input_dim: 2304 (1152 * 2, concat of ViT layers 18 and 24)
    - hidden_dim: 1152 (adapter hidden size)
    - num_heads: 16
    - head_dim: 72 (1152 / 16)

All wq, wk, wv, wo projections have bias.

Supports TP=8 tensor parallelism:
    - Q, K, V projections: column-parallel (sharded by heads)
    - Output projection: row-parallel with all_reduce
"""

import math
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

# Chunk size along batch_seq for TP all_reduce: full-sequence all_gather peak-allocates (~GB);
# lower if OOM on long video. Override with MOLMO2_IMAGE_POOLING_AR_CHUNK.
_DEFAULT_IMAGE_POOLING_AR_CHUNK = 512


class ImagePooling(LightweightModule):
    """
    Cross-attention based image pooling for Molmo2.

    Pools multi-scale ViT features using attention with gathered patch neighborhoods.

    Supports TP=8:
        - 16 heads / 8 devices = 2 heads per device
        - wq, wk, wv: column-parallel
        - wo: row-parallel with all_reduce
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        input_dim: int = 2304,
        hidden_dim: int = 1152,
        num_heads: int = 16,
        head_dim: int = 72,
        weight_cache_path=None,
        state_dict_prefix: str = "model.vision_backbone.image_pooling_2d",
        dtype=ttnn.bfloat16,  # Changed from bfloat8_b for better precision
    ):
        """
        Initialize ImagePooling.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            input_dim: Input dimension (2304 = 1152 * 2 for multi-scale)
            hidden_dim: Hidden dimension for attention (1152)
            num_heads: Number of attention heads (16)
            head_dim: Dimension per head (72)
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.tile_size = 32

        # TP=8 configuration
        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.num_devices = mesh_device.get_num_devices() if self.is_mesh_device else 1

        # Calculate local heads per device for TP
        assert (
            num_heads % self.num_devices == 0
        ), f"num_heads {num_heads} must be divisible by num_devices {self.num_devices}"
        self.n_local_heads = num_heads // self.num_devices  # 16/8 = 2 heads per device

        # Pad head_dim to tile boundary if needed
        self.padded_head_dim = math.ceil(head_dim / self.tile_size) * self.tile_size

        # Scale factor for attention
        self.scale = head_dim**-0.5

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        # Mesh mappers for TP
        if self.is_mesh_device:
            # Column-parallel for Q, K, V (shard output dimension = heads)
            col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
            # Row-parallel for output projection (shard input dimension)
            row_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
            # Bias mappers
            bias_col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
            bias_replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        else:
            col_mesh_mapper = None
            row_mesh_mapper = None
            bias_col_mapper = None
            bias_replicate_mapper = None

        # Load wq: input_dim (2304) -> hidden_dim (1152) - column-parallel
        wq = state_dict[f"{state_dict_prefix}.wq.weight"]
        bq = state_dict[f"{state_dict_prefix}.wq.bias"]

        # Handle head_dim padding if needed
        if self.head_dim != self.padded_head_dim:
            wq = self._pad_weight(wq)
            bq = self._pad_bias(bq)

        wq_t = torch.transpose(wq, -2, -1)

        self.wq = ttnn.as_tensor(
            wq_t.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=col_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wq.weight.tp8") if self.is_mesh_device else cache_name("wq.weight"),
        )

        self.bq = ttnn.as_tensor(
            bq,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=bias_col_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wq.bias.tp8") if self.is_mesh_device else cache_name("wq.bias"),
        )

        # Load wk: input_dim (2304) -> hidden_dim (1152) - column-parallel
        wk = state_dict[f"{state_dict_prefix}.wk.weight"]
        bk = state_dict[f"{state_dict_prefix}.wk.bias"]

        if self.head_dim != self.padded_head_dim:
            wk = self._pad_weight(wk)
            bk = self._pad_bias(bk)

        wk_t = torch.transpose(wk, -2, -1)

        self.wk = ttnn.as_tensor(
            wk_t.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=col_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wk.weight.tp8") if self.is_mesh_device else cache_name("wk.weight"),
        )

        self.bk = ttnn.as_tensor(
            bk,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=bias_col_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wk.bias.tp8") if self.is_mesh_device else cache_name("wk.bias"),
        )

        # Load wv: input_dim (2304) -> hidden_dim (1152) - column-parallel
        wv = state_dict[f"{state_dict_prefix}.wv.weight"]
        bv = state_dict[f"{state_dict_prefix}.wv.bias"]

        if self.head_dim != self.padded_head_dim:
            wv = self._pad_weight(wv)
            bv = self._pad_bias(bv)

        wv_t = torch.transpose(wv, -2, -1)

        self.wv = ttnn.as_tensor(
            wv_t.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=col_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wv.weight.tp8") if self.is_mesh_device else cache_name("wv.weight"),
        )

        self.bv = ttnn.as_tensor(
            bv,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=bias_col_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wv.bias.tp8") if self.is_mesh_device else cache_name("wv.bias"),
        )

        # Load wo: hidden_dim (1152) -> hidden_dim (1152) - row-parallel
        wo = state_dict[f"{state_dict_prefix}.wo.weight"]
        bo = state_dict[f"{state_dict_prefix}.wo.bias"]

        # Handle padding for output projection
        if self.head_dim != self.padded_head_dim:
            wo_reshaped = wo.reshape(-1, self.num_heads, self.head_dim)
            wo_padded = torch.nn.functional.pad(wo_reshaped, (0, self.padded_head_dim - self.head_dim))
            wo = wo_padded.reshape(-1, self.num_heads * self.padded_head_dim)

        wo_t = torch.transpose(wo, -2, -1)

        self.wo = ttnn.as_tensor(
            wo_t.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=row_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo.weight.tp8") if self.is_mesh_device else cache_name("wo.weight"),
        )

        # wo bias is replicated (added after all_reduce)
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

    def _pad_weight(self, w):
        """Pad weight tensor for non-tile-aligned head_dim."""
        w = w.reshape(self.num_heads, self.head_dim, -1)
        w = torch.nn.functional.pad(w, (0, 0, 0, self.padded_head_dim - self.head_dim))
        return w.reshape(self.num_heads * self.padded_head_dim, -1)

    def _pad_bias(self, b):
        """Pad bias tensor for non-tile-aligned head_dim."""
        b = b.reshape(self.num_heads, self.head_dim)
        b = torch.nn.functional.pad(b, (0, self.padded_head_dim - self.head_dim))
        return b.reshape(-1)

    def forward(
        self,
        query: ttnn.Tensor,
        key_value: ttnn.Tensor,
        attn_mask: ttnn.Tensor = None,
        debug_stats: bool = True,
    ) -> ttnn.Tensor:
        """
        Forward pass through cross-attention pooling.

        With TP=8, each device processes n_local_heads (2 heads).
        After output projection, all_reduce combines partial results.

        Args:
            query: Query tensor of shape [1, 1, num_queries, input_dim]
                   (typically mean of gathered features)
            key_value: Key/Value tensor of shape [1, 1, pool_size, input_dim]
                       (gathered patch features)
            attn_mask: Optional attention mask [1, 1, 1, pool_size]
            debug_stats: Log intermediate stats (disabled during trace capture)

        Returns:
            Pooled features of shape [1, 1, num_queries, hidden_dim]
        """
        from loguru import logger

        def _get_stats(tensor, name):
            try:
                mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if self.is_mesh_device else None
                t = ttnn.to_torch(tensor, mesh_composer=mesh_composer)
                if self.is_mesh_device:
                    t = t[0]
                t = t.float()
                return f"{name}: shape={list(t.shape)}, mean={t.mean():.4f}, std={t.std():.4f}, min={t.min():.4f}, max={t.max():.4f}"
            except Exception as e:
                return f"{name}: stats error - {e}"

        num_queries = query.shape[-2]
        pool_size = key_value.shape[-2]

        if debug_stats:
            logger.info(_get_stats(query, "ImagePooling query input"))
            logger.info(_get_stats(key_value, "ImagePooling kv input"))

        # Q projection (column-parallel: each device computes local heads)
        q = ttnn.linear(
            query,
            self.wq,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = q + self.bq

        # K projection (column-parallel)
        k = ttnn.linear(
            key_value,
            self.wk,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = k + self.bk

        # V projection (column-parallel)
        v = ttnn.linear(
            key_value,
            self.wv,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = v + self.bv

        if debug_stats:
            logger.info(_get_stats(q, "ImagePooling q after wq+bq"))
            logger.info(_get_stats(k, "ImagePooling k after wk+bk"))
            logger.info(_get_stats(v, "ImagePooling v after wv+bv"))

        # Reshape Q, K, V for multi-head attention using LOCAL heads
        # Note: Using ttnn ops for head splitting

        # For cross-attention, we need special handling
        # Get batch dimensions from input tensors
        batch_seq = query.shape[1]  # B*N_out from vision_backbone
        padded_local_hidden = self.n_local_heads * self.padded_head_dim  # Local heads only

        # Q: [1, batch_seq, num_queries, padded_local_hidden] -> [batch_seq, n_local_heads, num_queries, head_dim]
        # K: [1, batch_seq, pool_size, padded_local_hidden] -> [batch_seq, n_local_heads, pool_size, head_dim]
        # V: [1, batch_seq, pool_size, padded_local_hidden] -> [batch_seq, n_local_heads, pool_size, head_dim]

        q = ttnn.reshape(q, [batch_seq, num_queries, self.n_local_heads, self.padded_head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))
        q = ttnn.typecast(q, dtype=ttnn.bfloat16)  # Changed from bfloat8_b

        k = ttnn.reshape(k, [batch_seq, pool_size, self.n_local_heads, self.padded_head_dim])
        k = ttnn.permute(k, (0, 2, 1, 3))
        k = ttnn.typecast(k, dtype=ttnn.bfloat16)  # Changed from bfloat8_b

        v = ttnn.reshape(v, [batch_seq, pool_size, self.n_local_heads, self.padded_head_dim])
        v = ttnn.permute(v, (0, 2, 1, 3))
        v = ttnn.typecast(v, dtype=ttnn.bfloat16)  # Changed from bfloat8_b

        # Use manual attention computation to handle mask correctly
        # (TTNN SDPA has issues with additive masks in cross-attention)

        # Q @ K^T -> [batch_seq, n_local_heads, num_queries, pool_size]
        k_t = ttnn.permute(k, (0, 1, 3, 2))  # [batch_seq, n_local_heads, head_dim, pool_size]
        attn_weights = ttnn.matmul(
            q,
            k_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        ttnn.deallocate(k_t)

        # Scale
        attn_weights = ttnn.mul(attn_weights, self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if debug_stats:
            logger.info(_get_stats(attn_weights, "ImagePooling attn_weights (after scale)"))

        # Apply attention mask (additive mask: 0 for valid, -inf for invalid)
        if attn_mask is not None:
            # Expand mask from [batch_seq, 1, 1, pool_size] to [batch_seq, n_local_heads, num_queries, pool_size]
            # Broadcasting should handle this automatically
            attn_weights = ttnn.add(attn_weights, attn_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Softmax
        attn_probs = ttnn.softmax(attn_weights, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_weights)

        # Attention output: attn_probs @ V -> [batch_seq, n_local_heads, num_queries, head_dim]
        attn_output = ttnn.matmul(
            attn_probs,
            v,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        ttnn.deallocate(attn_probs)

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Reshape back: [batch_seq, n_local_heads, num_queries, head_dim] -> [1, batch_seq, num_queries, local_hidden_dim]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, [1, batch_seq, num_queries, self.n_local_heads * self.padded_head_dim])

        # Output projection (row-parallel: each device has partial weights)
        output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(attn_output)

        # TP: All-reduce combines row-parallel partials. all_reduce uses concat/all_gather internally
        # and can allocate multi-GB for long batch_seq (video); chunk along dim 1 to bound peak DRAM.
        if self.is_mesh_device and self.num_devices > 1:
            ar_chunk = int(os.environ.get("MOLMO2_IMAGE_POOLING_AR_CHUNK", str(_DEFAULT_IMAGE_POOLING_AR_CHUNK)))
            ar_chunk = max(1, ar_chunk)
            b1 = int(output.shape[1])
            nq = int(output.shape[2])
            ld = int(output.shape[3])
            if b1 <= ar_chunk:
                output = ttnn.all_reduce(
                    output,
                    cluster_axis=1,
                    num_links=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                reduced_parts = []
                for s in range(0, b1, ar_chunk):
                    e = min(s + ar_chunk, b1)
                    chunk_out = ttnn.to_memory_config(
                        ttnn.slice(output, (0, s, 0, 0), (1, e, nq, ld)),
                        ttnn.DRAM_MEMORY_CONFIG,
                    )
                    reduced_parts.append(
                        ttnn.all_reduce(
                            chunk_out,
                            cluster_axis=1,
                            num_links=1,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    )
                    ttnn.deallocate(chunk_out)
                ttnn.deallocate(output)
                output = ttnn.concat(reduced_parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for p in reduced_parts:
                    ttnn.deallocate(p)

        # Add output bias (replicated, added after all_reduce)
        output = output + self.bo

        if debug_stats:
            logger.info(_get_stats(output, "ImagePooling final output"))

        return output
