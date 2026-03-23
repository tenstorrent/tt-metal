# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Grouped-Query Attention for Molmo2 Text Model with Tensor Parallelism.

Implements GQA with:
- 32 query heads, 8 key/value heads (4:1 ratio)
- QK-norm (Qwen3-style normalization on Q and K)
- RoPE (Rotary Position Embeddings) with θ=1M using TTNN-native ops
- KV cache support for autoregressive generation
- Tensor parallelism: shard heads across devices

Weight layout:
- att_proj: fused QKV projection [hidden_dim, (num_heads + 2*num_kv_heads) * head_dim]
- attn_out: output projection [num_heads * head_dim, hidden_dim]

Uses ttnn.experimental.rotary_embedding (half-span RoPE) for device-side RoPE.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TextAttention(LightweightModule):
    """
    Grouped-Query Attention with QK-norm for Molmo2 text model.

    Supports tensor parallelism by sharding heads across devices.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        layer_num: int,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 8192,
        rope_theta: float = 1000000.0,
        weight_cache_path=None,
        state_dict_prefix: str = "model.transformer.blocks",
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize TextAttention with tensor parallelism support.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            layer_num: Layer number (0-35)
            hidden_dim: Hidden dimension (4096)
            num_heads: Number of query heads (32)
            num_kv_heads: Number of key/value heads (8)
            head_dim: Dimension per head (128)
            max_seq_len: Maximum sequence length (8192)
            rope_theta: RoPE theta (1,000,000)
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.dtype = dtype

        self.scale = head_dim**-0.5

        # Tile alignment
        self.tile_size = 32
        self.padded_head_dim = math.ceil(head_dim / self.tile_size) * self.tile_size

        # Layer prefix
        prefix = f"{state_dict_prefix}.{layer_num}.self_attn"

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{prefix}.{name}"

        # Determine tensor parallelism setup
        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.is_mesh_device = is_mesh_device

        if is_mesh_device:
            self.num_devices = mesh_device.get_num_devices()
            # Column parallel for Q/K/V: shard output (head) dimension
            col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
            # Row parallel for output projection: shard input dimension
            row_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
            # Update heads per device
            self.num_heads_per_device = num_heads // self.num_devices
            self.num_kv_heads_per_device = num_kv_heads // self.num_devices
        else:
            self.num_devices = 1
            col_mesh_mapper = None
            row_mesh_mapper = None
            self.num_heads_per_device = num_heads
            self.num_kv_heads_per_device = num_kv_heads

        # Load fused QKV projection: att_proj
        # Shape: [hidden_dim, (num_heads + 2*num_kv_heads) * head_dim]
        att_proj = state_dict[f"{prefix}.att_proj.weight"]
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        # Split into Q, K, V
        wq = att_proj[:q_dim, :]
        wk = att_proj[q_dim : q_dim + kv_dim, :]
        wv = att_proj[q_dim + kv_dim :, :]

        # Transpose for TTNN linear: [out, in] -> [1, 1, in, out]
        wq_t = torch.transpose(wq, -2, -1).unsqueeze(0).unsqueeze(0)
        wk_t = torch.transpose(wk, -2, -1).unsqueeze(0).unsqueeze(0)
        wv_t = torch.transpose(wv, -2, -1).unsqueeze(0).unsqueeze(0)

        # Fused QKV weights only (prefill + decode). Per-device Q/K/V slices are concatenated on dim=-1.
        # Format: per-device concatenation of [Q_heads, K_heads, V_heads]
        if is_mesh_device and self.num_devices > 1:
            # Multi-device: create fused QKV with per-device chunking
            qkv_list = []
            for i in range(self.num_devices):
                wq_chunk = torch.chunk(wq, self.num_devices, dim=0)[i]
                wk_chunk = torch.chunk(wk, self.num_devices, dim=0)[i]
                wv_chunk = torch.chunk(wv, self.num_devices, dim=0)[i]

                wq_chunk_t = torch.transpose(wq_chunk, -2, -1)
                wk_chunk_t = torch.transpose(wk_chunk, -2, -1)
                wv_chunk_t = torch.transpose(wv_chunk, -2, -1)

                qkv_chunk = torch.cat([wq_chunk_t, wk_chunk_t, wv_chunk_t], dim=-1)
                qkv_list.append(qkv_chunk)

            wqkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

            self.wqkv = ttnn.as_tensor(
                wqkv_cat,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name("wqkv.weight"),
            )
        else:
            # Single device: simple concatenation
            wqkv_t = torch.cat([wq_t, wk_t, wv_t], dim=-1)
            self.wqkv = ttnn.as_tensor(
                wqkv_t,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=None,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name("wqkv.weight"),
            )

        # QKV output size per device for nlp_create_qkv_heads_decode
        self.qkv_size_per_device = (self.num_heads_per_device + 2 * self.num_kv_heads_per_device) * head_dim

        # Load output projection: attn_out
        wo = state_dict[f"{prefix}.attn_out.weight"]
        wo_t = torch.transpose(wo, -2, -1).unsqueeze(0).unsqueeze(0)

        self.wo = ttnn.as_tensor(
            wo_t,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=row_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo.weight"),
        )

        # Load QK-norm weights (replicated across devices - they apply per head_dim)
        q_norm = state_dict[f"{prefix}.q_norm.weight"]
        k_norm = state_dict[f"{prefix}.k_norm.weight"]

        # QK norm weights are per head_dim, so replicate across devices
        if is_mesh_device:
            norm_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        else:
            norm_mesh_mapper = None

        self.q_norm_weight = ttnn.as_tensor(
            q_norm.reshape(1, 1, 1, -1),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=norm_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("q_norm.weight"),
        )

        self.k_norm_weight = ttnn.as_tensor(
            k_norm.reshape(1, 1, 1, -1),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=norm_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("k_norm.weight"),
        )

        # Compute kernel configs
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats: List[ttnn.Tensor],
        transformation_mats: Dict[str, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        trace_id: int = None,
        layer_idx: int = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Forward pass through GQA attention (prefill mode) with tensor parallelism.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]
            rot_mats: List of [cos, sin] rotation matrices (half-span RoPE; transformation_mats unused)
            attn_mask: Optional causal mask
            start_pos: Starting position for KV cache
            kv_cache: Optional (k_cache, v_cache) tuple - tensor parallel sharded

        Returns:
            Tuple of (output, updated_kv_cache)
        """
        seq_len = x.shape[-2]

        # Fused QKV: one matmul, then slice along output (column-parallel / mesh layout matches self.wqkv).
        qkv = ttnn.linear(
            x,
            self.wqkv,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_dim = self.num_heads_per_device * self.head_dim
        kv_dim = self.num_kv_heads_per_device * self.head_dim
        # Tile layout may pad seq; slice using actual qkv time dimension.
        sq = qkv.shape[2]
        # Slice last dim: [Q | K | V]
        q = ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, sq, q_dim))
        k = ttnn.slice(qkv, (0, 0, 0, q_dim), (1, 1, sq, q_dim + kv_dim))
        v = ttnn.slice(qkv, (0, 0, 0, q_dim + kv_dim), (1, 1, sq, q_dim + 2 * kv_dim))

        # Drop tile padding on sequence if present (reshape expects logical seq_len)
        if sq != seq_len:
            q = ttnn.slice(q, (0, 0, 0, 0), (1, 1, seq_len, q_dim))
            k = ttnn.slice(k, (0, 0, 0, 0), (1, 1, seq_len, kv_dim))
            v = ttnn.slice(v, (0, 0, 0, 0), (1, 1, seq_len, kv_dim))

        # Reshape for multi-head attention (using per-device head counts)
        # Q: [1, 1, seq_len, num_heads_per_device * head_dim] -> [1, num_heads_per_device, seq_len, head_dim]
        q = ttnn.reshape(q, [1, seq_len, self.num_heads_per_device, self.head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))

        # K, V: [1, 1, seq_len, num_kv_heads_per_device * head_dim] -> [1, num_kv_heads_per_device, seq_len, head_dim]
        k = ttnn.reshape(k, [1, seq_len, self.num_kv_heads_per_device, self.head_dim])
        k = ttnn.permute(k, (0, 2, 1, 3))

        v = ttnn.reshape(v, [1, seq_len, self.num_kv_heads_per_device, self.head_dim])
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Apply QK-norm (RMSNorm on Q and K)
        q = ttnn.rms_norm(q, weight=self.q_norm_weight, epsilon=1e-5)
        k = ttnn.rms_norm(k, weight=self.k_norm_weight, epsilon=1e-5)

        # Apply RoPE using TTNN half-span op (prefill mode)
        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, dtype=ttnn.bfloat16)
        print("first Run Demo : ", " : ", os.getenv("First_run"))
        if os.getenv("First_run", "false").lower() == "true" and trace_id is not None:
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            os.environ["First_run"] = "false"
        q = ttnn.experimental.rotary_embedding(
            q,
            rot_mats[0],  # cos
            rot_mats[1],  # sin
        )

        k = ttnn.experimental.rotary_embedding(
            k,
            rot_mats[0],  # cos
            rot_mats[1],  # sin
        )

        # rotary_embedding pads the sequence dim to TILE_HEIGHT; slice back to seq_len so Q,K match V for SDPA
        q = ttnn.slice(
            q,
            (0, 0, 0, 0),
            (1, self.num_heads_per_device, seq_len, self.head_dim),
        )
        k = ttnn.slice(
            k,
            (0, 0, 0, 0),
            (1, self.num_kv_heads_per_device, seq_len, self.head_dim),
        )

        # Update KV cache using fill_cache for prefill
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Fill cache at batch_idx=0 (we use single batch during prefill)
            ttnn.fill_cache(k_cache, k, batch_idx=0)
            ttnn.fill_cache(v_cache, v, batch_idx=0)

        new_kv_cache = (k, v)

        # Repeat K, V for GQA (within each device's subset of heads)
        if self.num_kv_groups > 1:
            # [1, num_kv_heads_per_device, seq_len, head_dim] -> [1, num_heads_per_device, seq_len, head_dim]
            k = ttnn.repeat_interleave(k, self.num_kv_groups, dim=1)
            v = ttnn.repeat_interleave(v, self.num_kv_groups, dim=1)

        # Convert to bfloat8_b for SDPA
        q = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
        k = ttnn.typecast(k, dtype=ttnn.bfloat8_b)
        v = ttnn.typecast(v, dtype=ttnn.bfloat8_b)

        # Scaled dot-product attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True if attn_mask is None else False,
            scale=self.scale,
            attn_mask=attn_mask,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.deallocate(qkv)

        # Reshape back: [1, num_heads_per_device, seq_len, head_dim] -> [1, 1, seq_len, hidden_dim_per_device]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, [1, 1, seq_len, self.num_heads_per_device * self.head_dim])

        # Output projection (row parallel - input is sharded)
        output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(attn_output)

        # All-reduce for tensor parallelism
        if self.is_mesh_device and self.num_devices > 1:
            output = ttnn.all_reduce(
                output,
                cluster_axis=1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return output, new_kv_cache

    def forward_decode(
        self,
        x: ttnn.Tensor,
        rot_mats: List[ttnn.Tensor],
        transformation_mat: ttnn.Tensor,
        kv_cache: Tuple[ttnn.Tensor, ttnn.Tensor],
        current_pos: ttnn.Tensor,
        page_table: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Decode-mode forward pass with KV cache update and tensor parallelism.

        Uses TTNN-native RoPE, paged_update_cache, and scaled_dot_product_attention_decode
        for efficient autoregressive generation with tracing support.

        Args:
            x: Input tensor of shape [1, 1, 1, hidden_dim] (single token)
            rot_mats: List of [cos, sin] for current position (interleaved; half-span RoPE)
            kv_cache: Tuple of (k_cache, v_cache) pre-allocated tensors
                      Shape per device: [batch, num_kv_heads_per_device, max_seq_len, head_dim]
            current_pos: Current decode position tensor [batch]
            page_table: Optional page table for paged attention (vLLM)
                Shape: [batch, max_num_blocks_per_req] mapping positions to block IDs

        Returns:
            Output tensor [1, 1, 1, hidden_dim]
        """
        batch_size = 1
        seq_len = 1  # Decode mode processes one token at a time

        # Fused QKV projection (single matmul instead of 3 separate ones)
        # Use L1 for decode mode - small tensors fit in L1
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # Use nlp_create_qkv_heads_decode for efficient reshape
        # Output: q [1, B, num_heads, d], k [1, B, num_kv_heads, d], v [1, B, num_kv_heads, d]
        # Use HEIGHT_SHARDED output for efficient paged_update_cache later
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=self.num_heads_per_device,
            num_kv_heads=self.num_kv_heads_per_device,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # Apply QK-norm (RMSNorm on Q and K)
        # RMSNorm doesn't support HEIGHT_SHARDED, convert to interleaved first
        q = ttnn.to_memory_config(q, ttnn.L1_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.L1_MEMORY_CONFIG)
        q = ttnn.rms_norm(q, weight=self.q_norm_weight, epsilon=1e-5)
        k = ttnn.rms_norm(k, weight=self.k_norm_weight, epsilon=1e-5)

        # Ensure bfloat16 for rotary_embedding
        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, dtype=ttnn.bfloat16)

        # Apply RoPE using TTNN rotary embedding
        # Note: rot_mats already contain cos/sin for the current position, so we pass 0
        # This avoids reading current_pos from device during traced execution
        q = ttnn.experimental.rotary_embedding(
            q,
            rot_mats[0],  # cos
            rot_mats[1],  # sin
            0,  # Position is already embedded in rot_mats
        )

        k = ttnn.experimental.rotary_embedding(
            k,
            rot_mats[0],  # cos
            rot_mats[1],  # sin
            0,  # Position is already embedded in rot_mats
        )

        # Reshape to handle padding from rotary_embedding (pads to 32 heads)
        # Output shape after RoPE: [1, B, 32, head_dim], need [1, B, num_heads_per_device, head_dim]
        q = ttnn.reshape(
            q,
            (1, batch_size, self.num_heads_per_device, self.head_dim),
            (1, batch_size, 32, self.head_dim),
        )
        k = ttnn.reshape(
            k,
            (1, batch_size, self.num_kv_heads_per_device, self.head_dim),
            (1, batch_size, 32, self.head_dim),
        )

        # Slice to actual number of heads
        # Note: nlp_create_qkv_heads_decode output shape is [1, H_padded, B, d]
        # After RoPE/reshape, tensors are [1, B, H_padded, d] for Q,K but V stays [1, H_padded, B, d]
        q = q[:, :, : self.num_heads_per_device]
        k = k[:, :, : self.num_kv_heads_per_device]
        # V hasn't been through reshape, so heads are in dim 1
        v = v[:, : self.num_kv_heads_per_device, :, :]

        # Get KV cache references
        k_cache, v_cache = kv_cache

        # Convert K, V to DRAM for permute operation
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)

        # Reshape V for cache - K is already in correct format [1, B, H, d]
        # V is [1, H, B, d] -> [1, B, H, d] to match K format
        # paged_update_cache expects input_tensor.shape[1] == page_table.shape[0] (batch_size)
        v = ttnn.permute(v, (0, 2, 1, 3))  # [1, H, B, d] -> [1, B, H, d]

        # Create sharded memory config for paged_update_cache
        # paged_update_cache requires HEIGHT sharded input tensors in [1, B, H, d] format
        # Shard across batch dimension (1 core per batch element)
        grid_size = ttnn.CoreCoord(8, 8)
        kv_num_cores = batch_size  # 1 core per batch
        kv_core_grid = ttnn.num_cores_to_corerangeset(kv_num_cores, grid_size, row_wise=True)
        # For [1, B, H, d], HEIGHT = B * H (padded), WIDTH = head_dim
        # Shard height must be tile-aligned (multiple of 32)
        kv_shard_height = ((self.num_kv_heads_per_device + 31) // 32) * 32  # Tile-aligned
        kv_shard_width = self.head_dim
        kv_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(kv_shard_height, kv_shard_width),
            core_grid=kv_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        # Convert K, V to sharded memory config for paged_update_cache
        k = ttnn.to_memory_config(k, kv_mem_cfg)
        v = ttnn.to_memory_config(v, kv_mem_cfg)

        # Update KV cache at current position using tensor-based indexing
        # paged_update_cache expects [1, B, K, D] format where B is batch_size
        # This matches the format used by tt_transformers (k_heads_1BKD)
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=current_pos, page_table=page_table)

        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Convert Q back to interleaved for SDPA (keep bfloat16 for GQA decode)
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)

        # Scaled dot-product attention decode
        # Uses the full KV cache up to current_pos
        # Configure SDPA for tensor parallel setup with few heads per device
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 4),  # Limit cores
            exp_approx_mode=False,
            q_chunk_size=256,
            k_chunk_size=256,
        )

        if page_table is not None:
            # Paged attention: use paged SDPA decode
            attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                k_cache,
                v_cache,
                page_table_tensor=page_table,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=sdpa_program_config,
            )  # Output: [1, B, H, d]
        else:
            # Non-paged attention: use standard SDPA decode
            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                k_cache,
                v_cache,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=sdpa_program_config,
            )  # Output: [1, B, H, d]

        ttnn.deallocate(q)

        # Convert SDPA output to sharded for nlp_concat_heads_decode
        # SDPA output: [1, B, H, d] needs HEIGHT sharded memory config
        sdpa_output_shard_config = ttnn.create_sharded_memory_config(
            shape=(
                (self.num_heads_per_device + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE,
                self.head_dim,
            ),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn_output = ttnn.to_memory_config(attn_output, sdpa_output_shard_config)

        # Concat heads: [1, B, H, d] -> [1, 1, B, H*d]
        # Use nlp_concat_heads_decode for proper head concatenation
        attn_output = ttnn.experimental.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.num_heads_per_device,
        )

        # Output projection (row parallel) - use L1 for decode
        output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        ttnn.deallocate(attn_output)

        # All-reduce for tensor parallelism
        # Note: all_reduce needs DRAM input, convert from L1 first
        if self.is_mesh_device and self.num_devices > 1:
            output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
            output = ttnn.all_reduce(
                output,
                cluster_axis=1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return output
