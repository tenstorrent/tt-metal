# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Attention implementation for Qwen3-TTS.

Note: Qwen3-TTS uses non-interleaved RoPE (pairs dims i and i+64),
while TTNN rotary_embedding_llama uses interleaved format (pairs dims 2i and 2i+1).
This module handles the necessary dimension rearrangement.

Supports both prefill mode (full sequence) and decode mode (single token with KV cache).

WHY NOT SDPA:
  ttnn.transformer.scaled_dot_product_attention requires bfloat16 inputs.
  Qwen3-TTS uses QK-norm with k_norm gamma weights up to 68, amplifying K values
  to ~260. In bfloat16 at magnitude 256, resolution is 2 units — enough precision
  loss that the model takes completely wrong decoding paths (no EOS, loops to max
  tokens). bfloat16 SDPA was tested and confirmed non-viable for this model.

WHY FLOAT32 MATMUL:
  ttnn.matmul supports float32 inputs/outputs. We typecast Q/K/V to float32 after
  QK-norm, run matmul+softmax in float32, and typecast the result back to bfloat16
  for the output projection. This matches PyTorch reference quality.

GQA EXPANSION:
  Unlike SDPA which handles GQA natively, manual matmul requires explicit head
  expansion. We use ttnn.slice (fresh tensor per group) + ttnn.concat to expand
  K/V from num_kv_heads to num_heads. Each group's slice is a distinct tensor object
  to avoid duplicate-reference deadlocks in ttnn.concat (Wormhole B0 constraint).

TRACE-COMPATIBLE DECODE:
  Passing cur_pos_tensor (int32 device tensor [1]) enables trace-compatible decode:
  - paged_update_cache uses cur_pos_tensor to write K/V at the correct position
    without baking a Python scalar into the trace.
  - Attention reads the FULL cache (fixed shape [1, kv_heads, max_seq, dim]) so
    no dynamic slice is needed in the trace.
  - decode_attn_mask (float32 [1, 1, 1, max_seq]) is pre-allocated on device and
    updated outside the trace each step to mask out future positions.
  When cur_pos_tensor is None, falls back to non-traceable update_cache+slice.
"""

from typing import Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.rope import ttnn_rearrange_to_interleaved, ttnn_rearrange_to_noninterleaved


class Attention(LightweightModule):
    """
    Multi-head attention with GQA and QK-norm for Qwen3-TTS.

    Features:
    - Grouped Query Attention (GQA) with 16 Q heads and 8 KV heads
    - QK-normalization (q_norm, k_norm) for stable training
    - RoPE positional embeddings
    - Float32 attention computation (required — bfloat16 SDPA loses precision)
    """

    def __init__(
        self,
        device,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        state_dict: dict,
        layer_prefix: str,
        rms_norm_eps: float = 1e-6,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim**-0.5
        self.rms_norm_eps = rms_norm_eps

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        def get_cache_name(name):
            if weight_cache_path is None:
                return None
            return weight_cache_path / f"{layer_prefix}_{name}".replace(".", "_")

        # Fuse QKV weights: [hidden_size, (num_heads + 2*num_kv_heads) * head_dim]
        q_proj_weight = state_dict[f"{layer_prefix}.self_attn.q_proj.weight"]
        k_proj_weight = state_dict[f"{layer_prefix}.self_attn.k_proj.weight"]
        v_proj_weight = state_dict[f"{layer_prefix}.self_attn.v_proj.weight"]
        o_proj_weight = state_dict[f"{layer_prefix}.self_attn.o_proj.weight"]

        qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
        qkv_weight = torch.transpose(qkv_weight, -2, -1).unsqueeze(0).unsqueeze(0)

        self.wqkv = ttnn.as_tensor(
            qkv_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("wqkv"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        o_proj_weight = torch.transpose(o_proj_weight, -2, -1).unsqueeze(0).unsqueeze(0)
        self.wo = ttnn.as_tensor(
            o_proj_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("wo"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        # QK-norm weights (per-head RMSNorm)
        q_norm_weight = state_dict[f"{layer_prefix}.self_attn.q_norm.weight"]
        k_norm_weight = state_dict[f"{layer_prefix}.self_attn.k_norm.weight"]

        TILE = 32
        q_norm_torch = q_norm_weight.unsqueeze(0).view(1, 1, head_dim).reshape([1, 1, head_dim // TILE, TILE])
        k_norm_torch = k_norm_weight.unsqueeze(0).view(1, 1, head_dim).reshape([1, 1, head_dim // TILE, TILE])

        self.q_norm_weight = ttnn.as_tensor(
            q_norm_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("q_norm"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.k_norm_weight = ttnn.as_tensor(
            k_norm_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("k_norm"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # HiFi4 + fp32 accumulation for float32 attention matmuls
        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Pre-compute HEIGHT_SHARDED memory config for paged_update_cache input.
        # paged_update_cache requires input in [1, batch, kv_heads, head_dim] HEIGHT_SHARDED on batch cores.
        # For batch=1: tensor [1, 1, num_kv_heads, head_dim] padded to [1, 1, tile(kv_heads), head_dim].
        # Physical height = tile(kv_heads), width = head_dim, 1 shard on 1 core.
        TILE = 32
        kv_shard_height = ((num_kv_heads + TILE - 1) // TILE) * TILE  # ceil to tile: 8→32
        compute_grid = device.compute_with_storage_grid_size()
        paged_shard_grid = ttnn.num_cores_to_corerangeset(1, compute_grid, True)
        self.paged_input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(paged_shard_grid, [kv_shard_height, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_attn_mask: Optional[ttnn.Tensor] = None,
        cp_prefill_mask: Optional[ttnn.Tensor] = None,
        prefill_attn_mask: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Apply multi-head attention with QK-norm and RoPE.

        Args:
            x: Input tensor [batch, 1, seq_len, hidden_size]
            cos, sin: RoPE frequencies [1, 1, seq_len, head_dim]
            transformation_mat: TTNN RoPE transformation matrix
            attention_mask: Optional attention mask
            kv_cache: (k_cache, v_cache) each [batch, num_kv_heads, max_seq, head_dim]
            start_pos: KV cache write position (decode, used only when cur_pos_tensor is None)
            mode: "prefill" or "decode"
            cur_pos_tensor: Optional int32 device tensor [1] for trace-compatible decode.
                When provided, uses paged_update_cache and attends over full cache.
            decode_attn_mask: Optional float32 device tensor [1,1,1,max_seq] for decode.
                Pre-allocated; caller updates it each step (0 for valid, -inf for future).
            cp_prefill_mask: Optional float32 device tensor [1,1,seq,max_seq] for trace-
                compatible CP prefill. When provided, writes K/V at positions 0 and 1
                using update_cache (constant scalars, trace-safe) and attends over the
                full cache masked by this tensor.
            prefill_attn_mask: Optional float32 device tensor [1,heads,padded_seq,max_seq]
                for trace-compatible Talker prefill. When provided, writes the full K/V
                sequence to cache at position 0 using update_cache (trace-safe) and
                attends over the full cache masked by this tensor. The mask encodes
                both causal constraints and padding.

        Returns:
            (output [batch, 1, seq_len, hidden_size], updated_kv_cache)
        """
        batch_size = x.shape[0]
        is_decode = mode == "decode"

        # QKV projection
        xqkv = ttnn.linear(
            x, self.wqkv, compute_kernel_config=self.compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Split: Q [b, num_heads, seq, head_dim], K/V [b, num_kv_heads, seq, head_dim]
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # QK-norm (per-head RMSNorm to stabilize attention with large logit scales)
        q = ttnn.rms_norm(
            q, epsilon=self.rms_norm_eps, weight=self.q_norm_weight, compute_kernel_config=self.compute_kernel_config
        )
        k = ttnn.rms_norm(
            k, epsilon=self.rms_norm_eps, weight=self.k_norm_weight, compute_kernel_config=self.compute_kernel_config
        )

        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, dtype=ttnn.bfloat16)

        # RoPE: rearrange to interleaved format for TTNN, apply, rearrange back.
        # Use is_decode_mode=False to work with DRAM layout for all sequence lengths.
        q = ttnn_rearrange_to_interleaved(q)
        k = ttnn_rearrange_to_interleaved(k)
        q = ttnn.experimental.rotary_embedding_llama(q, cos, sin, transformation_mat, is_decode_mode=False)
        k = ttnn.experimental.rotary_embedding_llama(k, cos, sin, transformation_mat, is_decode_mode=False)
        q = ttnn_rearrange_to_noninterleaved(q)
        k = ttnn_rearrange_to_noninterleaved(k)

        k_seq = k.shape[2]

        # Default: k/v are temporary tensors owned by this call, safe to free after typecast.
        # The trace-compatible decode path overrides k_for_attn/v_for_attn to alias k_cache
        # and sets k_is_cache_alias=True to prevent accidental deallocation.
        k_for_attn = k
        v_for_attn = v
        k_is_cache_alias = False

        # KV cache: store bfloat16, read for attention (then typecast to float32 below)
        updated_kv_cache = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache

            if is_decode:
                if cur_pos_tensor is not None:
                    # Trace-compatible path: paged_update_cache uses a device tensor for position.
                    # Reshape K/V from [batch, kv_heads, 1, dim] → [1, batch, kv_heads, dim]
                    # for paged_update_cache's expected input format.
                    k_paged = ttnn.transpose(k, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    v_paged = ttnn.transpose(v, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    # Move to HEIGHT_SHARDED L1 (required by paged_update_cache kernel)
                    k_paged_hs = ttnn.to_memory_config(k_paged, self.paged_input_mem_config)
                    v_paged_hs = ttnn.to_memory_config(v_paged, self.paged_input_mem_config)
                    ttnn.deallocate(k_paged)
                    ttnn.deallocate(v_paged)
                    ttnn.deallocate(k)
                    ttnn.deallocate(v)
                    # Write to cache at cur_pos_tensor position (in-place, trace-compatible)
                    ttnn.experimental.paged_update_cache(k_cache, k_paged_hs, update_idxs_tensor=cur_pos_tensor)
                    ttnn.experimental.paged_update_cache(v_cache, v_paged_hs, update_idxs_tensor=cur_pos_tensor)
                    ttnn.deallocate(k_paged_hs)
                    ttnn.deallocate(v_paged_hs)
                    # Read FULL cache for attention — fixed shape, trace-compatible.
                    # k_for_attn = k_cache is an ALIAS (not a copy). The code below
                    # must NOT deallocate it; k_is_cache_alias=True guards that.
                    k_for_attn = k_cache  # [batch, kv_heads, max_seq, dim]
                    v_for_attn = v_cache
                    k_is_cache_alias = True
                    k_seq = k_cache.shape[2]  # max_seq (constant)
                else:
                    # update_cache with Python scalar position (baked into trace as constant).
                    # Use this path when start_pos is a fixed int (e.g. 13 separate CP decode
                    # traces, one per position 2..14 — each bakes its own constant position).
                    ttnn.update_cache(k_cache, k, update_idx=start_pos)
                    ttnn.update_cache(v_cache, v, update_idx=start_pos)
                    ttnn.deallocate(k)
                    ttnn.deallocate(v)
                    if decode_attn_mask is not None:
                        # Trace-compatible full-cache attention: decode_attn_mask masks future
                        # positions.  k_seq = max_seq is a constant — trace-safe.
                        k_for_attn = k_cache
                        v_for_attn = v_cache
                        k_is_cache_alias = True
                        k_seq = k_cache.shape[2]
                    else:
                        # Non-trace path: slice cache to the valid prefix only.
                        cache_len = start_pos + 1
                        k_for_attn = ttnn.slice(
                            k_cache, [0, 0, 0, 0], [batch_size, self.num_kv_heads, cache_len, self.head_dim]
                        )
                        v_for_attn = ttnn.slice(
                            v_cache, [0, 0, 0, 0], [batch_size, self.num_kv_heads, cache_len, self.head_dim]
                        )
                        k_is_cache_alias = False
                        k_seq = cache_len
            elif prefill_attn_mask is not None:
                # Trace-compatible Talker prefill: write full padded K/V sequence to
                # cache at position 0. update_cache with update_idx=0 is a Python
                # constant baked into the trace — trace-safe.
                # k shape: [batch, kv_heads, padded_seq_len, head_dim]
                ttnn.update_cache(k_cache, k, update_idx=0)
                ttnn.update_cache(v_cache, v, update_idx=0)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                # Full-cache attention with prefill_attn_mask handles both causal
                # masking and padding: real positions only attend to prior real
                # positions; padding + empty cache positions are masked to -inf.
                k_for_attn = k_cache
                v_for_attn = v_cache
                k_is_cache_alias = True
                k_seq = k_cache.shape[2]
            elif cp_prefill_mask is not None:
                # Trace-compatible CP prefill: write K/V at constant positions 0 and 1.
                # We split the 2-token K/V into individual tokens and write each separately.
                # update_cache(cache, input, update_idx=0/1) is trace-compatible because
                # update_idx is a Python constant captured once in the trace — CP prefill
                # always writes exactly at positions 0 and 1.
                k0 = ttnn.slice(k, [0, 0, 0, 0], [batch_size, self.num_kv_heads, 1, self.head_dim])
                k1 = ttnn.slice(k, [0, 0, 1, 0], [batch_size, self.num_kv_heads, 2, self.head_dim])
                v0 = ttnn.slice(v, [0, 0, 0, 0], [batch_size, self.num_kv_heads, 1, self.head_dim])
                v1 = ttnn.slice(v, [0, 0, 1, 0], [batch_size, self.num_kv_heads, 2, self.head_dim])
                ttnn.update_cache(k_cache, k0, update_idx=0)
                ttnn.update_cache(k_cache, k1, update_idx=1)
                ttnn.update_cache(v_cache, v0, update_idx=0)
                ttnn.update_cache(v_cache, v1, update_idx=1)
                ttnn.deallocate(k0)
                ttnn.deallocate(k1)
                ttnn.deallocate(v0)
                ttnn.deallocate(v1)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                # Read full cache for attention — fixed shape, trace-compatible.
                # Positions 2..max_seq-1 may contain stale data from prior frames,
                # but cp_prefill_mask masks those positions to -inf.
                k_for_attn = k_cache
                v_for_attn = v_cache
                k_is_cache_alias = True
                k_seq = k_cache.shape[2]
            else:
                # Standard prefill: fill cache via PyTorch (not trace-compatible).
                # The returned k_cache/v_cache tensors may have new addresses if reallocated.
                # The decode trace is captured AFTER prefill, so it will capture the new addresses.
                max_seq = k_cache.shape[2]
                k_torch = ttnn.to_torch(k).to(torch.bfloat16)
                v_torch = ttnn.to_torch(v).to(torch.bfloat16)
                k_padded = torch.zeros(batch_size, self.num_kv_heads, max_seq, self.head_dim, dtype=torch.bfloat16)
                v_padded = torch.zeros(batch_size, self.num_kv_heads, max_seq, self.head_dim, dtype=torch.bfloat16)
                k_padded[:, :, :k_seq, :] = k_torch
                v_padded[:, :, :k_seq, :] = v_torch
                ttnn.deallocate(k_cache)
                ttnn.deallocate(v_cache)
                k_cache = ttnn.from_torch(
                    k_padded,
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                v_cache = ttnn.from_torch(
                    v_padded,
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            updated_kv_cache = (k_cache, v_cache)

        # Typecast to float32 for precise attention.
        # k_norm gamma up to 68 amplifies K to ~260; bfloat16 SDPA loses enough
        # precision to cause completely wrong token predictions (no EOS, model loops).
        q_f32 = ttnn.typecast(q, dtype=ttnn.float32)
        k_f32 = ttnn.typecast(k_for_attn, dtype=ttnn.float32)
        v_f32 = ttnn.typecast(v_for_attn, dtype=ttnn.float32)
        ttnn.deallocate(q)
        # IMPORTANT: only deallocate k_for_attn/v_for_attn when they are TEMPORARY
        # tensors (not aliases of the persistent KV cache).  In the trace-compatible
        # decode path, k_for_attn = k_cache and v_for_attn = v_cache — freeing them
        # would destroy the trace's KV cache buffers.
        if not k_is_cache_alias:
            ttnn.deallocate(k_for_attn)
            ttnn.deallocate(v_for_attn)

        # GQA expansion: replicate each KV head num_kv_groups times (interleaved).
        # k_f32 [b, num_kv_heads, k_seq, d] → [b, num_heads, k_seq, d]
        # Each group gets a fresh ttnn.slice to avoid duplicate-reference concat deadlock.
        if self.num_kv_groups > 1:
            k_parts = []
            v_parts = []
            for i in range(self.num_kv_heads):
                k_h = ttnn.slice(k_f32, [0, i, 0, 0], [batch_size, i + 1, k_seq, self.head_dim])
                v_h = ttnn.slice(v_f32, [0, i, 0, 0], [batch_size, i + 1, k_seq, self.head_dim])
                for g in range(self.num_kv_groups):
                    if g < self.num_kv_groups - 1:
                        k_parts.append(ttnn.clone(k_h, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                        v_parts.append(ttnn.clone(v_h, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                    else:
                        k_parts.append(k_h)
                        v_parts.append(v_h)
            k_exp = ttnn.concat(k_parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v_exp = ttnn.concat(v_parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for h in k_parts + v_parts:
                ttnn.deallocate(h)
            ttnn.deallocate(k_f32)
            ttnn.deallocate(v_f32)
        else:
            k_exp = k_f32
            v_exp = v_f32

        # Float32 scaled dot-product attention via ttnn.matmul + ttnn.softmax
        q_seq = q_f32.shape[2]
        k_t = ttnn.transpose(k_exp, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores = ttnn.matmul(
            q_f32, k_t, compute_kernel_config=self.sdpa_compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(k_t)
        ttnn.deallocate(q_f32)
        scores = ttnn.mul(scores, self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if decode_attn_mask is not None:
            # Trace-compatible decode mask: pre-allocated float32 [1,1,1,max_seq] device tensor.
            # Broadcast over num_heads dim: [1,1,1,max_seq] → [1,num_heads,1,max_seq].
            scores = ttnn.add(scores, decode_attn_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        elif cp_prefill_mask is not None:
            # Trace-compatible CP prefill mask: pre-allocated float32 [1,1,seq,max_seq].
            # Encodes causal masking over the full cache (positions beyond seq are -inf).
            scores = ttnn.add(scores, cp_prefill_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        elif prefill_attn_mask is not None:
            # Trace-compatible Talker prefill mask: [1, num_heads, padded_seq, max_seq].
            # Encodes causal + padding masking: real positions attend only to prior real
            # positions; padding query rows and empty cache columns are -inf.
            scores = ttnn.add(scores, prefill_attn_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        elif q_seq == k_seq and q_seq > 1:
            # Causal mask for standard prefill (full sequence self-attention)
            causal_mask = torch.triu(torch.full((1, 1, q_seq, k_seq), float("-inf")), diagonal=1)
            mask_tt = ttnn.from_torch(
                causal_mask,
                dtype=ttnn.float32,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            scores = ttnn.add(scores, mask_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(mask_tt)

        attn_weights = ttnn.softmax(scores, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scores)

        attn_output_f32 = ttnn.matmul(
            attn_weights,
            v_exp,
            compute_kernel_config=self.sdpa_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_weights)
        ttnn.deallocate(v_exp)

        # Cast back to bfloat16 for output projection
        attn_output = ttnn.typecast(attn_output_f32, dtype=ttnn.bfloat16)
        ttnn.deallocate(attn_output_f32)

        # Reshape: [b, num_heads, seq, head_dim] → [b, 1, seq, hidden_size]
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output)

        return output, updated_kv_cache
