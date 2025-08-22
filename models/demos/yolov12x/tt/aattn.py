# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


from tracy import signpost

import ttnn
from models.demos.yolov12x.tt.common import TtYOLOv12xConv2D


class TtnnAattn:
    def __init__(self, device, parameter, conv_pt, dim=384, num_heads=8, area=1, is_bk_enabled=False):
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = TtYOLOv12xConv2D(
            conv=parameter.qkv.conv,
            conv_pth=conv_pt.qkv.conv,
            device=device,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        self.proj = TtYOLOv12xConv2D(
            conv=parameter.proj.conv,
            conv_pth=conv_pt.proj.conv,
            device=device,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        self.pe = TtYOLOv12xConv2D(
            conv=parameter.pe.conv,
            conv_pth=conv_pt.pe.conv,
            device=device,
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )

    def __call__(self, x, i=0, j=0):
        signpost("Attn Start")
        batch_size, qkv_height, qkv_width, qkv_chan = x.shape
        qkv_n = qkv_height * qkv_width
        qkv = self.qkv(x)
        if qkv.is_sharded():
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat8_b)
        if qkv.layout == ttnn.TILE_LAYOUT:
            qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat8_b)

        if self.area > 1:
            qkv = ttnn.reshape(qkv, (1, batch_size * self.area, qkv_chan * 3, qkv_n // self.area))
            _, batch_size, _, qkv_n = qkv.shape

        # Using ttnn.reshape instead of view as "The last dimension can not change in view"
        qkv = ttnn.reshape(qkv, (batch_size, qkv_n, self.num_heads, self.head_dim * 3))
        qkv = ttnn.permute(qkv, (0, 2, 3, 1))  # [B, H, 3*D, S]
        qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        qkv = ttnn.to_memory_config(qkv, ttnn.DRAM_MEMORY_CONFIG)

        # Split into Q, K, V
        q, k, v = ttnn.split(qkv, qkv.shape[2] // 3, 2)  # each: [B, H, D, S]
        ttnn.deallocate(qkv)

        # Prepare for SDPA: [B, H, S, D]
        q = ttnn.permute(q, (0, 1, 3, 2))
        k = ttnn.permute(k, (0, 1, 3, 2))
        v = ttnn.permute(v, (0, 1, 3, 2))

        # Keep v_for_pe for positional embedding
        v_for_pe = ttnn.clone(v, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # Optimized SDPA config for better performance
        # Choose chunk sizes based on sequence length for optimal performance
        if qkv_n >= 512:
            q_chunk_size = 256
            k_chunk_size = 512
        elif qkv_n >= 256:
            q_chunk_size = 128
            k_chunk_size = 256
        elif qkv_n >= 128:
            q_chunk_size = 128
            k_chunk_size = 128
        else:
            q_chunk_size = 32
            k_chunk_size = 32

        pc_sdpa = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=[8, 8],  # Use full 8x8 grid
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=True,  # Use approximation for better performance
        )

        # Compute kernel config for better performance with BFP8
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,  # Use LoFi for BFP8 operations (faster)
            math_approx_mode=True,  # Enable approximations
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Run SDPA - process all batches together for better efficiency
        x_attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            program_config=pc_sdpa,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # Output to L1 for faster access
        )

        # Free Q/K/V tensors early to save memory
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Optimize: Combine permutations to reduce overhead
        # x_attn is [B, H, S, D], need [B, H, W, C]
        x = ttnn.permute(x_attn, (0, 2, 1, 3))  # [B, S, H, D]
        ttnn.deallocate(x_attn)

        # v_for_pe is [B, H, S, D], permute to match
        v_for_pe = ttnn.permute(v_for_pe, (0, 2, 1, 3))  # [B, S, H, D]

        if self.area > 1:
            x = ttnn.reshape(x, (batch_size // self.area, qkv_n * self.area, self.num_heads, self.head_dim))
            v_for_pe = ttnn.reshape(
                v_for_pe, (batch_size // self.area, qkv_n * self.area, self.num_heads, self.head_dim)
            )
            batch_size = batch_size // self.area
            qkv_n = qkv_n * self.area

        # Reshape to spatial dimensions
        x = ttnn.reshape(x, (batch_size, qkv_height, qkv_width, qkv_chan))
        v_for_pe = ttnn.reshape(v_for_pe, (batch_size, qkv_height, qkv_width, qkv_chan))

        # Apply positional embedding
        y = self.pe(v_for_pe)
        ttnn.deallocate(v_for_pe)

        if y.is_sharded():
            y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat8_b)
        x = ttnn.add(x, y, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(y)

        x = self.proj(x)
        signpost("Attn end")

        return x
