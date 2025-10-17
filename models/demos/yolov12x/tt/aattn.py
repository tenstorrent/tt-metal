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
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            enable_weights_double_buffer=True,
            core_count=64,
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
            enable_split_reader=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )

    def __call__(self, x, i=0, j=0):
        signpost("Attn Start")
        batch_size, qkv_height, qkv_width, qkv_chan = x.shape
        qkv_n = qkv_height * qkv_width
        qkv = self.qkv(x)

        if qkv.is_sharded():
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat8_b)
        """
            # Convert to ROW_MAJOR if currently TILE layout
            if qkv.layout == ttnn.TILE_LAYOUT:
                qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        elif qkv.layout == ttnn.TILE_LAYOUT:
            qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)


        if self.area > 1:
            qkv = ttnn.reshape(qkv, (1, batch_size * self.area, qkv_chan * 3, qkv_n // self.area))
            _, batch_size, _, qkv_n = qkv.shape

        qkv = ttnn.reshape(qkv, (batch_size, qkv_n, self.num_heads, self.head_dim * 3))
        qkv = ttnn.permute(qkv, (0, 2, 3, 1))  # [B, H, 3*D, S]
        # Combine layout and memory config operations
        qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Split into Q, K, V
        q, k, v = ttnn.split(qkv, qkv.shape[2] // 3, 2)  # each: [B, H, D, S]
        ttnn.deallocate(qkv)

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv,
        num_heads=self.num_heads,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

        print("🚀🚀🚀 QKV BEFORE RESHAPE:", qkv.shape, "🚀🚀🚀")
        qkv = ttnn.reshape(qkv, (qkv.shape[0]*self.area, qkv.shape[1], qkv.shape[2]//self.area, qkv.shape[3]))
        print("🚀🚀🚀 QKV AFTER RESHAPE:", qkv.shape, "🚀🚀🚀")
        """

        qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # print("🚀🚀🚀 Q K V SHAPES:", q.shape, k.shape, v.shape, "🚀🚀🚀")
        ttnn.deallocate(qkv)
        # Prepare for SDPA: [B, H, S, D]
        # q = ttnn.permute(q, (0, 1, 3, 2))
        # k = ttnn.permute(k, (0, 1, 3, 2))
        # v = ttnn.permute(v, (0, 1, 3, 2))

        # Keep v_for_pe for positional embedding - avoid clone by reusing v later
        v_for_pe = v

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
            v_for_pe,  # Use v_for_pe directly instead of v
            is_causal=False,
            scale=self.scale,
            program_config=pc_sdpa,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # Output to L1 for faster access
        )

        # Free Q/K tensors early to save memory (keep v_for_pe for later use)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        # Note: v_for_pe will be used later, so don't deallocate here

        # Optimize: Directly reshape without intermediate allocation for concatenate_heads
        # x_attn is [B, H, S, D] -> concatenate to [B, S, H*D] -> reshape to [1, 1, B*S, H*D]
        x = ttnn.transformer.concatenate_heads(x_attn, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x_attn)
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[2]))

        # Same optimization for v_for_pe
        v_for_pe = ttnn.transformer.concatenate_heads(v_for_pe, memory_config=ttnn.L1_MEMORY_CONFIG)
        v_for_pe = ttnn.reshape(v_for_pe, (1, 1, v_for_pe.shape[0] * v_for_pe.shape[1], v_for_pe.shape[2]))

        # Apply positional embedding
        y = self.pe(v_for_pe)
        ttnn.deallocate(v_for_pe)

        # Ensure both tensors have compatible memory layouts for add operation
        if y.is_sharded():
            y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat8_b)

        # Ensure x is also in compatible memory config - convert to L1 interleaved
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

        x = ttnn.add(x, y, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(y)

        x = self.proj(x)

        # Ensure output tensor has compatible memory layout for residual add in caller
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat8_b)

        # Ensure output is in TILE layout to match input expectations
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

        signpost("Attn end")

        return x
