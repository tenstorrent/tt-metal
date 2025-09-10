# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov12x.tt.common import TtYOLOv12xConv2D


class TtnnAattn:
    def __init__(self, device, parameter, conv_pt, dim=384, num_heads=8, area=1, is_bk_enabled=False):
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = TtYOLOv12xConv2D(conv=parameter.qkv.conv, conv_pth=conv_pt.qkv.conv, device=device)
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
            config_override={"act_block_h": 32},
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

    def __call__(self, x, i=0, j=0):
        batch_size, qkv_height, qkv_width, qkv_chan = x.shape
        qkv_n = qkv_height * qkv_width
        qkv = self.qkv(x)
        if qkv.is_sharded():
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.L1_MEMORY_CONFIG)
        qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)

        if self.area > 1:
            qkv = ttnn.reshape(qkv, (1, batch_size * self.area, qkv_chan * 3, qkv_n // self.area))
            _, batch_size, _, qkv_n = qkv.shape

        # Using ttnn.reshape instead of view as "The last dimension can not change in view"
        qkv = ttnn.reshape(qkv, (batch_size, qkv_n, self.num_heads, self.head_dim * 3))
        qkv = ttnn.permute(qkv, (0, 2, 3, 1))  # [B, H, 3*D, S]
        q, k, v = ttnn.split(qkv, qkv.shape[2] // 3, 2)  # each: [B, H, D, S]
        ttnn.deallocate(qkv)

        # Prepare Q/K/V for Flash Attention (prefill-style, non-causal): [1, H, S, D]
        q = ttnn.permute(q, (0, 1, 3, 2))  # [B, H, S, D]
        k_attn = ttnn.permute(k, (0, 1, 3, 2))  # [B, H, S, D]
        v_attn = ttnn.permute(v, (0, 1, 3, 2))  # [B, H, S, D]

        # Convert to TILE for SDPA and move to DRAM (SDPA requires DRAM buffers)
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k_attn = ttnn.to_layout(k_attn, ttnn.TILE_LAYOUT)
        v_attn = ttnn.to_layout(v_attn, ttnn.TILE_LAYOUT)
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k_attn = ttnn.to_memory_config(k_attn, ttnn.DRAM_MEMORY_CONFIG)
        v_attn = ttnn.to_memory_config(v_attn, ttnn.DRAM_MEMORY_CONFIG)

        # Choose SDPA program config based on sequence length
        def _choose_sdpa_pc(seq_len: int):
            q_chunk = 128 if (seq_len % 128 == 0) else 32
            if seq_len % 512 == 0:
                k_chunk = 512
            elif seq_len % 128 == 0:
                k_chunk = 128
            else:
                k_chunk = 32
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=[8, 7],
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )

        pc_sdpa = _choose_sdpa_pc(qkv_n)

        # Run SDPA per batch if needed to avoid cross-batch mixing
        if batch_size == 1:
            x_attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                is_causal=False,
                scale=self.scale,
                program_config=pc_sdpa,
            )  # [1, H, S, D]
            # Expand back to [B, H, S, D]
            x_attn = ttnn.reshape(x_attn, (batch_size, x_attn.shape[1], x_attn.shape[2], x_attn.shape[3]))
        else:
            outs = []
            for b in range(batch_size):
                qb = ttnn.slice(q, (b, 0, 0, 0), (b + 1, q.shape[1], q.shape[2], q.shape[3]))
                kb = ttnn.slice(k_attn, (b, 0, 0, 0), (b + 1, k_attn.shape[1], k_attn.shape[2], k_attn.shape[3]))
                vb = ttnn.slice(v_attn, (b, 0, 0, 0), (b + 1, v_attn.shape[1], v_attn.shape[2], v_attn.shape[3]))
                qb = ttnn.to_memory_config(qb, ttnn.DRAM_MEMORY_CONFIG)
                kb = ttnn.to_memory_config(kb, ttnn.DRAM_MEMORY_CONFIG)
                vb = ttnn.to_memory_config(vb, ttnn.DRAM_MEMORY_CONFIG)
                ob = ttnn.transformer.scaled_dot_product_attention(
                    qb,
                    kb,
                    vb,
                    is_causal=False,
                    scale=self.scale,
                    program_config=pc_sdpa,
                )  # [1, H, S, D]
                outs.append(ttnn.reshape(ob, (1, ob.shape[1], ob.shape[2], ob.shape[3])))
            x_attn = ttnn.concat(outs, dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)  # [B, H, S, D]

        # Free Q/K/V for attention path
        ttnn.deallocate(q)
        ttnn.deallocate(k_attn)
        ttnn.deallocate(v_attn)

        # Align with the rest of the pipeline which expects [B, H, D, S]
        x = ttnn.permute(x_attn, (0, 1, 3, 2))
        ttnn.deallocate(x_attn)

        x = ttnn.permute(x, (0, 3, 1, 2))
        v = ttnn.permute(v, (0, 3, 1, 2))

        if self.area > 1:
            x = ttnn.reshape(x, (1, batch_size // self.area, qkv_n * self.area, qkv_chan))
            v = ttnn.reshape(v, (1, batch_size // self.area, qkv_n * self.area, qkv_chan))
            batch_size, qkv_n, _, _ = x.shape

        x = ttnn.reshape(x, (batch_size, qkv_height, qkv_width, qkv_chan))
        v = ttnn.reshape(v, (batch_size, qkv_height, qkv_width, qkv_chan))
        y = self.pe(v)
        y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
        x = x + ttnn.reshape(y, x.shape)
        ttnn.deallocate(v)

        x = self.proj(x)

        return x
