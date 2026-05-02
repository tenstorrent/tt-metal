# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral TTS text decoder GQA attention for N150 (single device).

N150 layout (no tensor parallelism):
  - wq/wk/wv: single-device full weights, standard ttnn.linear
  - wo: single-device, standard ttnn.linear
  - SDPA: ttnn.transformer.scaled_dot_product_attention (prefill)
           ttnn.transformer.scaled_dot_product_attention_decode (decode)
  - RoPE: ttnn.experimental.rotary_embedding_llama (prefill)

Weight keys in consolidated.safetensors:
  layers.{N}.attention.wq.weight  [4096, 3072]
  layers.{N}.attention.wk.weight  [1024, 3072]
  layers.{N}.attention.wv.weight  [1024, 3072]
  layers.{N}.attention.wo.weight  [3072, 4096]

No QK-norm in text backbone.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtVoxtralTextAttention(LightweightModule):
    """Text decoder GQA attention for Voxtral-4B-TTS-2603 on N150.

    Single-device, no tensor parallelism, no CCL.
    """

    def __init__(
        self,
        device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
    ):
        super().__init__()

        self.device = device
        self.n_heads = configuration.n_heads
        self.n_kv_heads = configuration.n_kv_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.scale = self.head_dim**-0.5
        self.transformation_mats = transformation_mats
        self.model_config = configuration.get_model_config()
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN

        pfx = f"layers.{layer_num}.attention"
        cache_name = (
            (lambda name: weight_cache_path / f"{pfx}.{name}")
            if (not configuration.dummy_weights and weight_cache_path is not None)
            else (lambda _: None)
        )

        # ── Build fused wqkv: [3072, 6144] for single matmul ──────────────
        # wq: [4096, 3072] → need [3072, 4096] transposed for linear
        # wk: [1024, 3072] → [3072, 1024]
        # wv: [1024, 3072] → [3072, 1024]
        # Fuse: [3072, 6144]
        wq = state_dict[f"{pfx}.wq.weight"].T  # [3072, 4096]
        wk = state_dict[f"{pfx}.wk.weight"].T  # [3072, 1024]
        wv = state_dict[f"{pfx}.wv.weight"].T  # [3072, 1024]
        wqkv = torch.cat([wq, wk, wv], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3072, 6144]

        self.wqkv = ttnn.as_tensor(
            wqkv,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wqkv"),
        )

        # ── wo: [1, 1, 4096, 3072] ─────────────────────────────────────────
        wo = state_dict[f"{pfx}.wo.weight"].T.unsqueeze(0).unsqueeze(0)  # [1, 1, 4096, 3072]
        # wo original: [3072, 4096] → T → [4096, 3072]

        self.wo = ttnn.as_tensor(
            wo,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo"),
        )

        # ── KV cache (bfloat16 for precision) ─────────────────────────────
        cache_k = torch.zeros(self.max_batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim)
        self.layer_past = [
            ttnn.as_tensor(
                kv,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for kv in (cache_k, torch.zeros_like(cache_k))
        ]

    def forward_prefill(self, x_11SH, rot_mats, user_id=0, kv_cache=None, mask=None):
        """x_11SH: [1, 1, S, 3072]."""
        seq_len = x_11SH.shape[-2]

        qkv_chunked = seq_len > self.MAX_QKV_MM_SEQ_LEN and seq_len % self.MAX_QKV_MM_SEQ_LEN == 0
        if qkv_chunked:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        xqkv = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )

        if qkv_chunked:
            xqkv = ttnn.reshape(xqkv, [1, 1, seq_len, -1])
        ttnn.deallocate(x_11SH)

        # Split Q/K/V
        q_dim = self.n_heads * self.head_dim  # 4096
        kv_dim = self.n_kv_heads * self.head_dim  # 1024

        q_pre, k_pre, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_heads,
            num_kv_heads=self.n_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # RoPE
        q = ttnn.experimental.rotary_embedding_llama(q_pre, rot_mats[0], rot_mats[1], self.transformation_mats)
        k = ttnn.experimental.rotary_embedding_llama(k_pre, rot_mats[0], rot_mats[1], self.transformation_mats)
        ttnn.deallocate(q_pre)
        ttnn.deallocate(k_pre)

        # SDPA prefill — use current k/v (seq_len = S), not full cache (Sq must == Sk)
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None,
            scale=self.scale,
            program_config=self.model_config["SDPA_PROGCFG"](seq_len),
            compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)

        # Store k/v in cache after SDPA
        keys = kv_cache[0] if kv_cache else self.layer_past[0]
        values = kv_cache[1] if kv_cache else self.layer_past[1]
        ttnn.fill_cache(keys, k, user_id)
        ttnn.fill_cache(values, v, user_id)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # [1, n_heads, S, head_dim] → [1, 1, S, n_heads*head_dim]
        attn_11SH = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # Output projection
        out = ttnn.linear(
            attn_11SH,
            self.wo,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
        )
        ttnn.deallocate(attn_11SH)

        return out

    def forward_decode(self, x, current_pos, rot_mats=None, kv_cache=None):
        """x: [1, 1, batch, dim]."""
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(x)
        xqkv = ttnn.reshape(xqkv, (1, 1, self.max_batch_size, xqkv.shape[3]))

        q_pre, k_pre, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=self.n_heads,
            num_kv_heads=self.n_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        k_mem_cfg = k_pre.memory_config()
        v_mem_cfg = v.memory_config()

        q_pre_r = ttnn.reshape(q_pre, [1, self.n_heads, 1, self.head_dim])
        k_pre_r = ttnn.reshape(k_pre, [1, self.n_kv_heads, 1, self.head_dim])
        ttnn.deallocate(q_pre)
        ttnn.deallocate(k_pre)
        q_rotated = ttnn.experimental.rotary_embedding(q_pre_r, rot_mats[0], rot_mats[1])
        k_rotated = ttnn.experimental.rotary_embedding(k_pre_r, rot_mats[0], rot_mats[1])
        ttnn.deallocate(q_pre_r)
        ttnn.deallocate(k_pre_r)
        q = ttnn.reshape(q_rotated[:, :, :1, :], [1, 1, self.n_heads, self.head_dim])
        k = ttnn.reshape(k_rotated[:, :, :1, :], [1, 1, self.n_kv_heads, self.head_dim])
        ttnn.deallocate(q_rotated)
        ttnn.deallocate(k_rotated)

        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k_sharded = ttnn.to_memory_config(k, k_mem_cfg)
        v_sharded = ttnn.to_memory_config(v, v_mem_cfg)

        keys = kv_cache[0] if kv_cache else self.layer_past[0]
        values = kv_cache[1] if kv_cache else self.layer_past[1]
        ttnn.experimental.paged_update_cache(keys, k_sharded, update_idxs_tensor=current_pos)
        ttnn.experimental.paged_update_cache(values, v_sharded, update_idxs_tensor=current_pos)
        ttnn.deallocate(k_sharded)
        ttnn.deallocate(v_sharded)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            keys,
            values,
            cur_pos_tensor=current_pos,
            scale=self.scale,
            program_config=self.model_config["SDPA_DECODE_PROGCFG"],
            compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)

        sdpa_batch = attn_out.shape[1]
        sdpa_grid_x = min(8, sdpa_batch)
        sdpa_grid_y = (sdpa_batch + sdpa_grid_x - 1) // sdpa_grid_x
        sdpa_shard_cfg = ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=ttnn.CoreGrid(y=sdpa_grid_y, x=sdpa_grid_x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn_out = ttnn.to_memory_config(attn_out, sdpa_shard_cfg)
        attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=self.n_heads)

        if attn_out.shape[2] != self.max_batch_size:
            attn_out = ttnn.to_memory_config(attn_out, ttnn.DRAM_MEMORY_CONFIG)
            attn_out = attn_out[:, :, : self.max_batch_size, :]

        out = ttnn.linear(
            attn_out,
            self.wo,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
        )
        ttnn.deallocate(attn_out)
        return out

    def forward(self, x, current_pos=None, rot_mats=None, mode="prefill", kv_cache=None, mask=None):
        if mode == "prefill":
            return self.forward_prefill(x, rot_mats, kv_cache=kv_cache, mask=mask)
        return self.forward_decode(x, current_pos, rot_mats=rot_mats, kv_cache=kv_cache)
