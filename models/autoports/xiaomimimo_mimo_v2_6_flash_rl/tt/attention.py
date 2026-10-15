# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo attention (SWA and global layers), tensor-parallel over one Galaxy row.

Per chip (TP=8): 8 Q heads and the KV head they attend to. SWA layers have 8 KV heads, one per
chip; global layers have 4, so each KV head is stored on the two chips whose Q heads use it.
Head dim is 192 for Q/K. V has 128 dims; we zero-pad it to 192 so the SDPA kernels see one head
dim, and zero-pad the matching rows of ``o_proj`` so the padding never reaches the output.
``attention_value_scale`` is folded into the V weights. Sinks (SWA only) are pre-divided by the
softmax scale because the SDPA kernels apply ``scale`` to the sink logit too.
"""

import torch

import ttnn

from .common import TILE, as_weight, cache_name, compute_config
from .rope import permute_heads_for_rope


def kv_heads_for_chip(chip, tp, num_kv_heads):
    if tp >= num_kv_heads:
        return [chip // (tp // num_kv_heads)]
    per = num_kv_heads // tp
    return list(range(chip * per, (chip + 1) * per))


def prepare_qkv_o(w, dims, v_scale, tp):
    """Host-side per-chip fused QKV ([hidden, tp * local_qkv]) and padded o_proj ([heads * head_dim, hidden])."""
    hd, vd = dims.head_dim, dims.v_head_dim
    nh_local = dims.num_heads // tp
    q = permute_heads_for_rope(w["q_proj"], dims.num_heads, hd, dims.rope_dim)
    k = permute_heads_for_rope(w["k_proj"], dims.num_kv_heads, hd, dims.rope_dim)
    v = w["v_proj"] * (v_scale if v_scale is not None else 1.0)
    per_chip = []
    for c in range(tp):
        rows = [q[c * nh_local * hd : (c + 1) * nh_local * hd]]
        kv_ids = kv_heads_for_chip(c, tp, dims.num_kv_heads)
        rows += [k[h * hd : (h + 1) * hd] for h in kv_ids]
        rows += [torch.nn.functional.pad(v[h * vd : (h + 1) * vd], (0, 0, 0, hd - vd)) for h in kv_ids]
        per_chip.append(torch.cat(rows))
    wqkv = torch.cat(per_chip).T.contiguous()

    o = w["o_proj"].T.reshape(dims.num_heads, vd, -1)
    o = torch.nn.functional.pad(o, (0, 0, 0, hd - vd)).reshape(dims.num_heads * hd, -1)
    return wqkv, o.contiguous()


class TtMiMoAttention:
    def __init__(
        self,
        plan,
        cfg,
        layer_idx,
        weights,
        rope_setup,
        cache_dir=None,
        weight_dtype=ttnn.bfloat8_b,
    ):
        """``weights``: dict from MiMoCheckpoint.attention (or None to load from a warm cache)."""
        self.plan = plan
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.dims = dims = cfg.attention_dims(layer_idx)
        tp = plan.tp
        assert dims.num_heads % tp == 0
        assert tp % dims.num_kv_heads == 0 or dims.num_kv_heads % tp == 0
        self.nh_local = dims.num_heads // tp
        self.nkv_local = len(kv_heads_for_chip(0, tp, dims.num_kv_heads))
        self.head_dim = dims.head_dim
        self.scale = dims.scaling
        self.sliding_window = dims.sliding_window
        self.rope = rope_setup
        prefix = f"layer{layer_idx}_attn_tp{tp}"

        wqkv = wo = sinks = decode_sinks = None
        if weights is not None:
            wqkv, wo = prepare_qkv_o(weights, dims, cfg.attention_value_scale, tp)
            if dims.has_sink:
                s = weights["attention_sink_bias"].float() / self.scale
                sinks = s.reshape(1, dims.num_heads, 1, 1)
                decode_sinks = torch.nn.functional.pad(s.reshape(dims.num_heads, 1), (0, TILE - 1))

        self.wqkv = as_weight(wqkv, plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, prefix, "wqkv"))
        self.wo = as_weight(wo, plan, plan.shard(-2), weight_dtype, cache_name(cache_dir, prefix, "wo"))
        self.sinks = self.decode_sinks = None
        if dims.has_sink:
            self.sinks = as_weight(sinks, plan, plan.shard(1), ttnn.bfloat16, cache_name(cache_dir, prefix, "sinks"))
            self.decode_sinks = as_weight(
                decode_sinks, plan, plan.shard(0), ttnn.bfloat16, cache_name(cache_dir, prefix, "decode_sinks")
            )

        self.mm_config = compute_config(ttnn.MathFidelity.HiFi2)
        self.sdpa_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=False, packer_l1_acc=False)
        self.decode_sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

    # ---------------------------------------------------------------- KV cache
    def allocate_kv_cache(self, num_blocks, block_size, dtype=ttnn.bfloat8_b):
        shape = (num_blocks, self.nkv_local, block_size, self.head_dim)
        return [
            ttnn.as_tensor(
                torch.zeros(shape),
                device=self.plan.mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.plan.replicate(),
            )
            for _ in range(2)
        ]

    # ---------------------------------------------------------------- prefill
    def _prefill_sdpa_program(self, seq_len):
        chunk = 256 if seq_len >= 2048 else (128 if seq_len >= 128 else 32)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )

    def forward_prefill(self, x, user_id, page_table, kv_cache, start_pos=0):
        """x [1, 1, S, hidden] (S a multiple of 32) for one user starting at position 0."""
        assert start_pos == 0, "chunked prefill (start_pos > 0) is not implemented yet"
        seq_len = x.shape[-2]
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.nh_local,
            num_kv_heads=self.nkv_local,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        cos, sin = self.rope.get_prefill_rot_mats(start_pos, seq_len)
        trans = self.rope.transformation_mat_prefill
        q_rot = ttnn.experimental.rotary_embedding_llama(q, cos, sin, trans, is_decode_mode=False)
        k_rot = ttnn.experimental.rotary_embedding_llama(k, cos, sin, trans, is_decode_mode=False)
        q.deallocate(True)
        k.deallocate(True)

        k_cache, v_cache = kv_cache
        k_fill = ttnn.typecast(k_rot, k_cache.dtype)
        v_fill = ttnn.typecast(v, v_cache.dtype)
        ttnn.experimental.paged_fill_cache(k_cache, k_fill, page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(v_cache, v_fill, page_table, batch_idx=user_id)
        k_fill.deallocate(True)
        v_fill.deallocate(True)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            attention_sink=self.sinks,
            program_config=self._prefill_sdpa_program(seq_len),
            compute_kernel_config=self.sdpa_config,
        )
        q_rot.deallocate(True)
        k_rot.deallocate(True)
        v.deallocate(True)
        attn_cat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        out = ttnn.linear(
            attn_cat,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        attn_cat.deallocate(True)
        return self.plan.all_reduce(out)

    # ---------------------------------------------------------------- decode
    def forward_decode(self, x, rot_mats, cur_pos, page_table, kv_cache):
        """x [1, 1, B, hidden] (B users padded to a tile); cur_pos int32 [B] on device (-1 skips a slot)."""
        batch = x.shape[-2]
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=self.nh_local,
            num_kv_heads=self.nkv_local,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        trans = self.rope.transformation_mat
        q_rot = ttnn.experimental.rotary_embedding_llama(q, rot_mats[0], rot_mats[1], trans, is_decode_mode=True)
        k_rot = ttnn.experimental.rotary_embedding_llama(k, rot_mats[0], rot_mats[1], trans, is_decode_mode=True)
        q.deallocate(True)
        k.deallocate(True)

        k_cache, v_cache = kv_cache
        ttnn.experimental.paged_update_cache(k_cache, k_rot, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=cur_pos, page_table=page_table)
        k_rot.deallocate(True)
        v.deallocate(True)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_rot,
            k_cache,
            v_cache,
            cur_pos_tensor=cur_pos,
            page_table_tensor=page_table,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            attention_sink=self.decode_sinks,
            program_config=self.decode_sdpa_program,
            compute_kernel_config=self.sdpa_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_rot.deallocate(True)
        batch_grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        sharded = ttnn.create_sharded_memory_config(
            shape=(((self.nh_local + TILE - 1) // TILE) * TILE, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn_sharded = ttnn.to_memory_config(attn, sharded)
        attn.deallocate(True)
        attn_cat = ttnn.experimental.nlp_concat_heads_decode(attn_sharded, num_heads=self.nh_local)
        attn_sharded.deallocate(True)
        attn_cat_dram = ttnn.to_memory_config(attn_cat, ttnn.DRAM_MEMORY_CONFIG)
        attn_cat.deallocate(True)
        out = ttnn.linear(
            attn_cat_dram,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        attn_cat_dram.deallocate(True)
        return self.plan.all_reduce(out)
