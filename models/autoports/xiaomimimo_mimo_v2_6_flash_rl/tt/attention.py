# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo attention (SWA and global layers), tensor-parallel over one Galaxy row.

SWA layers (8 KV heads): each chip has 8 Q heads and one KV head, in prefill and decode. V has
128 dims; it is zero-padded to 192 (and so are the matching ``o_proj`` rows) so the SDPA kernels see
one head dim. The SWA cache is a 512-token ring per user, so the padding costs ~nothing.

Global layers (4 KV heads over 8 chips): chips 2g and 2g+1 serve KV head g and split its users
(see ``kv.py``). Prefill keeps 8 Q heads per chip on freshly computed K/V. Decode gives each chip all
16 Q heads of its group for the slots it owns, with an unpadded 128-wide V cache read by
``paged_flash_multi_latent_attention_decode`` (``head_dim_v=128``). Each KV head is stored once
and V is not padded: 320 values per token instead of 2 x 384.
``attention_value_scale`` is folded into the V weights. Sinks (SWA only) are pre-divided by the
softmax scale because the SDPA kernels apply ``scale`` to the sink logit too.
"""

import torch

import ttnn

from .common import TILE, as_weight, cache_name, compute_config
from .kv import kv_replication
from .rope import permute_heads_for_rope

# SWA layers keep a per-user ring of this many tokens (``cache_position_modulo``) instead of the full context.
# It must hold the 128-token window plus the prefill padding tail, which the fill writes into the ring
# after the real tokens (prefill pads by up to 255 tokens), and be a multiple of the block size.
SWA_RING_TOKENS = 512


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


def prepare_grouped_decode_qkv_o(w, dims, v_scale, tp):
    """Decode weights for chips that share a KV head: every chip gets all Q heads of its group.

    Fused QKV per chip: [16 Q heads x 192 | K 192 | V padded to 192] (the decode head-split op wants one
    head dim; only the first 128 V columns are cached). o_proj per chip: the group's 16 x 128 rows, unpadded."""
    hd, vd = dims.head_dim, dims.v_head_dim
    group = dims.num_heads // dims.num_kv_heads
    rep = kv_replication(tp, dims.num_kv_heads)
    q = permute_heads_for_rope(w["q_proj"], dims.num_heads, hd, dims.rope_dim)
    k = permute_heads_for_rope(w["k_proj"], dims.num_kv_heads, hd, dims.rope_dim)
    v = w["v_proj"] * (v_scale if v_scale is not None else 1.0)
    o = w["o_proj"].T.reshape(dims.num_heads, vd, -1)
    qkv_chips, o_chips = [], []
    for c in range(tp):
        g = c // rep
        qkv_chips.append(
            torch.cat(
                [
                    q[g * group * hd : (g + 1) * group * hd],
                    k[g * hd : (g + 1) * hd],
                    torch.nn.functional.pad(v[g * vd : (g + 1) * vd], (0, 0, 0, hd - vd)),
                ]
            )
        )
        o_chips.append(o[g * group : (g + 1) * group].reshape(group * vd, -1))
    return torch.cat(qkv_chips).T.contiguous(), torch.cat(o_chips).contiguous()


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
        self.cache_modulo = SWA_RING_TOKENS if dims.is_swa else None
        self.rep = kv_replication(tp, dims.num_kv_heads)
        self.split_users = self.rep > 1  # global layers on TP=8: the chip pair splits the users
        self.v_head_dim = dims.v_head_dim
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
        if self.split_users:
            self.nh_decode = dims.num_heads // dims.num_kv_heads
            wqkv_d = wo_d = None
            if weights is not None:
                wqkv_d, wo_d = prepare_grouped_decode_qkv_o(weights, dims, cfg.attention_value_scale, tp)
            self.wqkv_decode = as_weight(
                wqkv_d, plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, prefix, "wqkv_grouped")
            )
            self.wo_decode = as_weight(
                wo_d, plan, plan.shard(-2), weight_dtype, cache_name(cache_dir, prefix, "wo_grouped")
            )
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
    def kv_tokens_per_user(self, max_seq_len):
        """Cache capacity per user: the SWA ring, or the whole context for global layers."""
        return self.cache_modulo if self.cache_modulo is not None else max_seq_len

    def allocate_kv_cache(self, num_blocks, block_size, dtype=ttnn.bfloat8_b):
        """[K, V] caches ``[num_blocks, kv_heads, block_size, dim]`` per chip; V is 128 wide when users are split."""
        v_dim = self.v_head_dim if self.split_users else self.head_dim
        caches = []
        for dim in (self.head_dim, v_dim):
            # Allocate and zero on device: at 1M context a host torch.zeros + conversion would move ~GBs per layer.
            cache = ttnn.allocate_tensor_on_device(
                ttnn.Shape((num_blocks, self.nkv_local, block_size, dim)),
                dtype,
                ttnn.TILE_LAYOUT,
                self.plan.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.fill(cache, 0.0, output_tensor=cache)
            caches.append(cache)
        return caches

    # ---------------------------------------------------------------- prefill
    def _prefill_sdpa_program(self, seq_len):
        chunk = 256 if seq_len >= 2048 else (128 if seq_len >= 128 else 32)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )

    def forward_prefill(self, x, user_id, page_table, kv_cache, start_pos=0, real_len=None):
        """x [1, 1, S, hidden] (S a multiple of 32) for one user starting at position 0.
        ``real_len``: prompt length before padding (defaults to S)."""
        assert start_pos == 0, "chunked prefill (start_pos > 0) is not implemented yet"
        seq_len = x.shape[-2]
        real_len = seq_len if real_len is None else real_len
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
        cos.deallocate(True)
        sin.deallocate(True)

        k_cache, v_cache = kv_cache
        k_fill = ttnn.typecast(k_rot, k_cache.dtype)
        if self.split_users:
            v_real = ttnn.slice(v, (0, 0, 0, 0), (1, self.nkv_local, seq_len, self.v_head_dim))
            v_fill = ttnn.typecast(v_real, v_cache.dtype)
            v_real.deallocate(True)
        else:
            v_fill = ttnn.typecast(v, v_cache.dtype)
        if self.cache_modulo is not None:
            assert seq_len - real_len < self.cache_modulo - self.sliding_window, "padding would wrap over the window"
        fill = dict(batch_idx=user_id, cache_position_modulo=self.cache_modulo)
        ttnn.experimental.paged_fill_cache(k_cache, k_fill, page_table, **fill)
        ttnn.experimental.paged_fill_cache(v_cache, v_fill, page_table, **fill)
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
    def _height_sharded(self, batch, rows, width):
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        return ttnn.create_sharded_memory_config(
            shape=(rows, width),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def forward_decode(self, x, rot_mats, cur_pos, page_table, kv_cache):
        """x [1, 1, B, hidden] (B users padded to a tile).

        cur_pos: int32 [B] on device (-1 skips a slot), or a dict {"pos", "pos_owned", "owner_mask"}; global
        layers with split users read the per-chip owned view (see kv.py)."""
        if isinstance(cur_pos, dict):
            if self.split_users:
                return self._forward_decode_split(
                    x, rot_mats, cur_pos["pos_owned"], cur_pos["owner_mask"], page_table, kv_cache
                )
            cur_pos = cur_pos["pos"]
        assert not self.split_users, "global layers on TP=8 need the per-chip owned positions"
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
        update = dict(update_idxs_tensor=cur_pos, page_table=page_table, cache_position_modulo=self.cache_modulo)
        ttnn.experimental.paged_update_cache(k_cache, k_rot, **update)
        ttnn.experimental.paged_update_cache(v_cache, v, **update)
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
            cache_position_modulo=self.cache_modulo,
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

    def _forward_decode_split(self, x, rot_mats, pos_owned, owner_mask, page_table, kv_cache):
        """Global-layer decode: all 16 Q heads of this chip's KV group, for the slots this chip owns."""
        batch = x.shape[-2]
        xqkv = ttnn.linear(
            x,
            self.wqkv_decode,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=self.nh_decode,
            num_kv_heads=1,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        trans = self.rope.transformation_mat
        q_rot = ttnn.experimental.rotary_embedding_llama(q, rot_mats[0], rot_mats[1], trans, is_decode_mode=True)
        k_rot = ttnn.experimental.rotary_embedding_llama(k, rot_mats[0], rot_mats[1], trans, is_decode_mode=True)
        q.deallocate(True)
        k.deallocate(True)

        # The cache keeps only V's 128 real dims.
        v_inter = ttnn.to_memory_config(v, ttnn.L1_MEMORY_CONFIG)
        v.deallocate(True)
        v_real = ttnn.slice(v_inter, (0, 0, 0, 0), (1, batch, 1, self.v_head_dim))
        v_inter.deallocate(True)
        v_shard = ttnn.to_memory_config(v_real, self._height_sharded(batch, TILE, self.v_head_dim))
        v_real.deallocate(True)

        k_cache, v_cache = kv_cache
        update = dict(update_idxs_tensor=pos_owned, page_table=page_table)
        ttnn.experimental.paged_update_cache(k_cache, k_rot, **update)
        ttnn.experimental.paged_update_cache(v_cache, v_shard, **update)
        k_rot.deallocate(True)
        v_shard.deallocate(True)

        attn = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_rot,
            k_cache,
            v_cache,
            head_dim_v=self.v_head_dim,
            page_table_tensor=page_table,
            cur_pos_tensor=pos_owned,
            scale=self.scale,
            program_config=self.decode_sdpa_program,
            compute_kernel_config=self.sdpa_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, B, 16, 128]
        q_rot.deallocate(True)
        heads_pad = ((self.nh_decode + TILE - 1) // TILE) * TILE
        attn_sharded = ttnn.to_memory_config(attn, self._height_sharded(batch, heads_pad, self.v_head_dim))
        attn.deallocate(True)
        attn_cat = ttnn.experimental.nlp_concat_heads_decode(attn_sharded, num_heads=self.nh_decode)
        attn_sharded.deallocate(True)
        attn_cat_dram = ttnn.to_memory_config(attn_cat, ttnn.DRAM_MEMORY_CONFIG)
        attn_cat.deallocate(True)
        # Slots owned by the other chip of the pair were skipped here; zero them before the row all-reduce.
        owned = ttnn.mul(attn_cat_dram, owner_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_cat_dram.deallocate(True)
        out = ttnn.linear(
            owned,
            self.wo_decode,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        owned.deallocate(True)
        return self.plan.all_reduce(out)
