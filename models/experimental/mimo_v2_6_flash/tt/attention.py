# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo attention (SWA and global layers), tensor-parallel over one Galaxy row.

SWA layers (8 KV heads): each chip has 8 Q heads and one KV head, in prefill and decode. The cache is
paged like vLLM's ``SlidingWindowSpec`` group: addressed by absolute position, with only the blocks that
overlap the 128-token window mapped (see ``kv.py``). The SDPA kernels skip keys outside the window.

Global layers (4 KV heads over 8 chips): chips 2g and 2g+1 serve KV head g and split its users
(see ``kv.py``). The chip that owns a user keeps that user's whole-context KV and runs all 16 Q heads
of its group for it, in prefill and in decode. Both chips of the pair run the same program; the
non-owner's result for the user is zeroed before the row all-reduce. Each KV head is stored once.

V has 128 dims. It is zero-padded to 192 (and so are the matching ``o_proj`` rows) on both layer types,
so every SDPA kernel sees one head dim, including chunked prefill over the cache. On global layers that
costs 20% more KV than an unpadded V cache; a ``head_dim_v`` in chunked SDPA would remove it.
``attention_value_scale`` is folded into the V weights. Sinks (SWA only) are pre-divided by the
softmax scale because the SDPA kernels apply ``scale`` to the sink logit too.

Prefill takes one user's chunk at ``start_pos`` (a multiple of the block size). It writes the chunk's
K/V into the cache, then attends over its fresh K/V when ``start_pos == 0``, or over the cache with
``chunked_scaled_dot_product_attention`` when earlier chunks (or a prefix-cache hit) are already there.
"""

import torch

import ttnn

from .common import TILE, as_weight, cache_name, compute_config
from .kv import kv_replication, per_chip, slot_owner_masks
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
    per_chip_rows = []
    for c in range(tp):
        rows = [q[c * nh_local * hd : (c + 1) * nh_local * hd]]
        kv_ids = kv_heads_for_chip(c, tp, dims.num_kv_heads)
        rows += [k[h * hd : (h + 1) * hd] for h in kv_ids]
        rows += [torch.nn.functional.pad(v[h * vd : (h + 1) * vd], (0, 0, 0, hd - vd)) for h in kv_ids]
        per_chip_rows.append(torch.cat(rows))
    wqkv = torch.cat(per_chip_rows).T.contiguous()

    o = w["o_proj"].T.reshape(dims.num_heads, vd, -1)
    o = torch.nn.functional.pad(o, (0, 0, 0, hd - vd)).reshape(dims.num_heads * hd, -1)
    return wqkv, o.contiguous()


def prepare_grouped_qkv_o(w, dims, v_scale, tp):
    """Weights for chips that share a KV head: every chip gets all Q heads of its group.

    Fused QKV per chip: [16 Q heads x 192 | K 192 | V padded to 192]. o_proj per chip: the group's
    16 heads, each 128 rows padded to 192 with zeros to match the padded V."""
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
        o_group = torch.nn.functional.pad(o[g * group : (g + 1) * group], (0, 0, 0, hd - vd))
        o_chips.append(o_group.reshape(group * hd, -1))
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
        self.rep = kv_replication(tp, dims.num_kv_heads)
        self.split_users = self.rep > 1  # global layers on TP=8: the chip pair splits the users
        if self.split_users:
            self.nh_local = dims.num_heads // dims.num_kv_heads  # all Q heads of this chip's KV group
            self.nkv_local = 1
        else:
            self.nh_local = dims.num_heads // tp
            self.nkv_local = len(kv_heads_for_chip(0, tp, dims.num_kv_heads))
        self.head_dim = dims.head_dim
        # The kernels take scale as a float32. Round it here: chunked SDPA's binding rejects a Python float that
        # float32 cannot represent exactly (noconvert), and the sinks are divided by the same value.
        self.scale = float(torch.tensor(dims.scaling, dtype=torch.float32))
        self.sliding_window = dims.sliding_window
        self.rope = rope_setup
        prefix = f"layer{layer_idx}_attn_tp{tp}"

        wqkv = wo = sinks = decode_sinks = None
        if weights is not None:
            prepare = prepare_grouped_qkv_o if self.split_users else prepare_qkv_o
            wqkv, wo = prepare(weights, dims, cfg.attention_value_scale, tp)
            if dims.has_sink:
                s = weights["attention_sink_bias"].float() / self.scale
                sinks = s.reshape(1, dims.num_heads, 1, 1)
                decode_sinks = torch.nn.functional.pad(s.reshape(dims.num_heads, 1), (0, TILE - 1))
        names = ("wqkv_grouped", "wo_grouped_vpad") if self.split_users else ("wqkv", "wo")
        self.wqkv = as_weight(wqkv, plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, prefix, names[0]))
        self.wo = as_weight(wo, plan, plan.shard(-2), weight_dtype, cache_name(cache_dir, prefix, names[1]))
        self.sinks = self.decode_sinks = None
        if dims.has_sink:
            self.sinks = as_weight(sinks, plan, plan.shard(1), ttnn.bfloat16, cache_name(cache_dir, prefix, "sinks"))
            self.decode_sinks = as_weight(
                decode_sinks, plan, plan.shard(0), ttnn.bfloat16, cache_name(cache_dir, prefix, "decode_sinks")
            )
        self.prefill_owner_masks = None
        if self.split_users:
            self.prefill_owner_masks = {
                p: per_chip(plan, m, ttnn.bfloat16, ttnn.TILE_LAYOUT) for p, m in slot_owner_masks(tp, self.rep).items()
            }

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
        """[K, V] caches ``[num_blocks, kv_heads, block_size, 192]`` per chip (V zero-padded to 192)."""
        caches = []
        for _ in range(2):
            # Allocate and zero on device: at 1M context a host torch.zeros + conversion would move ~GBs per layer.
            cache = ttnn.allocate_tensor_on_device(
                ttnn.Shape((num_blocks, self.nkv_local, block_size, self.head_dim)),
                dtype,
                ttnn.TILE_LAYOUT,
                self.plan.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.fill(cache, 0.0, output_tensor=cache)
            caches.append(cache)
        return caches

    # ---------------------------------------------------------------- prefill
    def _prefill_sdpa_program(self, seq_len, start_pos):
        """Q/K chunk sizes; chunked SDPA also needs them to divide ``start_pos``."""
        chunk = 256 if seq_len >= 2048 else (128 if seq_len >= 128 else 32)
        while start_pos % chunk or seq_len % chunk:
            chunk //= 2
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )

    def forward_prefill(self, x, user_id, page_table, kv_cache, start_pos=0, real_len=None):
        """x [1, 1, S, hidden]: positions ``start_pos .. start_pos + S - 1`` of the user in batch slot ``user_id``.

        S is a multiple of 32; a prompt's last chunk is padded (``real_len`` real tokens; the padding's K/V
        land past the prompt, where decode overwrites them before reading). ``page_table`` is the layer
        group's per-chip [max_batch, logical blocks] table. ``start_pos`` must be a multiple of the block
        size, and the cache must already hold the user's K/V for positions before it."""
        seq_len = x.shape[-2]
        k_cache, v_cache = kv_cache
        block_size = k_cache.shape[2]
        assert start_pos % block_size == 0, f"start_pos {start_pos} must be a multiple of the block size {block_size}"
        first_block, end_block = start_pos // block_size, -(-(start_pos + seq_len) // block_size)
        assert end_block <= page_table.shape[-1], "chunk runs past the page table"

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

        # This chunk's blocks of the user's table: the fill writes the chunk from its first row.
        chunk_table = ttnn.slice(page_table, (user_id, first_block), (user_id + 1, end_block))
        k_fill = ttnn.typecast(k_rot, k_cache.dtype)
        v_fill = ttnn.typecast(v, v_cache.dtype)
        ttnn.experimental.paged_fill_cache(k_cache, k_fill, chunk_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(v_cache, v_fill, chunk_table, batch_idx=0)
        k_fill.deallocate(True)
        v_fill.deallocate(True)
        chunk_table.deallocate(True)

        program = self._prefill_sdpa_program(seq_len, start_pos)
        if start_pos == 0:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q_rot,
                k_rot,
                v,
                is_causal=True,
                scale=self.scale,
                sliding_window_size=self.sliding_window,
                attention_sink=self.sinks,
                program_config=program,
                compute_kernel_config=self.sdpa_config,
            )
        else:
            # Earlier positions come from the cache, so attend over the cache (the chunk is in it now too).
            user_table = ttnn.slice(page_table, (user_id, 0), (user_id + 1, end_block))
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                q_rot,
                k_cache,
                v_cache,
                user_table,
                start_pos,
                scale=self.scale,
                program_config=program,
                compute_kernel_config=self.sdpa_config,
                sliding_window_size=self.sliding_window,
                attention_sink=self.sinks,
            )
            user_table.deallocate(True)
        q_rot.deallocate(True)
        k_rot.deallocate(True)
        v.deallocate(True)
        attn_cat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        if self.split_users:
            # Only the owner chip of the pair holds this user's cache; the other chip's result is zeroed.
            owned = ttnn.mul(
                attn_cat, self.prefill_owner_masks[user_id % self.rep], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            attn_cat.deallocate(True)
            attn_cat = owned
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
        """x [1, 1, B, hidden] (B users padded to a tile).

        cur_pos: int32 [B] on device (-1 skips a slot), or a dict {"pos", "pos_owned", "owner_mask"}; global
        layers with split users read the per-chip owned view (see kv.py)."""
        owner_mask = None
        if isinstance(cur_pos, dict):
            if self.split_users:
                owner_mask = cur_pos["owner_mask"]
                cur_pos = cur_pos["pos_owned"]
            else:
                cur_pos = cur_pos["pos"]
        assert not self.split_users or owner_mask is not None, "global layers on TP=8 need the per-chip owned view"
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
        update = dict(update_idxs_tensor=cur_pos, page_table=page_table)
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
        if owner_mask is not None:
            # Slots owned by the other chip of the pair were skipped here; zero them before the row all-reduce.
            owned = ttnn.mul(attn_cat_dram, owner_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn_cat_dram.deallocate(True)
            attn_cat_dram = owned
        out = ttnn.linear(
            attn_cat_dram,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        attn_cat_dram.deallocate(True)
        return self.plan.all_reduce(out)
