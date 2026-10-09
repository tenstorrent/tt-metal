# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash decode attention, on top of the V4-Flash block.

V4.1 keeps V4's block -- q_a -> q_b, shared-KV MQA over a ``sliding_window`` ring, the
attention sink, inverse RoPE on the output and the grouped o_a -> o_b -- and replaces its
compressed path:

* ``compress_ratios[i]`` is 0 (ring only), 2 (softmax-gated pooling of 2 tokens) or 1
  (one latent per token). Only ``kv_source_layer_ids`` compress; later layers read the
  last source's compressed KV.
* The index key is ``k_norm(wk(latent))``, from the source's pre-RoPE latent.
* ``index_source_layer_ids`` score those keys and keep the top ``index_topk`` rows; later
  layers reuse that selection.
* q_b's per-head RMSNorm is gone.

A compressed layer attends ``[its ring | the selected rows]``. The selected rows are
gathered by ``fused_lightning_select_kv`` into a buffer every layer of the group shares,
so in each step the layers have to run in order. Compressed layers decode one user.
"""

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.common import _MASK_NEG
from models.experimental.deepseek_v4_flash.tt.decode.attention import (
    CSA_INDEX_BLOCK_SIZE,
    DeepSeekV4Attention,
    _StaticLayerCache,
    _apply_rope,
    _decode_activation,
    _index_key_pool,
    _one_row_per_user,
    _pack_tokens,
    _update_cache_at,
    _update_kv_at,
    int32_pos_tensor,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.decode.attention_csa import DeepSeekV4Indexer, _scale_linear_weight
from models.experimental.deepseek_v4_flash.tt.layers import DeepSeekV4RMSNorm, Linear, LinearDecode
from models.experimental.deepseek_v4_flash.tt.weight_cache import _as_cache

DECODE_LAYOUTS = {
    # 8 cores keep q_a's B grid inside q_b's 8x8 rectangle, the dest of its fused-norm mcast.
    "q_a_proj": {"K": 5120, "N": 1280, "n_blocks": 8},
    "q_b_proj": {"K": 1280, "N": 32768, "n_blocks": 64},
    "kv_proj": {"K": 5120, "N": 512, "n_blocks": 16},
    "o_b_proj": {"K": 8192, "N": 5120},
    # Same cut as kv, so all three read the one decode all-gather replica.
    "compressor.wkv": {"K": 5120, "N": 512, "n_blocks": 16},
    "compressor.wgate": {"K": 5120, "N": 512, "n_blocks": 16},
    # Same 8x8 rectangle as q_b, so it reads the q_a replica q_b already holds.
    "indexer.wq_b": {"K": 1280, "N": 4096, "n_blocks": 64},
    "indexer.weights_proj": {"K": 5120, "N": 32, "n_blocks": 1},
}
# ``fused_lightning_select_kv`` scores ``(cur_pos + 1) // 4`` keys whatever the ratio.
_SELECT_KV_RATE = 4


class DeepSeekV41Indexer(DeepSeekV4Indexer):
    """V4's :meth:`~DeepSeekV4Indexer.select_kv` over the index keys a kv source writes.

    V4.1 derives its keys from the attention compressor's latent, so only the query
    projection and the head weights live here, ``1/sqrt(index_head_dim * index_n_heads)``
    folded into the latter.
    """

    def __init__(self, config, weights: dict, device, rot, cache, weight_dtype):
        self.device = device
        self.rot = rot
        self.rope_dim = config.qk_rope_head_dim
        self.head_dim = config.index_head_dim
        self.num_heads = config.index_n_heads
        self.index_topk = config.index_topk
        self.sliding_window = config.sliding_window
        folded = (self.head_dim * self.num_heads) ** -0.5
        self.q_b_proj = LinearDecode(
            weights["indexer.wq_b.weight"],
            device,
            cache.file("indexer.wq_b"),
            dtype=weight_dtype,
            **DECODE_LAYOUTS["indexer.wq_b"],
            rectangle_b_grid=True,
            use_rm_hs=True,
        )
        self.weights_proj = LinearDecode(
            _scale_linear_weight(weights["indexer.weights_proj.weight"], folded),
            device,
            cache.file("indexer.weights_proj"),
            dtype=ttnn.bfloat16,
            **DECODE_LAYOUTS["indexer.weights_proj"],
            rectangle_b_grid=True,
            use_rm_hs=True,
        )


class DeepSeekV41Attention(DeepSeekV4Attention):
    """V4.1 decode attention for layer ``layer_idx``.

    ``weights`` comes from :func:`~..config.attention_weights`. Weights the checkpoint keeps
    in bf16 (compressor, index ``wk`` / ``weights_proj``) stay bf16; the rest use
    ``weight_dtype``.
    """

    decode_layouts = DECODE_LAYOUTS
    q_head_norm = False

    def __init__(self, config, layer_idx: int, weights: dict, device, cache=None, weight_dtype=ttnn.bfloat16):
        super().__init__(config, layer_idx, weights, device, cache=cache, weight_dtype=weight_dtype)
        cache = _as_cache(cache)
        self.window = config.sliding_window
        self.index_topk = config.index_topk
        self.compress_ratio = config.compress_ratios[layer_idx]
        assert self.compress_ratio in (0, 1, 2), f"no compressor for ratio {self.compress_ratio}"
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        # A source's new entry reaches the compressed KV through its own select.
        assert self.is_index_source or not self.is_kv_source
        if self.is_kv_source:
            self.comp_kv_proj = LinearDecode(
                weights["compressor.wkv.weight"],
                device,
                cache.file("compressor.wkv"),
                dtype=ttnn.bfloat16,
                **DECODE_LAYOUTS["compressor.wkv"],
                use_rm_hs=True,
            )
            self.comp_gate_proj = (
                LinearDecode(
                    weights["compressor.wgate.weight"],
                    device,
                    cache.file("compressor.wgate"),
                    dtype=ttnn.bfloat16,
                    **DECODE_LAYOUTS["compressor.wgate"],
                    use_rm_hs=True,
                )
                if self.compress_ratio > 1
                else None
            )
            self.comp_norm = DeepSeekV4RMSNorm(
                weights["compressor.norm.weight"], self.eps, device, cache.file("compressor.norm"), sharded=True
            )
            self.index_wk = Linear(weights["indexer.wk.weight"], device, cache.file("indexer.wk"))
            self.index_k_norm = DeepSeekV4RMSNorm(
                weights["indexer.k_norm.weight"], self.eps, device, cache.file("indexer.k_norm"), sharded=True
            )
        if self.is_index_source:
            self.indexer = DeepSeekV41Indexer(config, weights, device, self.rot, cache, weight_dtype)

    def _latent(self, tokens, scache, win_slot, pool: bool):
        """Pre-RoPE compressed latent ``[1, 1, 1, Dh]`` of the group this step closes, else ``None``."""
        kv = ttnn.to_memory_config(
            self.comp_kv_proj(_decode_activation(self.comp_kv_proj, tokens)), ttnn.DRAM_MEMORY_CONFIG
        )
        kv = ttnn.reshape(kv, [1, 1, 1, self.head_dim])
        if self.compress_ratio > 1:
            gate = self.comp_gate_proj(_decode_activation(self.comp_gate_proj, tokens))
            gate = ttnn.reshape(ttnn.to_memory_config(gate, ttnn.DRAM_MEMORY_CONFIG), [1, 1, 1, self.head_dim])
            _update_cache_at(scache.win_kv, _one_row_per_user(kv), win_slot)
            _update_cache_at(scache.win_gate, _one_row_per_user(gate), win_slot)
            if not pool:
                return None
            # softmax over a pair is sigmoid of the difference; a softmax over the 2-row
            # window axis would run into its tile padding.
            k0, k1, g0, g1 = (
                ttnn.slice(t, [0, 0, j, 0], [1, 1, j + 1, self.head_dim])
                for t in (scache.win_kv, scache.win_gate)
                for j in (0, 1)
            )
            w0 = ttnn.sigmoid(ttnn.subtract(g0, g1))
            kv = ttnn.add(k1, ttnn.multiply(w0, ttnn.subtract(k0, k1)))
        return self.comp_norm(kv)

    def _write_index_key(self, latent, cos_win, sin_win, scache, entry) -> None:
        """``k_norm(wk(latent))``, RoPE'd at the group's position, into index-key row ``entry``."""
        k = self.index_k_norm(self.index_wk(ttnn.to_memory_config(latent, ttnn.DRAM_MEMORY_CONFIG)))
        k = _one_row_per_user(_apply_rope(k, cos_win, sin_win, self.rot, self.rope_dim))
        ttnn.experimental.paged_update_cache(
            scache.idx_key_cache, k, update_idxs_tensor=entry, page_table=scache.idx_page_table
        )

    def decode(
        self,
        hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        scache: _StaticLayerCache,
        sliding_pos: ttnn.Tensor,
        sdpa_cur_pos: ttnn.Tensor | None = None,
        mask: ttnn.Tensor | None = None,
        cos_win: ttnn.Tensor | None = None,
        sin_win: ttnn.Tensor | None = None,
        win_slot: ttnn.Tensor | None = None,
        entry: ttnn.Tensor | None = None,
        index_pos: ttnn.Tensor | None = None,
        pool: bool = False,
        select: bool = False,
    ) -> ttnn.Tensor:
        """One decode step: ``hidden`` ``[B, 1, 1, D]`` -> the block's output, same shape.

        The arguments are what :meth:`decode_inputs` builds for a position. ``pool`` closes a
        compressor group (``cos_win`` / ``sin_win`` its RoPE row, ``entry`` its index);
        ``select`` re-selects the top-k rows. Exactly one of ``sdpa_cur_pos`` / ``mask``
        bounds the KV axis.
        """
        if self.compress_ratio == 0:
            return self.decode_static(
                hidden, cos, sin, neg_sin, None, None, None, scache, sliding_pos, None, None, sdpa_cur_pos=sdpa_cur_pos
            )
        b, s, _, d = hidden.shape
        assert b * s == 1, "fused_lightning_select_kv decodes one user"
        tokens = ttnn.experimental.deepseek.all_gather_for_matmul(_pack_tokens(hidden), self._decode_activation_grid())
        latent = self._latent(tokens, scache, win_slot, pool) if self.is_kv_source else None
        assert latent is None or select, "a closed group is written by the select that follows it"
        q, kv, q_a = self._qkv(tokens, cos, sin)
        _update_kv_at(scache.kv, kv, sliding_pos)
        ttnn.deallocate(kv)
        if select:
            # The select runs on the whole grid; keep q out of its L1 until SDPA.
            q_l1 = q.memory_config()
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            row = None
            if latent is not None:
                self._write_index_key(latent, cos_win, sin_win, scache, entry)
                row = _apply_rope(latent, cos_win, sin_win, self.rot, self.rope_dim)
            self.indexer.select_kv(tokens, q_a, cos, sin, scache, index_pos, new_row=row, new_row_index=entry)
            q = ttnn.to_memory_config(q, q_l1)
        ttnn.deallocate(q_a)
        ttnn.deallocate(tokens)
        kv = self._selected_kv(scache, self.window)
        out = self._attend(q, kv, mask, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos)
        ttnn.deallocate(kv)
        return ttnn.reshape(out, [b, s, 1, d])

    def decode_inputs(self, pos: int, cos_half: torch.Tensor, sin_half: torch.Tensor) -> dict:
        """Host-built :meth:`decode` keyword arguments for absolute position ``pos``.

        ``cos_half`` / ``sin_half`` are this layer's ``[max_seq, Rd / 2]`` RoPE tables.
        """
        device, window, r = self.device, self.window, self.compress_ratio

        def to_device(t):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        def rope_rows(p):
            cos, sin = make_rope_table(cos_half[p : p + 1], sin_half[p : p + 1])
            return to_device(cos), to_device(sin), to_device(-sin)

        cos, sin, neg_sin = rope_rows(pos)
        kw = dict(cos=cos, sin=sin, neg_sin=neg_sin, sliding_pos=int32_pos_tensor(pos % window, device))
        if r == 0:
            kw["sdpa_cur_pos"] = int32_pos_tensor(min(pos, window - 1), device)
            return kw
        n = (pos + 1) // r
        if self.is_kv_source:
            kw["pool"] = (pos + 1) % r == 0
            if r > 1:
                kw["win_slot"] = int32_pos_tensor(pos % r, device)
            if kw["pool"]:
                # Group j stands at its first token, j * ratio.
                kw["cos_win"], kw["sin_win"], _ = rope_rows(pos + 1 - r)
                kw["entry"] = int32_pos_tensor(pos // r, device)
        if self.is_index_source and n > 0:
            kw["select"] = True
            kw["index_pos"] = int32_pos_tensor(_SELECT_KV_RATE * n - 1, device)
        selected = min(n, self.index_topk)
        if pos >= window - 1:
            kw["sdpa_cur_pos"] = int32_pos_tensor(window + selected - 1, device)
        else:
            mask = torch.full((1, 1, 1, window + self.index_topk), _MASK_NEG)
            mask[..., : pos + 1] = 0
            mask[..., window : window + selected] = 0
            kw["mask"] = to_device(mask)
        return kw


def build_decode_caches(device, config, layers, max_seq: int) -> dict:
    """Empty batch-1 decode caches for ``layers``, by layer id.

    Every layer owns its ``kv`` ring. A compressed layer aliases its kv source's window,
    compressed-KV, index-key and selection buffers, so its kv and index sources must be
    in ``layers`` too.
    """
    window, dh = config.sliding_window, config.head_dim
    # Below this length the candidate pre-filter (layers past candidate_source_layer_id) keeps every block.
    assert max_seq <= config.candidate_topk_blocks * config.candidate_block_size

    def zeros(shape, layout):
        return ttnn.from_torch(
            torch.zeros(shape), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    shared, caches = {}, {}
    for i in sorted(layers):
        r = config.compress_ratios[i]
        if r:
            src = max(s for s in config.kv_source_layer_ids if s <= i)
            index_src = max(s for s in config.index_source_layer_ids if s <= i)
            assert {src, index_src} <= set(layers), f"layer {i} reads layers {src} / {index_src}; decode them too"
            if src not in shared:
                blocks = -(-(max_seq // r) // CSA_INDEX_BLOCK_SIZE)
                shared[src] = dict(
                    win_kv=zeros((1, 1, r, dh), ttnn.TILE_LAYOUT) if r > 1 else None,
                    win_gate=zeros((1, 1, r, dh), ttnn.TILE_LAYOUT) if r > 1 else None,
                    idx_key_cache=_index_key_pool(device, blocks, CSA_INDEX_BLOCK_SIZE, config.index_head_dim),
                    comp_kv=zeros((blocks, 1, CSA_INDEX_BLOCK_SIZE, dh), ttnn.ROW_MAJOR_LAYOUT),
                    idx_page_table=ttnn.from_torch(
                        torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    sel_kv=zeros((1, 1, window + config.index_topk, dh), ttnn.ROW_MAJOR_LAYOUT),
                )
        caches[i] = _StaticLayerCache(kv=zeros((1, 1, window, dh), ttnn.TILE_LAYOUT), **(shared[src] if r else {}))
    return caches
