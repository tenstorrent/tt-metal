# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash decode attention.

V4.1 keeps V4-Flash's block -- q_a -> q_b, shared-KV MQA over a ``sliding_window`` ring, the
attention sink, inverse RoPE on the output and the grouped o_a -> o_b -- and replaces its
compressed path:

* ``compress_ratios[i]`` is 0 (ring only), 2 (softmax-gated pooling of 2 tokens) or 1
  (one latent per token). Only ``kv_source_layer_ids`` compress; later layers read the
  last source's compressed KV.
* The index key is ``k_norm(wk(latent))``, from the source's pre-RoPE latent.
* ``index_source_layer_ids`` score those keys and keep the top ``index_topk`` rows; later
  layers reuse that selection.
* Past ``candidate_topk_blocks * candidate_block_size`` keys, the select is two-level. The
  ``candidate_source_layer_id`` keeps its best ``candidate_topk_blocks`` blocks of
  ``candidate_block_size`` keys (each scored by its best key, the newest block always kept);
  the later index sources select only among those blocks' keys, which
  ``build_candidate_pool`` copies into a pool. Below that every block is kept, so the plain
  select gives the same rows.
* q_b's per-head RMSNorm is gone.

A compressed layer attends ``[its ring | the selected rows]``. The selected rows are
gathered by ``fused_lightning_select_kv`` into a buffer every layer of the group shares,
so in each step the layers have to run in order. Compressed layers decode one user.

SDPA reads a tiled ``[ring | selected]`` buffer the group shares (``sel_tile``). Its selected
rows are refreshed only by a select; every compressed layer then copies its own tiled ring
over rows ``[0, window)``.
"""

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.common import _HIFI4_SDPA, _MASK_NEG, DeepSeekV4Module, _profile
from models.experimental.deepseek_v4_flash.tt.decode.attention import (
    CSA_INDEX_BLOCK_SIZE,
    _StaticLayerCache,
    _apply_rope,
    _decode_activation,
    _index_key_pool,
    _interleaved_rotate_matrix,
    _one_row_per_user,
    _pack_tokens,
    _packed_users,
    _sdpa_decode_output_config,
    _tp_cluster_axis,
    _update_cache_at,
    int32_pos_tensor,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.decode.attention_csa import DeepSeekV4Indexer, _scale_linear_weight
from models.experimental.deepseek_v4_flash.tt.decode.decode_prefetch import (
    check_decode_layout,
    make_decode_prefetch_buffers,
    ring_prefetch,
)
from models.experimental.deepseek_v4_flash.tt.layers import (
    BatchedLinearDecode,
    DeepSeekV4RMSNorm,
    Linear,
    LinearDecode,
    _core_grid_contains,
)
from models.experimental.deepseek_v4_flash.tt.system_config import active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import _as_cache, _load_weight

from .decode_prefetch import prefetch_rings

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

    Layer ``layer_idx`` is the candidate source when it is ``candidate_source_layer_id``, and a
    candidate consumer when it comes after it.
    """

    def __init__(self, config, layer_idx: int, weights: dict, device, rot, cache, weight_dtype):
        self.device = device
        self.rot = rot
        self.rope_dim = config.qk_rope_head_dim
        self.head_dim = config.index_head_dim
        self.num_heads = config.index_n_heads
        self.index_topk = config.index_topk
        self.sliding_window = config.sliding_window
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
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

    def select_kv(
        self,
        tokens: ttnn.Tensor,
        q_a: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        scache: "DeepSeekV41LayerCache",
        compress_pos: ttnn.Tensor,
        new_row: ttnn.Tensor | None = None,
        new_row_index: ttnn.Tensor | None = None,
        candidates: bool = False,
        build_pool: bool = False,
    ) -> ttnn.Tensor:
        """V4's :meth:`~DeepSeekV4Indexer.select_kv`, two-level when ``candidates`` is set.

        The candidate source then also writes its blocks into ``scache.cand_blocks`` /
        ``pool_count``. A consumer selects from ``scache.pool_keys`` and gathers ``comp_kv`` rows
        through ``pool_rows``; ``build_pool`` first builds that pool from the source's blocks. The
        pool's row table bypasses the page table, so ``new_row_index`` is a ``comp_kv`` row, as it
        is with the identity ``idx_page_table``.
        """
        users = _packed_users(tokens)
        assert users == 1, f"fused_lightning_select_kv decodes one user, got {users}"
        assert not candidates or self.is_candidate_source or self.uses_candidates
        q = self.q_b_proj(_decode_activation(self.q_b_proj, q_a))
        q = _apply_rope(q, cos_row, sin_row, self.rot, self.rope_dim, head_dim=self.head_dim)
        w = self.weights_proj(_decode_activation(self.weights_proj, tokens))
        query = self._replicate_heads(q, self.num_heads, self.head_dim)
        weights = self._replicate_heads(w, 1, self.num_heads)
        cur_pos = ttnn.reshape(compress_pos, [1])

        extra = {}
        if new_row is not None:
            row = ttnn.to_memory_config(new_row, ttnn.DRAM_MEMORY_CONFIG)
            if row.layout != ttnn.ROW_MAJOR_LAYOUT:
                row = ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT)
            extra = {
                "new_kv_row": ttnn.reshape(row, [1, 1, 1, row.shape[-1]]),
                "new_kv_row_index": ttnn.reshape(new_row_index, [1]),
            }
        key_cache, page_table = scache.idx_key_cache, scache.idx_page_table
        if candidates and self.is_candidate_source:
            extra.update(
                candidate_topk_blocks=self.candidate_topk_blocks,
                candidate_block_size=self.candidate_block_size,
                candidate_blocks_output=scache.cand_blocks,
                pool_count_output=scache.pool_count,
            )
        elif candidates:
            if build_pool:
                ttnn.experimental.deepseek.build_candidate_pool(
                    scache.cand_blocks,
                    scache.pool_count,
                    scache.idx_key_cache,
                    scache.idx_page_table,
                    self.candidate_block_size,
                    pool_keys_output=scache.pool_keys,
                    pool_rows_output=scache.pool_rows,
                )
            key_cache, page_table = scache.pool_keys, scache.pool_page_table
            extra.update(num_keys_tensor=scache.pool_count, row_table=scache.pool_rows)
        sel_kv, scores, *_ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            query,
            key_cache,
            weights,
            scache.comp_kv,
            page_table,
            cur_pos,
            self.index_topk,
            output_tensor=scache.sel_kv,
            output_row_offset=self.sliding_window,
            **extra,
        )
        ttnn.deallocate(scores)
        ttnn.deallocate(query)
        ttnn.deallocate(weights)
        if new_row is not None and extra["new_kv_row"].buffer_address() != new_row.buffer_address():
            ttnn.deallocate(extra["new_kv_row"])
        return sel_kv


class DeepSeekV41LayerCache(_StaticLayerCache):
    """V4's layer cache plus V4.1's group buffers.

    ``sel_tile`` is the TILE DRAM ``[1, 1, window + index_topk, Dh]`` SDPA reads. ``new_kv_row`` /
    ``new_key_row`` are ROW_MAJOR DRAM ``[1, 1, 1, Dh]`` / ``[1, 1, 1, index_head_dim]``: the
    compressed-KV row and index key the kv source wrote this step, carried to stages that select
    from the group without holding its source. They are ``None`` where no such stage exists.

    ``win_hist_kv`` / ``win_hist_gate`` (speculative decode, ratio > 1 kv sources) are TILE DRAM
    ``[1, 1, hist, Dh]``: every step's compressor-window row again, at ``pos % hist``, so a
    rejected draft's write into the window can be undone.

    The candidate source's group, when the context outgrows the candidate pool, also has the
    source's ``cand_blocks`` (INT32 ROW_MAJOR ``[1, candidate_topk_blocks]``) and ``pool_count``
    (INT32 ``[1]``), carried to every stage of the group. Stages with a candidate consumer add the
    pool: ``pool_keys`` (TILE ``[P / 64, 1, 64, index_head_dim]``, ``P = candidate_topk_blocks *
    candidate_block_size``), ``pool_rows`` (INT32 ``[1, P]``, the ``comp_kv`` row of each pool key)
    and ``pool_page_table`` (the identity over the pool's blocks).
    """

    __slots__ = (
        "sel_tile",
        "new_kv_row",
        "new_key_row",
        "win_hist_kv",
        "win_hist_gate",
        "cand_blocks",
        "pool_count",
        "pool_keys",
        "pool_rows",
        "pool_page_table",
    )

    def __init__(
        self,
        sel_tile=None,
        new_kv_row=None,
        new_key_row=None,
        cand_blocks=None,
        pool_count=None,
        pool_keys=None,
        pool_rows=None,
        pool_page_table=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sel_tile = sel_tile
        self.new_kv_row = new_kv_row
        self.new_key_row = new_key_row
        self.win_hist_kv = None
        self.win_hist_gate = None
        self.cand_blocks = cand_blocks
        self.pool_count = pool_count
        self.pool_keys = pool_keys
        self.pool_rows = pool_rows
        self.pool_page_table = pool_page_table


def _stash_row(row: ttnn.Tensor, buffer: ttnn.Tensor) -> None:
    """Copy the one-row ``row`` (any layout / memory) into the persistent ROW_MAJOR DRAM ``buffer``."""
    dram = ttnn.to_memory_config(row, ttnn.DRAM_MEMORY_CONFIG)
    rm = ttnn.to_layout(dram, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.copy(ttnn.reshape(rm, list(buffer.shape)), buffer)
    if dram.buffer_address() != row.buffer_address():
        ttnn.deallocate(dram)
    ttnn.deallocate(rm)


class DeepSeekV41Attention(DeepSeekV4Module):
    """V4.1 decode attention for layer ``layer_idx``.

    ``weights`` comes from :func:`~..config.attention_weights`. Weights the checkpoint keeps
    in bf16 (compressor, index ``wk`` / ``weights_proj``) stay bf16; the rest use
    ``weight_dtype``.

    ``tp_size > 1`` expects a 1xTP ``device``: q_a and kv stay full-width on every rank, q_b
    is head-sharded, o_a runs on each rank's complete output groups and the row-parallel o_b
    all-reduces its partials. The compressor and indexer are replicated, so every rank writes
    the same caches and selects the same rows.

    ``use_prefetcher`` streams q_a, q_b, kv, o_a and o_b through the rings of
    :func:`~.decode_prefetch.prefetch_rings`, in ``prefetch_buffers``.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device,
        cache=None,
        weight_dtype=ttnn.bfloat16,
        tp_size: int = 1,
        use_prefetcher: bool = False,
        prefetch_buffers=None,
    ):
        cache = _as_cache(cache)
        sys_cfg = active_system_config()
        self.layer_idx = layer_idx
        self.device = device
        self.tp_size = tp_size
        self.num_heads = config.num_attention_heads
        self.local_num_heads = self.num_heads // tp_size
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.o_groups = config.o_groups
        self.local_o_groups = self.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        self.scaling = self.head_dim**-0.5

        if use_prefetcher and prefetch_buffers is None:
            prefetch_buffers = make_decode_prefetch_buffers(device, weight_dtype, sys_cfg.prefetcher.num_prefetch_pages)
        rings = prefetch_rings(tp_size) if use_prefetcher else {}
        replicate = ttnn.ReplicateTensorToMesh(device) if tp_size > 1 else None
        self.qkv_tp_strategy = "replicated"
        tp_tag = f".tp{tp_size}.{self.qkv_tp_strategy}" if tp_size > 1 else ""

        # q_a's producer grid must sit inside q_b's filled output-mcast rectangle.
        self.q_a_proj = LinearDecode(
            weights["q_a_proj.weight"],
            device,
            cache.file(f"q_a_proj{tp_tag}.full"),
            dtype=weight_dtype,
            mesh_mapper=replicate,
            **DECODE_LAYOUTS["q_a_proj"],
            **ring_prefetch(rings, "q_a_proj", prefetch_buffers, device, weight_dtype),
            rectangle_b_grid=True,
            use_rm_hs=True,
        )
        self.kv_proj = LinearDecode(
            weights["kv_proj.weight"],
            device,
            cache.file(f"kv_proj{tp_tag}.full"),
            dtype=weight_dtype,
            mesh_mapper=replicate,
            **DECODE_LAYOUTS["kv_proj"],
            **ring_prefetch(rings, "kv_proj", prefetch_buffers, device, weight_dtype),
            use_rm_hs=True,
        )
        # One replica of kv on core (0,0): the writer unicasts producer slices into that dest bbox.
        self.kv_proj.set_output_core_grid(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        )
        self.q_b_proj = LinearDecode(
            weights["q_b_proj.weight"],
            device,
            cache.file(f"q_b_proj{tp_tag}.rect"),
            dtype=weight_dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1) if tp_size > 1 else None,
            **{**DECODE_LAYOUTS["q_b_proj"], "N": DECODE_LAYOUTS["q_b_proj"]["N"] // tp_size},
            **ring_prefetch(rings, "q_b_proj", prefetch_buffers, device, weight_dtype),
            rectangle_b_grid=True,
            use_rm_hs=True,
        )
        self.o_b_proj = LinearDecode(
            weights["o_b_proj.weight"],
            device,
            cache.file(f"o_b_proj.tp{tp_size}.row.rect"),
            dtype=weight_dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-2),
            **{**DECODE_LAYOUTS["o_b_proj"], "K": DECODE_LAYOUTS["o_b_proj"]["K"] // tp_size},
            **ring_prefetch(rings, "o_b_proj", prefetch_buffers, device, weight_dtype),
            rectangle_b_grid=True,
            use_rm_hs=True,
        )
        # q_a's and kv's RMSNorms ride their matmul epilogues; q_a's mcasts a ROW_MAJOR
        # HEIGHT_SHARDED replica onto every q_b B core, which q_b reads as A.
        assert self.q_a_proj.can_fuse_rms_norm() and self.kv_proj.can_fuse_rms_norm()
        self.q_a_proj.set_output_core_grid(self.q_b_proj.b_core_grid())
        self.q_a_proj.enable_fused_rms_norm(self.eps, weights["q_a_norm.weight"])
        self.kv_proj.enable_fused_rms_norm(self.eps, weights["kv_norm.weight"])

        # Grouped o_a, block-diagonal over o_groups: one batched matmul_decode, batch axis = group.
        in_per_group = (self.num_heads * self.head_dim) // self.o_groups
        o_a_layout = check_decode_layout("o_a_proj", in_per_group, self.o_lora_rank, batch=self.o_groups)
        b_blocks = o_a_layout["b_blocks"] // tp_size
        n_blocks = o_a_layout["n_blocks"] * tp_size
        self.o_a_proj = BatchedLinearDecode(
            weights["o_a_proj.weight"],
            device,
            cache.file(f"o_a_proj.tp{tp_size}.b{b_blocks}n{n_blocks}" if tp_size > 1 else "o_a_proj"),
            dtype=weight_dtype,
            batch=self.local_o_groups,
            global_batch=self.o_groups,
            K=in_per_group,
            N=self.o_lora_rank,
            b_blocks=b_blocks,
            n_blocks=n_blocks,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3) if tp_size > 1 else None,
            preprocess=lambda w: w.reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).contiguous(),
            **ring_prefetch(rings, "o_a_proj", prefetch_buffers, device, weight_dtype),
        )
        # o_b reads o_a's result as replicated ROW_MAJOR HEIGHT_SHARDED A; NOC mcast needs a filled rectangle.
        self.o_a_proj.set_output_core_grid(ttnn.CoreRangeSet({self.o_b_proj.b_core_grid().bounding_box()}))

        # SDPA-decode multiplies ``scale`` into the sink too; the reference leaves it unscaled.
        sinks = weights["sinks"]
        sinks = sinks() if callable(sinks) else sinks
        sdpa_sink = sinks.reshape(self.num_heads, 1).float() / self.scaling
        self.sdpa_sinks_tt = ttnn.from_torch(
            torch.nn.functional.pad(sdpa_sink, (0, ttnn.TILE_SIZE - 1), "constant", value=0.0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0) if tp_size > 1 else None,
        )
        self._sdpa_pcfg = sys_cfg.attention.sdpa_program_config(device)
        # The rotate-half matrix must stay precise (a bf4 rotation would corrupt RoPE).
        self.rot = _load_weight(_interleaved_rotate_matrix(self.rope_dim), device, cache_file_name=cache.file("rot"))

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
            self.indexer = DeepSeekV41Indexer(config, layer_idx, weights, device, self.rot, cache, weight_dtype)

    def prefetch_weights(self):
        """Queue the prefetched projections in the order :meth:`decode` pops them."""
        for proj in (self.q_a_proj, self.q_b_proj, self.kv_proj, self.o_a_proj, self.o_b_proj):
            if proj.use_prefetcher:
                proj.fetch_weights()

    def _decode_activation_grid(self) -> ttnn.CoreRangeSet:
        """Cores that receive the decode all-gather replica of ``tokens``: q_a's B grid, or kv's if it contains it."""
        grid = self.q_a_proj.b_core_grid()
        kv_grid = self.kv_proj.b_core_grid()
        if kv_grid.num_cores() > grid.num_cores() and _core_grid_contains(kv_grid, grid):
            return kv_grid
        return grid

    def _qkv(self, tokens, cos, sin):
        """RoPE'd ``q`` ``[1, 1, B, H_local*Dh]``, RoPE'd ``kv`` ``[1, 1, B, Dh]`` and the normalized q_a latent."""
        _profile(self.device)
        q_a = self.q_a_proj(_decode_activation(self.q_a_proj, tokens))
        q_b_input = _decode_activation(self.q_b_proj, q_a)
        q = self.q_b_proj(q_b_input)
        if q_b_input is not q_a:
            ttnn.deallocate(q_b_input)
        q = _apply_rope(q, cos, sin, self.rot, self.rope_dim, head_dim=self.head_dim)
        # kv runs after q_b: a ring is one FIFO, so the matmuls pop it in prefetch order.
        kv = self.kv_proj(_decode_activation(self.kv_proj, tokens))
        kv = _apply_rope(kv, cos, sin, self.rot, self.rope_dim)
        return q, kv, q_a

    def _attend(self, q, kv, mask, cos, neg_sin, sdpa_cur_pos=None):
        """SDPA-decode of packed ``q`` over the dense K == V ``kv``, output RoPE, then o_a and o_b.

        Exactly one of ``sdpa_cur_pos`` (causal) / ``mask`` (additive ``[1, 1, 1, Skv]``) bounds
        the KV axis. Returns the block's output on packed rows, ``[1, 1, B, D]``.
        """
        h, dh = self.local_num_heads, self.head_dim
        batch = _packed_users(q)
        out_mem = _sdpa_decode_output_config(batch, h, dh, self._sdpa_pcfg.compute_with_storage_grid_size)
        gathered = ttnn.experimental.deepseek.all_gather_for_matmul(q, out_mem.shard_spec.grid)
        if gathered is not q:
            ttnn.deallocate(q)
        q = ttnn.experimental.view(gathered, [1, batch, h, dh])
        sdpa_kwargs = dict(
            attention_sink=self.sdpa_sinks_tt,
            scale=self.scaling,
            program_config=self._sdpa_pcfg,
            compute_kernel_config=_HIFI4_SDPA,
            memory_config=out_mem,
        )
        if sdpa_cur_pos is not None:
            # No V tensor: the reader builds each V chunk from the K chunk already in L1, so KV is read once.
            attn = ttnn.transformer.flash_multi_latent_attention_decode(
                q, kv, head_dim_v=dh, is_causal=True, cur_pos_tensor=sdpa_cur_pos, **sdpa_kwargs
            )
        else:
            # The V-from-K mode only accepts causal, so masked steps read K and V separately.
            # The op compares logical head counts, so the mask is materialised across H.
            if mask.shape[-2] != h:
                mask = ttnn.repeat(mask, ttnn.Shape([1, 1, h, 1]))
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q, kv, kv, is_causal=False, attn_mask=mask, **sdpa_kwargs
            )
        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim)

        # At M == 1, [1, 1, H, Dh] is already group-major: each group's K features are contiguous.
        _, m, heads, _ = attn.shape
        assert m == 1, f"batched o_a is decode-only (M == 1), got M={m}"
        in_per_group = (heads * dh) // self.local_o_groups
        oa_dest = ttnn.CoreRangeSet({self.o_a_proj.b_core_grid().bounding_box()})
        gathered = ttnn.experimental.deepseek.all_gather_for_matmul(attn, oa_dest)
        if gathered is not attn:
            ttnn.deallocate(attn)
        attn = ttnn.experimental.view(gathered, [1, self.local_o_groups, oa_dest.num_cores(), in_per_group])
        y = self.o_a_proj(attn)
        y = ttnn.experimental.view(y, [1, 1, y.shape[-2], self.local_o_groups * self.o_lora_rank])
        out = self.o_b_proj(_decode_activation(self.o_b_proj, y))
        if self.tp_size > 1:
            reduced = ttnn.experimental.deepseek.width_sharded_all_reduce(
                out, cluster_axis=_tp_cluster_axis(self.device), num_links=2, topology=ttnn.Topology.Ring
            )
            ttnn.deallocate(out)
            out = reduced
        return out

    def _latent(self, tokens, scache, win_slot, pool: bool, hist_slot=None):
        """Pre-RoPE compressed latent ``[1, 1, 1, Dh]`` of the group this step closes, else ``None``.

        ``hist_slot`` also writes the window rows into ``scache.win_hist_kv`` / ``win_hist_gate``.
        """
        kv = ttnn.to_memory_config(
            self.comp_kv_proj(_decode_activation(self.comp_kv_proj, tokens)), ttnn.DRAM_MEMORY_CONFIG
        )
        kv = ttnn.reshape(kv, [1, 1, 1, self.head_dim])
        if self.compress_ratio > 1:
            gate = self.comp_gate_proj(_decode_activation(self.comp_gate_proj, tokens))
            gate = ttnn.reshape(ttnn.to_memory_config(gate, ttnn.DRAM_MEMORY_CONFIG), [1, 1, 1, self.head_dim])
            kv_row, gate_row = _one_row_per_user(kv), _one_row_per_user(gate)
            _update_cache_at(scache.win_kv, kv_row, win_slot)
            _update_cache_at(scache.win_gate, gate_row, win_slot)
            if hist_slot is not None:
                _update_cache_at(scache.win_hist_kv, kv_row, hist_slot)
                _update_cache_at(scache.win_hist_gate, gate_row, hist_slot)
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
        k = _apply_rope(k, cos_win, sin_win, self.rot, self.rope_dim)
        if scache.new_key_row is not None:
            _stash_row(k, scache.new_key_row)
        k = _one_row_per_user(k)
        ttnn.experimental.paged_update_cache(
            scache.idx_key_cache, k, update_idxs_tensor=entry, page_table=scache.idx_page_table
        )
        ttnn.deallocate(k)

    def _write_received_key(self, scache, entry) -> None:
        """Write the index key another stage's kv source sent (``scache.new_key_row``) into row ``entry``."""
        k = _one_row_per_user(scache.new_key_row)
        ttnn.experimental.paged_update_cache(
            scache.idx_key_cache, k, update_idxs_tensor=entry, page_table=scache.idx_page_table
        )
        if k.buffer_address() != scache.new_key_row.buffer_address():
            ttnn.deallocate(k)

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
        received_entry: ttnn.Tensor | None = None,
        hist_slot: ttnn.Tensor | None = None,
        candidates: bool = False,
        build_pool: bool = False,
    ) -> ttnn.Tensor:
        """One decode step: ``hidden`` ``[B, 1, 1, D]`` -> the block's output, same shape.

        The arguments are what :meth:`decode_inputs` builds for a position. ``pool`` closes a
        compressor group (``cos_win`` / ``sin_win`` its RoPE row, ``entry`` its index);
        ``select`` re-selects the top-k rows. Exactly one of ``sdpa_cur_pos`` / ``mask``
        bounds the KV axis. ``received_entry`` (an index source on a stage without its kv
        source, on a step the source pooled) writes the rows in ``scache.new_kv_row`` /
        ``new_key_row`` at that index before selecting. ``hist_slot`` (speculative decode) is
        where a ratio > 1 kv source keeps this step's window rows (see :class:`DeepSeekV41LayerCache`).
        ``candidates`` makes the select two-level and ``build_pool`` has a consumer build the pool
        first (see :meth:`DeepSeekV41Indexer.select_kv`).
        """
        b, s, _, d = hidden.shape
        assert s == 1, f"decode attends one token per user, but S == {s}"
        tokens = ttnn.experimental.deepseek.all_gather_for_matmul(_pack_tokens(hidden), self._decode_activation_grid())
        if self.compress_ratio == 0:
            q, kv, q_a = self._qkv(tokens, cos, sin)
            ttnn.deallocate(q_a)
            if b > 1:
                kv = _one_row_per_user(kv)
            _update_cache_at(scache.kv, kv, sliding_pos)
            ttnn.deallocate(kv)
            out = self._attend(q, scache.kv, None, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos)
            ttnn.deallocate(tokens)
            return ttnn.reshape(out, [b, s, 1, d])
        assert b == 1, "fused_lightning_select_kv decodes one user"
        latent = self._latent(tokens, scache, win_slot, pool, hist_slot) if self.is_kv_source else None
        assert latent is None or select, "a closed group is written by the select that follows it"
        assert received_entry is None or (select and latent is None), "received rows are written by a non-source select"
        assert select or not candidates, "candidates are a select mode"
        q, kv, q_a = self._qkv(tokens, cos, sin)
        _update_cache_at(scache.kv, kv, sliding_pos)
        ttnn.deallocate(kv)
        if select:
            # The select runs on the whole grid; keep q out of its L1 until SDPA.
            q_l1 = q.memory_config()
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            row = None
            if latent is not None:
                self._write_index_key(latent, cos_win, sin_win, scache, entry)
                row = _apply_rope(latent, cos_win, sin_win, self.rot, self.rope_dim)
                if scache.new_kv_row is not None:
                    _stash_row(row, scache.new_kv_row)
            elif received_entry is not None:
                self._write_received_key(scache, received_entry)
                row, entry = scache.new_kv_row, received_entry
            self.indexer.select_kv(
                tokens,
                q_a,
                cos,
                sin,
                scache,
                index_pos,
                new_row=row,
                new_row_index=entry,
                candidates=candidates,
                build_pool=build_pool,
            )
            if latent is not None:
                ttnn.deallocate(row)
            ttnn.tilize(scache.sel_kv, output_tensor=scache.sel_tile)
            q = ttnn.to_memory_config(q, q_l1)
        ttnn.deallocate(q_a)
        ttnn.deallocate(tokens)
        ttnn.fill_cache(scache.sel_tile, scache.kv, 0)
        out = self._attend(q, scache.sel_tile, mask, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos)
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
            indexer = self.indexer
            if n > indexer.candidate_topk_blocks * indexer.candidate_block_size and (
                indexer.is_candidate_source or indexer.uses_candidates
            ):
                kw["candidates"] = True
                kw["build_pool"] = indexer.uses_candidates
        selected = min(n, self.index_topk)
        if pos >= window - 1:
            kw["sdpa_cur_pos"] = int32_pos_tensor(window + selected - 1, device)
        else:
            mask = torch.full((1, 1, 1, window + self.index_topk), _MASK_NEG)
            mask[..., : pos + 1] = 0
            mask[..., window : window + selected] = 0
            kw["mask"] = to_device(mask)
        return kw


def candidate_buffers(device, config, with_pool: bool) -> dict:
    """The candidate source's outputs and, ``with_pool``, a consumer's pool, as
    :class:`DeepSeekV41LayerCache` keyword arguments (all zero, in DRAM)."""

    def int32_zeros(shape):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    buffers = dict(cand_blocks=int32_zeros((1, config.candidate_topk_blocks)), pool_count=int32_zeros((1,)))
    if with_pool:
        pool_size = config.candidate_topk_blocks * config.candidate_block_size
        assert pool_size % CSA_INDEX_BLOCK_SIZE == 0, f"the {pool_size}-key pool must fill whole index blocks"
        pool_blocks = pool_size // CSA_INDEX_BLOCK_SIZE
        buffers.update(
            pool_keys=_index_key_pool(device, pool_blocks, CSA_INDEX_BLOCK_SIZE, config.index_head_dim),
            pool_rows=int32_zeros((1, pool_size)),
            pool_page_table=ttnn.from_torch(
                torch.arange(pool_blocks, dtype=torch.int32).reshape(1, pool_blocks),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
    return buffers


def uses_candidate_pool(config, max_seq: int) -> bool:
    """Whether the candidate source's group outgrows the candidate pool within ``max_seq`` tokens."""
    source = config.candidate_source_layer_id
    if source < 0:
        return False
    ratio = config.compress_ratios[source]
    return max_seq // ratio > config.candidate_topk_blocks * config.candidate_block_size


def build_decode_caches(device, config, layers, max_seq: int) -> dict:
    """Empty batch-1 decode caches for ``layers``, by layer id.

    Every layer owns its ``kv`` ring. A compressed layer aliases its kv source's window,
    compressed-KV, index-key and selection buffers, so its kv and index sources must be
    in ``layers`` too. Past the candidate pool the candidate source's group also gets
    :func:`candidate_buffers`, so the candidate source must be in ``layers`` with its consumers.
    """
    window, dh = config.sliding_window, config.head_dim
    two_level = uses_candidate_pool(config, max_seq)

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
                    sel_tile=zeros((1, 1, window + config.index_topk, dh), ttnn.TILE_LAYOUT),
                )
                source = config.candidate_source_layer_id
                if two_level and src == max(s for s in config.kv_source_layer_ids if s <= source):
                    consumers = [li for li in layers if li in config.index_source_layer_ids and li > source]
                    assert source in layers or not consumers, f"candidate consumers {consumers} need layer {source}"
                    shared[src].update(candidate_buffers(device, config, with_pool=bool(consumers)))
        caches[i] = DeepSeekV41LayerCache(kv=zeros((1, 1, window, dh), ttnn.TILE_LAYOUT), **(shared[src] if r else {}))
    return caches
