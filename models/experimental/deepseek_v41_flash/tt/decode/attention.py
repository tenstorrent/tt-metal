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

``decode`` with ``S > 1`` steps ``S`` consecutive positions of one session at once (DSpark's
parallel verify, ``DSPARK_PARALLEL_VERIFY.md``), equal to ``S`` single-row ``decode`` steps.
"""

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.common import (
    CCL_TOPOLOGY,
    _HIFI4,
    _HIFI4_SDPA,
    _MASK_NEG,
    DeepSeekV4Module,
    _profile,
    _region,
    width_sharded_l1_config,
)
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
# A key-split select (``num_parts`` select, ``all_gather``, ``merge_select_kv``) has a fixed cost of
# its own and loses below ~45k keys (~31k for the candidate source). The split is fixed per layer, so
# a layer splits only when its maximum key count is past where the split wins averaged over a run
# from position 0.
KEY_SPLIT_MIN_KEYS = 80 * 1024
CANDIDATE_SOURCE_KEY_SPLIT_MIN_KEYS = 48 * 1024


def _rope_tail(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, rot: ttnn.Tensor) -> ttnn.Tensor:
    """Interleaved RoPE on the trailing ``Rd`` columns of every row of ``x`` ``[1, 1, R, W]`` (TILE).

    ``cos`` / ``sin`` are ``[1, 1, R, Rd]`` (one position per row); passing ``-sin`` undoes the rotation.
    ``fused_partial_rope`` takes a per-row table only for TILE input it can shard, so rows at different
    positions go through this instead.
    """
    r, w = x.shape[2], x.shape[3]
    rd = rot.shape[-1]
    nope = ttnn.slice(x, [0, 0, 0, 0], [1, 1, r, w - rd])
    tail = ttnn.slice(x, [0, 0, 0, w - rd], [1, 1, r, w])
    rotated = ttnn.matmul(tail, rot, compute_kernel_config=_HIFI4)
    tail = ttnn.add(ttnn.multiply(tail, cos), ttnn.multiply(rotated, sin))
    return ttnn.concat([nope, tail], dim=-1)


class DeepSeekV41Indexer(DeepSeekV4Indexer):
    """V4's :meth:`~DeepSeekV4Indexer.select_kv` over the index keys a kv source writes.

    V4.1 derives its keys from the attention compressor's latent, so only the query
    projection and the head weights live here, ``1/sqrt(index_head_dim * index_n_heads)``
    folded into the latter.

    Layer ``layer_idx`` is the candidate source when it is ``candidate_source_layer_id``, and a
    candidate consumer when it comes after it.

    ``tp_size > 1`` (a 1xTP ``device``) splits the select's keys over the ranks: each scores its
    contiguous share of the key blocks and keeps its own top-k (and, as the candidate source, its
    best candidate blocks); ``all_gather`` and ``merge_select_kv`` then give every rank the unsplit
    selection. The projections, caches and the merged rows stay replicated. ``split_keys`` turns the
    split off (every rank then scores every key); set it before compiling, e.g. from
    :func:`key_split_layers`.
    """

    def __init__(self, config, layer_idx: int, weights: dict, device, rot, cache, weight_dtype, tp_size: int = 1):
        self.device = device
        self.tp_size = tp_size
        self.split_keys = tp_size > 1
        self.cluster_axis = _tp_cluster_axis(device) if tp_size > 1 else None
        # Rank r scores part r: a trace runs one program on every rank, so the part is read on device.
        self.part_index = (
            ttnn.from_torch(
                torch.arange(tp_size, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
            )
            if tp_size > 1
            else None
        )
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
            dram_sharded_weight=True,
        )
        self.weights_proj = LinearDecode(
            _scale_linear_weight(weights["indexer.weights_proj.weight"], folded),
            device,
            cache.file("indexer.weights_proj"),
            dtype=ttnn.bfloat16,
            **DECODE_LAYOUTS["indexer.weights_proj"],
            rectangle_b_grid=True,
            use_rm_hs=True,
            dram_sharded_weight=True,
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
        page_row: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """V4's :meth:`~DeepSeekV4Indexer.select_kv`, two-level when ``candidates`` is set.

        ``tokens`` / ``q_a`` may hold ``T`` rows of one session (``decode`` with ``S == T``): ``cos_row`` /
        ``sin_row`` are then ROW_MAJOR ``[1, 1, T, Rd]``, ``compress_pos`` ``[T]``, and ``new_row`` /
        ``new_row_index`` ``[1, 1, n, Dh]`` / ``[n]``. One call selects every row's top-k at its own
        position, row ``j`` into its ``window + index_topk`` rows of ``scache.sel_kv_rows`` instead of
        ``sel_kv``. Such a select is one-level and unsplit.

        The candidate source then also writes its blocks into ``scache.cand_blocks`` /
        ``pool_count``. A consumer selects from ``scache.pool_keys`` and gathers ``comp_kv`` rows
        through ``pool_rows``; ``build_pool`` first builds that pool from the source's blocks.

        ``page_row`` (INT32 ROW_MAJOR ``[1]``) picks the session's row of a shared ``[N, L]``
        ``idx_page_table``; without it the table is ``[1, L]``. Block ids are logical (a session's
        entry ``w`` is at logical block ``w // 64``) and ``pool_rows`` are physical ``comp_kv`` rows,
        so a consumer's ``new_row_index`` is translated through ``idx_page_table`` (``kv_page_table``,
        which the op takes only with ``page_row``) while its selected rows are not. A consumer given
        ``new_row`` therefore needs ``page_row``, also with one session.
        """
        rows = _packed_users(tokens)
        assert not candidates or self.is_candidate_source or self.uses_candidates
        assert rows == 1 or not (candidates or self.split_keys), "a multi-row select is one-level and unsplit"
        q = self.q_b_proj(_decode_activation(self.q_b_proj, q_a))
        q = _apply_rope(q, cos_row, sin_row, self.rot, self.rope_dim, head_dim=self.head_dim)
        w = self.weights_proj(_decode_activation(self.weights_proj, tokens))
        # Row j's heads are rows [j * Hi, (j + 1) * Hi) of the query replica.
        query = self._replicate_heads(q, rows * self.num_heads, self.head_dim)
        weights = self._replicate_heads(w, rows, self.num_heads)
        weights = ttnn.experimental.view(weights, [weights.shape[0], 1, rows, self.num_heads])
        cur_pos = ttnn.reshape(compress_pos, [rows])

        extra = {}
        if new_row is not None:
            row = ttnn.to_memory_config(new_row, ttnn.DRAM_MEMORY_CONFIG)
            if row.layout != ttnn.ROW_MAJOR_LAYOUT:
                row = ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT)
            n = row.shape[-2]
            extra = {
                "new_kv_row": ttnn.reshape(row, [1, 1, n, row.shape[-1]]),
                "new_kv_row_index": ttnn.reshape(new_row_index, [n]),
            }
        key_cache, page_table = scache.idx_key_cache, scache.idx_page_table
        if page_row is not None:
            extra["page_table_row_tensor"] = page_row
        split = self.split_keys
        # Under a split the merge writes the selected rows and the candidate blocks.
        candidate_outputs = {}
        if candidates and self.is_candidate_source:
            candidate_outputs = dict(
                candidate_topk_blocks=self.candidate_topk_blocks,
                candidate_block_size=self.candidate_block_size,
                candidate_blocks_output=scache.cand_blocks,
                pool_count_output=scache.pool_count,
            )
            extra.update(
                candidate_topk_blocks=self.candidate_topk_blocks, candidate_block_size=self.candidate_block_size
            )
            if not split:
                extra.update(candidate_blocks_output=scache.cand_blocks, pool_count_output=scache.pool_count)
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
                    page_table_row_tensor=page_row,
                )
            key_cache, page_table = scache.pool_keys, scache.pool_page_table
            extra.update(num_keys_tensor=scache.pool_count, row_table=scache.pool_rows)
            if new_row is not None:
                # Without kv_page_table the op writes new_kv_row at new_row_index as a raw comp_kv row.
                assert page_row is not None, "a consumer writing new_row needs page_row to map its logical entry"
                extra["kv_page_table"] = scache.idx_page_table
            elif page_row is not None:
                del extra["page_table_row_tensor"]
        output = scache.sel_kv if rows == 1 else scache.sel_kv_rows
        output_row_offset = self.sliding_window
        if split:
            extra.update(num_parts=self.tp_size, part_index_tensor=self.part_index)
        else:
            extra.update(
                output_tensor=output,
                output_row_offset=output_row_offset,
                output_row_stride=self.sliding_window + self.index_topk,
            )
        sel_kv, scores, *_ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            query,
            key_cache,
            weights,
            scache.comp_kv,
            page_table,
            cur_pos,
            self.index_topk,
            **extra,
        )
        ttnn.deallocate(scores)
        ttnn.deallocate(query)
        ttnn.deallocate(weights)
        if split:
            local = sel_kv
            gathered = ttnn.all_gather(local, dim=2, cluster_axis=self.cluster_axis)
            ttnn.deallocate(local)
            sel_kv = ttnn.experimental.deepseek.merge_select_kv(
                gathered,
                scache.comp_kv,
                self.index_topk,
                output_tensor=output,
                output_row_offset=output_row_offset,
                **candidate_outputs,
            )
            ttnn.deallocate(gathered)
        if new_row is not None and extra["new_kv_row"].buffer_address() != new_row.buffer_address():
            ttnn.deallocate(extra["new_kv_row"])
        return sel_kv


class DeepSeekV41LayerCache(_StaticLayerCache):
    """V4's layer cache plus V4.1's group buffers.

    ``sel_tile`` is the TILE DRAM ``[1, 1, window + index_topk, Dh]`` SDPA reads. ``new_kv_row`` /
    ``new_key_row`` are ROW_MAJOR DRAM ``[1, 1, 1, Dh]`` / ``[1, 1, 1, index_head_dim]``: the
    compressed-KV row and index key the kv source wrote this step, carried to stages that select
    from the group without holding its source. They are ``None`` where no such stage exists.

    Under speculative decode a ratio > 1 kv source's ``win_kv`` / ``win_gate`` are deeper than the ratio
    (``[1, 1, rows, Dh]``, row ``pos % rows``): a rejected draft's row then lands in a slot of its own, and
    the next group's pool reads the row before it through ``win_prev`` (see :meth:`DeepSeekV41Attention.decode`).

    The candidate source's group, when the context outgrows the candidate pool, also has the
    source's ``cand_blocks`` (INT32 ROW_MAJOR ``[1, candidate_topk_blocks]``) and ``pool_count``
    (INT32 ``[1]``), carried to every stage of the group. Stages with a candidate consumer add the
    pool: ``pool_keys`` (TILE ``[P / 64, 1, 64, index_head_dim]``, ``P = candidate_topk_blocks *
    candidate_block_size``), ``pool_rows`` (INT32 ``[1, P]``, the ``comp_kv`` row of each pool key)
    and ``pool_page_table`` (the identity over the pool's blocks).

    With several sessions, ``kv`` is TILE ``[1, 1, N * window, Dh]`` (session slot ``s`` at rows
    ``[s * window, (s + 1) * window)``) and ``win_kv`` / ``win_gate`` are ``[1, 1, N * 32, Dh]``
    (slot ``s`` at row ``s * 32``). A step copies its slot's rows out with ``fill_cache``'s
    ``input_row_offset_tensor``: a compressed layer straight into ``sel_tile``, a sliding layer into
    ``ring_view`` (TILE ``[1, 1, window, Dh]``, which SDPA reads), a kv source's window into
    ``win_view_kv`` /     ``win_view_gate`` (TILE ``[1, 1, 32, Dh]``). The views are ``None`` with
    one session's dense buffers.

    A cache built for ``T``-row steps (:meth:`DeepSeekV41Attention.decode` with ``S == T``) also has
    ``sel_kv_rows`` (ROW_MAJOR) and ``sel_tile_rows`` (TILE), DRAM ``[1, 1, T * (window + index_topk), Dh]``:
    row ``j``'s ``[ring | selected]`` at rows ``[j * (window + index_topk), (j + 1) * (window + index_topk))``.
    Single-row steps keep using ``sel_kv`` / ``sel_tile``, so the two kinds of step interleave. Where the
    group has ``new_kv_row``, a ``T``-row step of its kv source leaves the rows of the ``n`` groups it closed
    in ``new_kv_rows`` / ``new_key_rows`` (DRAM ``[1, 1, n, Dh]`` / ``[1, 1, n, index_head_dim]``, fresh each
    step), and a stage without the source receives them there.
    """

    __slots__ = (
        "sel_tile",
        "sel_kv_rows",
        "sel_tile_rows",
        "new_kv_row",
        "new_key_row",
        "new_kv_rows",
        "new_key_rows",
        "cand_blocks",
        "pool_count",
        "pool_keys",
        "pool_rows",
        "pool_page_table",
        "ring_view",
        "win_view_kv",
        "win_view_gate",
    )

    def __init__(
        self,
        sel_tile=None,
        sel_kv_rows=None,
        sel_tile_rows=None,
        new_kv_row=None,
        new_key_row=None,
        cand_blocks=None,
        pool_count=None,
        pool_keys=None,
        pool_rows=None,
        pool_page_table=None,
        ring_view=None,
        win_view_kv=None,
        win_view_gate=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sel_tile = sel_tile
        self.sel_kv_rows = sel_kv_rows
        self.sel_tile_rows = sel_tile_rows
        self.new_kv_row = new_kv_row
        self.new_key_row = new_key_row
        self.new_kv_rows = None
        self.new_key_rows = None
        self.cand_blocks = cand_blocks
        self.pool_count = pool_count
        self.pool_keys = pool_keys
        self.pool_rows = pool_rows
        self.pool_page_table = pool_page_table
        self.ring_view = ring_view
        self.win_view_kv = win_view_kv
        self.win_view_gate = win_view_gate


def _rows_on_one_core(x: ttnn.Tensor) -> ttnn.Tensor:
    """``[1, 1, R, F]`` -> ROW_MAJOR, height-sharded on one core (shard ``[R, F]``): one user's ``R`` rows,
    which ``paged_update_cache`` writes at ``R`` positions in one call. :func:`_one_row_per_user` at ``R == 1``."""
    r, f = x.shape[-2], x.shape[-1]
    grid = ttnn.num_cores_to_corerangeset(1, x.device().compute_with_storage_grid_size(), row_wise=True)
    want = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [r, f], ttnn.ShardOrientation.ROW_MAJOR),
    )
    if x.layout == ttnn.ROW_MAJOR_LAYOUT and x.memory_config() == want:
        return x
    if x.is_sharded():
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.to_memory_config(x, want)


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
    all-reduces its partials. The compressor and the indexer projections are replicated, so every
    rank writes the same caches; the select scores a share of the keys on each rank and merges
    (see :class:`DeepSeekV41Indexer`), so every rank still selects the same rows.

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
            dram_sharded_weight=True,
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
            dram_sharded_weight=True,
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
            dram_sharded_weight=True,
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
            dram_sharded_weight=True,
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
            dram_sharded_weight=True,
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
            # On kv's B cores (its ring's receivers under the prefetcher), which the decode
            # all-gather replica of the tokens already covers.
            self.comp_kv_proj = LinearDecode(
                weights["compressor.wkv.weight"],
                device,
                cache.file("compressor.wkv"),
                dtype=ttnn.bfloat16,
                **DECODE_LAYOUTS["compressor.wkv"],
                use_rm_hs=True,
                dram_sharded_weight=True,
                b_core_grid=self.kv_proj.b_core_grid(),
            )
            self.comp_gate_proj = (
                LinearDecode(
                    weights["compressor.wgate.weight"],
                    device,
                    cache.file("compressor.wgate"),
                    dtype=ttnn.bfloat16,
                    **DECODE_LAYOUTS["compressor.wgate"],
                    use_rm_hs=True,
                    dram_sharded_weight=True,
                    b_core_grid=self.kv_proj.b_core_grid(),
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
            self.indexer = DeepSeekV41Indexer(
                config, layer_idx, weights, device, self.rot, cache, weight_dtype, tp_size=tp_size
            )

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

    def _project_qkv(self, tokens):
        """``q`` ``[M, H_local*Dh]``, ``kv`` ``[M, Dh]`` (both before RoPE) and the normalized q_a latent."""
        _profile(self.device)
        q_a = self.q_a_proj(_decode_activation(self.q_a_proj, tokens))
        q_b_input = _decode_activation(self.q_b_proj, q_a)
        q = self.q_b_proj(q_b_input)
        if q_b_input is not q_a:
            ttnn.deallocate(q_b_input)
        # kv runs after q_b: a ring is one FIFO, so the matmuls pop it in prefetch order.
        kv = self.kv_proj(_decode_activation(self.kv_proj, tokens))
        return q, kv, q_a

    def _qkv(self, tokens, cos, sin):
        """RoPE'd ``q`` ``[1, 1, B, H_local*Dh]``, RoPE'd ``kv`` ``[1, 1, B, Dh]`` and the normalized q_a latent."""
        q, kv, q_a = self._project_qkv(tokens)
        q = _apply_rope(q, cos, sin, self.rot, self.rope_dim, head_dim=self.head_dim)
        kv = _apply_rope(kv, cos, sin, self.rot, self.rope_dim)
        return q, kv, q_a

    def _sdpa(self, q, kv, mask, sdpa_cur_pos, memory_config):
        """SDPA-decode of ``q`` ``[1, B, H_local, Dh]`` over K == V ``kv`` ``[B or 1, 1, Skv, Dh]``.

        Exactly one of ``sdpa_cur_pos`` (causal, ``[B]``) / ``mask`` (additive ``[B or 1, 1, 1, Skv]``)
        bounds the KV axis.
        """
        sdpa_kwargs = dict(
            attention_sink=self.sdpa_sinks_tt,
            scale=self.scaling,
            program_config=self._sdpa_pcfg,
            compute_kernel_config=_HIFI4_SDPA,
            memory_config=memory_config,
        )
        if sdpa_cur_pos is not None:
            # No V tensor: the reader builds each V chunk from the K chunk already in L1, so KV is read once.
            return ttnn.transformer.flash_multi_latent_attention_decode(
                q, kv, head_dim_v=self.head_dim, is_causal=True, cur_pos_tensor=sdpa_cur_pos, **sdpa_kwargs
            )
        # The V-from-K mode only accepts causal, so masked steps read K and V separately.
        # The op compares logical head counts, so the mask is materialised across H.
        h = self.local_num_heads
        if mask.shape[-2] != h:
            mask = ttnn.repeat(mask, ttnn.Shape([1, 1, h, 1]))
        return ttnn.transformer.scaled_dot_product_attention_decode(
            q, kv, kv, is_causal=False, attn_mask=mask, **sdpa_kwargs
        )

    def _attend(self, q, kv, mask, cos, neg_sin, sdpa_cur_pos=None):
        """SDPA-decode of packed ``q`` ``[1, 1, B, H_local*Dh]`` over the dense K == V ``kv``, output RoPE,
        then o_a and o_b.

        ``B`` is the users, or one session's ``B`` rows (``kv`` ``[B, 1, Skv, Dh]``, row ``j``'s keys at batch
        ``j``). Exactly one of ``sdpa_cur_pos`` (causal, ``[B]``) / ``mask`` (additive ``[1 or B, 1, 1, Skv]``)
        bounds the KV axis. ``cos`` / ``neg_sin`` are :meth:`decode`'s. Returns the block's output on packed
        rows, ``[1, 1, B, D]``.
        """
        h, dh = self.local_num_heads, self.head_dim
        batch = _packed_users(q)
        out_mem = _sdpa_decode_output_config(batch, h, dh, self._sdpa_pcfg.compute_with_storage_grid_size)
        if batch == 1:
            gathered = ttnn.experimental.deepseek.all_gather_for_matmul(q, out_mem.shard_spec.grid)
        else:
            # The gather replicates every row onto every core; SDPA reads row b alone on core b.
            dram = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            one_row_per_core = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(out_mem.shard_spec.grid, [1, h * dh], ttnn.ShardOrientation.ROW_MAJOR),
            )
            gathered = ttnn.to_memory_config(dram, one_row_per_core)
            if dram.buffer_address() != q.buffer_address():
                ttnn.deallocate(dram)
        if gathered is not q:
            ttnn.deallocate(q)
        q = ttnn.experimental.view(gathered, [1, batch, h, dh])
        attn = self._sdpa(q, kv, mask, sdpa_cur_pos, out_mem)
        # One row per core with its heads side by side, so a per-row RoPE table row covers all of them.
        attn = ttnn.experimental.view(attn, [1, 1, batch, h * dh])
        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim, head_dim=dh)
        return self._project_out(attn)

    def _project_out(self, attn):
        """Grouped o_a and row-parallel o_b of ``M`` rows of attention ``[1, 1, M, H_local*Dh]`` (RoPE undone),
        ROW_MAJOR height-sharded one row per core -> ``[1, 1, M, D]``."""
        m, width = attn.shape[-2], attn.shape[-1]
        in_per_group = width // self.local_o_groups
        oa_dest = ttnn.CoreRangeSet({self.o_a_proj.b_core_grid().bounding_box()})
        if m > 1:
            # all_gather_for_matmul takes a height-sharded input only on one core.
            dram = ttnn.to_memory_config(attn, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn)
            attn = ttnn.to_memory_config(
                dram, width_sharded_l1_config(m, width, self.device, num_cores=self.local_o_groups, tile_height=1)
            )
            ttnn.deallocate(dram)
        gathered = ttnn.experimental.deepseek.all_gather_for_matmul(attn, oa_dest)
        if gathered is not attn:
            ttnn.deallocate(attn)
        # A row's groups are contiguous, so the replica's [M * groups, K] rows are ordered [M][group],
        # the order batched o_a reads.
        attn = ttnn.experimental.view(gathered, [1, self.local_o_groups, oa_dest.num_cores() * m, in_per_group])
        y = self.o_a_proj(attn)
        y = ttnn.experimental.view(y, [1, 1, y.shape[-2], self.local_o_groups * self.o_lora_rank])
        out = self.o_b_proj(_decode_activation(self.o_b_proj, y))
        if self.tp_size > 1:
            reduced = ttnn.experimental.deepseek.width_sharded_all_reduce(
                out, cluster_axis=_tp_cluster_axis(self.device), num_links=2, topology=CCL_TOPOLOGY
            )
            ttnn.deallocate(out)
            out = reduced
        return out

    def _compressor_rows(self, tokens):
        """The compressor's ``kv`` / ``gate`` projections of the packed tokens, DRAM ``[1, 1, M, Dh]``
        each (``gate`` is ``None`` at ratio 1)."""
        m = _packed_users(tokens)
        kv = ttnn.to_memory_config(
            self.comp_kv_proj(_decode_activation(self.comp_kv_proj, tokens)), ttnn.DRAM_MEMORY_CONFIG
        )
        kv = ttnn.reshape(kv, [1, 1, m, self.head_dim])
        if self.compress_ratio == 1:
            return kv, None
        gate = self.comp_gate_proj(_decode_activation(self.comp_gate_proj, tokens))
        gate = ttnn.reshape(ttnn.to_memory_config(gate, ttnn.DRAM_MEMORY_CONFIG), [1, 1, m, self.head_dim])
        return kv, gate

    def _latent(self, kv, gate, scache, win_slot, pool: bool, win_prev=None, win_offset=None):
        """Pre-RoPE compressed latent ``[1, 1, 1, Dh]`` of the group this step closes, else ``None``.

        ``kv`` / ``gate`` are one row of :meth:`_compressor_rows`.
        ``win_prev`` (a window deeper than the ratio) is the one-hot ``[1, 1, 1, rows]`` of the previous
        position's window slot: the pool pairs that row with this step's own.
        ``win_offset`` (INT32 ``[1]``) is the first window row of the session's slot, already folded
        into ``win_slot``; the pool reads the slot through ``win_view_kv`` / ``win_view_gate``.
        """
        if self.compress_ratio > 1:
            kv_row, gate_row = _one_row_per_user(kv), _one_row_per_user(gate)
            _update_cache_at(scache.win_kv, kv_row, win_slot)
            _update_cache_at(scache.win_gate, gate_row, win_slot)
            if not pool:
                return None
            windows = (scache.win_kv, scache.win_gate)
            if win_prev is not None:
                # One-hot products are exact.
                k0, g0 = (ttnn.matmul(win_prev, w, compute_kernel_config=_HIFI4) for w in windows)
                k1, g1 = (ttnn.to_layout(x, ttnn.TILE_LAYOUT) for x in (kv, gate))
            else:
                assert (
                    win_offset is not None or scache.win_kv.shape[2] == self.compress_ratio
                ), "a window deeper than the ratio pairs through win_prev"
                if win_offset is not None:
                    rows = scache.win_view_kv.shape[2]
                    for view, win in zip((scache.win_view_kv, scache.win_view_gate), windows):
                        ttnn.fill_cache(view, win, 0, input_row_offset_tensor=win_offset, num_input_rows=rows)
                    windows = (scache.win_view_kv, scache.win_view_gate)
                k0, k1, g0, g1 = (
                    ttnn.slice(t, [0, 0, j, 0], [1, 1, j + 1, self.head_dim]) for t in windows for j in (0, 1)
                )
            # softmax over a pair is sigmoid of the difference; a softmax over the 2-row
            # window axis would run into its tile padding.
            w0 = ttnn.sigmoid(ttnn.subtract(g0, g1))
            kv = ttnn.add(k1, ttnn.multiply(w0, ttnn.subtract(k0, k1)))
        return self.comp_norm(kv)

    def _latent_rows(self, kv, gate, scache, win_slot, pool_rows: tuple, win_prev=None):
        """:meth:`_latent` for ``T`` consecutive rows: the pre-RoPE latents ``[1, 1, n, Dh]`` of the ``n``
        groups that rows ``pool_rows`` close, in row order, else ``None``.

        ``kv`` / ``gate`` are :meth:`_compressor_rows` of the ``T`` rows; ``win_slot`` is ``[T]``. ``win_prev``
        (a window deeper than the ratio) is the one-hot ``[1, 1, 1, rows]`` of the window slot of the position
        before row 0. The window ends as ``T`` single-row steps leave it.
        """
        if self.compress_ratio == 1:
            assert pool_rows == tuple(range(kv.shape[-2])), "at ratio 1 every row closes its group"
            return self.comp_norm(kv)
        t, dh = kv.shape[-2], self.head_dim
        windows = (scache.win_kv, scache.win_gate)
        # Row 0 may close the group the previous step opened; read that row before the writes.
        if win_prev is not None:
            prevs = [
                ttnn.to_layout(ttnn.matmul(win_prev, w, compute_kernel_config=_HIFI4), ttnn.ROW_MAJOR_LAYOUT)
                for w in windows
            ]
        else:
            assert (
                scache.win_kv.shape[2] == self.compress_ratio
            ), "a window deeper than the ratio pairs through win_prev"
            prevs = [ttnn.to_layout(ttnn.slice(w, [0, 0, 0, 0], [1, 1, 1, dh]), ttnn.ROW_MAJOR_LAYOUT) for w in windows]
        rows = [_rows_on_one_core(kv), _rows_on_one_core(gate)]
        # A slot named twice keeps the later row, as the single-row steps would.
        for win, row in zip(windows, rows):
            ttnn.experimental.paged_update_cache(win, row, update_idxs_tensor=win_slot)
        for row in rows:
            ttnn.deallocate(row)
        if not pool_rows:
            return None
        first, n = pool_rows[0], len(pool_rows)
        assert pool_rows == tuple(range(first, t, 2)), f"ratio-2 groups close every other row, got {pool_rows}"
        # Closing row j pairs xs[j] (the row before it) with xs[j + 1] (itself).
        pairs = []
        for prev, x in zip(prevs, (kv, gate)):
            xs = ttnn.concat([prev, x], dim=2)
            ttnn.deallocate(prev)
            for start in (first, first + 1):
                pick = ttnn.slice(xs, [0, 0, start, 0], [1, 1, start + 2 * n - 1, dh], [1, 1, 2, 1])
                pairs.append(ttnn.to_layout(pick, ttnn.TILE_LAYOUT))
            ttnn.deallocate(xs)
        k0, k1, g0, g1 = pairs
        w0 = ttnn.sigmoid(ttnn.subtract(g0, g1))
        return self.comp_norm(ttnn.add(k1, ttnn.multiply(w0, ttnn.subtract(k0, k1))))

    def _write_causal_rows(self, kv, copies, ring_rows) -> None:
        """Write row ``i`` of ``kv`` ``[1, 1, T, Dh]`` (ROW_MAJOR, on one core) into ring copy ``j`` of
        ``copies`` ``[T, 1, Skv, Dh]`` (TILE) at row ``ring_rows[j * T + i]``, ``-1`` skipping it: copy ``j``
        then holds rows ``0 .. j`` in their slots and the older keys in the rest."""
        t = kv.shape[-2]
        grid = ttnn.num_cores_to_corerangeset(t, self.device.compute_with_storage_grid_size(), row_wise=True)
        # paged_update_cache takes copy j's rows on core j.
        replicas = ttnn.experimental.deepseek.all_gather_for_matmul(kv, grid)
        ttnn.experimental.paged_update_cache(
            copies, ttnn.experimental.view(replicas, [1, t, t, kv.shape[-1]]), update_idxs_tensor=ring_rows
        )
        ttnn.deallocate(replicas)

    def _write_index_key(self, latent, cos_win, sin_win, scache, entry, page_row=None, multi_row=False) -> None:
        """``k_norm(wk(latent))``, RoPE'd at the groups' positions, into index-key rows ``entry`` (logical,
        through row ``page_row`` of ``idx_page_table`` when given). ``latent`` is ``[1, 1, n, Dh]``, ``entry``
        ``[n]``. ``multi_row`` (a ``T``-row step) keeps the keys for other stages in ``scache.new_key_rows``."""
        k = self.index_k_norm(self.index_wk(ttnn.to_memory_config(latent, ttnn.DRAM_MEMORY_CONFIG)))
        k = _apply_rope(k, cos_win, sin_win, self.rot, self.rope_dim)
        if scache.new_key_row is not None:
            if multi_row:
                scache.new_key_rows = ttnn.clone(ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG))
            else:
                _stash_row(k, scache.new_key_row)
        k = _rows_on_one_core(k)
        ttnn.experimental.paged_update_cache(
            scache.idx_key_cache,
            k,
            update_idxs_tensor=entry,
            page_table=scache.idx_page_table,
            page_table_row_tensor=page_row,
        )
        ttnn.deallocate(k)

    def _write_received_key(self, scache, entry, page_row=None, multi_row=False) -> None:
        """Write the index key another stage's kv source sent (``scache.new_key_row``) into row ``entry``;
        ``multi_row``: the ``n`` keys of ``scache.new_key_rows`` into rows ``entry`` ``[n]``."""
        received = scache.new_key_rows if multi_row else scache.new_key_row
        k = _rows_on_one_core(received) if multi_row else _one_row_per_user(received)
        ttnn.experimental.paged_update_cache(
            scache.idx_key_cache,
            k,
            update_idxs_tensor=entry,
            page_table=scache.idx_page_table,
            page_table_row_tensor=page_row,
        )
        if k.buffer_address() != received.buffer_address():
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
        win_prev: ttnn.Tensor | None = None,
        candidates: bool = False,
        build_pool: bool = False,
        page_row: ttnn.Tensor | None = None,
        ring_offset: ttnn.Tensor | None = None,
        win_offset: ttnn.Tensor | None = None,
        ring_rows: ttnn.Tensor | None = None,
        pool_rows: tuple = (),
    ) -> ttnn.Tensor:
        """One decode step: ``hidden`` ``[B, S, 1, D]`` -> the block's output, same shape.

        ``S > 1`` (``B == 1``) steps one session's ``S`` consecutive positions at once: row ``j`` is position
        ``p + j`` and sees what a single-row step at ``p + j`` would after rows ``0 .. j - 1``, so the output and
        every cache equal ``S`` single-row steps (up to bf16), e.g. a DSpark verify of ``[t, d1 .. d(S-1)]``.
        ``hidden`` is ROW_MAJOR, width-sharded in L1 with all ``S`` rows on every core. Each row attends its own
        copy of the ring with the slots of rows ``0 .. j`` replaced (``ring_rows``, see :meth:`_write_causal_rows`),
        and a compressed layer reads row ``j``'s ``[ring | selected]`` from ``scache.sel_tile_rows``. Such a
        step needs one session's dense caches and has no two-level select; its cross-stage rows go through
        ``scache.new_kv_rows`` / ``new_key_rows`` (``received_entry`` ``[n]`` on the receiving stage).

        The arguments are what :meth:`decode_inputs` builds for a position (``rows=S`` for ``S > 1``,
        where ``pool_rows`` replaces ``pool``). ``pool`` closes a compressor group (``cos_win`` /
        ``sin_win`` its RoPE row, ``entry`` its index); ``select`` re-selects the top-k rows. Exactly
        one of ``sdpa_cur_pos`` / ``mask`` bounds the KV axis. ``received_entry`` (an index source on a stage without its kv
        source, on a step the source pooled) writes the rows in ``scache.new_kv_row`` /
        ``new_key_row`` at that index before selecting. ``win_prev`` (speculative decode, whose ratio > 1
        windows are deeper than the ratio, see :class:`DeepSeekV41LayerCache`) is the bf16 one-hot
        ``[1, 1, 1, rows]`` of the window slot of position ``pos - 1``, which ``win_slot`` (``pos % rows``)
        no longer implies; a kv source needs it whenever its first row closes a group.
        ``candidates`` makes the select two-level and ``build_pool`` has a consumer build the pool
        first (see :meth:`DeepSeekV41Indexer.select_kv`).

        ``page_row`` (INT32 ROW_MAJOR ``[1]``) is the session's slot, its row of every ``idx_page_table``
        (0 with one session). Several sessions (see :class:`DeepSeekV41LayerCache`): ``ring_offset`` /
        ``win_offset`` are the first rows of its slot in ``kv`` and in ``win_kv`` / ``win_gate``.
        ``sliding_pos`` and ``win_slot`` already include those offsets, and ``entry`` stays the
        session's logical row.
        """
        b, s, _, d = hidden.shape
        if s > 1:
            assert b == 1, f"S > 1 steps the rows of one session, got B == {b}"
            assert s <= self.window, f"{s} rows would overwrite each other's ring slots"
            assert ring_offset is None and win_offset is None, "several sessions' rings decode one row at a time"
            assert ring_rows is not None, "decode_inputs(..., rows=S) builds the rows' ring copies"
        tokens = ttnn.experimental.deepseek.all_gather_for_matmul(_pack_tokens(hidden), self._decode_activation_grid())
        if self.compress_ratio == 0:
            q, kv, q_a = self._qkv(tokens, cos, sin)
            ttnn.deallocate(q_a)
            ttnn.deallocate(tokens)
            if s > 1:
                rings = ttnn.repeat(scache.kv, ttnn.Shape([s, 1, 1, 1]))
                self._write_causal_rows(kv, rings, ring_rows)
                ttnn.experimental.paged_update_cache(scache.kv, kv, update_idxs_tensor=sliding_pos)
                ttnn.deallocate(kv)
                out = self._attend(q, rings, None, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos)
                ttnn.deallocate(rings)
                return ttnn.reshape(out, [b, s, 1, d])
            if b > 1:
                kv = _one_row_per_user(kv)
            _update_cache_at(scache.kv, kv, sliding_pos)
            ttnn.deallocate(kv)
            ring = scache.kv
            if ring_offset is not None:
                ring = scache.ring_view
                ttnn.fill_cache(ring, scache.kv, 0, input_row_offset_tensor=ring_offset, num_input_rows=ring.shape[2])
            out = self._attend(q, ring, None, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos)
            return ttnn.reshape(out, [b, s, 1, d])
        assert b == 1, "fused_lightning_select_kv decodes one user"
        if s > 1:
            assert not candidates, "S > 1 selects one level"
            stride = self.window + self.index_topk
            assert (
                scache.sel_tile_rows is not None and scache.sel_tile_rows.shape[2] == s * stride
            ), f"build_decode_caches(..., verify_rows={s}) holds this step's selections"
        latent = None
        if self.is_kv_source:
            # V4 marks its CSA/HCA compressor's decode the same way; V4.1 inlines the compressor
            # into this method, so without this the whole compressed path is one ATTENTION blob.
            with _region("COMPRESSOR"):
                kv, gate = self._compressor_rows(tokens)
                if s == 1:
                    latent = self._latent(kv, gate, scache, win_slot, pool, win_prev, win_offset=win_offset)
                else:
                    latent = self._latent_rows(kv, gate, scache, win_slot, pool_rows, win_prev)
        assert latent is None or select, "a closed group is written by the select that follows it"
        assert received_entry is None or (select and latent is None), "received rows are written by a non-source select"
        assert select or not candidates, "candidates are a select mode"
        q, kv, q_a = self._qkv(tokens, cos, sin)
        if s == 1:
            _update_cache_at(scache.kv, kv, sliding_pos)
            ttnn.deallocate(kv)
        if select:
            # The select runs on the whole grid; keep q out of its L1 until SDPA.
            q_l1 = q.memory_config()
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            row = None
            if latent is not None:
                self._write_index_key(latent, cos_win, sin_win, scache, entry, page_row, multi_row=s > 1)
                row = _apply_rope(latent, cos_win, sin_win, self.rot, self.rope_dim)
                if scache.new_kv_row is not None:
                    if s == 1:
                        _stash_row(row, scache.new_kv_row)
                    else:
                        scache.new_kv_rows = ttnn.clone(ttnn.to_memory_config(row, ttnn.DRAM_MEMORY_CONFIG))
            elif received_entry is not None:
                self._write_received_key(scache, received_entry, page_row, multi_row=s > 1)
                row, entry = (scache.new_kv_row if s == 1 else scache.new_kv_rows), received_entry
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
                page_row=page_row,
            )
            if latent is not None:
                ttnn.deallocate(row)
            if s == 1:
                ttnn.tilize(scache.sel_kv, output_tensor=scache.sel_tile)
                q = ttnn.to_memory_config(q, q_l1)
            else:
                # _attend reshards the rows one per core from DRAM.
                ttnn.tilize(scache.sel_kv_rows, output_tensor=scache.sel_tile_rows)
        ttnn.deallocate(q_a)
        ttnn.deallocate(tokens)
        if s > 1:
            sel = ttnn.experimental.view(scache.sel_tile_rows, [s, 1, stride, self.head_dim])
            # Every row's copy starts from the ring before this step.
            for j in range(s):
                ttnn.fill_cache(sel, scache.kv, j)
            self._write_causal_rows(kv, sel, ring_rows)
            ttnn.experimental.paged_update_cache(scache.kv, kv, update_idxs_tensor=sliding_pos)
            ttnn.deallocate(kv)
            out = self._attend(q, sel, mask, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos)
            return ttnn.reshape(out, [b, s, 1, d])
        if ring_offset is None:
            ttnn.fill_cache(scache.sel_tile, scache.kv, 0)
        else:
            ttnn.fill_cache(
                scache.sel_tile, scache.kv, 0, input_row_offset_tensor=ring_offset, num_input_rows=self.window
            )
        out = self._attend(q, scache.sel_tile, mask, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos)
        return ttnn.reshape(out, [b, s, 1, d])

    def decode_inputs(self, pos: int, cos_half: torch.Tensor, sin_half: torch.Tensor, rows: int = 1) -> dict:
        """Host-built :meth:`decode` keyword arguments for absolute positions ``pos .. pos + rows - 1``.

        ``cos_half`` / ``sin_half`` are this layer's ``[max_seq, Rd / 2]`` RoPE tables. Every per-position
        argument has one entry per row: ``sliding_pos`` / ``win_slot`` / ``index_pos`` / ``sdpa_cur_pos``
        ``[T]``, ``mask`` ``[T, 1, 1, Skv]``, and ``cos_win`` / ``sin_win`` ``[1, 1, n, Rd]`` / ``entry`` ``[n]``
        for the ``n`` groups the rows close. ``cos`` / ``sin`` / ``neg_sin`` are TILE ``[1, 1, 1, Rd]`` for one
        row and ROW_MAJOR ``[1, 1, T, Rd]`` (one row per position) otherwise. ``rows > 1`` also gives
        ``ring_rows`` (:meth:`_write_causal_rows`) and ``pool_rows``, the rows that close a group.
        """
        device, window, r = self.device, self.window, self.compress_ratio
        multi = rows > 1
        positions = range(pos, pos + rows)

        def to_device(t, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device)

        def int32_rows(values):
            return ttnn.from_torch(
                torch.tensor(values, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )

        # A TILE table carries per-row positions only for TILE input, so ROW_MAJOR rows take a ROW_MAJOR one.
        rope_layout = ttnn.ROW_MAJOR_LAYOUT if multi else ttnn.TILE_LAYOUT
        cos, sin = make_rope_table(cos_half[pos : pos + rows], sin_half[pos : pos + rows])
        kw = dict(
            cos=to_device(cos, rope_layout),
            sin=to_device(sin, rope_layout),
            neg_sin=to_device(-sin, rope_layout),
            sliding_pos=int32_rows([p % window for p in positions]),
        )
        if multi:
            assert rows <= window, f"{rows} rows would overwrite each other's ring slots"
            # Copy j holds rows 0 .. j in their slots.
            kw["ring_rows"] = int32_rows(
                [(pos + i) % window if i <= j else -1 for j in range(rows) for i in range(rows)]
            )
        if r == 0:
            kw["sdpa_cur_pos"] = int32_rows([min(p, window - 1) for p in positions])
            return kw
        counts = [(p + 1) // r for p in positions]
        if self.is_kv_source:
            closing = tuple(j for j, p in enumerate(positions) if (p + 1) % r == 0)
            if multi:
                kw["pool_rows"] = closing
            else:
                kw["pool"] = bool(closing)
            if r > 1:
                kw["win_slot"] = int32_rows([p % r for p in positions])
            if closing:
                # Group j stands at its first token, j * ratio.
                starts = [positions[j] + 1 - r for j in closing]
                cos_win, sin_win = make_rope_table(cos_half[starts], sin_half[starts])
                kw["cos_win"], kw["sin_win"] = to_device(cos_win), to_device(sin_win)
                kw["entry"] = int32_rows([positions[j] // r for j in closing])
        if self.is_index_source and counts[-1] > 0:
            kw["select"] = True
            # A row with no key yet still selects one; its mask hides the result.
            kw["index_pos"] = int32_rows([_SELECT_KV_RATE * max(n, 1) - 1 for n in counts])
            indexer = self.indexer
            if counts[-1] > indexer.candidate_topk_blocks * indexer.candidate_block_size and (
                indexer.is_candidate_source or indexer.uses_candidates
            ):
                assert not multi, "the two-level select decodes one row at a time"
                kw["candidates"] = True
                kw["build_pool"] = indexer.uses_candidates
        selected = [min(n, self.index_topk) for n in counts]
        if pos >= window - 1:
            kw["sdpa_cur_pos"] = int32_rows([window + s - 1 for s in selected])
        else:
            mask = torch.full((rows, 1, 1, window + self.index_topk), _MASK_NEG)
            for j, (p, s) in enumerate(zip(positions, selected)):
                mask[j, ..., : min(p + 1, window)] = 0
                mask[j, ..., window : window + s] = 0
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


def key_split_layers(config, layers, max_seq: int) -> set[int]:
    """The index sources in ``layers`` that score more than :data:`KEY_SPLIT_MIN_KEYS` keys
    (:data:`CANDIDATE_SOURCE_KEY_SPLIT_MIN_KEYS` for the candidate source) within ``max_seq`` tokens.
    A candidate consumer scores at most the candidate pool."""
    source = config.candidate_source_layer_id
    pool_size = config.candidate_topk_blocks * config.candidate_block_size
    split = set()
    for li in layers:
        if li not in config.index_source_layer_ids:
            continue
        max_keys = max_seq // config.compress_ratios[li]
        if 0 <= source < li:
            max_keys = min(max_keys, pool_size)
        threshold = (
            CANDIDATE_SOURCE_KEY_SPLIT_MIN_KEYS
            if li == source and uses_candidate_pool(config, max_seq)
            else KEY_SPLIT_MIN_KEYS
        )
        if max_keys > threshold:
            split.add(li)
    return split


def build_decode_caches(device, config, layers, max_seq: int, verify_rows: int = 1) -> dict:
    """Empty batch-1 decode caches for ``layers``, by layer id.

    Every layer owns its ``kv`` ring. A compressed layer aliases its kv source's window,
    compressed-KV, index-key and selection buffers, so its kv and index sources must be
    in ``layers`` too. Past the candidate pool the candidate source's group also gets
    :func:`candidate_buffers`, so the candidate source must be in ``layers`` with its consumers.
    ``verify_rows > 1`` adds the selection buffers of a :meth:`~DeepSeekV41Attention.decode` step with
    ``S == verify_rows``.
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
                if verify_rows > 1:
                    rows = verify_rows * (window + config.index_topk)
                    shared[src].update(
                        sel_kv_rows=zeros((1, 1, rows, dh), ttnn.ROW_MAJOR_LAYOUT),
                        sel_tile_rows=zeros((1, 1, rows, dh), ttnn.TILE_LAYOUT),
                    )
                source = config.candidate_source_layer_id
                if two_level and src == max(s for s in config.kv_source_layer_ids if s <= source):
                    consumers = [li for li in layers if li in config.index_source_layer_ids and li > source]
                    assert source in layers or not consumers, f"candidate consumers {consumers} need layer {source}"
                    shared[src].update(candidate_buffers(device, config, with_pool=bool(consumers)))
        caches[i] = DeepSeekV41LayerCache(kv=zeros((1, 1, window, dh), ttnn.TILE_LAYOUT), **(shared[src] if r else {}))
    return caches
