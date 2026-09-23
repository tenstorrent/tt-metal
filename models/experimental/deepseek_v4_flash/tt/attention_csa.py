"""CSA (compressed sparse attention) decode compressor and its window-buffer helpers.

The compressor projects every source token to a ``2*Dh`` Ca/Cb pair and pools each closed
window of ``compress_rate`` (4) tokens into a single compressed KV entry, using the overlap
scheme: entry ``w`` combines window ``w-1``'s Ca slice with window ``w``'s Cb slice. Only the
window being filled and the one just closed are held, because pooling is incremental.

Window state is kept as ROW_MAJOR L1 WIDTH_SHARDED ``[B*compress_rate, 1, 1, 2*Dh]`` buffers
so the fused ``csa_pool_window`` op can consume them in place
(:func:`~.attention.build_static_layer_cache` allocates them); user ``u``'s token ``t`` sits at row
``u*compress_rate + t``, and :func:`_update_window_at` / :func:`_scatter_window_rows` are the
two ways those packed rows are written. ``prev_gate`` starts at ``_MASK_NEG`` so window 0's
absent Ca half carries softmax weight 0.

Pooling runs only on the steps that close a window -- a compressor emits an entry once every
``compress_rate`` tokens and the additive block-bias exposes entries
``w < (pos+1)//compress_rate``, which is constant between closures -- so a step costs
``O(compress_rate)`` rather than ``O(max_seq)``; the caller drives that via the ``pool`` flag
(see ``DeepSeekV4Model._compressor_pool_due``).

The CSA Lightning Indexer (:class:`DeepSeekV4Indexer`) is a second CSA compressor at
``index_head_dim`` plus ``indexer_score_dsa`` / ``topk_large_indices``. It reuses
:class:`DeepSeekV4CSACompressor` for key writes. While ``seq_len < compress_rate * index_topk``
there are fewer closed windows than ``index_topk``, so attention stays dense causal SDPA
(its own trace). At and above that length the indexer picks the top-k compressed entries
and core attention is ``sparse_sdpa`` over the sliding ring plus those entries.

HCA has its own compressor (:mod:`.attention_hca`). The helpers shared with the rest of the
attention block -- the decode activation layouts, the rope wrapper, the in-place cache writers --
live in :mod:`.attention`, which imports this module lazily (see ``attention._compressor_class``)
because these modules need those helpers in turn.
"""

from typing import Optional

import torch
import ttnn

from .attention import (
    _StaticLayerCache,
    _apply_rope,
    _compressor_projections,
    _decode_activation,
    _one_row_per_user,
    _packed_users,
    _tp_cluster_axis,
    _update_cache_at,
    _update_kv_at,
)
from .common import _profile, _signpost, width_sharded_l1_config
from .decode_prefetch import (
    check_decode_layout,
    decode_prefetch_page_bytes,
    make_decode_prefetch_buffers,
)
from .layers import DeepSeekV4RMSNorm, LinearDecode
from .paged_cache import PagedLayerView
from .system_config import active_system_config
from .weight_cache import WeightCache, _as_cache, _load_weight, _materialize

INDEX_SENTINEL = 0xFFFFFFFF


def _scatter_window_rows(cache: ttnn.Tensor, rows: ttnn.Tensor, index: ttnn.Tensor) -> None:
    """In-place ``cache[index[i]] = rows[i]`` on a packed CSA window buffer.

    ``cache`` is ROW_MAJOR L1 WIDTH_SHARDED ``[B*cr, 1, 1, F]``, ``rows`` an
    INTERLEAVED ``[n, 1, 1, F]`` and ``index`` an INT32 ROW_MAJOR ``[n]``.
    ``paged_update_cache`` cannot target a width-sharded L1 cache, so the scatter
    goes through ``indexed_fill``, which needs ``dim == 0`` to take its shard-local
    path -- at any other dim it falls back to a generic path that is documented as
    wrong for a sharded destination. ``indexed_fill`` returns a fresh tensor, so
    the result is copied back into the persistent buffer.
    """
    written = ttnn.indexed_fill(index, cache, rows, memory_config=cache.memory_config(), dim=0)
    ttnn.copy(written, cache)
    ttnn.deallocate(written)


def _update_window_at(cache: ttnn.Tensor, row: ttnn.Tensor, index: ttnn.Tensor) -> None:
    """Write ``row`` ``[1, B, 1, F]`` into a CSA window at the packed rows ``index``.

    CSA windows are user-major: user ``u``'s token ``t`` sits at row ``u*cr + t``, so
    ``index`` carries one row per user. Projections arrive as packed ``[1, 1, B, F]``
    and are spread to height-sharded ``[1, B, 1, F]`` for ``paged_update_cache``.
    """
    packed = _one_row_per_user(row)
    ttnn.experimental.paged_update_cache(cache, packed, update_idxs_tensor=index)
    if packed is not row:
        ttnn.deallocate(packed)


def _rm_width_sharded(tensor: ttnn.Tensor, height: int, width: int) -> ttnn.Tensor:
    """ROW_MAJOR WIDTH_SHARDED ``[1, 1, height, width]`` in L1 (1-high faces)."""
    if list(tensor.shape) != [1, 1, height, width]:
        tensor = ttnn.reshape(tensor, [1, 1, height, width])
    if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        tensor = ttnn.to_layout(ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)
    cfg = width_sharded_l1_config(height, width, tensor.device(), tile_height=1)
    if tensor.memory_config() != cfg:
        tensor = ttnn.to_memory_config(tensor, cfg)
    return tensor


def _retire_window(prev: ttnn.Tensor, current: ttnn.Tensor) -> None:
    """Copy the just-closed window buffer ``current`` into ``prev``, in place.

    CSA windows are the same ROW_MAJOR L1 WIDTH_SHARDED ``[B*cr, 1, 1, 2*Dh]`` spec, so a
    device copy is a whole-buffer write into the persistent ``prev`` address. TILE DRAM
    windows (unused by CSA) still go through ``fill_cache``, which writes a single batch
    index per call.
    """
    if current.is_sharded():
        ttnn.copy(current, prev)
        return
    users, heads, rows, width = current.shape
    if users == 1:
        ttnn.fill_cache(prev, current, 0)
        return
    for user in range(users):
        one = ttnn.slice(current, [user, 0, 0, 0], [user + 1, heads, rows, width])
        ttnn.fill_cache(prev, one, user)
        ttnn.deallocate(one)


class DeepSeekV4CSACompressor:
    """Compressed-Sparse-Attention compressor (decode, running KV cache).

    Like HCA but with the two-series Ca/Cb overlap scheme: each token projects to
    ``2*Dh`` (Ca = its contribution to the *next* window, Cb = to the *current*
    window). Compressed entry ``w`` pools window ``w-1``'s Ca slice with window
    ``w``'s Cb slice over a width-``2*compress_rate`` window. Window 0's Ca half
    is zero-kv / ``-inf``-gate (softmax weight 0), since there is no prior window.

    The CSA Lightning Indexer (:class:`DeepSeekV4Indexer`) scores queries against
    these compressed keys and gathers a top-k; this compressor still emits every
    closed window. For ``seq_len < index_topk * compress_rate`` that top-k would
    exceed the closed windows, so the caller keeps dense causal SDPA instead.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device,
        rot,
        rope_dim: int,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        use_prefetcher: bool = False,
        num_prefetch_pages: Optional[int] = None,
        prefetch_buffers: Optional[dict] = None,
        *,
        head_dim: Optional[int] = None,
        layout_name: str = "compressed_sparse_attention",
        weight_prefix: str = "compressor",
        attach_indexer: bool = True,
        tp_size: int = 1,
    ):
        """Build the compressor's projections, norm and position bias from ``weights``.

        As :meth:`~.attention_hca.DeepSeekV4HCACompressor.__init__`, except that the kv/gate weights are
        ``[D, 2*Dh]`` (the Ca/Cb pair) and ``compressor.position_bias`` is reshaped to
        ``[1, 1, compress_rate, 2*Dh]`` and kept ROW_MAJOR L1 WIDTH_SHARDED for the fused
        pool. The shared ``[Rd, Rd]`` rotate matrix is passed in.
        """
        self.device = device
        self.rope_dim = rope_dim
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim if head_dim is None else head_dim
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        cache = _as_cache(cache)
        if num_prefetch_pages is None:
            num_prefetch_pages = active_system_config().prefetcher.num_prefetch_pages
        self.kv_proj, self.gate_proj = _compressor_projections(
            "compressed_sparse_attention",
            config,
            weights,
            device,
            cache,
            weight_dtype,
            use_prefetcher,
            num_prefetch_pages,
            prefetch_buffers,
            feat=2 * self.head_dim,
            layout_name=layout_name,
            weight_prefix=weight_prefix,
        )
        self.kv_norm = DeepSeekV4RMSNorm(
            weights[f"{weight_prefix}.kv_norm.weight"],
            self.eps,
            device,
            cache.file(f"{weight_prefix}.kv_norm"),
            sharded=True,
        )
        pb = _materialize(
            weights[f"{weight_prefix}.position_bias"], cache.file(f"{weight_prefix}.position_bias"), ttnn.bfloat16
        )
        self.position_bias = _load_weight(
            pb.reshape(1, 1, self.compress_rate, 2 * self.head_dim) if pb is not None else None,
            device,
            cache_file_name=cache.file(f"{weight_prefix}.position_bias"),
        )
        if self.position_bias is not None:
            self.position_bias = _rm_width_sharded(self.position_bias, self.compress_rate, 2 * self.head_dim)
        self._win_offsets: ttnn.Tensor | None = None
        self.indexer: DeepSeekV4Indexer | None = None
        # ``indexer: true`` (the default) attaches the indexer when the checkpoint
        # has its weights. ``false`` attaches nothing, so the prefetcher never queues
        # weights no matmul pops. Decode writes index keys every step and scores
        # them once the sequence is long enough for top-k (see
        # :meth:`DeepSeekV4Attention.decode_static`).
        if attach_indexer and active_system_config().attention.indexer_enabled() and _has_indexer_weights(weights):
            self.indexer = DeepSeekV4Indexer(
                config,
                weights,
                device,
                rot,
                rope_dim,
                cache=cache.sub("indexer"),
                weight_dtype=weight_dtype,
                use_prefetcher=use_prefetcher,
                num_prefetch_pages=num_prefetch_pages,
                prefetch_buffers=prefetch_buffers,
                tp_size=tp_size,
            )

    def _win_index(self, win_slot: ttnn.Tensor, users: int) -> ttnn.Tensor:
        """Packed window rows ``u*compress_rate + pos % compress_rate`` for each user.

        ``win_slot`` is the ``[B]`` slot vector the caller already built. Above batch 1
        the user offsets are added on device, from a vector allocated on the first call
        -- eagerly, since the traced path compiles each step before capturing it, and a
        host-to-device write inside a capture is rejected.
        """
        if users == 1:
            return win_slot
        if self._win_offsets is None:
            self._win_offsets = ttnn.from_torch(
                torch.arange(users, dtype=torch.int32) * self.compress_rate,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        return ttnn.add(win_slot, self._win_offsets)

    def prefetch_weights(self, *, index_sparse: bool = False):
        """Stage the two projection weights ahead of the :meth:`decode_static` that uses them.

        Queued kv before gate, the order :meth:`decode_static` pops them off their shared GCB
        (q_a's 32-receiver ring); both stay DRAM ND-sharded ``[D, 2*Dh]``. ``index_sparse``
        also stages the indexer's score projections; key compression is staged either way,
        because both the short-sequence and the indexer trace write index keys.
        """
        self.kv_proj.fetch_weights()
        self.gate_proj.fetch_weights()
        if self.indexer is not None:
            # q_b was queued by attention, between its own q_b and o_a: it shares that ring.
            self.indexer.prefetch_weights(score=index_sparse, q_b=False)

    def _write_projection(self, proj, tokens: ttnn.Tensor, window: ttnn.Tensor, win_index: ttnn.Tensor) -> None:
        """Project ``tokens`` with ``proj`` into ``window`` at ``win_index``, then free the row.

        kv and gate share a GCB, and ``matmul_decode`` places its circular buffers on the
        bounding box of every core the activation and the weight touch. That box is the
        whole compute grid for the indexer's 8-core ring. The kv row lives in that box;
        leaving it allocated while the gate program is built drops the lowest L1 address
        into the buffer region and the program is rejected. The window already holds the
        row, so the projection result is free to go before the next matmul.
        """
        row = proj(_decode_activation(proj, tokens))
        _update_window_at(window, row, win_index)
        ttnn.deallocate(row)

    def _pool_window(
        self,
        prev_kv: ttnn.Tensor,
        prev_gate: ttnn.Tensor,
        win_kv: ttnn.Tensor,
        win_gate: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Pool each user's closing window ``w`` into its single compressed entry,
        returned as ``[1, B, 1, Dh]`` ready for the cache write.

        ``win_*`` / ``prev_*`` are the persistent ROW_MAJOR L1 WIDTH_SHARDED
        ``[B*compress_rate, 1, 1, 2*Dh]`` buffers allocated by
        :func:`~.attention.build_static_layer_cache`. The fused op consumes them in place.
        On the very first window ``prev_gate`` is still ``_MASK_NEG``, which
        gives the absent Ca half softmax weight 0.
        """
        compressed = ttnn.experimental.deepseek.csa_pool_window(
            prev_kv, prev_gate, win_kv, win_gate, self.position_bias
        )
        _profile(self.device)
        compressed = self.kv_norm(compressed)
        compressed = _apply_rope(compressed, cos_row, sin_row, self.rot, self.rope_dim)
        return _one_row_per_user(compressed)

    def _step(
        self,
        tokens: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        scache: "_StaticLayerCache",
        win_slot: ttnn.Tensor,
        pool: bool,
    ) -> ttnn.Tensor | None:
        """Write each user's ``2*Dh`` token projection in place at ``win_slot`` into the
        one-window ROW_MAJOR L1 WIDTH_SHARDED ``[B*cr, 1, 1, 2*Dh]`` buffers, and -- on the
        step that closes the window -- pool just that window (Ca/Cb overlap against the
        retained previous window).

        Returns the pooled entry ``[1, B, 1, Dh]`` for the caller to write and free, or
        ``None`` when ``pool`` is false. After pooling, the closing window becomes the
        ``prev_*`` the *next* window will overlap with.
        """
        users = _packed_users(tokens)
        win_index = self._win_index(win_slot, users)
        # kv before gate: that is the order :meth:`prefetch_weights` queued them on the
        # shared ring. Each row is freed before the next matmul; see :meth:`_write_projection`.
        self._write_projection(self.kv_proj, tokens, scache.win_kv, win_index)
        self._write_projection(self.gate_proj, tokens, scache.win_gate, win_index)
        if win_index is not win_slot:
            ttnn.deallocate(win_index)
        if not pool:
            return None
        pooled = self._pool_window(scache.prev_kv, scache.prev_gate, scache.win_kv, scache.win_gate, cos_row, sin_row)
        _retire_window(scache.prev_kv, scache.win_kv)
        _retire_window(scache.prev_gate, scache.win_gate)
        return pooled

    def decode_static(
        self,
        tokens: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        scache: "_StaticLayerCache",
        kv: "PagedLayerView | ttnn.Tensor",
        win_slot: ttnn.Tensor,
        win_row: ttnn.Tensor | None = None,
        pool: bool = True,
    ) -> None:
        """Trace-safe decode: :meth:`_step`, then append the pooled entry at row
        ``win_row`` of the layer's KV axis (``kv``, the dense ``[1, 1, rows, Dh]`` buffer).

        ``tokens`` is the block's packed-row hidden ``[1, 1, B, D]``, already gathered onto
        the decode activation grid when the caller used
        :meth:`~.attention.DeepSeekV4Attention.decode_static`; ``cos_row`` / ``sin_row`` are the closing
        window's RoPE row ``[1, 1, 1, Rd]`` and ``win_slot`` / ``win_row`` INT32 ``[B]``
        row-index vectors. See :meth:`~.attention_hca.DeepSeekV4HCACompressor.decode_static`.
        """
        _signpost("CSA_START")
        pooled = self._step(tokens, cos_row, sin_row, scache, win_slot, pool)
        if pooled is not None:
            _update_kv_at(kv, pooled, win_row)
            ttnn.deallocate(pooled)
        if self.indexer is not None and getattr(scache, "idx_key_cache", None) is not None:
            self.indexer.write_keys(tokens, cos_row, sin_row, scache, win_slot, win_row, pool=pool)
        _signpost("CSA_END")


_INDEXER_WEIGHT_KEYS = {
    "kv_proj": (
        "compressor.indexer.kv_proj.weight",
        "indexer.kv_proj.weight",
        "indexer.compressor.wkv.weight",
    ),
    "gate_proj": (
        "compressor.indexer.gate_proj.weight",
        "indexer.gate_proj.weight",
        "indexer.compressor.wgate.weight",
    ),
    "position_bias": (
        "compressor.indexer.position_bias",
        "indexer.position_bias",
        "indexer.compressor.ape",
    ),
    "kv_norm": (
        "compressor.indexer.kv_norm.weight",
        "indexer.kv_norm.weight",
        "indexer.compressor.norm.weight",
    ),
    "q_b_proj": (
        "compressor.indexer.q_b_proj.weight",
        "indexer.q_b_proj.weight",
        "indexer.wq_b.weight",
    ),
    "weights_proj": (
        "compressor.indexer.weights_proj.weight",
        "compressor.indexer.scorer.weights_proj.weight",
        "indexer.weights_proj.weight",
    ),
}


def _has_indexer_weights(weights: dict) -> bool:
    """True when ``weights`` carries any of the Lightning Indexer tensors."""
    return any(key in weights for aliases in _INDEXER_WEIGHT_KEYS.values() for key in aliases)


def _indexer_weight(weights: dict, name: str):
    """The first present alias of indexer tensor ``name``."""
    for key in _INDEXER_WEIGHT_KEYS[name]:
        if key in weights:
            return weights[key]
    raise KeyError(f"indexer weight {name} not found; tried {_INDEXER_WEIGHT_KEYS[name]}")


def _release_if_distinct(src: ttnn.Tensor, dst: ttnn.Tensor) -> None:
    """Free ``src`` only when ``dst`` owns a different buffer.

    ``to_memory_config`` returns the input unchanged when it is already in the
    requested config, but as a new Python object. ``src is not dst`` is then
    true while both still name the same allocation, and freeing ``src`` frees
    ``dst`` as well.
    """
    if src is dst or not src.is_allocated() or not dst.is_allocated():
        return
    if src.buffer_address() == dst.buffer_address():
        return
    ttnn.deallocate(src)


def _scale_linear_weight(weight, scale: float):
    """Fold ``scale`` into a ``[N, K]`` weight (or thunk) for ``indexer_score_dsa``."""
    if callable(weight):
        return lambda: weight() * scale
    return weight * scale


class _IndexerWindowView:
    """Expose indexer window buffers under the CSA compressor's ``win_*`` / ``prev_*`` names."""

    def __init__(self, scache: "_StaticLayerCache"):
        self.win_kv = scache.idx_win_kv
        self.win_gate = scache.idx_win_gate
        self.prev_kv = scache.idx_prev_kv
        self.prev_gate = scache.idx_prev_gate


class DeepSeekV4Indexer:
    """CSA Lightning Indexer on device: CSA compressor at ``index_head_dim``, then
    ``indexer_score_dsa`` / ``ring_indexer_score_dsa`` + ``topk_large_indices``.

    Key compression is :class:`DeepSeekV4CSACompressor` with ``head_dim=index_head_dim``
    (same Ca/Cb pool, ``csa_pool_window``, RoPE, cache write). Queries come from the
    shared q_a latent via :class:`~.layers.LinearDecode` ``q_b_proj``. Both
    ``1/√n_heads`` and ``1/√index_head_dim`` are folded into ``weights_proj`` because
    the score op applies no scale of its own.

    The index-key cache is replicated on every TP rank. Decode scores it with
    ``indexer_score_dsa`` on each rank: the query is one replicated token, so a
    sequence-sharded cache cannot be updated at the global window row a large
    ``start_pos`` names.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device,
        rot,
        rope_dim: int,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        use_prefetcher: bool = False,
        num_prefetch_pages: Optional[int] = None,
        prefetch_buffers: Optional[dict] = None,
        tp_size: int = 1,
    ):
        # LinearDecode projections always prefetch (ROW_MAJOR HEIGHT_SHARDED, fully
        # width-sharded). ``use_prefetcher`` is kept for the CSA constructor's kwargs.
        del use_prefetcher
        self.device = device
        self.tp_size = int(tp_size)
        self.cluster_axis = _tp_cluster_axis(device) if self.tp_size > 1 else None
        self._ag_sub_device_id = None
        self.head_dim = config.index_head_dim
        self.num_heads = config.index_n_heads
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.index_topk = int(getattr(config, "index_topk", 512))
        self.sliding_window = config.sliding_window
        self.rope_dim = rope_dim
        self.rot = rot
        cache = _as_cache(cache)
        folded = (self.head_dim**-0.5) * (self.num_heads**-0.5)
        if num_prefetch_pages is None:
            num_prefetch_pages = active_system_config().prefetcher.num_prefetch_pages
        if prefetch_buffers is None:
            prefetch_buffers = make_decode_prefetch_buffers(device, weight_dtype, num_prefetch_pages)

        compressor_weights = {
            "compressor.kv_proj.weight": _indexer_weight(weights, "kv_proj"),
            "compressor.gate_proj.weight": _indexer_weight(weights, "gate_proj"),
            "compressor.kv_norm.weight": _indexer_weight(weights, "kv_norm"),
            "compressor.position_bias": _indexer_weight(weights, "position_bias"),
        }
        self.compressor = DeepSeekV4CSACompressor(
            config,
            compressor_weights,
            device,
            rot,
            rope_dim,
            cache=cache.sub("compressor"),
            weight_dtype=weight_dtype,
            use_prefetcher=True,
            num_prefetch_pages=num_prefetch_pages,
            prefetch_buffers=prefetch_buffers,
            head_dim=self.head_dim,
            layout_name="indexer.kv_proj",
            attach_indexer=False,
        )
        for name, proj in (("kv_proj", self.compressor.kv_proj), ("gate_proj", self.compressor.gate_proj)):
            assert (
                proj.use_prefetcher and proj.use_rm_hs and not proj.partial_width_sharded
            ), f"indexer compressor {name} must be prefetched, ROW_MAJOR HEIGHT_SHARDED, fully width-sharded"

        def head_proj(layout_name: str, weight, K: int, N: int, cache_key: str) -> LinearDecode:
            layout = dict(check_decode_layout(layout_name, K, N))
            if layout.get("partial_width_sharded", False):
                raise ValueError(f"{layout_name} must be fully width-sharded for hub-mode matmul_decode")
            # q_b shares the 64-receiver ring. weights_proj is 2 cores, which does
            # not divide the DRAM banks, so it cannot take a prefetch ring.
            prefetch = layout_name == "indexer.q_b_proj"
            extra = {}
            if prefetch:
                extra = {
                    "global_cb": prefetch_buffers["q_b_proj"],
                    "global_cb_page_bytes": decode_prefetch_page_bytes(weight_dtype),
                }
            proj = LinearDecode(
                weight,
                device,
                cache.file(f"{cache_key}.full"),
                dtype=weight_dtype,
                **layout,
                use_prefetcher=prefetch,
                rectangle_b_grid=True,
                use_rm_hs=True,
                **extra,
            )
            assert (
                proj.use_rm_hs and not proj.partial_width_sharded
            ), f"{layout_name} must be ROW_MAJOR HEIGHT_SHARDED and fully width-sharded"
            assert proj._can_matmul_decode_rm_hs(), f"{layout_name} cannot take a ROW_MAJOR HEIGHT_SHARDED A"
            return proj

        self.q_b_proj = head_proj(
            "indexer.q_b_proj",
            _indexer_weight(weights, "q_b_proj"),
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            "q_b_proj",
        )
        self.weights_proj = head_proj(
            "indexer.weights_proj",
            _scale_linear_weight(_indexer_weight(weights, "weights_proj"), folded),
            config.hidden_size,
            self.num_heads,
            "weights_proj",
        )
        self._window_ids: ttnn.Tensor | None = None

    def prefetch_weights(self, *, score: bool = False, q_b: bool = True):
        """Stage compressor kv/gate, and the score projections when ``score`` is set.

        The short-sequence trace writes index keys but does not score, and a queued
        page that no matmul pops desynchronizes that GCB. Score weights are staged
        only on the indexer trace. ``q_b`` is false when the caller already queued
        ``q_b_proj`` on the shared ring (it has to precede o_a).
        """
        self.compressor.prefetch_weights()
        if score and q_b:
            self.q_b_proj.fetch_weights()
        # weights_proj has no prefetch ring (2 cores do not divide the DRAM banks).
        # Its forward copies DRAM -> L1 itself; queueing that here would allocate
        # inside the trace.
        if score and self.weights_proj.use_prefetcher:
            self.weights_proj.fetch_weights()

    def write_keys(
        self,
        tokens: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        scache: "_StaticLayerCache",
        win_slot: ttnn.Tensor,
        win_row: ttnn.Tensor | None = None,
        pool: bool = True,
    ) -> None:
        """Same schedule as CSA :meth:`DeepSeekV4CSACompressor.decode_static`, writing
        pooled index keys into ``scache.idx_key_cache``.

        ``win_row`` addresses the attention KV axis (``sliding_window + w``). The index
        cache holds only compressed keys, so the entry lands at row ``w``.
        """
        idx_row = win_row
        if pool and win_row is not None:
            idx_row = ttnn.subtract(win_row, self.sliding_window)
        pooled = self.compressor._step(tokens, cos_row, sin_row, _IndexerWindowView(scache), win_slot, pool)
        if pooled is not None:
            _update_cache_at(scache.idx_key_cache, pooled, idx_row)
            ttnn.deallocate(pooled)
        if idx_row is not win_row and idx_row is not None:
            ttnn.deallocate(idx_row)

    def select(
        self,
        query: ttnn.Tensor,
        key_cache: ttnn.Tensor,
        head_weights: ttnn.Tensor,
        logits_out: ttnn.Tensor | None = None,
        valid_length_tensor: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """``indexer_score_dsa`` then ``topk_large_indices``.

        ``query`` ``[B, Hi, Sq, D]`` TILE, ``key_cache`` the full ``[B, 1, T, D]``
        TILE (replicated across TP), ``head_weights`` ``[B, 1, Sq, Hi]`` TILE with
        both scales already folded.

        ``Sq`` must be a multiple of the 32-row tile. Decode uses ``Sq == 32`` and puts
        the real query on the last row: ``chunk_start = T - Sq`` is tile-aligned and
        constant, so that row sees every allocated key (``t <= T - 1``). Columns past
        the closed-window count are dropped by ``valid_length_tensor`` (a 1-element
        uint32 DRAM tensor), which is what keeps one captured program correct at
        every later position, including a non-zero ``start_pos``. Do not pass a host
        ``kv_len``. Returns uint32 ``[B, 1, Sq, k]``.
        """
        sq = list(query.shape)[2]
        assert sq % 32 == 0, (
            f"indexer score needs a tile-aligned query block, got Sq={sq} "
            "(1-row decode queries are rejected on device)"
        )
        t_full = list(key_cache.shape)[2]
        assert t_full >= sq, f"index-key cache T={t_full} is shorter than the score query block Sq={sq}"
        chunk_start = t_full - sq
        scores = ttnn.experimental.indexer_score_dsa(query, key_cache, head_weights, chunk_start_idx=chunk_start)
        if logits_out is not None:
            ttnn.copy(scores, logits_out)
            scores = logits_out
        return ttnn.experimental.topk_large_indices(
            scores,
            k=self.index_topk,
            valid_length_tensor=valid_length_tensor,
            subdevice_id=self._ag_sub_device_id,
        )

    def _closed_windows(self, compress_pos: ttnn.Tensor) -> ttnn.Tensor:
        """``(pos + 1) // compress_rate`` as a 1-element uint32 DRAM row, from ``compress_pos`` ``[B]``.

        Top-k reads this on device (``valid_length_tensor``), so a traced step can grow
        the searchable prefix without a host scalar baked into the capture. The batch
        shares one position; the count is taken from user 0.
        """
        users = list(compress_pos.shape)[0]
        pos = ttnn.reshape(compress_pos, [1, 1, 1, users])
        scalar = ttnn.slice(pos, [0, 0, 0, 0], [1, 1, 1, 1])
        scalar_f = ttnn.typecast(ttnn.to_layout(scalar, ttnn.TILE_LAYOUT), ttnn.float32)
        closed = ttnn.floor(ttnn.multiply(ttnn.add(scalar_f, 1.0), 1.0 / self.compress_rate))
        closed_rm = ttnn.to_layout(ttnn.typecast(closed, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        closed_1 = ttnn.reshape(closed_rm, [1])
        return ttnn.to_memory_config(closed_1, ttnn.DRAM_MEMORY_CONFIG)

    def _score_block(self, row: ttnn.Tensor) -> ttnn.Tensor:
        """TILE ``[B, ..., 32, D]`` with the real row at index 31.

        ``indexer_score_dsa`` rejects ``Sq == 1`` and wants TILE input. Padding is
        applied in row-major, before the tilize, so the 31 leading rows are real
        zeros rather than tile padding. With ``chunk_start = T - 32`` that last row
        is the one whose causal window covers the whole allocated cache.
        """
        dram = ttnn.to_memory_config(row, ttnn.DRAM_MEMORY_CONFIG)
        # A no-op copy (already DRAM) comes back as a new Python object over the
        # same buffer. Freeing ``row`` would free ``dram`` too.
        _release_if_distinct(row, dram)
        if dram.layout != ttnn.ROW_MAJOR_LAYOUT:
            rm = ttnn.to_layout(dram, ttnn.ROW_MAJOR_LAYOUT)
            _release_if_distinct(dram, rm)
            dram = rm
        padded = ttnn.pad(dram, padding=[(0, 0), (0, 0), (ttnn.TILE_SIZE - 1, 0), (0, 0)], value=0.0)
        _release_if_distinct(dram, padded)
        tiled = ttnn.to_layout(padded, ttnn.TILE_LAYOUT)
        _release_if_distinct(padded, tiled)
        return tiled

    def score_and_select(
        self,
        tokens: ttnn.Tensor,
        q_a: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        key_cache: ttnn.Tensor,
        compress_pos: ttnn.Tensor,
        logits_out: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Project indexer q / folded weights, RoPE q, then :meth:`select`.

        Returns uint32 ROW_MAJOR ``[B, 1, 1, sliding_window + index_topk]``: the sliding
        ring followed by the selected compressed entries, which is the index list
        ``sparse_sdpa`` consumes against the combined KV axis. The real query occupies
        score-row 31; top-k runs on the padded block and only that row is kept.
        """
        q = self.q_b_proj(_decode_activation(self.q_b_proj, q_a))
        q = _apply_rope(q, cos_row, sin_row, self.rot, self.rope_dim, head_dim=self.head_dim)
        users = _packed_users(tokens)
        # RoPE leaves a width-sharded L1 row. Reshaping that shard into heads
        # asks for a page size L1 will not take, so interleave in DRAM first.
        w = self.weights_proj(_decode_activation(self.weights_proj, tokens))
        topk = self.select(
            q, key_cache, w, logits_out=logits_out, valid_length_tensor=self._closed_windows(compress_pos)
        )
        ttnn.deallocate(q)
        ttnn.deallocate(w)
        row = ttnn.slice(topk, [0, 0, ttnn.TILE_SIZE - 1, 0], [users, 1, ttnn.TILE_SIZE, self.index_topk])
        ttnn.deallocate(topk)
        return self.index_list(row)

    def index_list(self, topk_indices: ttnn.Tensor) -> ttnn.Tensor:
        """``[0..window)`` plus ``window + selected``; sentinels stay ``INDEX_SENTINEL``.

        ``topk_indices`` is uint32 ROW_MAJOR ``[B, 1, Sq, k]``; the result is uint32
        ROW_MAJOR ``[B, 1, Sq, window + k]`` (``sparse_sdpa``'s index contract). The window
        prefix is replicated across the leading dims so concat is along the last axis only.

        ROW_MAJOR concat on the last axis requires each input row to be DRAM-aligned, i.e.
        ``window`` and ``k`` multiples of 16 uint32 elements: the production pair (128, 512)
        qualifies, and the 640-wide result is aligned too.
        """
        window = self._window_prefix(topk_indices)
        offset = self._offset_selected(topk_indices)
        return ttnn.concat([window, offset], dim=-1)

    def _offset_selected(self, topk_indices: ttnn.Tensor) -> ttnn.Tensor:
        """``window + selected`` in uint32, sentinel-preserving, ROW_MAJOR.

        ``where(sel == INDEX_SENTINEL, sel, sel + window)`` is *not* usable here: the ternary
        device op requires TILE for interleaved input, and it picks the compute kernel data
        format from the predicate dtype, mapping UINT32 to ``DataFormat::Float16_b``
        (``ternary_op_utils.cpp::get_compute_defines``). That template loads and stores only
        the low 16 bits of every 32-bit lane, so a uint32 row would come back high-half
        clobbered -- the sentinel would decode as ``0x0000FFFF``, not ``0xFFFFFFFF``.

        ``add`` and ``maximum`` both have native uint32 kernels -- ``add_int_tile<UInt32>``
        and ``binary_max_uint32_tile`` (unsigned compare) -- so the branch-free equivalent
        below is exact for the full uint32 range:

        * ``sel != INDEX_SENTINEL``: ``sel + window`` does not wrap and is ``> sel``, so the
          unsigned maximum selects the shifted index;
        * ``sel == INDEX_SENTINEL``: ``sel + window`` wraps to ``window - 1``, so the maximum
          selects ``sel`` itself and the sentinel survives unchanged.

        The trick only relies on the wrap for a value within ``window`` of the uint32 ceiling;
        every entry that can land there is a sentinel by construction, so no real index is
        ever left unshifted. Arithmetic runs on TILE (the layout the uint32 kernels are tested
        in); ``to_layout`` pads/untilizes, so ``k`` need not be tile-aligned.
        """
        tile = ttnn.to_layout(topk_indices, ttnn.TILE_LAYOUT)
        shifted = ttnn.add(tile, self.sliding_window)
        return ttnn.to_layout(ttnn.maximum(shifted, tile), ttnn.ROW_MAJOR_LAYOUT)

    def _window_prefix(self, topk_indices: ttnn.Tensor) -> ttnn.Tensor:
        """``arange(sliding_window)`` broadcast to ``[*leading, window]`` of ``topk_indices``."""
        leading = list(topk_indices.shape)[:-1]
        if self._window_ids is None or list(self._window_ids.shape)[:-1] != leading:
            ids = torch.arange(self.sliding_window, dtype=torch.int64)
            ids = ids.view(*([1] * len(leading)), self.sliding_window).expand(*leading, self.sliding_window)
            self._window_ids = ttnn.from_torch(
                ids.contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        return self._window_ids
