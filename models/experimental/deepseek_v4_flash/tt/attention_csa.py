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
:class:`DeepSeekV4CSACompressor` for key writes. For ``seq_len <= index_topk * compress_rate``
its top-k selects every closed window, so the block bias reduces to causal masking.

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
)
from .common import _profile, _signpost, width_sharded_l1_config
from .decode_prefetch import (
    check_decode_layout,
    decode_prefetch_page_bytes,
    ensure_named_gcb,
    make_decode_prefetch_buffers,
)
from .layers import DeepSeekV4RMSNorm, LinearDecode, decode_gcb_page_bytes
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
    closed window. For ``seq_len <= index_topk * compress_rate`` that top-k is
    all entries, so the block bias reduces to causal masking over windows.
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
        # The indexer's compute (``write_keys`` / ``score_and_select``) is not wired into the
        # decode step yet, so attaching it only stages weights that no matmul ever drains. The
        # prefetcher's credit accounting assumes every queued weight is popped (see
        # ``LinearDecode._queue_prefetch``), so those orphan pages cost decode throughput for
        # weights nothing reads. Gate on the profile switch so ``indexer: "off"`` (the default)
        # is genuinely off and the build path matches the pre-indexer behaviour; ``always`` or
        # ``auto`` opt back in once the indexer is actually consumed.
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

    def prefetch_weights(self):
        """Stage the two projection weights ahead of the :meth:`decode_static` that uses them.

        Queued kv before gate, the order :meth:`_project` pops them off their shared GCB
        (q_a's 32-receiver ring); both stay DRAM ND-sharded ``[D, 2*Dh]``.
        """
        self.kv_proj.fetch_weights()
        self.gate_proj.fetch_weights()
        if self.indexer is not None:
            self.indexer.prefetch_weights()

    def _project(self, tokens: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``tokens`` ``[1, 1, B, D]`` -> per-token ``(kv, gate)`` ``[1, 1, B, 2*Dh]`` each.

        Returned in whatever layout ``LinearDecode`` leaves them in -- unlike HCA's, they
        are not moved to DRAM here, because :func:`_update_window_at` reshapes and
        reshards them for the width-sharded window buffer anyway.
        """
        return (
            self.kv_proj(_decode_activation(self.kv_proj, tokens)),
            self.gate_proj(_decode_activation(self.gate_proj, tokens)),
        )

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

    def decode_static(
        self,
        tokens: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        scache: "_StaticLayerCache",
        combined_cache: ttnn.Tensor | None,
        win_slot: ttnn.Tensor,
        win_row: ttnn.Tensor | None = None,
        pool: bool = True,
        paged: PagedLayerView | None = None,
    ) -> None:
        """Trace-safe decode: write each user's ``2*Dh`` token projection in place at
        ``win_slot`` into the one-window ROW_MAJOR L1 WIDTH_SHARDED
        ``[B*cr, 1, 1, 2*Dh]`` buffers, and -- on the step that closes the window -- pool
        just that window (Ca/Cb overlap against the retained previous window) and append
        its single entry at row ``win_row`` of the layer's KV axis (``combined_cache``,
        or ``paged``'s pool).

        ``tokens`` is the block's packed-row hidden ``[1, 1, B, D]``, already gathered onto
        the decode activation grid when the caller used
        :meth:`~.attention.DeepSeekV4Attention.decode_static`; ``cos_row`` / ``sin_row`` are the closing
        window's RoPE row ``[1, 1, 1, Rd]`` and ``win_slot`` / ``win_row`` INT32 ``[B]``
        row-index vectors.

        After pooling, the closing window becomes the ``prev_*`` the *next* window
        will overlap with. See :meth:`~.attention_hca.DeepSeekV4HCACompressor.decode_static`.
        """
        _signpost("CSA_START")
        users = _packed_users(tokens)
        kv, gate = self._project(tokens)  # [1, 1, B, 2*Dh]
        win_index = self._win_index(win_slot, users)
        _update_window_at(scache.win_kv, kv, win_index)
        _update_window_at(scache.win_gate, gate, win_index)
        if pool and (combined_cache is not None or paged is not None):
            pooled = self._pool_window(
                scache.prev_kv, scache.prev_gate, scache.win_kv, scache.win_gate, cos_row, sin_row
            )
            _update_cache_at(combined_cache, pooled, win_row, paged=paged)
            ttnn.deallocate(pooled)
            _retire_window(scache.prev_kv, scache.win_kv)
            _retire_window(scache.prev_gate, scache.win_gate)
        if win_index is not win_slot:
            ttnn.deallocate(win_index)
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

    At ``tp_size > 1`` the index-key cache is sequence-sharded on the TP mesh axis
    (``T / tp`` per rank). Scoring uses ``ring_indexer_score_dsa``, which all-gathers
    those shards into a persistent full-T buffer while overlapping the gather with
    compute. Query and weights must be sequence-sharded on the same axis (the fused
    op's SP contract); each rank's local ``Sq`` stays tile-aligned.
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
        self._k_gathered: ttnn.Tensor | None = None
        self._ag_semaphores: list | None = None
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
            if layout_name == "indexer.q_b_proj":
                global_cb = prefetch_buffers["q_b_proj"]
                page_bytes = decode_prefetch_page_bytes(weight_dtype)
            else:
                global_cb = ensure_named_gcb(prefetch_buffers, layout_name, device, [layout], weight_dtype)
                page_bytes = decode_gcb_page_bytes([layout], weight_dtype)
            proj = LinearDecode(
                weight,
                device,
                cache.file(f"{cache_key}.full"),
                dtype=weight_dtype,
                **layout,
                use_prefetcher=True,
                global_cb=global_cb,
                global_cb_page_bytes=page_bytes,
                rectangle_b_grid=True,
                use_rm_hs=True,
            )
            assert (
                proj.use_prefetcher and proj.use_rm_hs and not proj.partial_width_sharded
            ), f"{layout_name} must be prefetched, ROW_MAJOR HEIGHT_SHARDED, fully width-sharded"
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

    def _ensure_ring(self, k_local: ttnn.Tensor) -> None:
        """Persistent full-T gather buffer and the two AG direction semaphores.

        ``k_local`` is this rank's ``[B, 1, T/tp, D]`` shard. The fused op gathers into
        a replicated ``[1, 1, T, D]`` scratch (batch-1, matching indexed-cache mode).
        """
        t_local, dim = k_local.shape[2], k_local.shape[3]
        t_full = t_local * self.tp_size
        if self._k_gathered is None or list(self._k_gathered.shape) != [1, 1, t_full, dim]:
            self._k_gathered = ttnn.from_torch(
                torch.zeros(1, 1, t_full, dim, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        if self._ag_semaphores is None:
            grid = self.device.compute_with_storage_grid_size()
            cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
            # Same full-grid worker subdevice the ring-indexer unit tests load: AG cores
            # stay in that stall group so the fused gather and score wait on each other.
            if self._ag_sub_device_id is None:
                mgr = self.device.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
                self.device.load_sub_device_manager(mgr)
                self._ag_sub_device_id = ttnn.SubDeviceId(0)
                self.device.set_sub_device_stall_group([self._ag_sub_device_id])
            self._ag_semaphores = [ttnn.create_global_semaphore(self.device, cores, 0) for _ in range(2)]

    def prefetch_weights(self):
        """Stage compressor kv/gate then q_b / weights_proj."""
        self.compressor.prefetch_weights()
        self.q_b_proj.fetch_weights()
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
        pooled index keys into ``scache.idx_key_cache``."""
        self.compressor.decode_static(
            tokens,
            cos_row,
            sin_row,
            _IndexerWindowView(scache),
            scache.idx_key_cache,
            win_slot,
            win_row,
            pool=pool,
        )

    def select(
        self,
        query: ttnn.Tensor,
        key_cache: ttnn.Tensor,
        head_weights: ttnn.Tensor,
        logits_out: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """``indexer_score_dsa`` (TP1) or ``ring_indexer_score_dsa`` (TP>1) then ``topk_large_indices``.

        ``query`` ``[B, Hi, Sq, D]`` TILE, ``key_cache`` TILE, ``head_weights``
        ``[B, 1, Sq, Hi]`` TILE with both scales already folded.
        At TP1 ``key_cache`` is the full ``[B, 1, T, D]``. At TP>1 it is this rank's
        ``[B, 1, T/tp, D]`` shard on the TP mesh axis; the fused op all-gathers into a
        persistent full-T buffer. Query / weights use the same sequence-shard (local
        ``Sq`` tile-aligned). Future / pad columns must already be ``-inf`` on the
        score path (mask on device; do not pass host ``kv_len``). Returns uint32
        ``[B, 1, Sq, k]``.

        ``Sq`` must be a multiple of the 32-row tile: the score ops reject anything
        shorter ("Sq 1, T .., D .. must be tile-aligned"). One call therefore scores
        ONE key-cache slot against a whole tile of query rows, so a packed decode
        step cannot score one query row per user -- see :meth:`score_and_select`.
        Checked here so the caller gets a message instead of a device abort.
        """
        assert list(query.shape)[2] % 32 == 0, (
            f"indexer score needs a tile-aligned query block, got Sq={list(query.shape)[2]} "
            "(1-row decode queries are rejected on device)"
        )
        if self.tp_size > 1:
            t_local = list(key_cache.shape)[2]
            assert t_local % 32 == 0, f"TP{self.tp_size} index-key shard T/tp={t_local} must be tile-aligned"
            self._ensure_ring(key_cache)
            # A 1xTP submesh of Galaxy is a line: TORUS_X wraps the full 8-chip X axis, not
            # these four ranks, so Topology.Ring hangs on a wrap link that does not exist.
            scores = ttnn.experimental.ring_indexer_score_dsa(
                query,
                self._k_gathered,
                head_weights,
                key_cache,
                self._ag_semaphores,
                cluster_axis=self.cluster_axis,
                topology=ttnn.Topology.Linear,
                num_links=1,
                ag_sub_device_id=self._ag_sub_device_id,
                program_config=ttnn.IndexerScoreProgramConfig(head_group_size=0),
            )
        else:
            scores = ttnn.experimental.indexer_score_dsa(query, key_cache, head_weights)
        if logits_out is not None:
            ttnn.copy(scores, logits_out)
            scores = logits_out
        return ttnn.experimental.topk_large_indices(
            scores,
            k=self.index_topk,
            subdevice_id=self._ag_sub_device_id,
        )

    def score_and_select(
        self,
        tokens: ttnn.Tensor,
        q_a: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        key_cache: ttnn.Tensor,
        logits_out: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Project indexer q / folded weights, RoPE q, then :meth:`select`.

        NOT usable as written for a packed decode step: it reshapes to ``[users, Hi, 1, D]``, i.e.
        ``Sq == 1``, which ``indexer_score_dsa`` rejects (see :meth:`select`). The users axis has to
        become the query block (a tile of rows, pad rows masked) and the key cache has to be
        addressed per slot, since one call scores exactly one cache slot. Resolve that when wiring
        this into the decode loop.
        """
        q = self.q_b_proj(_decode_activation(self.q_b_proj, q_a))
        q = _apply_rope(q, cos_row, sin_row, self.rot, self.rope_dim, head_dim=self.head_dim)
        users = _packed_users(tokens)
        q = ttnn.reshape(q, [users, self.num_heads, 1, self.head_dim])
        q = ttnn.to_layout(ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
        w = self.weights_proj(_decode_activation(self.weights_proj, tokens))
        w = ttnn.reshape(w, [users, 1, 1, self.num_heads])
        w = ttnn.to_layout(ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
        return self.select(q, key_cache, w, logits_out=logits_out)

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
