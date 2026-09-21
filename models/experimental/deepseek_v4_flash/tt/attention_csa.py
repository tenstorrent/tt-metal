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

The CSA Lightning Indexer only affects *which* compressed entries each query may see (the
``block_bias``); for ``seq_len <= index_topk * compress_rate`` its top-k selects every entry, so
the block bias reduces to plain causal masking over windows, which the caller builds on host.
The compressed KV values themselves (this module's output) do not depend on the indexer.

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
    _update_cache_at,
)
from .common import _profile, _signpost, width_sharded_l1_config
from .layers import DeepSeekV4RMSNorm
from .paged_cache import PagedLayerView
from .system_config import active_system_config
from .weight_cache import WeightCache, _as_cache, _load_weight, _materialize


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

    The CSA Lightning Indexer only affects *which* compressed entries each query
    may see (the ``block_bias``); for ``seq_len <= index_topk * compress_rate``
    its top-k selects every entry, so the block_bias reduces to plain causal
    masking over windows, which the caller builds on host. The compressed KV
    values themselves (this module's output) do not depend on the indexer.
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
        self.head_dim = config.head_dim
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
        )
        self.kv_norm = DeepSeekV4RMSNorm(
            weights["compressor.kv_norm.weight"], self.eps, device, cache.file("compressor.kv_norm"), sharded=True
        )
        pb = _materialize(weights["compressor.position_bias"], cache.file("compressor.position_bias"), ttnn.bfloat16)
        self.position_bias = _load_weight(
            pb.reshape(1, 1, self.compress_rate, 2 * self.head_dim) if pb is not None else None,
            device,
            cache_file_name=cache.file("compressor.position_bias"),
        )
        if self.position_bias is not None:
            self.position_bias = _rm_width_sharded(self.position_bias, self.compress_rate, 2 * self.head_dim)
        # Per-user row offsets into the packed window buffer, built on first use and
        # then reused (see :meth:`_win_index`). Batch 1 needs none: user 0's row is
        # the window slot itself.
        self._win_offsets: ttnn.Tensor | None = None

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
