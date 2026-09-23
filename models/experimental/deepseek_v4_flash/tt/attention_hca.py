"""HCA (heavily compressed attention) decode compressor.

Compresses every complete window of ``compress_rate`` (128) source tokens into a single
softmax-gated KV entry with the Welford-free shared pooling (``gate`` softmaxed over the window
axis, convex-combining the ``kv`` rows), then RoPEs that entry at its window's absolute position
and appends it to the compressed region of the layer's paged KV axis. Only the window currently being filled is buffered, so a step
costs ``O(compress_rate)``.

HCA has no lightning indexer: every closed window is visible to every query, which is why the
valid KV set is a contiguous prefix and the caller can bound SDPA by a position instead of an
additive mask. CSA's short-sequence trace (sequence
length below ``compress_rate * index_topk``) and its long-sequence indexer trace both leave
HCA on that dense path. The compressor's state is the one-window ``win_kv`` / ``win_gate``
pair, kept as TILE DRAM ``[B, 1, compress_rate, Dh]`` for ``paged_update_cache``.

Pooling runs only on the steps that close a window -- an entry is emitted once every
``compress_rate`` tokens and the additive block-bias exposes entries
``w < (pos+1)//compress_rate``, constant between closures -- so pooling at each closure and
reusing the result in between is bit-identical to pooling every step at ``1/compress_rate`` of
the cost; the caller drives that via the ``pool`` flag (see
``DeepSeekV4Model._compressor_pool_due``).

CSA has its own compressor (:mod:`.attention_csa`). The helpers shared with the rest of the
attention block live in :mod:`.attention`, which imports this module lazily (see
``attention._compressor_class``) because this module needs those helpers in turn.
"""

from typing import Optional

import ttnn

from .attention import (
    _StaticLayerCache,
    _apply_rope,
    _compressor_projections,
    _decode_activation,
    _one_row_per_user,
    _packed_users,
    _update_cache_at,
    _update_kv_at,
)
from .common import _signpost
from .layers import DeepSeekV4RMSNorm
from .paged_cache import PagedLayerView
from .system_config import active_system_config
from .weight_cache import WeightCache, _as_cache, _load_weight, _materialize


def _softmax_weighted_sum(kv: ttnn.Tensor, gate: ttnn.Tensor, window_axis: int) -> ttnn.Tensor:
    """``sum_w softmax(gate, axis=w) * kv`` over the window axis: ``[..., W, Dh]`` -> ``[..., Dh]``.

    Shared compressor pooling (``DeepseekV4*Compressor``): the gate logits are
    softmaxed over the per-window token axis and used to convex-combine the kv
    rows into one compressed entry per window. The reduced axis is dropped
    (``ttnn.sum`` defaults to ``keepdim=False``), so HCA's ``[B, 1, cr, Dh]``
    window comes back ``[B, 1, Dh]``.
    """
    weights = ttnn.softmax(gate, dim=window_axis)
    return ttnn.sum(ttnn.multiply(kv, weights), dim=window_axis)


class DeepSeekV4HCACompressor:
    """Heavily-Compressed-Attention compressor (decode, running KV cache).

    Compresses every complete window of ``compress_rate`` (m'=128) source tokens
    into a single softmax-gated KV entry, then RoPEs that entry at its window's
    absolute position and appends it to the compressed region of the layer's
    paged KV axis. Only the window currently
    being filled is buffered, so a step costs ``O(compress_rate)``.
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
        """Build the compressor's projections, norm and position bias from ``weights``.

        ``weights`` holds the ``compressor.kv_proj`` / ``gate_proj`` weights ``[D, Dh]``
        (DRAM ND-sharded under the prefetcher), ``compressor.kv_norm``, and
        ``compressor.position_bias``, reshaped here to ``[1, 1, compress_rate, Dh]`` --
        indexed by a token's offset *within* its window. ``rot`` is the shared
        ``[Rd, Rd]`` rotate matrix.
        """
        self.device = device
        self.rope_dim = rope_dim
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        cache = _as_cache(cache)
        if num_prefetch_pages is None:
            num_prefetch_pages = active_system_config().prefetcher.num_prefetch_pages
        self.kv_proj, self.gate_proj = _compressor_projections(
            "heavily_compressed_attention",
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
        # position_bias: [compress_rate, head_dim] -> broadcast over [B, n_win].
        pb = _materialize(weights["compressor.position_bias"], cache.file("compressor.position_bias"), ttnn.bfloat16)
        self.position_bias = _load_weight(
            pb.reshape(1, 1, self.compress_rate, self.head_dim) if pb is not None else None,
            device,
            cache_file_name=cache.file("compressor.position_bias"),
        )

    def prefetch_weights(self, *, index_sparse: bool = False):
        """Stage the two projection weights ahead of the :meth:`decode_static` that uses them.

        Queued kv before gate, the order :meth:`_project` pops them off their shared GCB;
        both stay DRAM ND-sharded ``[D, Dh]``, the prefetcher pushing their pages into the
        matmul's in1 buffer. ``index_sparse`` is the CSA indexer trace flag; HCA has no
        indexer, so it is ignored.
        """
        del index_sparse
        self.kv_proj.fetch_weights()
        self.gate_proj.fetch_weights()

    def _project(self, tokens: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``tokens`` ``[1, 1, B, D]`` -> per-token ``(kv, gate)`` ``[1, 1, B, Dh]`` each.

        ``LinearDecode`` leaves its result width-sharded over the cores it reduced onto, while
        the callers reshape these and reshard them height-wise for the cache write, so hand
        back the DRAM-interleaved form they expect (as ``_o_proj`` does for o_b_proj).
        """
        return (
            ttnn.to_memory_config(self.kv_proj(_decode_activation(self.kv_proj, tokens)), ttnn.DRAM_MEMORY_CONFIG),
            ttnn.to_memory_config(self.gate_proj(_decode_activation(self.gate_proj, tokens)), ttnn.DRAM_MEMORY_CONFIG),
        )

    def _pool_window(
        self, win_kv: ttnn.Tensor, win_gate: ttnn.Tensor, cos_row: ttnn.Tensor, sin_row: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Pool each user's closed window ``[B, 1, compress_rate, Dh]`` into that window's
        single compressed entry, returned as ``[1, B, 1, Dh]`` (RoPE'd at ``cos_row`` /
        ``sin_row``, the window's own position) ready for the cache write.

        The buffer shape doubles as the ``[B, n_win, compress_rate, Dh]`` the pool wants with
        ``n_win == 1``, so ``position_bias`` (indexed by a token's offset *within* its window)
        broadcasts over both users and windows unchanged.
        """
        users = win_kv.shape[0]
        gate = ttnn.add(win_gate, self.position_bias)
        compressed = _softmax_weighted_sum(win_kv, gate, window_axis=2)
        # Back onto packed rows for the norm + RoPE, which are per-token arithmetic.
        compressed = ttnn.reshape(compressed, [1, 1, users, self.head_dim])
        compressed = self.kv_norm(compressed)
        compressed = _apply_rope(compressed, cos_row, sin_row, self.rot, self.rope_dim)
        return _one_row_per_user(compressed)

    def decode_static(
        self,
        tokens: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        scache: "_StaticLayerCache",
        paged: PagedLayerView,
        win_slot: ttnn.Tensor,
        win_row: ttnn.Tensor | None = None,
        pool: bool = True,
    ) -> None:
        """Trace-safe decode: write each user's token projection in place at ``win_slot``
        (``pos % compress_rate``) into the one-window ``[B, 1, compress_rate, Dh]``
        buffers, and -- on the step that closes the window -- pool just that window
        and append its single entry at row ``win_row`` of the layer's paged KV axis.

        ``tokens`` is the block's packed-row hidden ``[1, 1, B, D]``, already gathered
        onto the decode activation grid when the caller used :meth:`~.attention.DeepSeekV4Attention.decode_static`;
        ``cos_row`` / ``sin_row`` are the closing window's RoPE row ``[1, 1, 1, Rd]`` and
        ``win_slot`` / ``win_row`` INT32 ``[B]`` row-index vectors, ``win_row`` read only
        when ``pool``.

        ``pool`` is set by the caller only on the steps that close a window, so the
        cost per step is ``O(compress_rate)`` rather than ``O(max_seq)``: in between,
        the KV axis already holds exactly the entries the block-bias exposes
        (see the module header).
        """
        _signpost("HCA_START")
        users = _packed_users(tokens)
        kv, gate = self._project(tokens)  # [1, 1, B, Dh]
        kv = _one_row_per_user(ttnn.reshape(kv, [1, 1, users, self.head_dim]))
        gate = _one_row_per_user(ttnn.reshape(gate, [1, 1, users, self.head_dim]))
        _update_cache_at(scache.win_kv, kv, win_slot)
        _update_cache_at(scache.win_gate, gate, win_slot)
        if pool:
            pooled = self._pool_window(scache.win_kv, scache.win_gate, cos_row, sin_row)
            _update_kv_at(paged, pooled, win_row)
            ttnn.deallocate(pooled)
        _signpost("HCA_END")
