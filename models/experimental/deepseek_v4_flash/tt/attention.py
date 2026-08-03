from typing import Optional

import ttnn
import torch

from .common import DeepSeekV4Module, _HIFI4_SDPA, _MASK_NEG, _profile, width_sharded_l1_config
from .layers import BatchedLinearDecode, DeepSeekV4RMSNorm, Linear, LinearDecode, _rms_norm_unweighted
from .paged_cache import PagedLayerView
from .weight_cache import WeightCache, _as_cache, _load_weight, _materialize


# ---------------------------------------------------------------------------- #
# DeepSeek-V4-Flash attention (decode, running KV cache)
#
# ttnn port of ``DeepseekV4Attention`` (and its CSA / HCA compressors) from
# ``modular_deepseek_v4.py``. Scope is *decode only*: each step appends the new
# token's K=V (and compressor projections) to the running cache and attends the
# tokens-so-far, via the fused ``scaled_dot_product_attention_decode`` op.
#
# Layout conventions, matching the reference:
#   B = batch, S = query/seq length, H = num_attention_heads, Dh = head_dim,
#   Rd = qk_rope_head_dim (the trailing RoPE slice of each head).
# V4 is shared-KV MQA (one KV head broadcast to all query heads) and lays each
# head out as ``[nope | rope]`` with interleaved RoPE on the trailing ``Rd``.
# ---------------------------------------------------------------------------- #
# KV / compressor cache (decode)
#
# The only cross-token state in the V4-Flash stack lives in attention: the
# hyper-connection streams, RMSNorms, the routed/shared MoE and the MLP are all
# strictly per-token. So a single-token decode step only needs to remember, per
# decoder layer:
#
#   * the rotated sliding K=V entries (shared-KV MQA, K==V), capped to the
#     ``sliding_window`` most recent tokens, and
#   * for CSA / HCA layers, every source token's compressor projections
#     (``kv`` / ``gate``); the compressed long-range entries are re-pooled from
#     these with the exact prefill pooling, so decode is bit-for-bit the same
#     function of the tokens-so-far as a full prefill over them (no separate
#     rolling-window / overlap / entry-count bookkeeping needed).
#
# The pool runs over the whole fixed capacity, so its cost scales with
# ``max_seq``, not with the current position. It is therefore run only on the
# steps where it can change: a compressor emits a new entry once every
# ``compress_rate`` tokens, and the additive block-bias exposes entries
# ``w < (pos+1)//compress_rate`` -- a quantity that is constant across the
# ``compress_rate`` steps between two window closures. So pooling at each
# closure and reusing the result in between is bit-identical to pooling every
# step, at ``1/compress_rate`` of the cost. The pooled entries are kept in the
# persistent ``compressed`` cache below; callers drive the schedule via the
# ``pool`` flag (see ``DeepSeekV4Model._compressor_pool_due``).
#
# Cache updates follow the GPT-OSS / tt-transformers paged-KV pattern: fixed-size
# DRAM buffers written in place each step via ``paged_update_cache`` (with a
# device-tensor index, trace-safe). The traced decode path additionally requires
# step-invariant shapes / addresses; the eager path uses the same buffers and
# ops but builds the additive mask on host.
# ---------------------------------------------------------------------------- #
class _StaticLayerCache:
    """Fixed-size, in-place per-layer decode caches (eager + traced decode).

    DRAM tensors of a fixed capacity written in place at the new token's position
    by ``paged_update_cache`` (a device-tensor index):

      * ``sliding`` ``[1, 1, window, Dh]`` -- a ring buffer (slot ``pos % window``);
        attention masks unwritten / out-of-window slots. Sliding-only layers only;
        for CSA/HCA the ring lives in ``combined`` (below).
      * ``win_kv`` / ``win_gate`` ``[1, 1, compress_rate, feat]`` -- the compressor
        projections of the window *currently being filled*, at slot
        ``pos % compress_rate``. Only the ``compress_rate`` tokens of one window are
        held, because pooling is incremental: the step that closes a window pools
        just that window and appends its single entry (see :meth:`_pool_window`).
        ``None`` for sliding-only layers.
      * ``prev_kv`` / ``prev_gate`` ``[1, 1, compress_rate, 2*Dh]`` -- the previous
        window's projections, kept only by CSA because its entry ``w`` also needs
        window ``w-1``'s Ca slice. Refreshed from ``win_*`` after each pool.
        ``prev_gate`` starts at ``_MASK_NEG`` so window 0's absent Ca half carries
        softmax weight 0. ``None`` for HCA and sliding-only layers.
      * ``combined`` ``[1, 1, window + cap // compress_rate, Dh]`` -- the single
        K==V buffer a CSA/HCA layer hands to SDPA, holding *both* regions of the
        attention axis: the sliding ring in rows ``[0, window)`` and the pooled
        (normed, RoPE'd) compressed entries in rows ``[window, ...)``.
        ``None`` for sliding-only layers.

    Keeping both regions in one buffer removes a per-step ``concat``: the ring
    slot is ``pos % window``, already inside the prefix, so the ordinary
    ``paged_update_cache`` write lands in the right place, and each pooled entry
    goes in at row ``window + w`` by the same in-place write.
    That the sliding region comes *first* is also what makes the valid set a
    contiguous prefix, and hence causal SDPA possible (:func:`sdpa_causal_ok`).

    Built empty (all-zero) by :func:`build_static_layer_cache` /
    :meth:`DeepSeekV4Model.reset_caches`; the prompt is written in by replaying
    decode one token at a time.
    """

    __slots__ = ("sliding", "win_kv", "win_gate", "prev_kv", "prev_gate", "combined")

    def __init__(
        self,
        sliding: Optional[ttnn.Tensor],
        win_kv: Optional[ttnn.Tensor],
        win_gate: Optional[ttnn.Tensor],
        prev_kv: Optional[ttnn.Tensor] = None,
        prev_gate: Optional[ttnn.Tensor] = None,
        combined: Optional[ttnn.Tensor] = None,
    ):
        self.sliding = sliding
        self.win_kv = win_kv
        self.win_gate = win_gate
        self.prev_kv = prev_kv
        self.prev_gate = prev_gate
        self.combined = combined


def build_static_layer_cache(
    device: ttnn.MeshDevice,
    sliding_window: int,
    layer_type: str,
    head_dim: int,
    max_seq: int,
    compress_rates: dict,
    paged: bool = False,
) -> _StaticLayerCache:
    """Allocate a layer's fixed-size in-place caches empty (all-zero).

    ``paged`` leaves the KV buffers (``sliding`` / ``combined``) unallocated: those
    reads and writes go through the shared block pool instead (see
    :mod:`.paged_cache`), and only the small compressor window buffers -- which are
    per-session state swapped outside the trace -- are still owned per layer.
    """

    def _filled(rows: int, width: int, value: float = 0.0) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.full((1, 1, rows, width), value),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # CSA/HCA layers keep the sliding ring inside ``combined`` rather than in its
    # own buffer, so only sliding-only layers allocate ``sliding``.
    sliding = None if paged or layer_type != "sliding_attention" else _filled(sliding_window, head_dim)
    win_kv = win_gate = prev_kv = prev_gate = combined = None
    if layer_type != "sliding_attention":
        cr = compress_rates[layer_type]
        cap = max_seq
        is_csa = layer_type == "compressed_sparse_attention"
        feat = (2 if is_csa else 1) * head_dim
        # Only one window's worth of projections: pooling is incremental.
        win_kv = _filled(cr, feat)
        win_gate = _filled(cr, feat)
        if is_csa:
            # Entry w pools window w-1's Ca with window w's Cb, so CSA also keeps the
            # previous window. ``-inf`` gates give window 0's absent Ca weight 0.
            prev_kv = _filled(cr, feat)
            prev_gate = _filled(cr, feat, _MASK_NEG)
        # ``[sliding ring | compressed entries]`` on one axis. The width matches the
        # mask that :func:`host_decode_mask` builds for this layer.
        if not paged:
            combined = _filled(sliding_window + max(cap // cr, 0), head_dim)
    return _StaticLayerCache(sliding, win_kv, win_gate, prev_kv, prev_gate, combined)


def int32_pos_tensor(pos: int, device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Single INT32 position scalar ``[1]`` on ``device`` (for ``paged_update_cache``)."""
    return ttnn.from_torch(
        torch.tensor([pos], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def host_decode_mask(
    sliding_window: int,
    layer_type: str,
    compress_rate: int | None,
    pos: int,
    max_seq: int,
    device: ttnn.MeshDevice,
) -> ttnn.Tensor:
    """Host-built additive decode mask for one absolute position ``pos``.

    Mirrors the on-device mask in :meth:`DeepSeekV4Model._device_mask`: sliding
    columns mask slots with index ``> pos``; compressor columns mask windows with
    index ``>= (pos+1)//cr``.
    """
    if layer_type == "sliding_attention":
        invalid = torch.arange(sliding_window, dtype=torch.float32) > pos
        width = sliding_window
    else:
        n_win_cap = max_seq // compress_rate
        a = torch.cat([torch.arange(sliding_window), torch.full((n_win_cap,), -1.0)]).float()
        b = torch.cat([torch.full((sliding_window,), -1.0), torch.arange(n_win_cap)]).float()
        thr = (pos + 1) // compress_rate
        invalid = (a > pos) | (b >= thr)
        width = sliding_window + n_win_cap
    mask = torch.zeros(1, 1, 1, width, dtype=torch.float32)
    mask.masked_fill_(invalid.view(1, 1, 1, -1), _MASK_NEG)
    return ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def sdpa_causal_ok(sliding_window: int, layer_type: str, pos: int) -> bool:
    """Can this step's valid set be expressed as a single SDPA-decode ``cur_pos``?

    The CSA/HCA KV axis is ``[sliding 0..W) | compressor 0..n_win_cap)`` and the
    valid set (see :func:`host_decode_mask`) is sliding slot ``i <= pos`` plus
    compressor window ``j < (pos+1)//cr``. Once the ring is full every sliding slot
    is valid, so the union is the contiguous prefix ``[0, W + (pos+1)//cr)`` -- which
    a single ``cur_pos`` describes exactly. Below that the set has a hole (slots
    ``pos+1 .. W-1`` are still unwritten) that no ``cur_pos`` can express, so those
    steps must keep the additive mask.
    """
    return layer_type != "sliding_attention" and pos + 1 >= sliding_window


def sdpa_causal_cur_pos(sliding_window: int, compress_rate: int, pos: int) -> int:
    """Inclusive last-valid index on the ``[sliding | compressor]`` KV axis at ``pos``.

    Only meaningful when :func:`sdpa_causal_ok`. Note the ``-1``: ``(pos+1)//cr`` is
    the *count* of closed windows, and ``cur_pos`` is inclusive. Dropping it (i.e.
    using ``W + pos//cr``) happens to agree only at window boundaries and otherwise
    exposes the still-open window, whose entry is unpooled -- a silent accuracy loss
    rather than an error.
    """
    return sliding_window + (pos + 1) // compress_rate - 1


def _interleaved_rotate_matrix(rope_dim: int) -> torch.Tensor:
    """Fixed ``[Rd, Rd]`` matrix ``R`` s.t. ``x @ R == rotate_half(x)``.

    GLM/V4 interleaved ``rotate_half`` maps each consecutive pair
    ``(x_{2p}, x_{2p+1}) -> (-x_{2p+1}, x_{2p})``. As a right-multiply that is a
    block-diagonal matrix of ``[[0, 1], [-1, 0]]`` blocks, which lets us express
    the rotation as a single on-device matmul instead of strided gathers.
    """
    r = torch.zeros(rope_dim, rope_dim, dtype=torch.float32)
    for p in range(rope_dim // 2):
        r[2 * p, 2 * p + 1] = 1.0
        r[2 * p + 1, 2 * p] = -1.0
    return r


def make_rope_table(cos_half: torch.Tensor, sin_half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand half-sized rotary ``(cos, sin)`` to full ``Rd`` and shape ``[1,1,L,Rd]``.

    ``DeepseekV4RotaryEmbedding`` emits one entry per interleaved pair; the
    reference ``apply_rotary_pos_emb`` does ``repeat_interleave(2)`` before the
    rotation. We bake that into the host-side table (broadcast over batch/heads).
    """
    cos = cos_half.repeat_interleave(2, dim=-1)
    sin = sin_half.repeat_interleave(2, dim=-1)
    cos = cos.reshape(1, 1, cos.shape[-2], cos.shape[-1]).float()
    sin = sin.reshape(1, 1, sin.shape[-2], sin.shape[-1]).float()
    return cos, sin


# ``rot`` (the ``[Rd, Rd]`` interleaved rotate matrix) is block-diagonal in 32-wide
# blocks, so the single top-left ``[32, 32]`` tile is the per-tile ``rotate_half`` the
# fused device op applies to every rope tile. Derive + cache it once per ``rot`` object.
_TRANS_MAT_CACHE: dict[int, ttnn.Tensor] = {}


def _trans_mat_for(rot: ttnn.Tensor) -> ttnn.Tensor:
    tm = _TRANS_MAT_CACHE.get(id(rot))
    if tm is None:
        tm = ttnn.reshape(
            ttnn.slice(rot, [0, 0], [ttnn.TILE_SIZE, ttnn.TILE_SIZE]), [1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE]
        )
        # The fused op reads trans_mat from a DRAM-interleaved source.
        tm = ttnn.to_memory_config(tm, ttnn.DRAM_MEMORY_CONFIG)
        _TRANS_MAT_CACHE[id(rot)] = tm
    return tm


def _rope_height_sharded_config(width: int, num_cores: int, device) -> ttnn.MemoryConfig:
    """Height-sharded L1 config: one tile-row (32 rows) per core over ``num_cores`` cores."""
    grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    shard_spec = ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, rot: ttnn.Tensor, rope_dim: int) -> ttnn.Tensor:
    """Interleaved RoPE on the trailing ``rope_dim`` channels of ``x`` ([.., D]).

    ``cos`` / ``sin`` are ``[1,1,L,rope_dim]`` tables (broadcast over batch/heads);
    ``rot`` is the ``[rope_dim, rope_dim]`` ``rotate_half`` matrix. Leading "nope"
    channels pass through untouched.

    Delegates the whole calc to the fused ``ttnn.experimental.fused_partial_rope`` device
    op: ``x`` is height-sharded one tile-row per core while ``cos`` / ``sin`` / ``trans_mat``
    are DRAM-interleaved (the reader streams each core's rope tile-row), then the sharded
    output is converted back to ``x``'s original memory config.
    """
    device = x.device()
    d = x.shape[-1]
    rows = x.shape[-2]

    # The op reads one cos/sin tile-row per core, or a single tile-row broadcast across all
    # rows on device (e.g. a shared decode position over heads). So cos/sin must cover either
    # every input row or exactly one row.
    assert cos.shape[-2] in (rows, 1), f"{cos.shape} not broadcastable to rows={rows}"

    # cos/sin must already be DRAM-interleaved (the fused op's reader streams them from DRAM).
    assert cos.memory_config().buffer_type == ttnn.BufferType.DRAM, "cos must be DRAM-interleaved"
    assert sin.memory_config().buffer_type == ttnn.BufferType.DRAM, "sin must be DRAM-interleaved"

    if not x.is_sharded():
        x = ttnn.to_memory_config(x, width_sharded_l1_config(rows, d, device))

    out_sh = ttnn.experimental.fused_partial_rope(x, cos, sin, _trans_mat_for(rot), rope_dim)
    return out_sh


# ---------------------------------------------------------------------------- #
# Traced-decode helpers (fixed-size, in-place KV cache via ``paged_update_cache``)
#
# A reusable ``ttnn`` trace requires fixed tensor shapes / addresses and no host
# round-trips inside the captured region, so the traced decode swaps the eager
# concat-grown caches for fixed-size DRAM buffers that are written *in place*
# every step at the new token's position (a device-tensor index, so the same
# trace serves every step). ``paged_update_cache`` is the canonical trace-safe
# in-place KV writer (it mutates the persistent cache buffer during capture,
# unlike ``ttnn.copy`` which is rejected mid-capture).
# ---------------------------------------------------------------------------- #
def _height_sharded_l1_config(width: int) -> ttnn.MemoryConfig:
    """Single-core height-sharded L1 config for a ``[1, 1, 1, width]`` decode row.

    ``paged_update_cache`` requires its (single-token) input to be height-sharded
    with one core per batch user (B == 1 here -> one core), shard width == the
    last dim, ROW_MAJOR orientation.
    """
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    shard_spec = ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _update_cache_at(
    cache: ttnn.Tensor,
    row: ttnn.Tensor,
    pos_tensor: ttnn.Tensor,
    paged: PagedLayerView | None = None,
) -> None:
    """In-place write ``row`` ``[1, 1, 1, F]`` into a KV cache at ``pos_tensor`` ``[1]``
    (INT32), either the layer's own dense buffer or -- when ``paged`` is given --
    ``paged.pool`` through the active session's page table.

    ``paged.position_modulo`` wraps the logical position into a bounded capacity
    before the page-table lookup, which is what makes a sliding-window session need
    only ``window / block_size`` blocks; without it any position past that capacity
    resolves through the row's unmapped tail (see :mod:`.paged_cache`).
    """
    width = row.shape[-1]
    row_sharded = ttnn.to_memory_config(row, _height_sharded_l1_config(width))
    if paged is None:
        ttnn.experimental.paged_update_cache(cache, row_sharded, update_idxs_tensor=pos_tensor)
    else:
        ttnn.experimental.paged_update_cache(
            paged.pool,
            row_sharded,
            update_idxs_tensor=pos_tensor,
            page_table=paged.page_table,
            cache_position_modulo=paged.position_modulo,
        )
    ttnn.deallocate(row_sharded)


def _softmax_weighted_sum(kv: ttnn.Tensor, gate: ttnn.Tensor, window_axis: int) -> ttnn.Tensor:
    """``sum_w softmax(gate, axis=w) * kv`` over the window axis.

    Shared compressor pooling (``DeepseekV4*Compressor``): the gate logits are
    softmaxed over the per-window token axis and used to convex-combine the kv
    rows into one compressed entry per window.
    """
    weights = ttnn.softmax(gate, dim=window_axis)
    return ttnn.sum(ttnn.multiply(kv, weights), dim=window_axis)


class DeepSeekV4HCACompressor:
    """Heavily-Compressed-Attention compressor (decode, running KV cache).

    Compresses every complete window of ``compress_rate`` (m'=128) source tokens
    into a single softmax-gated KV entry, then RoPEs that entry at its window's
    absolute position and appends it to the compressed region of the layer's
    combined KV buffer (see :class:`_StaticLayerCache`). Only the window currently
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
    ):
        self.device = device
        self.rope_dim = rope_dim
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        cache = _as_cache(cache)
        self.kv_proj = Linear(
            weights["compressor.kv_proj.weight"], device, cache.file("compressor.kv_proj"), dtype=weight_dtype
        )
        self.gate_proj = Linear(
            weights["compressor.gate_proj.weight"], device, cache.file("compressor.gate_proj"), dtype=weight_dtype
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

    def _project(self, hidden: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``hidden`` ``[B, S, D]`` -> per-token ``(kv, gate)`` ``[B, S, Dh]`` each."""
        return self.kv_proj(hidden), self.gate_proj(hidden)

    def _pool_window(
        self, win_kv: ttnn.Tensor, win_gate: ttnn.Tensor, cos_row: ttnn.Tensor, sin_row: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Pool one closed window's ``[1, 1, compress_rate, Dh]`` projections into that
        window's single compressed entry ``[1, 1, 1, Dh]``, RoPE'd at ``cos_row`` /
        ``sin_row`` (the window's own position).

        The buffer shape doubles as the ``[B, n_win, compress_rate, Dh]`` the pool
        wants with ``B == n_win == 1``, so ``position_bias`` (indexed by a token's
        offset *within* its window) broadcasts unchanged.
        """
        gate = ttnn.add(win_gate, self.position_bias)
        compressed = _softmax_weighted_sum(win_kv, gate, window_axis=2)
        compressed = ttnn.reshape(compressed, [1, 1, 1, self.head_dim])
        compressed = self.kv_norm(compressed)
        return _apply_rope(compressed, cos_row, sin_row, self.rot, self.rope_dim)

    def decode_static(
        self,
        hidden: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        scache: "_StaticLayerCache",
        combined_cache: ttnn.Tensor | None,
        win_slot: ttnn.Tensor,
        win_row: ttnn.Tensor | None = None,
        pool: bool = True,
        paged: PagedLayerView | None = None,
    ) -> None:
        """Trace-safe decode: write this token's projection in place at ``win_slot``
        (``pos % compress_rate``) into the one-window ``[1, 1, compress_rate, Dh]``
        buffers, and -- on the step that closes the window -- pool just that window
        and append its single entry at row ``win_row`` of the layer's KV axis
        (``combined_cache``, or ``paged``'s block pool).

        ``pool`` is set by the caller only on the steps that close a window, so the
        cost per step is ``O(compress_rate)`` rather than ``O(max_seq)``: in between,
        the KV axis already holds exactly the entries the block-bias exposes
        (see the module header).
        """
        kv, gate = self._project(hidden)  # [1, 1, 1, Dh]
        kv = ttnn.reshape(kv, [1, 1, 1, self.head_dim])
        gate = ttnn.reshape(gate, [1, 1, 1, self.head_dim])
        _update_cache_at(scache.win_kv, kv, win_slot)
        _update_cache_at(scache.win_gate, gate, win_slot)
        if pool and (combined_cache is not None or paged is not None):
            pooled = self._pool_window(scache.win_kv, scache.win_gate, cos_row, sin_row)
            _update_cache_at(combined_cache, pooled, win_row, paged=paged)
            ttnn.deallocate(pooled)


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
    ):
        self.device = device
        self.rope_dim = rope_dim
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        cache = _as_cache(cache)
        self.kv_proj = Linear(
            weights["compressor.kv_proj.weight"], device, cache.file("compressor.kv_proj"), dtype=weight_dtype
        )
        self.gate_proj = Linear(
            weights["compressor.gate_proj.weight"], device, cache.file("compressor.gate_proj"), dtype=weight_dtype
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

    def _project(self, hidden: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``hidden`` ``[B, S, D]`` -> per-token ``(kv, gate)`` ``[B, S, 2*Dh]`` each."""
        return self.kv_proj(hidden), self.gate_proj(hidden)

    def _pool_window(
        self,
        prev_kv: ttnn.Tensor,
        prev_gate: ttnn.Tensor,
        win_kv: ttnn.Tensor,
        win_gate: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Pool the closing window ``w`` into its single compressed entry ``[1, 1, 1, Dh]``.

        ``win_*`` hold window ``w``'s ``[1, 1, compress_rate, 2*Dh]`` projections and
        ``prev_*`` window ``w-1``'s; the entry is the softmax-gated combination of
        window ``w-1``'s Ca half with window ``w``'s Cb half over a width-``2*cr``
        window. On the very first window ``prev_gate`` is still ``_MASK_NEG``, which
        gives the absent Ca half softmax weight 0.

        Both buffers are shaped like the ``[B, n_win, compress_rate, 2*Dh]`` the pool
        wants with ``B == n_win == 1``, so ``position_bias`` -- indexed by a token's
        offset within its own window -- broadcasts over each half unchanged.
        """
        dh = self.head_dim
        prev_g = ttnn.add(prev_gate, self.position_bias)
        cur_g = ttnn.add(win_gate, self.position_bias)
        _profile(self.device)

        ca_prev, _ = ttnn.split(prev_kv, dh, dim=3)
        cag_prev, _ = ttnn.split(prev_g, dh, dim=3)
        _, cb_cur = ttnn.split(win_kv, dh, dim=3)
        _, cbg_cur = ttnn.split(cur_g, dh, dim=3)

        new_kv = ttnn.concat([ca_prev, cb_cur], dim=2)  # [1, 1, 2*cr, Dh]
        new_gate = ttnn.concat([cag_prev, cbg_cur], dim=2)
        compressed = _softmax_weighted_sum(new_kv, new_gate, window_axis=2)
        compressed = ttnn.reshape(compressed, [1, 1, 1, dh])
        compressed = self.kv_norm(compressed)
        return _apply_rope(compressed, cos_row, sin_row, self.rot, self.rope_dim)

    def decode_static(
        self,
        hidden: ttnn.Tensor,
        cos_row: ttnn.Tensor,
        sin_row: ttnn.Tensor,
        scache: "_StaticLayerCache",
        combined_cache: ttnn.Tensor | None,
        win_slot: ttnn.Tensor,
        win_row: ttnn.Tensor | None = None,
        pool: bool = True,
        paged: PagedLayerView | None = None,
    ) -> None:
        """Trace-safe decode: write this token's ``2*Dh`` projection in place at
        ``win_slot`` into the one-window ``[1, 1, compress_rate, 2*Dh]`` buffers, and
        -- on the step that closes the window -- pool just that window (Ca/Cb overlap
        against the retained previous window) and append its single entry at row
        ``win_row`` of the layer's KV axis (``combined_cache``, or ``paged``'s pool).

        After pooling, the closing window becomes the ``prev_*`` the *next* window
        will overlap with. See :meth:`DeepSeekV4HCACompressor.decode_static`.
        """
        feat = 2 * self.head_dim
        kv, gate = self._project(hidden)  # [1, 1, 1, 2*Dh]
        kv = ttnn.reshape(kv, [1, 1, 1, feat])
        gate = ttnn.reshape(gate, [1, 1, 1, feat])
        _update_cache_at(scache.win_kv, kv, win_slot)
        _update_cache_at(scache.win_gate, gate, win_slot)
        if pool and (combined_cache is not None or paged is not None):
            pooled = self._pool_window(
                scache.prev_kv, scache.prev_gate, scache.win_kv, scache.win_gate, cos_row, sin_row
            )
            _update_cache_at(combined_cache, pooled, win_row, paged=paged)
            ttnn.deallocate(pooled)
            # Retire the closed window into ``prev_*`` for the next overlap. ``fill_cache``
            # is an in-place whole-tensor cache writer, so (unlike ``ttnn.copy``) it is
            # accepted mid trace capture, and both buffers are distinct tensors.
            ttnn.fill_cache(scache.prev_kv, scache.win_kv, 0)
            ttnn.fill_cache(scache.prev_gate, scache.win_gate, 0)


_COMPRESSORS = {
    "compressed_sparse_attention": DeepSeekV4CSACompressor,
    "heavily_compressed_attention": DeepSeekV4HCACompressor,
}


class DeepSeekV4Attention(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4Attention`` (decode only, running KV cache).

    Construct from a ``config`` (the HF ``DeepseekV4Config`` or any object
    exposing the same attributes), the layer's torch ``weights`` (HF-named
    ``state_dict`` entries), and a device. :meth:`decode` / :meth:`decode_static`
    consume pre-built RoPE tables (see :func:`make_rope_table`); these are inputs
    because the rotary embedding is owned by the surrounding model in the
    reference, not by the attention block.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        self.scaling = self.head_dim**-0.5
        cache = _as_cache(cache)
        print(f"weight_dtype: {weight_dtype}")

        self.o_b_proj = LinearDecode(
            weights["o_b_proj.weight"], device, cache.file("o_b_proj"), dtype=weight_dtype, K=8192, N=4096
        )
        self.kv_proj = LinearDecode(
            weights["kv_proj.weight"],
            device,
            cache.file("kv_proj"),
            dtype=weight_dtype,
            partial_width_sharded=True,
            k_blocks=4,
            n_blocks=16,
            N=512,
            K=4096,
        )
        self.q_b_proj = LinearDecode(
            weights["q_b_proj.weight"], device, cache.file("q_b_proj"), dtype=weight_dtype, n_blocks=64, K=1024, N=32768
        )
        self.q_a_proj = LinearDecode(
            weights["q_a_proj.weight"],
            device,
            cache.file("q_a_proj"),
            dtype=weight_dtype,
            partial_width_sharded=True,
            k_blocks=2,
            n_blocks=32,
            K=4096,
            N=1024,
        )
        # self.q_a_proj = Linear(weights["q_a_proj.weight"], device, cache.file("q_a_proj"), dtype=weight_dtype)
        # self.q_b_proj = Linear(weights["q_b_proj.weight"], device, cache.file("q_b_proj"), dtype=weight_dtype)
        # self.kv_proj = Linear(weights["kv_proj.weight"], device, cache.file("kv_proj"), dtype=weight_dtype)
        self.q_a_norm = DeepSeekV4RMSNorm(
            weights["q_a_norm.weight"], self.eps, device, cache.file("q_a_norm"), sharded=True
        )
        self.kv_norm = DeepSeekV4RMSNorm(
            weights["kv_norm.weight"], self.eps, device, cache.file("kv_norm"), sharded=True
        )

        # Grouped output projection (``DeepseekV4GroupedLinear``): block-diagonal over o_groups,
        # run as a single batched ``matmul_decode`` (batch axis = group). ``BatchedLinearDecode``
        # folds the weight along BOTH batch (group) and N into the width-sharded layout the op
        # expects. The raw torch weight is [g*o_lora_rank, (H*Dh)//g]; ``preprocess`` normalizes it
        # to the per-group [g, K, N] the class folds from (applied only on a cache miss).
        in_per_group = (self.num_heads * self.head_dim) // self.o_groups  # K
        self.o_a_proj = BatchedLinearDecode(
            weights["o_a_proj.weight"],
            device,
            cache.file("o_a_proj"),
            dtype=weight_dtype,
            batch=self.o_groups,
            K=in_per_group,
            N=self.o_lora_rank,
            preprocess=lambda w: w.reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).contiguous(),
        )

        # sinks live on host (folded into the softmax denominator), so there is
        # no tile cache for them -- always materialise.
        sinks = weights["sinks"]
        sinks = sinks() if callable(sinks) else sinks
        self.sinks_torch = sinks.reshape(1, self.num_heads, 1, 1).float()
        # Sink for the fused SDPA-decode op (:meth:`_sdpa_decode`). That kernel
        # multiplies ``scale`` into BOTH the QK logits and the sink before the
        # exp, but the reference leaves the sink un-scaled, so we pre-divide by
        # ``scaling`` to cancel it. Shape ``[H, TILE]``
        # (per-head, tile-padded width), resident so the call stays trace-safe.
        sdpa_sink = self.sinks_torch.reshape(self.num_heads, 1) / self.scaling
        sdpa_sink = torch.nn.functional.pad(sdpa_sink, (0, ttnn.TILE_SIZE - 1), "constant", value=0.0)
        self.sdpa_sinks_tt = ttnn.from_torch(sdpa_sink, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # SDPA-decode needs an explicit program config (k_chunk_size) when given an
        # attn_mask. The K=V sequence (sliding window + compressor windows) is a
        # multiple of the tile size, so a 32-wide chunk divides it cleanly.
        #
        # ``max_cores_per_head_batch`` (NOT the grid) is the L1 lever here: this is
        # MQA (one shared KV head) at batch 1, so there is a single reduction group
        # and the op assigns ``min(grid, max_cores_per_head_batch)`` cores to reduce
        # that one head. Its per-core reduction-scratch CB grows as
        # ``(out_tiles + 2*PNHt) * (cores_per_head - 1)``; with the default 16 and
        # ``head_dim == 256`` that overflows L1 (~1.8 MB > 1.5 MB), independent of
        # the grid. Capping it to 4 shrinks that CB ~5x while still parallelising
        # the KV reduction 4 ways.
        self._sdpa_pcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=0,
            k_chunk_size=32,
            exp_approx_mode=False,
            max_cores_per_head_batch=4,
        )

        # The rotate-half matrix must stay precise (a bf4 rotation would corrupt RoPE).
        self.rot = _load_weight(_interleaved_rotate_matrix(self.rope_dim), device, cache_file_name=cache.file("rot"))
        compressor_cls = _COMPRESSORS.get(self.layer_type)
        print(f"Attn with {compressor_cls} compressor at layer {self.layer_idx}. {self.layer_type}")
        self.compressor = (
            compressor_cls(config, weights, device, self.rot, self.rope_dim, cache=cache, weight_dtype=weight_dtype)
            if compressor_cls is not None
            else None
        )

    def prefetch_weights(self):
        # self.o_b_proj.fetch_weights()
        self.kv_proj.fetch_weights()
        self.q_b_proj.fetch_weights()
        self.q_a_proj.fetch_weights()

    def _sdpa_decode(
        self,
        q: ttnn.Tensor,
        kv: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        cur_pos: ttnn.Tensor | None = None,
        paged: PagedLayerView | None = None,
        sliding_window: int | None = None,
    ) -> ttnn.Tensor:
        """Single-token (``S == 1``) attention via the fused SDPA-decode op.

        Drop-in for :meth:`_attention` on the decode paths: fuses the scale, the
        masking, the per-head sink, and both matmuls into one device op.

        ``q`` ``[1, 1, H, Dh]`` (already the op's ``[1, B, H, Dh]`` decode head
        layout, produced by :meth:`_qkv`); ``kv`` is the shared K==V
        ``[1, 1, Skv, Dh]`` (MQA, one KV head). The op emits ``[1, 1, H, Dh]`` too, so
        no head/seq transposes are needed around the call.

        Two mutually exclusive ways to bound the KV axis (the op rejects an
        ``attn_mask`` in causal mode, so this is a real branch):

        * ``cur_pos`` ``[1]`` INT32 -- causal mode. The kernel derives its chunk
          range from the position, so it never reads or computes the chunks past it:
          cost tracks the *actual* position instead of the ``max_seq``-sized axis.
          Requires the valid set to be a contiguous prefix
          (:func:`sdpa_causal_ok`) and is exact even mid-chunk, since the kernel
          generates a partial mask for the final chunk.
        * ``mask`` ``[1, 1, 1, Skv]`` additive (``0`` valid / ``_MASK_NEG`` masked) --
          the fallback for the steps whose valid set has a hole. The mask is *data*,
          not control flow, so the kernel always walks the whole axis. The op wants
          the mask to carry Q's (padded) head count, so the head-independent row is
          broadcast across ``H`` first -- a materialisation the causal path avoids.

        ``paged`` swaps ``kv`` for the layer's block pool read through the active
        session's page table; the bounding modes above are unchanged by it, except
        that a bounded ring (``paged.position_modulo``) additionally passes
        ``sliding_window_size`` so the kernel attends the last ``window`` positions
        rather than the whole (wrapped) capacity.
        """
        # sdpa_decode requires its K/V operands in DRAM.
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        bounds = (
            {"is_causal": True, "cur_pos_tensor": cur_pos}
            if cur_pos is not None
            else {"is_causal": False, "attn_mask": ttnn.repeat(mask, ttnn.Shape([1, 1, self.num_heads, 1]))}
        )
        if paged is not None:
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                paged.pool,
                paged.pool,  # K == V (shared single KV head)
                paged.page_table,
                attention_sink=self.sdpa_sinks_tt,
                scale=self.scaling,
                sliding_window_size=sliding_window,
                cache_position_modulo=paged.position_modulo,
                program_config=self._sdpa_pcfg,
                compute_kernel_config=_HIFI4_SDPA,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                **bounds,
            )
        kv = ttnn.to_memory_config(kv, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            kv,
            kv,  # K == V (shared single KV head)
            attention_sink=self.sdpa_sinks_tt,
            scale=self.scaling,
            program_config=self._sdpa_pcfg,
            compute_kernel_config=_HIFI4_SDPA,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            **bounds,
        )  # [1, 1, H, Dh]

    def _grouped_output(self, attn: ttnn.Tensor) -> ttnn.Tensor:
        """``DeepseekV4GroupedLinear`` (o_a) + ``o_b_proj``.

        ``attn`` is ``[B, S, H, Dh]``. Reshape to per-group feature blocks, run the
        batched ``matmul_decode`` over the group axis (batch = o_groups, weights
        folded along group + N), then mix groups back to hidden via ``o_b_proj``.
        """
        b, s, h, dh = attn.shape
        in_per_group = (h * dh) // self.o_groups
        m = b * s
        # Rank-4 activation [1, g, M, K] (batch = g = o_groups) for the batched matmul_decode; the
        # op folds the group axis to match the folded (b_blocks x n_blocks) weight layout.
        x = ttnn.reshape(attn, [m, self.o_groups, in_per_group])
        x = ttnn.permute(x, [1, 0, 2])  # [g, M, K]
        x = ttnn.reshape(x, [1, self.o_groups, m, in_per_group])  # [1, g, M, K]
        y = self.o_a_proj(x)  # DRAM-interleaved [1, g, M, N]
        y = ttnn.permute(y, [0, 2, 1, 3])  # [1, M, g, N]
        y = ttnn.reshape(y, [b, s, 1, self.o_groups * self.o_lora_rank])
        return ttnn.to_memory_config(self.o_b_proj(y), ttnn.DRAM_MEMORY_CONFIG)

    def _attend(
        self,
        q: ttnn.Tensor,
        kv: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        cos: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        sdpa_cur_pos: ttnn.Tensor | None = None,
        paged: PagedLayerView | None = None,
        sliding_window: int | None = None,
    ) -> ttnn.Tensor:
        """Fused SDPA-decode + output RoPE + grouped output projection.

        Shared tail of :meth:`decode` / :meth:`decode_static`: ``q`` ``[B,1,H,Dh]``,
        the shared K==V ``kv`` ``[B,1,Skv,Dh]`` (or ``paged``'s block pool) and either
        ``sdpa_cur_pos`` or the additive ``mask`` ``[1,1,1,Skv]`` -> the block's hidden
        output ``[B,1,1,D]``. ``kv`` is the layer's persistent buffer, updated in place;
        the only per-path difference is where ``mask`` / ``sdpa_cur_pos`` come from
        (host-built for eager, device-generated for the traced path).
        """
        attn = self._sdpa_decode(
            q, kv, mask, cur_pos=sdpa_cur_pos, paged=paged, sliding_window=sliding_window
        )  # [B, 1, H, Dh]
        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim)
        return self._grouped_output(attn)  # already [B, 1, H, Dh] for the grouped proj

    def _qkv(self, hidden: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Project + RoPE the query and (shared) K=V for ``hidden`` ``[B, S, 1, D]``.

        Returns ``q`` ``[B, 1, H, Dh]`` (the SDPA-decode head layout) and the
        rotated ``kv`` ``[B, 1, S, Dh]`` (pre-compressor, pre-cache). Shared by the
        decode paths.

        The per-head split uses the fused ``nlp_create_qkv_heads_decode`` op (as in
        the gpt-oss decode attention) instead of manual ``reshape``/``transpose``:
        Q and the shared K=V are concatenated into one ``[1, 1, B, (H+2)*Dh]`` row
        (K==V, so the single KV head is duplicated for the op's K and V slices) and
        split into the ``[1, B, H, Dh]`` decode layout in one device op. Producing Q
        directly in this layout also removes the head/seq transposes that previously
        wrapped the SDPA-decode call.
        """
        b, s, _, hidden_width = hidden.shape  # B == 1, S == 1 (decode)
        h, dh = self.num_heads, self.head_dim
        # width_sharded_l1_config = _width_sharded_l1_config(b * s, hidden_width, self.device)
        # hidden = ttnn.to_memory_config(hidden, width_sharded_l1_config)
        _profile(self.device)
        # hidden_input_memory_config = self.q_a_proj.get_input_memory_config(1, hidden.shape[3])
        # hidden = ttnn.to_memory_config(hidden, hidden_input_memory_config)
        q_a = self.q_a_norm(self.q_a_proj(hidden))
        q = self.q_b_proj(q_a)  # [B, S, H*Dh]
        q = ttnn.reshape(q, [1, 1, h, dh], memory_config=width_sharded_l1_config(b * s * h, dh, self.device))

        q = _rms_norm_unweighted(q, self.eps)
        q = _apply_rope(q, cos, sin, self.rot, self.rope_dim)  # [B, 1, H, Dh]

        kv = self.kv_norm(self.kv_proj(hidden))  # [B, S, Dh]

        kv = _apply_rope(kv, cos, sin, self.rot, self.rope_dim)  # [B, 1, S, Dh]
        return q, kv

    def decode(
        self,
        hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        mask: ttnn.Tensor | None,
        scache: "_StaticLayerCache",
        sliding_pos: ttnn.Tensor,
        compress_pos: ttnn.Tensor,
        paged: PagedLayerView | None = None,
        pool_compressor: bool = True,
        sdpa_cur_pos: ttnn.Tensor | None = None,
        win_slot: ttnn.Tensor | None = None,
        win_row: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Single-token decode attention against the in-place ``scache`` (or ``paged``).

        Same as :meth:`decode_static`; the eager model path builds ``mask`` and the
        position tensors on host while the traced path generates them on device.
        """
        return self.decode_static(
            hidden,
            cos,
            sin,
            neg_sin,
            cos_win,
            sin_win,
            mask,
            scache,
            sliding_pos,
            compress_pos,
            paged=paged,
            pool_compressor=pool_compressor,
            sdpa_cur_pos=sdpa_cur_pos,
            win_slot=win_slot,
            win_row=win_row,
        )

    def decode_static(
        self,
        hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        mask: ttnn.Tensor | None,
        scache: "_StaticLayerCache",
        sliding_pos: ttnn.Tensor,
        compress_pos: ttnn.Tensor,
        paged: PagedLayerView | None = None,
        pool_compressor: bool = True,
        sdpa_cur_pos: ttnn.Tensor | None = None,
        win_slot: ttnn.Tensor | None = None,
        win_row: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Trace-safe single-token decode against fixed-size in-place caches.

        ``paged`` replaces the layer's dense KV buffer (``scache.sliding`` /
        ``scache.combined``) with a shared block pool read through the active
        session's page table, which is what lets several sessions share one captured
        trace (see :mod:`.paged_cache`). The compressor's own window buffers stay in
        ``scache`` either way -- they are small enough to be swapped per session
        outside the trace.

        ``pool_compressor`` selects whether this step closes (and so pools) a
        compressor window; it is ignored by sliding layers. On CSA/HCA layers
        ``win_slot`` is this token's slot in the window buffer (``pos % compress_rate``)
        and, when pooling, ``win_row`` is the ``combined`` row the new entry lands in
        (``sliding_window + w``) and ``cos_win`` / ``sin_win`` are window ``w``'s
        single RoPE row.

        ``sdpa_cur_pos``, when set, replaces ``mask`` with causal-mode SDPA bounded
        by that position (see :meth:`_sdpa_decode` and :func:`sdpa_causal_ok`).
        """
        q, kv_new = self._qkv(hidden, cos, sin)  # q [1,1,H,Dh], kv_new [1,1,1,Dh]

        if self.compressor is None:
            # The KV axis is the sliding ring alone. Paged: the *absolute* position,
            # which ``paged.position_modulo`` wraps into the bounded ring, read in causal
            # mode so the kernel honours ``cur_pos`` -- non-causal ignores it and walks
            # the whole (wrapped) capacity, double-counting the tail. Dense: the ring
            # slot, with the additive mask hiding the not-yet-written slots.
            if paged is not None:
                _update_cache_at(None, kv_new, compress_pos, paged=paged)
                return self._attend(
                    q,
                    None,
                    None,
                    cos,
                    neg_sin,
                    sdpa_cur_pos=compress_pos,
                    paged=paged,
                    sliding_window=self.config.sliding_window,
                )
            _update_cache_at(scache.sliding, kv_new, sliding_pos)
            return self._attend(q, scache.sliding, mask, cos, neg_sin)

        # One KV axis holds both regions, so there is no per-step concat: the ring slot
        # ``pos % window`` lands in the prefix and each pooled entry is appended after
        # it at row ``window + w``. Both indices are pre-wrapped, so the paged reads
        # need no ``cache_position_modulo``.
        kv = None if paged is not None else scache.combined  # [1, 1, window + n_win, Dh]
        _update_cache_at(kv, kv_new, sliding_pos, paged=paged)
        self.compressor.decode_static(
            hidden,
            cos_win,
            sin_win,
            scache,
            kv,
            win_slot,
            win_row=win_row,
            pool=pool_compressor,
            paged=paged,
        )
        return self._attend(q, kv, mask, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos, paged=paged)
