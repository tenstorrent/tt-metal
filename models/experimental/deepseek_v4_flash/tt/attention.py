from typing import Optional

import ttnn
import torch

from .common import DeepSeekV4Module, _HIFI4, _HIFI4_SDPA, _MASK_NEG, _profile
from .layers import DeepSeekV4RMSNorm, Linear, LinearDecode, _rms_norm_unweighted
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
#     these each step with the exact prefill pooling, so decode is bit-for-bit
#     the same function of the tokens-so-far as a full prefill over them (no
#     separate rolling-window / overlap / entry-count bookkeeping needed).
#
# Re-pooling the (small, ``head_dim``-wide) compressor each step is cheap next
# to the per-token MoE / projection matmuls, which now run at ``S = 1`` instead
# of over the whole growing context as in the repeated-prefill demo.
# ---------------------------------------------------------------------------- #
class _SlidingKVCache:
    """Append-only rotated K=V cache, capped to the last ``sliding_window`` rows."""

    def __init__(self, sliding_window: int):
        self.window = sliding_window
        self.kv: Optional[ttnn.Tensor] = None  # [B, 1, L, Dh]

    def append(self, kv_new: ttnn.Tensor) -> ttnn.Tensor:
        """Append ``kv_new`` ``[B, 1, n, Dh]`` and return the (capped) cache."""
        self.kv = kv_new if self.kv is None else ttnn.concat([self.kv, kv_new], dim=2)
        b, _, length, dh = self.kv.shape
        if length > self.window:
            self.kv = ttnn.slice(self.kv, [0, 0, length - self.window, 0], [b, 1, length, dh])
        return self.kv


class _CompressorCache:
    """All source tokens' compressor ``kv`` / ``gate`` projections ``[B, T, c*Dh]``."""

    def __init__(self):
        self.kv: Optional[ttnn.Tensor] = None
        self.gate: Optional[ttnn.Tensor] = None

    def append(self, kv_new: ttnn.Tensor, gate_new: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        self.kv = kv_new if self.kv is None else ttnn.concat([self.kv, kv_new], dim=1)
        self.gate = gate_new if self.gate is None else ttnn.concat([self.gate, gate_new], dim=1)
        return self.kv, self.gate


class _LayerKVCache:
    """Per-decoder-layer decode state: a sliding K=V cache + optional compressor."""

    def __init__(self, sliding_window: int, has_compressor: bool):
        self.sliding = _SlidingKVCache(sliding_window)
        self.compressor = _CompressorCache() if has_compressor else None


class _StaticLayerCache:
    """Fixed-size, in-place per-layer decode caches for the *traced* decode path.

    Unlike :class:`_LayerKVCache` (which grows via ``concat`` each step), these are
    DRAM tensors of a fixed capacity written in place at the new token's position
    by ``paged_update_cache`` (a device-tensor index), so the captured trace's
    shapes / addresses are step-invariant:

      * ``sliding`` ``[1, 1, window, Dh]`` -- a ring buffer (slot ``pos % window``);
        attention masks unwritten / out-of-window slots.
      * ``compressor_kv`` / ``compressor_gate`` ``[1, 1, cap, feat]`` -- every
        source token's compressor projection at its absolute position; the pool
        runs over the whole buffer and the block-bias mask drops the windows past
        the current position. ``None`` for sliding-only layers.

    Built empty (all-zero) by :meth:`DeepSeekV4Model.prepare_static_decode`; the
    prompt is written in by replaying :meth:`decode_traced` per prompt token.
    """

    __slots__ = ("sliding", "compressor_kv", "compressor_gate")

    def __init__(
        self, sliding: ttnn.Tensor, compressor_kv: Optional[ttnn.Tensor], compressor_gate: Optional[ttnn.Tensor]
    ):
        self.sliding = sliding
        self.compressor_kv = compressor_kv
        self.compressor_gate = compressor_gate


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
    orig_mem = x.memory_config()
    d = x.shape[-1]
    rows = x.shape[-2]

    # The op reads one cos/sin tile-row per core, or a single tile-row broadcast across all
    # rows on device (e.g. a shared decode position over heads). So cos/sin must cover either
    # every input row or exactly one row.
    assert cos.shape[-2] in (rows, 1), f"{cos.shape} not broadcastable to rows={rows}"

    # cos/sin must already be DRAM-interleaved (the fused op's reader streams them from DRAM).
    assert cos.memory_config().buffer_type == ttnn.BufferType.DRAM, "cos must be DRAM-interleaved"
    assert sin.memory_config().buffer_type == ttnn.BufferType.DRAM, "sin must be DRAM-interleaved"

    num_cores = (rows + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    x_sh = ttnn.to_memory_config(x, _rope_height_sharded_config(d, num_cores, device))

    out_sh = ttnn.experimental.fused_partial_rope(x_sh, cos, sin, _trans_mat_for(rot), rope_dim)
    return ttnn.to_memory_config(out_sh, orig_mem)


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


def _update_cache_at(cache: ttnn.Tensor, row: ttnn.Tensor, pos_tensor: ttnn.Tensor) -> None:
    """In-place write ``row`` ``[1, 1, 1, F]`` into ``cache`` ``[1, 1, L, F]`` at the
    sequence index held (on device) by ``pos_tensor`` ``[1]`` (INT32). Trace-safe."""
    width = row.shape[-1]
    row_sharded = ttnn.interleaved_to_sharded(row, _height_sharded_l1_config(width))
    ttnn.experimental.paged_update_cache(cache, row_sharded, update_idxs_tensor=pos_tensor)
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
    into a single softmax-gated KV entry, then RoPEs each entry at its window's
    absolute position. Returns ``compressed_kv`` shaped ``[B, 1, n_windows, Dh]``
    ready to concat onto the sliding KV axis.
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
            weights["compressor.kv_norm.weight"], self.eps, device, cache.file("compressor.kv_norm")
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

    def _pool(
        self, kv: ttnn.Tensor, gate: ttnn.Tensor, cos_win: ttnn.Tensor, sin_win: ttnn.Tensor
    ) -> ttnn.Tensor | None:
        """Pool the projected ``(kv, gate)`` ``[B, T, Dh]`` into compressed entries.

        Compresses every complete window of ``compress_rate`` tokens into one
        softmax-gated entry and RoPEs it at its window position. Shared by the
        :meth:`decode` and :meth:`decode_static` paths.
        """
        b = kv.shape[0]
        t = kv.shape[1]
        cr = self.compress_rate
        n_win = t // cr
        if n_win == 0:
            return None
        usable = n_win * cr
        if usable != t:
            kv = ttnn.slice(kv, [0, 0, 0], [b, usable, self.head_dim])
            gate = ttnn.slice(gate, [0, 0, 0], [b, usable, self.head_dim])
        kv = ttnn.reshape(kv, [b, n_win, cr, self.head_dim])
        gate = ttnn.reshape(gate, [b, n_win, cr, self.head_dim])
        gate = ttnn.add(gate, self.position_bias)
        compressed = _softmax_weighted_sum(kv, gate, window_axis=2)  # [B, n_win, Dh]
        compressed = self.kv_norm(compressed)
        compressed = ttnn.reshape(compressed, [b, 1, n_win, self.head_dim])
        return _apply_rope(compressed, cos_win, sin_win, self.rot, self.rope_dim)

    def decode(
        self, hidden: ttnn.Tensor, cos_win: ttnn.Tensor, sin_win: ttnn.Tensor, cache: "_CompressorCache"
    ) -> ttnn.Tensor | None:
        """Project the new token(s), append to ``cache``, re-pool all entries.

        ``cos_win`` / ``sin_win`` must cover every window emittable from the full
        cached projection length (``n_win`` rows); the caller slices them.
        """
        kv, gate = self._project(hidden)
        if len(kv.shape) == 4:
            b, s, _, feat = kv.shape
            kv = ttnn.reshape(kv, [b, s, feat])
            gate = ttnn.reshape(gate, [b, s, feat])
        kv_all, gate_all = cache.append(kv, gate)
        return self._pool(kv_all, gate_all, cos_win, sin_win)

    def decode_static(
        self,
        hidden: ttnn.Tensor,
        cos_win: ttnn.Tensor,
        sin_win: ttnn.Tensor,
        kv_cache: ttnn.Tensor,
        gate_cache: ttnn.Tensor,
        pos_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Trace-safe decode: write this token's projection in place at ``pos_tensor``
        into the fixed ``[1, 1, cap, Dh]`` caches, then pool over the *whole* buffer.

        ``cos_win`` / ``sin_win`` cover every window of the fixed capacity
        (``cap // compress_rate`` rows); windows past the current position are
        pooled from zero-filled (unwritten) projections and dropped by the caller's
        additive block-bias mask.
        """
        kv, gate = self._project(hidden)  # [1, 1, 1, Dh]
        kv = ttnn.reshape(kv, [1, 1, 1, self.head_dim])
        gate = ttnn.reshape(gate, [1, 1, 1, self.head_dim])
        _update_cache_at(kv_cache, kv, pos_tensor)
        _update_cache_at(gate_cache, gate, pos_tensor)
        cap = kv_cache.shape[2]
        kv_all = ttnn.reshape(kv_cache, [1, cap, self.head_dim])
        gate_all = ttnn.reshape(gate_cache, [1, cap, self.head_dim])
        return self._pool(kv_all, gate_all, cos_win, sin_win)


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
            weights["compressor.kv_norm.weight"], self.eps, device, cache.file("compressor.kv_norm")
        )
        pb = _materialize(weights["compressor.position_bias"], cache.file("compressor.position_bias"), ttnn.bfloat16)
        self.position_bias = _load_weight(
            pb.reshape(1, 1, self.compress_rate, 2 * self.head_dim) if pb is not None else None,
            device,
            cache_file_name=cache.file("compressor.position_bias"),
        )
        # Persistent window-0 Ca filler (zero kv / ``-inf`` gate, softmax weight 0).
        # ``_pool`` re-uploaded these from host on every call -- a host transfer
        # that is illegal inside a trace -- so keep them resident (B == 1).
        self._zeros_filler = ttnn.from_torch(
            torch.zeros(1, 1, self.compress_rate, self.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._neg_filler = ttnn.from_torch(
            torch.full((1, 1, self.compress_rate, self.head_dim), _MASK_NEG),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def _project(self, hidden: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``hidden`` ``[B, S, D]`` -> per-token ``(kv, gate)`` ``[B, S, 2*Dh]`` each."""
        return self.kv_proj(hidden), self.gate_proj(hidden)

    def _pool(
        self, kv: ttnn.Tensor, gate: ttnn.Tensor, cos_win: ttnn.Tensor, sin_win: ttnn.Tensor
    ) -> ttnn.Tensor | None:
        """Pool the projected ``(kv, gate)`` ``[B, T, 2*Dh]`` into compressed entries
        (two-series Ca/Cb overlap). Shared by the decode paths."""
        b = kv.shape[0]
        t = kv.shape[1]
        cr = self.compress_rate
        dh = self.head_dim
        n_win = t // cr
        if n_win == 0:
            return None
        usable = n_win * cr
        if usable != t:
            kv = ttnn.slice(kv, [0, 0, 0], [b, usable, 2 * dh])
            gate = ttnn.slice(gate, [0, 0, 0], [b, usable, 2 * dh])
        kv = ttnn.reshape(kv, [b, n_win, cr, 2 * dh])
        gate = ttnn.reshape(gate, [b, n_win, cr, 2 * dh])
        gate = ttnn.add(gate, self.position_bias)

        ca, cb = ttnn.split(kv, dh, dim=3)
        ca_g, cb_g = ttnn.split(gate, dh, dim=3)
        _profile(self.device)

        # Shift Ca down one window: entry w sees window w-1's Ca; entry 0 sees a
        # zero-kv / -inf-gate filler (softmax weight 0). Fillers are resident
        # (B == 1) so the pool stays trace-safe (no per-call host transfer).
        zeros = self._zeros_filler
        neg = self._neg_filler
        ca_prev_src = ttnn.slice(ca, [0, 0, 0, 0], [b, n_win - 1, cr, dh])
        cag_prev_src = ttnn.slice(ca_g, [0, 0, 0, 0], [b, n_win - 1, cr, dh])
        ca_prev = ttnn.concat([zeros, ca_prev_src], dim=1)
        cag_prev = ttnn.concat([neg, cag_prev_src], dim=1)

        new_kv = ttnn.concat([ca_prev, cb], dim=2)  # [B, n_win, 2*cr, Dh]
        new_gate = ttnn.concat([cag_prev, cb_g], dim=2)
        compressed = _softmax_weighted_sum(new_kv, new_gate, window_axis=2)  # [B, n_win, Dh]
        compressed = self.kv_norm(compressed)
        compressed = ttnn.reshape(compressed, [b, 1, n_win, dh])
        return _apply_rope(compressed, cos_win, sin_win, self.rot, self.rope_dim)

    def decode(
        self, hidden: ttnn.Tensor, cos_win: ttnn.Tensor, sin_win: ttnn.Tensor, cache: "_CompressorCache"
    ) -> ttnn.Tensor | None:
        kv, gate = self._project(hidden)
        if len(kv.shape) == 4:
            b, s, _, feat = kv.shape
            kv = ttnn.reshape(kv, [b, s, feat])
            gate = ttnn.reshape(gate, [b, s, feat])
        kv_all, gate_all = cache.append(kv, gate)
        return self._pool(kv_all, gate_all, cos_win, sin_win)

    def decode_static(
        self,
        hidden: ttnn.Tensor,
        cos_win: ttnn.Tensor,
        sin_win: ttnn.Tensor,
        kv_cache: ttnn.Tensor,
        gate_cache: ttnn.Tensor,
        pos_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Trace-safe decode: write this token's ``2*Dh`` projection in place at
        ``pos_tensor`` into the fixed ``[1, 1, cap, 2*Dh]`` caches, then pool the
        whole buffer (Ca/Cb overlap). See :meth:`DeepSeekV4HCACompressor.decode_static`."""
        feat = 2 * self.head_dim
        kv, gate = self._project(hidden)  # [1, 1, 1, 2*Dh]
        kv = ttnn.reshape(kv, [1, 1, 1, feat])
        gate = ttnn.reshape(gate, [1, 1, 1, feat])
        _update_cache_at(kv_cache, kv, pos_tensor)
        _update_cache_at(gate_cache, gate, pos_tensor)
        cap = kv_cache.shape[2]
        kv_all = ttnn.reshape(kv_cache, [1, cap, feat])
        gate_all = ttnn.reshape(gate_cache, [1, cap, feat])
        return self._pool(kv_all, gate_all, cos_win, sin_win)


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
        self.q_a_norm = DeepSeekV4RMSNorm(weights["q_a_norm.weight"], self.eps, device, cache.file("q_a_norm"))
        self.kv_norm = DeepSeekV4RMSNorm(weights["kv_norm.weight"], self.eps, device, cache.file("kv_norm"))

        # Grouped output projection (``DeepseekV4GroupedLinear``): block-diagonal
        # over o_groups. Store the per-group weight as [g, in_per_group, out_per_group]
        # so a single batched matmul (batch axis = group) does all groups at once.
        # oa: [g*o_lora_rank, (H*Dh)//g] -> per-group [g, in_per_group, o_lora_rank].
        oa = _materialize(weights["o_a_proj.weight"], cache.file("o_a_proj"), weight_dtype)
        in_per_group = (self.num_heads * self.head_dim) // self.o_groups
        if oa is not None:
            oa = oa.reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).contiguous()
        self.o_a_weight = _load_weight(oa, device, cache_file_name=cache.file("o_a_proj"), dtype=weight_dtype)

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
        self.compressor = (
            compressor_cls(config, weights, device, self.rot, self.rope_dim, cache=cache, weight_dtype=weight_dtype)
            if compressor_cls is not None
            else None
        )

    def _sdpa_decode(self, q: ttnn.Tensor, kv: ttnn.Tensor, mask: ttnn.Tensor) -> ttnn.Tensor:
        """Single-token (``S == 1``) attention via the fused SDPA-decode op.

        Drop-in for :meth:`_attention` on the decode paths: fuses the scale, the
        additive ``mask``, the per-head sink, and both matmuls into one device op.

        ``q`` ``[1, 1, H, Dh]`` (already the op's ``[1, B, H, Dh]`` decode head
        layout, produced by :meth:`_qkv`); ``kv`` is the shared K==V
        ``[1, 1, Skv, Dh]`` (MQA, one KV head); ``mask`` ``[1, 1, 1, Skv]`` additive
        (``0`` valid / ``_MASK_NEG`` masked). The op emits ``[1, 1, H, Dh]`` too, so
        no head/seq transposes are needed around the call.

        The op requires the mask to carry the same (padded) head count as Q, so the
        head-independent ``mask`` is broadcast across the ``H`` head axis first.
        """
        mask_h = ttnn.repeat(mask, ttnn.Shape([1, 1, self.num_heads, 1]))  # [1, 1, H, Skv]
        # sdpa_decode requires its K/V operands in DRAM.
        return ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            kv,
            kv,  # K == V (shared single KV head)
            is_causal=False,
            attn_mask=mask_h,
            attention_sink=self.sdpa_sinks_tt,
            scale=self.scaling,
            program_config=self._sdpa_pcfg,
            compute_kernel_config=_HIFI4_SDPA,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # [1, 1, H, Dh]

    def _grouped_output(self, attn: ttnn.Tensor) -> ttnn.Tensor:
        """``DeepseekV4GroupedLinear`` (o_a) + ``o_b_proj``.

        ``attn`` is ``[B, S, H, Dh]``. Reshape to per-group feature blocks, run a
        batched matmul over the group axis, then mix groups back to hidden.
        """
        b, s, h, dh = attn.shape
        in_per_group = (h * dh) // self.o_groups
        x = ttnn.reshape(attn, [b * s, self.o_groups, in_per_group])
        x = ttnn.permute(x, [1, 0, 2])  # [g, B*S, in_per_group]
        y = ttnn.matmul(x, self.o_a_weight, compute_kernel_config=_HIFI4)  # [g, B*S, o_lora_rank]
        y = ttnn.permute(y, [1, 0, 2])  # [B*S, g, o_lora_rank]
        y = ttnn.reshape(y, [b, s, 1, self.o_groups * self.o_lora_rank])
        return self.o_b_proj(y)

    def _attend(
        self, q: ttnn.Tensor, kv: ttnn.Tensor, mask: ttnn.Tensor, cos: ttnn.Tensor, neg_sin: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Fused SDPA-decode + output RoPE + grouped output projection.

        Shared tail of :meth:`decode` / :meth:`decode_static`: ``q`` ``[B,1,H,Dh]``,
        the shared K==V ``kv`` ``[B,1,Skv,Dh]`` and the additive ``mask``
        ``[1,1,1,Skv]`` -> the block's hidden output ``[B,1,1,D]``. The only
        per-path difference is how ``kv`` / ``mask`` are assembled (concat-grown
        cache + implicit-zero mask for eager vs. fixed in-place cache + device mask
        for the traced path); the attention compute itself is identical.
        """
        attn = self._sdpa_decode(q, kv, mask)  # [B, 1, H, Dh]
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
        q = ttnn.reshape(q, [1, 1, h, dh])

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
        kv_cache: "_LayerKVCache",
    ) -> ttnn.Tensor:
        """Single-token decode attention against the running ``kv_cache``.

        ``hidden`` is ``[B, 1, 1, D]`` (the new token); ``cos`` / ``sin`` / ``neg_sin``
        are the single RoPE rows ``[1,1,1,Rd]`` at this token's absolute position;
        ``cos_win`` / ``sin_win`` cover every currently-emittable compressor window.

        The mask is implicitly zero: a single causal query at the front of its
        (window-capped) cache sees every cached sliding key and every existing
        compressed entry, so no additive masking is needed (valid while the
        sequence stays within ``index_topk * compress_rate``, matching the
        prefill port's degenerate-indexer assumption).
        """
        b, s, _, _ = hidden.shape  # s == 1

        q, kv_new = self._qkv(hidden, cos, sin)  # q [B,1,H,Dh], kv_new [B,1,1,Dh]
        kv = kv_cache.sliding.append(kv_new)  # [B, 1, L_sld, Dh]

        if self.compressor is not None:
            compressed = self.compressor.decode(hidden, cos_win, sin_win, kv_cache.compressor)
            if compressed is not None:
                kv = ttnn.concat([kv, compressed], dim=2)  # [B, 1, L_sld + n_win, Dh]
        _profile(self.device)

        mask = ttnn.zeros([1, 1, s, kv.shape[2]], ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device)
        return self._attend(q, kv, mask, cos, neg_sin)

    def decode_static(
        self,
        hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        mask: ttnn.Tensor,
        scache: "_StaticLayerCache",
        sliding_pos: ttnn.Tensor,
        compress_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Trace-safe single-token decode against fixed-size in-place caches.

        Writes the new token's rotated K=V into the ``sliding_window``-sized ring
        buffer (slot = ``pos % window`` carried by ``sliding_pos``) and, for
        CSA/HCA layers, its compressor projections at absolute ``compress_pos``;
        attends the *whole* fixed cache (sliding slots ++ all compressor windows)
        under the supplied additive ``mask`` (zeros for valid slots / windows,
        ``_MASK_NEG`` for unwritten slots and not-yet-emittable windows). Equivalent
        to :meth:`decode` but with static shapes / addresses for a reusable trace.
        """
        q, kv_new = self._qkv(hidden, cos, sin)  # q [1,1,H,Dh], kv_new [1,1,1,Dh]
        _update_cache_at(scache.sliding, kv_new, sliding_pos)
        kv = scache.sliding  # [1, 1, window, Dh] (updated in place)

        if self.compressor is not None:
            compressed = self.compressor.decode_static(
                hidden, cos_win, sin_win, scache.compressor_kv, scache.compressor_gate, compress_pos
            )
            kv = ttnn.concat([kv, compressed], dim=2)  # [1, 1, window + n_win, Dh]

        return self._attend(q, kv, mask, cos, neg_sin)
