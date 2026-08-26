from typing import Optional

import ttnn
import torch

from .common import DeepSeekV4Module, _HIFI4_SDPA, _MASK_NEG, _profile, width_sharded_l1_config, _signpost
from .decode_prefetch import (
    DECODE_LAYOUTS,
    check_decode_layout,
    decode_prefetch_page_bytes,
    make_decode_prefetch_buffers,
)
from .layers import (
    BatchedLinearDecode,
    DeepSeekV4RMSNorm,
    LinearDecode,
    _rms_norm_unweighted,
)
from .l1_weights import packed_weight_spec
from .paged_cache import PagedLayerView
from .system_config import active_system_config
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
#
# A step decodes ``B`` users at once (``S == 1``), all of them at the same absolute
# position: the RoPE rows, the additive mask and the cache indices are shared, which
# is what a batch stepped in lockstep from position 0 looks like. Users at *differing*
# positions would need per-user RoPE rows and masks and are not supported here.
#
# Two activation layouts appear throughout, and which one a tensor is in matters
# because tiles are 32 rows tall:
#
#   * *packed rows* ``[1, 1, B, F]`` -- the B users on consecutive rows of a single
#     tile-row. Everything that is per-token arithmetic (projections, norms, RoPE)
#     runs here, so one decode step costs one tile-row of work rather than B of them.
#     This is what caps a step at ``TILE_SIZE`` users.
#   * *per-user tile-rows* ``[1, B, 1, F]`` / ``[B, 1, ..., F]`` -- one tile-row per
#     user. The KV-cache ops require it (``paged_update_cache`` dispatches one user per
#     core, SDPA-decode indexes K/V by a leading batch), and the surrounding block hands
#     ``hidden`` in as ``[B, S, 1, D]``.
#
# :func:`_pack_tokens` / :func:`_one_row_per_user` convert between them; both are view
# reshapes at ``B == 1`` and relayouts above it.
#
# Two separate things bound ``B``, and the smaller one is not the one in the assert:
#
#   * the *layout* bound, ``TILE_SIZE`` (32), enforced by :func:`_pack_tokens` -- a step's
#     tokens have to fit one tile-row.
#   * an *L1* bound, well under that. Almost nothing here grows with ``B`` (the projections
#     and norms all run on the one tile-row, which is the point of the packed layout), but
#     the query does: SDPA-decode wants a head axis, so ``q`` is width-sharded over ``B*H``
#     rows, i.e. ``B * H * Dh * 2`` bytes -- 64 KB per core at ``B == 16``, ``head_dim 512``.
#     Past a handful of users that crowds out SDPA-decode's own statically-allocated
#     circular buffers and the op fails to build. Measured on a Blackhole grid at
#     ``head_dim 512``: 8 users fit, 16 do not. It surfaces as "statically allocated circular
#     buffers ... clash with L1 buffers", not as a clean batch-size error, so treat 8 as the
#     supported ceiling until the query's residency is reworked.
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
    by ``paged_update_cache`` (a device-tensor index). Every one of them carries the
    batch on dim 0 and the (single, shared) KV head on dim 1, which is the
    ``[B, heads, rows, feat]`` layout both ``paged_update_cache`` and SDPA-decode read:

      * ``sliding`` ``[B, 1, window, Dh]`` -- a ring buffer (slot ``pos % window``);
        attention masks unwritten / out-of-window slots. Sliding-only layers only;
        for CSA/HCA the ring lives in ``combined`` (below).
      * ``win_kv`` / ``win_gate`` ``[B, 1, compress_rate, feat]`` -- the compressor
        projections of the window *currently being filled*, at slot
        ``pos % compress_rate``. Only the ``compress_rate`` tokens of one window are
        held, because pooling is incremental: the step that closes a window pools
        just that window and appends its single entry (see :meth:`_pool_window`).
        ``None`` for sliding-only layers.
      * ``prev_kv`` / ``prev_gate`` ``[B, 1, compress_rate, 2*Dh]`` -- the previous
        window's projections, kept only by CSA because its entry ``w`` also needs
        window ``w-1``'s Ca slice. Refreshed from ``win_*`` after each pool.
        ``prev_gate`` starts at ``_MASK_NEG`` so window 0's absent Ca half carries
        softmax weight 0. ``None`` for HCA and sliding-only layers.
      * ``combined`` ``[B, 1, window + cap // compress_rate, Dh]`` -- the single
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
    batch: int = 1,
) -> _StaticLayerCache:
    """Allocate a layer's fixed-size in-place caches empty (all-zero), for ``batch`` users.

    ``paged`` leaves the KV buffers (``sliding`` / ``combined``) unallocated: those
    reads and writes go through the shared block pool instead (see
    :mod:`.paged_cache`), and only the small compressor window buffers -- which are
    per-session state swapped outside the trace -- are still owned per layer.
    """

    def _filled(rows: int, width: int, value: float = 0.0) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.full((batch, 1, rows, width), value),
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


def int32_pos_tensor(pos: int, device: ttnn.MeshDevice, batch: int = 1) -> ttnn.Tensor:
    """INT32 position vector ``[batch]`` on ``device`` (for ``paged_update_cache`` / SDPA).

    Both ops index per user, so the vector carries one entry per user; the batch decodes
    in lockstep, so every entry is the same ``pos``.
    """
    return ttnn.from_torch(
        torch.full((batch,), pos, dtype=torch.int32),
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
    """Host-built additive decode mask ``[1, 1, 1, Skv]`` for one absolute position ``pos``.

    Mirrors the on-device mask in :meth:`DeepSeekV4Model._device_mask`: sliding
    columns mask slots with index ``> pos``; compressor columns mask windows with
    index ``>= (pos+1)//cr``.

    Batch-independent: the users of a step share an absolute position, and SDPA-decode
    broadcasts a mask whose leading dim is 1 over the batch, so one row serves them all.
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
_TRANS_MAT_CACHE: dict[int, tuple[ttnn.Tensor, ttnn.Tensor]] = {}


def _trans_mat_for(rot: ttnn.Tensor) -> ttnn.Tensor:
    """The cached per-tile ``rotate_half`` for ``rot``, derived on first use.

    The entry keeps ``rot`` itself alive alongside the tile, because the key is its
    ``id()``: let the last reference go and CPython is free to hand that address to the
    next ``rot`` allocated, which would turn this into a silent hit returning a tile
    belonging to a different (possibly already closed) device. Holding it makes the
    address unique for as long as the entry lives, and costs one small tensor per distinct
    rope matrix -- one per layer.
    """
    cached = _TRANS_MAT_CACHE.get(id(rot))
    if cached is not None and cached[0] is rot:
        return cached[1]
    tm = ttnn.reshape(ttnn.slice(rot, [0, 0], [ttnn.TILE_SIZE, ttnn.TILE_SIZE]), [1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE])
    # The fused op reads trans_mat from a DRAM-interleaved source.
    tm = ttnn.to_memory_config(tm, ttnn.DRAM_MEMORY_CONFIG)
    _TRANS_MAT_CACHE[id(rot)] = (rot, tm)
    return tm


def _pack_tokens(hidden: ttnn.Tensor) -> ttnn.Tensor:
    """``[B, S, 1, D]`` -> ``[1, 1, B*S, D]``: the block's tokens packed onto rows.

    The surrounding block keeps one *tile-row* per token, of which a decode step fills a
    single row. Packing them onto consecutive rows of one tile-row is what makes a
    B-user step cost the same projections / norms / RoPE as a one-user step, and is
    why the batch is capped at ``TILE_SIZE`` users.
    """
    b, s, _, d = hidden.shape
    tokens = b * s
    assert (
        tokens <= ttnn.TILE_SIZE
    ), f"a decode step packs its {tokens} tokens onto one tile-row, so B*S must be at most {ttnn.TILE_SIZE}"
    return ttnn.reshape(hidden, [1, 1, tokens, d])


def _one_row_per_user(x: ttnn.Tensor) -> ttnn.Tensor:
    """``[1, 1, B, F]`` -> ``[1, B, 1, F]``, the layout the KV-cache writer indexes by user.

    ``paged_update_cache`` dispatches one user per core and reads its input with the batch
    on dim 1, so the packed rows the projections produce have to be spread back over one
    tile-row each. A view at ``B == 1``, a relayout above it.
    """
    return ttnn.reshape(x, [1, x.shape[-2], 1, x.shape[-1]])


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

    ``rows`` is every leading dim multiplied out, not just ``x.shape[-2]``: the batched
    inputs here are ``[1, B, H, Dh]`` (SDPA-decode's head layout), whose ``B*H`` rows are
    contiguous because ``H`` is tile-aligned.
    """
    device = x.device()
    d = x.shape[-1]
    shape = list(x.shape)
    rows = 1
    for dim in shape[:-1]:
        rows *= dim

    # The op reads one cos/sin tile-row per core, or a single tile-row broadcast across all
    # rows on device (e.g. a shared decode position over heads). So cos/sin must cover either
    # every input row or exactly one row.
    assert cos.shape[-2] in (rows, 1), f"{cos.shape} not broadcastable to rows={rows}"

    # cos/sin must already be DRAM-interleaved (the fused op's reader streams them from DRAM).
    assert cos.memory_config().buffer_type == ttnn.BufferType.DRAM, "cos must be DRAM-interleaved"
    assert sin.memory_config().buffer_type == ttnn.BufferType.DRAM, "sin must be DRAM-interleaved"

    if not x.is_sharded():
        x = ttnn.to_memory_config(x, width_sharded_l1_config(rows, d, device))

    # The op takes its row count off dim -2 alone, so a batched ``[1, B, H, Dh]`` has to be
    # folded onto that dim first. The shard already spans all ``rows``, so both reshapes are
    # metadata-only and the caller gets its own shape back.
    mem_config = x.memory_config()
    folded = shape[-2] != rows
    if folded:
        x = ttnn.reshape(x, [1, 1, rows, d], memory_config=mem_config)
    out_sh = ttnn.experimental.fused_partial_rope(x, cos, sin, _trans_mat_for(rot), rope_dim)
    if folded:
        out_sh = ttnn.reshape(out_sh, shape, memory_config=mem_config)
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
def _height_sharded_l1_config(num_users: int, width: int, device) -> ttnn.MemoryConfig:
    """Height-sharded L1 config for a ``[1, B, 1, width]`` decode row: one core per user.

    ``paged_update_cache`` requires its (single-token) input to be height-sharded with the
    core count equal to the number of batch users -- it dispatches one user per core --
    shard width == the last dim, ROW_MAJOR orientation.
    """
    grid = ttnn.num_cores_to_corerangeset(num_users, device.compute_with_storage_grid_size(), row_wise=True)
    shard_spec = ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _update_cache_at(
    cache: ttnn.Tensor,
    row: ttnn.Tensor,
    pos_tensor: ttnn.Tensor,
    paged: PagedLayerView | None = None,
) -> None:
    """In-place write ``row`` ``[1, B, 1, F]`` into a KV cache at ``pos_tensor`` ``[B]``
    (INT32), either the layer's own dense buffer or -- when ``paged`` is given --
    ``paged.pool`` through the active session's page table.

    ``paged.position_modulo`` wraps the logical position into a bounded capacity
    before the page-table lookup, which is what makes a sliding-window session need
    only ``window / block_size`` blocks; without it any position past that capacity
    resolves through the row's unmapped tail (see :mod:`.paged_cache`).
    """
    num_users, width = row.shape[1], row.shape[-1]
    row_sharded = ttnn.to_memory_config(row, _height_sharded_l1_config(num_users, width, row.device()))
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


def _retire_window(prev: ttnn.Tensor, current: ttnn.Tensor) -> None:
    """Copy the just-closed window buffer ``current`` into ``prev``, in place.

    ``fill_cache`` is an in-place whole-tensor cache writer, so (unlike ``ttnn.copy``) it is
    accepted mid trace capture, and both buffers are distinct tensors. It writes a single
    batch index per call, though, so a batched layer retires its users one slice at a time.
    """
    users, heads, rows, width = current.shape
    if users == 1:
        ttnn.fill_cache(prev, current, 0)
        return
    for user in range(users):
        one = ttnn.slice(current, [user, 0, 0, 0], [user + 1, heads, rows, width])
        ttnn.fill_cache(prev, one, user)
        ttnn.deallocate(one)


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
        use_prefetcher: bool = False,
        num_prefetch_pages: Optional[int] = None,
        prefetch_buffers: Optional[dict] = None,
        packed_weights=None,
    ):
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
            packed_weights,
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

    def prefetch_weights(self):
        """Stage the two projection weights ahead of the :meth:`decode_static` that uses them.

        Queued kv before gate, the order :meth:`_project` pops them off their shared GCB.
        """
        self.kv_proj.fetch_weights()
        self.gate_proj.fetch_weights()

    def _project(self, tokens: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``tokens`` ``[1, 1, B, D]`` -> per-token ``(kv, gate)`` ``[1, 1, B, Dh]`` each.

        ``LinearDecode`` leaves its result width-sharded over the cores it reduced onto, while
        the callers reshape these and reshard them height-wise for the cache write, so hand
        back the DRAM-interleaved form they expect (as ``_o_proj`` does for o_b_proj).
        """
        return (
            ttnn.to_memory_config(self.kv_proj(tokens), ttnn.DRAM_MEMORY_CONFIG),
            ttnn.to_memory_config(self.gate_proj(tokens), ttnn.DRAM_MEMORY_CONFIG),
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
        combined_cache: ttnn.Tensor | None,
        win_slot: ttnn.Tensor,
        win_row: ttnn.Tensor | None = None,
        pool: bool = True,
        paged: PagedLayerView | None = None,
    ) -> None:
        """Trace-safe decode: write each user's token projection in place at ``win_slot``
        (``pos % compress_rate``) into the one-window ``[B, 1, compress_rate, Dh]``
        buffers, and -- on the step that closes the window -- pool just that window
        and append its single entry at row ``win_row`` of the layer's KV axis
        (``combined_cache``, or ``paged``'s block pool).

        ``tokens`` is the block's packed-row hidden ``[1, 1, B, D]``.

        ``pool`` is set by the caller only on the steps that close a window, so the
        cost per step is ``O(compress_rate)`` rather than ``O(max_seq)``: in between,
        the KV axis already holds exactly the entries the block-bias exposes
        (see the module header).
        """
        _signpost("HCA_START")
        users = tokens.shape[-2]
        kv, gate = self._project(tokens)  # [1, 1, B, Dh]
        kv = _one_row_per_user(ttnn.reshape(kv, [1, 1, users, self.head_dim]))
        gate = _one_row_per_user(ttnn.reshape(gate, [1, 1, users, self.head_dim]))
        _update_cache_at(scache.win_kv, kv, win_slot)
        _update_cache_at(scache.win_gate, gate, win_slot)
        if pool and (combined_cache is not None or paged is not None):
            pooled = self._pool_window(scache.win_kv, scache.win_gate, cos_row, sin_row)
            _update_cache_at(combined_cache, pooled, win_row, paged=paged)
            ttnn.deallocate(pooled)
        _signpost("HCA_END")


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
        packed_weights=None,
    ):
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
            packed_weights,
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

    def prefetch_weights(self):
        """Stage the two projection weights ahead of the :meth:`decode_static` that uses them.

        Queued kv before gate, the order :meth:`_project` pops them off their shared GCB.
        """
        self.kv_proj.fetch_weights()
        self.gate_proj.fetch_weights()

    def _project(self, tokens: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``tokens`` ``[1, 1, B, D]`` -> per-token ``(kv, gate)`` ``[1, 1, B, 2*Dh]`` each.

        DRAM-interleaved for the same reason as
        :meth:`DeepSeekV4HCACompressor._project`.
        """
        return (
            ttnn.to_memory_config(self.kv_proj(tokens), ttnn.DRAM_MEMORY_CONFIG),
            ttnn.to_memory_config(self.gate_proj(tokens), ttnn.DRAM_MEMORY_CONFIG),
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

        ``win_*`` hold window ``w``'s ``[B, 1, compress_rate, 2*Dh]`` projections and
        ``prev_*`` window ``w-1``'s; the entry is the softmax-gated combination of
        window ``w-1``'s Ca half with window ``w``'s Cb half over a width-``2*cr``
        window. On the very first window ``prev_gate`` is still ``_MASK_NEG``, which
        gives the absent Ca half softmax weight 0.

        Both buffers are shaped like the ``[B, n_win, compress_rate, 2*Dh]`` the pool
        wants with ``n_win == 1``, so ``position_bias`` -- indexed by a token's offset
        within its own window -- broadcasts over users and windows for each half.
        """
        dh = self.head_dim
        users = win_kv.shape[0]
        prev_g = ttnn.add(prev_gate, self.position_bias)
        cur_g = ttnn.add(win_gate, self.position_bias)
        _profile(self.device)

        ca_prev, _ = ttnn.split(prev_kv, dh, dim=3)
        cag_prev, _ = ttnn.split(prev_g, dh, dim=3)
        _, cb_cur = ttnn.split(win_kv, dh, dim=3)
        _, cbg_cur = ttnn.split(cur_g, dh, dim=3)

        new_kv = ttnn.concat([ca_prev, cb_cur], dim=2)  # [B, 1, 2*cr, Dh]
        new_gate = ttnn.concat([cag_prev, cbg_cur], dim=2)
        compressed = _softmax_weighted_sum(new_kv, new_gate, window_axis=2)
        # Back onto packed rows for the norm + RoPE, which are per-token arithmetic.
        compressed = ttnn.reshape(compressed, [1, 1, users, dh])
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
        ``win_slot`` into the one-window ``[B, 1, compress_rate, 2*Dh]`` buffers, and
        -- on the step that closes the window -- pool just that window (Ca/Cb overlap
        against the retained previous window) and append its single entry at row
        ``win_row`` of the layer's KV axis (``combined_cache``, or ``paged``'s pool).

        ``tokens`` is the block's packed-row hidden ``[1, 1, B, D]``.

        After pooling, the closing window becomes the ``prev_*`` the *next* window
        will overlap with. See :meth:`DeepSeekV4HCACompressor.decode_static`.
        """
        _signpost("CSA_START")
        feat = 2 * self.head_dim
        users = tokens.shape[-2]
        kv, gate = self._project(tokens)  # [1, 1, B, 2*Dh]
        kv = _one_row_per_user(ttnn.reshape(kv, [1, 1, users, feat]))
        gate = _one_row_per_user(ttnn.reshape(gate, [1, 1, users, feat]))
        _update_cache_at(scache.win_kv, kv, win_slot)
        _update_cache_at(scache.win_gate, gate, win_slot)
        if pool and (combined_cache is not None or paged is not None):
            pooled = self._pool_window(
                scache.prev_kv, scache.prev_gate, scache.win_kv, scache.win_gate, cos_row, sin_row
            )
            _update_cache_at(combined_cache, pooled, win_row, paged=paged)
            ttnn.deallocate(pooled)
            _retire_window(scache.prev_kv, scache.win_kv)
            _retire_window(scache.prev_gate, scache.win_gate)
        _signpost("CSA_END")


_COMPRESSORS = {
    "compressed_sparse_attention": DeepSeekV4CSACompressor,
    "heavily_compressed_attention": DeepSeekV4HCACompressor,
}


def _compressor_projections(
    layer_type: str,
    config,
    weights: dict,
    device: ttnn.MeshDevice,
    cache: WeightCache,
    weight_dtype: ttnn.DataType,
    use_prefetcher: bool,
    num_prefetch_pages: int,
    prefetch_buffers: Optional[dict],
    packed_weights=None,
):
    """The compressor's ``(kv_proj, gate_proj)``, both projecting the block's ``hidden``.

    Shared by the two compressor kinds, which differ only in the projected width -- HCA
    projects a token to ``Dh``, CSA to ``2*Dh`` for its Ca/Cb pair -- which is why the layout
    is keyed by ``layer_type``. Under the prefetcher the pair streams through the device's one
    GCB, so they must be queued in the order :meth:`DeepSeekV4HCACompressor._project` runs
    them, kv before gate, in the block's turn (see ``decode_prefetch``).
    """
    feat = config.head_dim * (2 if layer_type == "compressed_sparse_attention" else 1)
    layout = check_decode_layout(layer_type, config.hidden_size, feat)
    if use_prefetcher and prefetch_buffers is None:
        prefetch_buffers = make_decode_prefetch_buffers(device, weight_dtype, num_prefetch_pages)
    prefetch = {"use_prefetcher": use_prefetcher}
    if use_prefetcher:
        prefetch["global_cb"] = prefetch_buffers[layer_type]
        prefetch["global_cb_page_bytes"] = decode_prefetch_page_bytes(weight_dtype)

    def projection(name):
        packed = {}
        if packed_weights is not None:
            tensor, packed_layout, packed_slot = packed_weights
            packed = {
                "packed_weight_tensor": tensor,
                "packed_weight_spec": packed_weight_spec(packed_layout, packed_slot, f"compressor.{name}"),
            }
        return LinearDecode(
            weights[f"compressor.{name}.weight"],
            device,
            cache.file(f"compressor.{name}"),
            dtype=weight_dtype,
            **layout,
            **prefetch,
            **packed,
        )

    return projection("kv_proj"), projection("gate_proj")


def _concat_weight(*sources):
    """A lazy ``[sum(out), in]`` weight from ``[out, in]`` sources stacked row-wise.

    Fusing projections that read the same activation means concatenating their torch
    weights; behind a thunk, so a populated tile cache still never touches the checkpoint.
    """

    def build():
        return torch.cat([w() if callable(w) else w for w in sources], dim=0)

    return build


def _interleave_tp_weights(q_source, kv_source, tp_size: int):
    """Rank-major ``[q_rank, kv_rank]`` output chunks for one TP-sharded matmul."""

    def build():
        q = q_source() if callable(q_source) else q_source
        kv = kv_source() if callable(kv_source) else kv_source
        q_chunks = q.chunk(tp_size, dim=0)
        kv_chunks = kv.chunk(tp_size, dim=0)
        return torch.cat([chunk for rank in range(tp_size) for chunk in (q_chunks[rank], kv_chunks[rank])], dim=0)

    return build


def _tp_group_slot_weight(source, groups: int, tp_size: int, slot: int):
    """One local group per rank, packed along output N for mesh sharding."""

    def build():
        weight = source() if callable(source) else source
        grouped = weight.reshape(groups, weight.shape[0] // groups, weight.shape[1])
        local_groups = groups // tp_size
        return torch.cat([grouped[rank * local_groups + slot] for rank in range(tp_size)], dim=0)

    return build


def _tp_cluster_axis(device: ttnn.MeshDevice) -> int:
    """Mesh axis of a 1xN (or flattened N-device) tensor-parallel group."""
    shape = tuple(device.shape)
    return 1 if len(shape) == 2 and shape[1] > 1 else 0


def _tp_rank_coord(device: ttnn.MeshDevice, rank: int) -> ttnn.MeshCoordinate:
    """Coordinate of a rank in the attention layer's one-dimensional TP mesh."""
    shape = tuple(device.shape)
    return ttnn.MeshCoordinate(0, rank) if len(shape) == 2 and shape[1] > 1 else ttnn.MeshCoordinate(rank, 0)


def _replicate_from_tp_rank(
    tensor: ttnn.Tensor, device: ttnn.MeshDevice, sender_rank: int, tp_size: int
) -> ttnn.Tensor:
    """Broadcast one rank's restricted matmul result with explicit P2P copies."""
    source = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tensor)
    output = ttnn.assign(source, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sender = _tp_rank_coord(device, sender_rank)
    for rank in range(tp_size):
        if rank == sender_rank:
            continue
        output = ttnn.point_to_point(
            source,
            sender,
            _tp_rank_coord(device, rank),
            output_tensor=output,
            topology=ttnn.Topology.Linear,
        )
    ttnn.deallocate(source)
    return output


def _gather_tp_width(tensor: ttnn.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Gather an N-sharded projection into one replicated DRAM tensor."""
    local = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tensor)
    gathered = ttnn.all_gather(
        local,
        dim=3,
        cluster_axis=_tp_cluster_axis(device),
        num_links=1,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(local)
    return gathered


class DeepSeekV4Attention(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4Attention`` (decode only, running KV cache).

    Construct from a ``config`` (the HF ``DeepseekV4Config`` or any object
    exposing the same attributes), the layer's torch ``weights`` (HF-named
    ``state_dict`` entries), and a device. :meth:`decode` / :meth:`decode_static`
    consume pre-built RoPE tables (see :func:`make_rope_table`); these are inputs
    because the rotary embedding is owned by the surrounding model in the
    reference, not by the attention block.

    ``tp_size > 1`` expects a 1xTP mesh and replicated hidden/KV inputs.
    ``qkv_tp_strategy`` controls the two small input projections. The default
    fuses them into one replicated full-width matmul; balanced N-sharding and
    dedicated q_a/KV ranks remain available for measured comparisons. Query
    heads and complete output groups are sharded across the mesh. By default,
    each local output group uses an ordinary matmul and ``o_b`` is row-parallel:
    it consumes those local groups directly and all-reduces the full-hidden
    partials. Column-parallel ``o_b`` remains available as an alternative.

    ``use_prefetcher=True`` switches the decode projections that still fit the shared
    64-receiver GCB (q_b, row-parallel o_b, the compressor's kv/gate pair) onto
    DRISC-prefetched weights. Sequential o_a stays on the DRAM->L1 copy: a private
    32-core GCB on the same cores as the shared ring (and the pipeline socket at
    ``(0,0)``) collides with ``fused_hyperconnection`` static CBs. Fused q_a+kv also
    stays on the transient L1 path: its 1536-wide weight cannot share the decode GCB's
    page size. Each prefetched weight stays DRAM ND-sharded and the tensor prefetcher
    pushes it into the matmul's in1 buffer, instead of copying DRAM -> L1 before every
    call. Two things come with it:

    * The caller must open a prefetcher session around the decode steps
      (``ttnn.experimental.start_tensor_prefetcher`` / ``stop_tensor_prefetcher``, with a
      ``wait_for_cq_on_tensor_prefetcher`` after the weights are written), because one
      session should span a whole model step rather than a single block. Check
      ``ttnn.experimental.is_tensor_prefetcher_supported(device)`` first.
    * A GCB is a permanent L1 allocation, not a transient staging copy. Pass
      ``prefetch_buffers`` from :func:`~.decode_prefetch.make_decode_prefetch_buffers` so one
      buffer is shared by every layer on the device: left to build its own, each block costs
      288 KB per receiver core plus a slice of a DRISC state zone that only fits about six
      GCBs, neither of which scales past a handful of layers.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        use_prefetcher: bool = False,
        num_prefetch_pages: Optional[int] = None,
        prefetch_buffers: Optional[dict] = None,
        system_config=None,
        packed_weights=None,
        tp_size: int = 1,
        qkv_tp_strategy: str = "fused_replicated_full",
        o_b_tp_strategy: str = "row",
        o_a_tp_strategy: str = "sequential",
    ):
        # SDPA program config, the resident-weight choice and the prefetch ring depth all
        # come from the system profile unless the caller pinned them.
        sys_cfg = system_config or active_system_config()
        self.system_config = sys_cfg
        if num_prefetch_pages is None:
            num_prefetch_pages = sys_cfg.prefetcher.num_prefetch_pages
        self.use_prefetcher = use_prefetcher
        self.use_packed_l1_weights = packed_weights is not None
        if self.use_packed_l1_weights and use_prefetcher:
            raise ValueError("packed L1 attention weights are incompatible with the weight prefetcher")
        if self.use_packed_l1_weights and tp_size > 1:
            raise ValueError("packed L1 attention weights do not support tensor parallelism")
        if self.use_packed_l1_weights and weight_dtype != ttnn.bfloat4_b:
            raise ValueError("packed L1 attention weights require weight_dtype=ttnn.bfloat4_b")
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if self.num_heads % tp_size:
            raise ValueError(f"num_attention_heads {self.num_heads} is not divisible by tp_size {tp_size}")
        if tp_size > 1 and device.get_num_devices() != tp_size:
            raise ValueError(
                f"tensor-parallel attention expects one device per TP rank, got tp_size={tp_size} "
                f"on a {device.get_num_devices()}-device mesh"
            )
        self.tp_size = tp_size
        self.local_num_heads = self.num_heads // tp_size
        if o_b_tp_strategy not in ("column", "row"):
            raise ValueError(f"o_b_tp_strategy must be column or row, got {o_b_tp_strategy!r}")
        if o_a_tp_strategy not in ("batched", "sequential"):
            raise ValueError(f"o_a_tp_strategy must be batched or sequential, got {o_a_tp_strategy!r}")
        self.o_b_tp_strategy = o_b_tp_strategy if tp_size > 1 else "column"
        self.row_parallel_o_b = self.o_b_tp_strategy == "row"
        self.o_a_tp_strategy = o_a_tp_strategy if tp_size > 1 else "batched"
        self.sequential_o_a = self.o_a_tp_strategy == "sequential"
        if qkv_tp_strategy not in (
            "fused_balanced",
            "fused_replicated_full",
            "fused_replicated",
            "balanced",
            "dedicated",
            "replicated",
        ):
            raise ValueError(
                "qkv_tp_strategy must be fused_balanced, fused_replicated_full, fused_replicated, "
                f"balanced, dedicated, or replicated, got {qkv_tp_strategy!r}"
            )
        self.qkv_tp_strategy = qkv_tp_strategy if tp_size > 1 else "replicated"
        self.dedicated_qkv_ranks = self.qkv_tp_strategy == "dedicated"
        self.balanced_qkv = self.qkv_tp_strategy == "balanced"
        self.fused_balanced_qkv = self.qkv_tp_strategy == "fused_balanced"
        self.fused_full_qkv = self.qkv_tp_strategy == "fused_replicated_full"
        self.q_projection_rank = 0
        self.kv_projection_rank = 1 if tp_size > 1 else 0
        self.q_projection_mesh_coords = (
            [_tp_rank_coord(device, self.q_projection_rank)] if self.dedicated_qkv_ranks else None
        )
        self.kv_projection_mesh_coords = (
            [_tp_rank_coord(device, self.kv_projection_rank)] if self.dedicated_qkv_ranks else None
        )
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.o_groups = config.o_groups
        if self.num_heads % self.o_groups:
            raise ValueError(f"num_attention_heads {self.num_heads} is not divisible by o_groups {self.o_groups}")
        if self.o_groups % tp_size:
            raise ValueError(f"o_groups {self.o_groups} is not divisible by tp_size {tp_size}")
        self.local_o_groups = self.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        self.scaling = self.head_dim**-0.5
        cache = _as_cache(cache)
        print(f"weight_dtype: {weight_dtype}")
        self.packed_weights = packed_weights

        if use_prefetcher and prefetch_buffers is None:
            prefetch_buffers = make_decode_prefetch_buffers(device, weight_dtype, num_prefetch_pages)

        def projection(name):
            # A restricted matmul cannot consume a mesh-wide prefetch request:
            # pages sent to inactive ranks would never be acknowledged. q_a/kv
            # therefore use their ordinary transient L1 weight path under TP.
            # Column-parallel o_b and balanced q_a/kv cut N, so their B-core
            # count no longer matches the shared 64-receiver GCB. Row-parallel
            # o_b only cuts K and stays on 64 cores, so it can share that buffer.
            restricted_projection = self.dedicated_qkv_ranks and name in ("q_a_proj", "kv_proj")
            local_receiver_grid = tp_size > 1 and (
                (name == "o_b_proj" and not self.row_parallel_o_b)
                or (self.balanced_qkv and name in ("q_a_proj", "kv_proj"))
            )
            projection_uses_prefetcher = use_prefetcher and not (restricted_projection or local_receiver_grid)
            prefetch = {"use_prefetcher": projection_uses_prefetcher}
            if projection_uses_prefetcher:
                prefetch["global_cb"] = prefetch_buffers[name]
                prefetch["global_cb_page_bytes"] = decode_prefetch_page_bytes(weight_dtype)
            packed = {}
            if self.packed_weights is not None:
                tensor, packed_layout, packed_slot = self.packed_weights
                packed = {
                    "packed_weight_tensor": tensor,
                    "packed_weight_spec": packed_weight_spec(packed_layout, packed_slot, name),
                }
            layout = dict(DECODE_LAYOUTS[name])
            mapper = None
            cache_name = name
            shard_projection = (
                name == "q_b_proj"
                or (name == "o_b_proj" and not self.row_parallel_o_b)
                or (self.balanced_qkv and name in ("q_a_proj", "kv_proj"))
            )
            if shard_projection and tp_size > 1:
                # Cut the full [K, N] host tensor into contiguous output ranges.
                # For q_b these are query-head ranges; for o_b they are hidden
                # features. Balanced q_a/KV use a full-K layout because folding
                # the global partial-K weight before mesh sharding would mix
                # output ranges from different ranks.
                layout["N"] //= tp_size
                if name in ("q_a_proj", "kv_proj"):
                    layout.pop("partial_width_sharded", None)
                    layout.pop("k_blocks", None)
                    layout["n_blocks"] = layout["N"] // ttnn.TILE_SIZE
                mapper = ttnn.ShardTensorToMesh(device, dim=-1)
                cache_name = f"{name}.tp{tp_size}.{self.qkv_tp_strategy}"
            elif name == "o_b_proj" and self.row_parallel_o_b:
                layout["K"] //= tp_size
                mapper = ttnn.ShardTensorToMesh(device, dim=-2)
                cache_name = f"{name}.tp{tp_size}.row"
            return LinearDecode(
                weights[f"{name}.weight"],
                device,
                cache.file(cache_name),
                dtype=weight_dtype,
                mesh_mapper=mapper,
                **layout,
                **prefetch,
                **packed,
            )

        # q_a and kv both read the block's hidden, so one matmul over their concatenated
        # weight replaces two and ``_qkv`` splits the halves back out. Only on the L1 path:
        # the fused weight cannot join the shared decode GCB (see DECODE_LAYOUTS), so under
        # the prefetcher the two projections stay separate. ``attention.fuse_qa_kv_proj`` in
        # the system profile picks between that and never fusing (see
        # :meth:`AttentionSettings.resolve_fuse_qa_kv`).
        # The split width is q_a's output, read off the layout registry rather than the
        # config: the layouts are fixed constants anyway (``check_decode_layout`` below is
        # what ties them to this config), and not every caller's config object carries
        # ``q_lora_rank``.
        self.q_lora_rank = DECODE_LAYOUTS["q_a_proj"]["N"]
        self.fused_qa_kv = (
            self.qkv_tp_strategy == "fused_replicated"
            or self.fused_full_qkv
            or self.fused_balanced_qkv
            or (sys_cfg.attention.resolve_fuse_qa_kv(use_prefetcher) and self.qkv_tp_strategy == "replicated")
        )
        if self.use_packed_l1_weights and self.fused_qa_kv:
            raise ValueError("packed L1 attention weights require attention.fuse_qa_kv_proj=false")
        if self.fused_qa_kv:
            if self.fused_balanced_qkv:
                local_qkv_width = (self.q_lora_rank + self.head_dim) // tp_size
                self.qa_kv_proj = LinearDecode(
                    _interleave_tp_weights(weights["q_a_proj.weight"], weights["kv_proj.weight"], tp_size),
                    device,
                    cache.file(f"qa_kv_proj.tp{tp_size}.fused_balanced"),
                    dtype=weight_dtype,
                    K=config.hidden_size,
                    N=local_qkv_width,
                    n_blocks=local_qkv_width // ttnn.TILE_SIZE,
                    mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1),
                )
            elif self.fused_full_qkv:
                qkv_width = self.q_lora_rank + self.head_dim
                self.qa_kv_proj = LinearDecode(
                    _concat_weight(weights["q_a_proj.weight"], weights["kv_proj.weight"]),
                    device,
                    cache.file("qa_kv_proj.full.n48"),
                    dtype=weight_dtype,
                    K=config.hidden_size,
                    N=qkv_width,
                    n_blocks=qkv_width // ttnn.TILE_SIZE,
                )
            else:
                check_decode_layout("qa_kv_proj", config.hidden_size, self.q_lora_rank + self.head_dim)
                self.qa_kv_proj = LinearDecode(
                    _concat_weight(weights["q_a_proj.weight"], weights["kv_proj.weight"]),
                    device,
                    cache.file("qa_kv_proj"),
                    dtype=weight_dtype,
                    keep_weights_in_l1=sys_cfg.attention.keep_qa_kv_weights_in_l1,
                    **DECODE_LAYOUTS["qa_kv_proj"],
                )
        else:
            self.q_a_proj = projection("q_a_proj")
            self.kv_proj = projection("kv_proj")
        self.q_b_proj = projection("q_b_proj")
        self.o_b_proj = projection("o_b_proj")
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
        #
        # Under the prefetcher it streams through the device's one shared GCB like every other
        # decode weight, at the fixed b_blocks/n_blocks the registry sizes that buffer against
        # -- passed explicitly rather than left to the class's own defaults, so a device with a
        # different grid still gets the geometry the buffer was actually built for.
        in_per_group = (self.num_heads * self.head_dim) // self.o_groups  # K; unchanged by group sharding
        o_a_layout = check_decode_layout("o_a_proj", in_per_group, self.o_lora_rank, batch=self.o_groups)
        if self.sequential_o_a:
            # Two [K, o_lora_rank] matmuls on 32 cores cannot join the shared 64-receiver
            # decode GCB. A private GCB on those cores also collides with
            # fused_hyperconnection static CBs (core (0,0) is a pipeline socket), so
            # they stay on the transient DRAM->L1 path.
            self.o_a_projs = [
                LinearDecode(
                    _tp_group_slot_weight(weights["o_a_proj.weight"], self.o_groups, tp_size, slot),
                    device,
                    cache.file(f"o_a_proj.tp{tp_size}.slot{slot}.n32"),
                    dtype=weight_dtype,
                    K=in_per_group,
                    N=self.o_lora_rank,
                    n_blocks=self.o_lora_rank // ttnn.TILE_SIZE,
                    mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1),
                )
                for slot in range(self.local_o_groups)
            ]
        elif tp_size > 1:
            o_a_layout = {
                **o_a_layout,
                "b_blocks": o_a_layout["b_blocks"] // tp_size,
                "n_blocks": o_a_layout["n_blocks"] * tp_size,
            }
        if not self.sequential_o_a:
            o_a_prefetch = {"use_prefetcher": use_prefetcher}
            if use_prefetcher:
                o_a_prefetch["global_cb"] = prefetch_buffers["o_a_proj"]
                o_a_prefetch["global_cb_page_bytes"] = decode_prefetch_page_bytes(weight_dtype)
            self.o_a_proj = BatchedLinearDecode(
                weights["o_a_proj.weight"],
                device,
                cache.file(
                    f"o_a_proj.tp{tp_size}.b{o_a_layout['b_blocks']}n{o_a_layout['n_blocks']}"
                    if tp_size > 1
                    else "o_a_proj"
                ),
                dtype=weight_dtype,
                batch=self.local_o_groups,
                global_batch=self.o_groups,
                K=in_per_group,
                N=self.o_lora_rank,
                b_blocks=o_a_layout["b_blocks"],
                n_blocks=o_a_layout["n_blocks"],
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3) if tp_size > 1 else None,
                preprocess=lambda w: w.reshape(self.o_groups, self.o_lora_rank, in_per_group)
                .transpose(1, 2)
                .contiguous(),
                **o_a_prefetch,
                **(
                    {
                        "packed_weight_tensor": self.packed_weights[0],
                        "packed_weight_spec": packed_weight_spec(
                            self.packed_weights[1], self.packed_weights[2], "o_a_proj"
                        ),
                    }
                    if self.packed_weights is not None
                    else {}
                ),
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
        self.sdpa_sinks_tt = ttnn.from_torch(
            sdpa_sink,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0) if tp_size > 1 else None,
        )
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
        # the grid.
        #
        # The profile's default is 2 rather than 4 because these CBs are *statically*
        # allocated and so have to fit under whatever L1 buffers are live at the call --
        # which now includes a resident projection weight (``keep_weights_in_l1``, ~55
        # KB/core at bf4). At 4 the CB region runs ~60 KB past the resident buffer and the
        # program fails to build; the term above is linear in ``cores_per_head - 1``, so
        # dropping to 2 cuts that scratch CB to a third of what 4 asks for. The cost is the
        # KV reduction splitting 2 ways instead of 4, which is the part of the op that
        # scales with the (short) KV axis.
        self._sdpa_pcfg = sys_cfg.attention.sdpa_program_config(device)

        # The rotate-half matrix must stay precise (a bf4 rotation would corrupt RoPE).
        self.rot = _load_weight(_interleaved_rotate_matrix(self.rope_dim), device, cache_file_name=cache.file("rot"))
        compressor_cls = _COMPRESSORS.get(self.layer_type)
        print(f"Attn with {compressor_cls} compressor at layer {self.layer_idx}. {self.layer_type}")
        self.compressor = (
            compressor_cls(
                config,
                weights,
                device,
                self.rot,
                self.rope_dim,
                cache=cache,
                weight_dtype=weight_dtype,
                use_prefetcher=use_prefetcher,
                num_prefetch_pages=num_prefetch_pages,
                prefetch_buffers=prefetch_buffers,
                packed_weights=self.packed_weights,
            )
            if compressor_cls is not None
            else None
        )

    def prefetch_weights(self):
        """Stage this block's projection weights ahead of the :meth:`decode` that uses them.

        On the L1 path this copies DRAM -> L1 width-sharded and so is bounded by L1: o_a_proj
        and o_b_proj are left out because their weights do not fit alongside the others.

        On the prefetcher path it instead queues each projection configured for the shared
        GCB. Column-parallel o_b, balanced q_a/kv, and sequential o_a keep a local receiver
        grid and stay on the transient L1 path because they cannot consume the global
        64-receiver buffer. Requires an open prefetcher session (see the class docstring),
        and every queued request must be consumed by a matching ``decode``: queueing without
        the follow-up matmul leaves pages nobody drains.

        Projections on the shared GCB use one FIFO, so they are queued here in the order
        ``decode`` calls them -- which puts the compressor between kv_proj and o_a_proj, since
        ``decode_static`` runs it after ``_qkv`` and before ``_attend`` reaches the output
        projections (see ``_grouped_output``). Nothing checks the shared FIFO order: a
        projection whose matmul runs out of turn pops its own page size off the head of another
        weight's slab, which is wrong results rather than an error.
        """
        if self.fused_qa_kv:
            self.qa_kv_proj.fetch_weights()
            self.q_b_proj.fetch_weights()
        else:
            if not (self.dedicated_qkv_ranks and self.use_prefetcher):
                self.q_a_proj.fetch_weights()
            self.q_b_proj.fetch_weights()
            if not (self.dedicated_qkv_ranks and self.use_prefetcher):
                self.kv_proj.fetch_weights()
        if self.compressor is not None:
            self.compressor.prefetch_weights()
        if self.sequential_o_a:
            for proj in self.o_a_projs:
                if proj.use_prefetcher:
                    proj.fetch_weights()
        elif self.o_a_proj.use_prefetcher:
            self.o_a_proj.fetch_weights()
        if self.o_b_proj.use_prefetcher:
            self.o_b_proj.fetch_weights()

    def _sdpa_decode(
        self,
        q: ttnn.Tensor,
        kv: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        cur_pos: ttnn.Tensor | None = None,
        paged: PagedLayerView | None = None,
        sliding_window: int | None = None,
    ) -> ttnn.Tensor:
        """Single-token (``S == 1``) attention over the batch via the fused SDPA-decode op.

        Drop-in for :meth:`_attention` on the decode paths: fuses the scale, the
        masking, the per-head sink, and both matmuls into one device op.

        ``q`` ``[1, B, H, Dh]`` (already the op's decode head layout, produced by
        :meth:`_qkv`); ``kv`` is the shared K==V ``[B, 1, Skv, Dh]`` (MQA, one KV head).
        The op emits ``[1, B, H, Dh]`` too, so no head/seq transposes are needed around
        the call.

        Two mutually exclusive ways to bound the KV axis (the op rejects an
        ``attn_mask`` in causal mode, so this is a real branch):

        * ``cur_pos`` ``[B]`` INT32 -- causal mode. The kernel derives its chunk
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
          Its leading dim stays 1 and the op broadcasts it over the batch, which the
          users of a step can share because they are all at the same position.

        ``paged`` swaps ``kv`` for the layer's block pool read through the active
        session's page table; the bounding modes above are unchanged by it, except
        that a bounded ring (``paged.position_modulo``) additionally passes
        ``sliding_window_size`` so the kernel attends the last ``window`` positions
        rather than the whole (wrapped) capacity.

        Under tensor parallelism Q and the per-head sink are sharded on the head
        axis while the shared MQA KV cache, positions, page table, and mask are
        replicated. Each rank therefore runs SDPA for ``H / TP`` heads independently;
        no collective is needed in this primitive. The result stays head-sharded through
        output RoPE and the group-local ``o_a`` projection; :meth:`_grouped_output`
        gathers those projected groups before the global ``o_b`` mix, then gathers
        the N/TP ``o_b`` outputs to restore a replicated hidden state.
        """
        # sdpa_decode requires its K/V operands in DRAM.
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        bounds = (
            {"is_causal": True, "cur_pos_tensor": cur_pos}
            if cur_pos is not None
            else {
                "is_causal": False,
                "attn_mask": ttnn.repeat(mask, ttnn.Shape([1, 1, self.local_num_heads, 1])),
            }
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
        )  # [1, B, H, Dh]

    def _grouped_output(self, attn: ttnn.Tensor) -> ttnn.Tensor:
        """``DeepseekV4GroupedLinear`` (o_a) + ``o_b_proj``.

        ``attn`` is SDPA-decode's ``[1, B, H, Dh]``; returns the block's hidden output
        back on packed rows, ``[1, 1, B, D]``. Reshape to per-group feature blocks, run
        the batched ``matmul_decode`` over the group axis (batch = o_groups, weights
        folded along group + N), then mix groups back to hidden via ``o_b_proj``.

        The groups partition the heads, so splitting ``[B, H, Dh]`` into
        ``[B, g, (H/g)*Dh]`` is a plain reshape and only the group / token axes have to
        be swapped to give the batched matmul its group-major activation.

        With TP, each rank owns a contiguous set of complete groups. ``o_a`` is
        consequently group-sharded and runs locally. In the default row-parallel
        mode, ``o_b`` consumes those local groups, computes full-N partials, and
        all-reduces them. Column mode instead gathers all groups, computes N/TP
        hidden features per rank, and gathers the final hidden state.
        """
        _, m, h, dh = attn.shape
        in_per_group = (h * dh) // self.local_o_groups
        # Rank-4 activation [1, g, M, K] (batch = g = o_groups) for the batched matmul_decode; the
        # op folds the group axis to match the folded (b_blocks x n_blocks) weight layout.
        x = ttnn.reshape(attn, [m, self.local_o_groups, in_per_group])
        x = ttnn.permute(x, [1, 0, 2])  # [g, M, K]
        x = ttnn.reshape(x, [1, self.local_o_groups, m, in_per_group])  # [1, g_local, M, K]
        if self.sequential_o_a:
            group_inputs = ttnn.split(x, 1, dim=1)
            group_outputs = [
                ttnn.to_memory_config(proj(group_input), ttnn.DRAM_MEMORY_CONFIG)
                for proj, group_input in zip(self.o_a_projs, group_inputs)
            ]
            y = ttnn.concat(group_outputs, dim=1)
            for tensor in group_outputs:
                ttnn.deallocate(tensor)
        else:
            y = self.o_a_proj(x)  # DRAM-interleaved [1, g, M, N]
        y = ttnn.permute(y, [0, 2, 1, 3])  # [1, M, g, N]
        y = ttnn.reshape(y, [1, 1, m, self.local_o_groups * self.o_lora_rank])
        if self.tp_size > 1 and not self.row_parallel_o_b:
            gathered = ttnn.all_gather(
                y,
                dim=3,
                cluster_axis=_tp_cluster_axis(self.device),
                num_links=1,
                topology=ttnn.Topology.Linear,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(y)
            y = gathered
        output = ttnn.to_memory_config(self.o_b_proj(y), ttnn.DRAM_MEMORY_CONFIG)
        if self.tp_size > 1:
            if self.row_parallel_o_b:
                gathered = ttnn.all_reduce(
                    output,
                    cluster_axis=_tp_cluster_axis(self.device),
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                gathered = ttnn.all_gather(
                    output,
                    dim=3,
                    cluster_axis=_tp_cluster_axis(self.device),
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            ttnn.deallocate(output)
            output = gathered
        return output

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

        Shared tail of :meth:`decode` / :meth:`decode_static`: ``q`` ``[1,B,H,Dh]``,
        the shared K==V ``kv`` ``[B,1,Skv,Dh]`` (or ``paged``'s block pool) and either
        ``sdpa_cur_pos`` or the additive ``mask`` ``[1,1,1,Skv]`` -> the block's hidden
        output on packed rows, ``[1,1,B,D]``. ``kv`` is the layer's persistent buffer,
        updated in place; the only per-path difference is where ``mask`` /
        ``sdpa_cur_pos`` come from (host-built for eager, device-generated for the
        traced path).
        """
        attn = self._sdpa_decode(
            q, kv, mask, cur_pos=sdpa_cur_pos, paged=paged, sliding_window=sliding_window
        )  # [1, B, H, Dh]
        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim)
        return self._grouped_output(attn)

    def _qkv(self, tokens: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Project + RoPE the query and (shared) K=V for the packed-row ``tokens`` ``[1, 1, B, D]``.

        Returns ``q`` ``[1, B, H_local, Dh]`` (``H_local == H / TP``; equal to
        ``H`` without TP) and the rotated, replicated ``kv`` ``[1, 1, B, Dh]``,
        still on packed rows (pre-compressor, pre-cache). Shared by the decode paths.

        Only ``q`` leaves the packed layout, because SDPA-decode is the one op here that
        wants a head axis: the projections and norms all run over the single tile-row the
        batch occupies, so a B-user step issues the same ops a one-user step does.
        """
        _, _, tokens_n, _ = tokens.shape
        h, dh = self.local_num_heads, self.head_dim
        _profile(self.device)
        kv_raw = None
        if self.fused_qa_kv:
            # One matmul for both projections of ``tokens``. Its output is width-sharded
            # over the reduction cores and the split point is not a shard boundary, so the
            # halves are cut in DRAM; both norms reshard their own input anyway.
            qa_kv = self.qa_kv_proj(tokens)
            if self.fused_balanced_qkv:
                local_q_width = self.q_lora_rank // self.tp_size
                qa_kv = _gather_tp_width(qa_kv, self.device)
                qa_kv = ttnn.reshape(
                    qa_kv,
                    [1, tokens_n, self.tp_size, local_q_width + dh // self.tp_size],
                )
                q_a_raw, kv_raw = ttnn.split(
                    qa_kv,
                    [local_q_width, dh // self.tp_size],
                    dim=3,
                )
                q_a_raw = ttnn.reshape(q_a_raw, [1, 1, tokens_n, self.q_lora_rank])
                kv_raw = ttnn.reshape(kv_raw, [1, 1, tokens_n, dh])
            else:
                qa_kv = ttnn.to_memory_config(qa_kv, ttnn.DRAM_MEMORY_CONFIG)
                q_a_raw, kv_raw = ttnn.split(qa_kv, [self.q_lora_rank, dh], dim=3)
            ttnn.deallocate(qa_kv)
        else:
            q_a_raw = self.q_a_proj(tokens, mesh_coords=self.q_projection_mesh_coords)
            if self.dedicated_qkv_ranks:
                q_a_raw = _replicate_from_tp_rank(q_a_raw, self.device, self.q_projection_rank, self.tp_size)
            elif self.balanced_qkv:
                q_a_raw = _gather_tp_width(q_a_raw, self.device)

        q_a = self.q_a_norm(q_a_raw)
        q = self.q_b_proj(q_a)  # [1, 1, B, H*Dh]
        q = ttnn.reshape(q, [1, tokens_n, h, dh], memory_config=width_sharded_l1_config(tokens_n * h, dh, self.device))

        q = _rms_norm_unweighted(q, self.eps)
        q = _apply_rope(q, cos, sin, self.rot, self.rope_dim)  # [1, B, H, Dh]

        # Unfused, kv_proj runs here rather than beside q_a_proj: one GCB is one FIFO, so a
        # prefetched matmul that runs out of turn pops another weight's page (see
        # ``prefetch_weights``).
        if kv_raw is None:
            kv_raw = self.kv_proj(tokens, mesh_coords=self.kv_projection_mesh_coords)
            if self.dedicated_qkv_ranks:
                kv_raw = _replicate_from_tp_rank(kv_raw, self.device, self.kv_projection_rank, self.tp_size)
            elif self.balanced_qkv:
                kv_raw = _gather_tp_width(kv_raw, self.device)
        kv = self.kv_norm(kv_raw)  # [1, 1, B, Dh]

        kv = _apply_rope(kv, cos, sin, self.rot, self.rope_dim)
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

        ``hidden`` is ``[B, 1, 1, D]`` and the block decodes all ``B`` users in one step,
        every one of them at the same absolute position: ``mask``, the RoPE rows and the
        position tensors are shared, and the latter carry one (identical) entry per user
        because the cache and SDPA ops index per user.
        """
        b, s, _, d = hidden.shape
        assert s == 1, f"decode attends one token per user, but S == {s}"
        tokens = _pack_tokens(hidden)  # [1, 1, B, D]
        q, kv_new = self._qkv(tokens, cos, sin)  # q [1,B,H,Dh], kv_new [1,1,B,Dh]
        kv_new = _one_row_per_user(kv_new)  # [1, B, 1, Dh], one core per user for the write

        if self.compressor is None:
            # The KV axis is the sliding ring alone. Paged: the *absolute* position,
            # which ``paged.position_modulo`` wraps into the bounded ring, read in causal
            # mode so the kernel honours ``cur_pos`` -- non-causal ignores it and walks
            # the whole (wrapped) capacity, double-counting the tail. Dense: the ring
            # slot, with the additive mask hiding the not-yet-written slots.
            if paged is not None:
                _update_cache_at(None, kv_new, compress_pos, paged=paged)
                ttnn.deallocate(kv_new)
                out = self._attend(
                    q,
                    None,
                    None,
                    cos,
                    neg_sin,
                    sdpa_cur_pos=compress_pos,
                    paged=paged,
                    sliding_window=self.config.sliding_window,
                )
            else:
                _update_cache_at(scache.sliding, kv_new, sliding_pos)
                ttnn.deallocate(kv_new)
                out = self._attend(q, scache.sliding, mask, cos, neg_sin)
            return ttnn.reshape(out, [b, s, 1, d])

        # One KV axis holds both regions, so there is no per-step concat: the ring slot
        # ``pos % window`` lands in the prefix and each pooled entry is appended after
        # it at row ``window + w``. Both indices are pre-wrapped, so the paged reads
        # need no ``cache_position_modulo``.
        kv = None if paged is not None else scache.combined  # [B, 1, window + n_win, Dh]
        _update_cache_at(kv, kv_new, sliding_pos, paged=paged)
        # Written, and one row per user is a whole tile of L1 each -- worth handing
        # back before the compressor and SDPA below ask for their own.
        ttnn.deallocate(kv_new)
        # ``q`` is width-sharded over B*H rows of L1 and nothing reads it until the SDPA
        # below, while the compressor in between is the step's L1 high-water mark. At a
        # wide batch holding both at once is what leaves an op's circular buffers
        # nowhere to go, so park q in DRAM across the compressor and bring it back in
        # the layout SDPA expects. At batch 1 both fit and the round trip is dead cost.
        q_config = q.memory_config() if b > 1 else None
        if q_config is not None:
            spilled = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(q)
            q = spilled
        self.compressor.decode_static(
            tokens,
            cos_win,
            sin_win,
            scache,
            kv,
            win_slot,
            win_row=win_row,
            pool=pool_compressor,
            paged=paged,
        )
        if q_config is not None:
            q = ttnn.to_memory_config(q, q_config)
        out = self._attend(q, kv, mask, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos, paged=paged)
        return ttnn.reshape(out, [b, s, 1, d])
