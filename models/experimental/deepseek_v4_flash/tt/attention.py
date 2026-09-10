from typing import Optional

import ttnn
import torch

from .common import DeepSeekV4Module, _HIFI4_SDPA, _MASK_NEG, _profile, _signpost, width_sharded_l1_config
from .decode_prefetch import (
    DECODE_LAYOUTS,
    HC_FN_GCB,
    HC_FN_GCB_PAGES,
    KV_GCB,
    Q_A_GCB,
    balanced_qkv_layout,
    check_decode_layout,
    decode_prefetch_page_bytes,
    ensure_named_gcb,
    hc_fn_page_bytes,
    hc_fn_ring_specs,
    kv_page_bytes,
    make_decode_prefetch_buffers,
    q_a_page_bytes,
)
from .layers import (
    BatchedLinearDecode,
    DeepSeekV4RMSNorm,
    LinearDecode,
    _core_grid_contains,
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
#   * *per-user rows* ``[1, B, 1, F]`` / ``[B, 1, ..., F]`` -- one row per user.
#     The KV-cache ops require it (``paged_update_cache`` dispatches one user per
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
      * ``win_kv`` / ``win_gate`` -- the compressor projections of the window
        *currently being filled*, at slot ``pos % compress_rate``. HCA keeps
        these as TILE DRAM ``[B, 1, compress_rate, Dh]`` for ``paged_update_cache``.
        CSA keeps them (and ``prev_*``) as ROW_MAJOR L1 WIDTH_SHARDED
        ``[B*compress_rate, 1, 1, 2*Dh]`` so ``csa_pool_window`` can consume them
        in place. Only one window is held, because pooling is incremental.
        ``None`` for sliding-only layers.
      * ``prev_kv`` / ``prev_gate`` -- CSA only: previous window's projections
        (same layout as CSA ``win_*``), because entry ``w`` also needs window
        ``w-1``'s Ca slice. Refreshed from ``win_*`` after each pool.
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

    def _csa_window(rows: int, width: int, value: float = 0.0) -> ttnn.Tensor:
        """ROW_MAJOR L1 WIDTH_SHARDED ``[batch*rows, 1, 1, width]`` (1-high faces).

        The width split ``csa_pool_window`` consumes: ``width // 32`` cores, one
        32-column shard each (32 cores at CSA ``2*Dh == 1024``). The op only reads
        the packed row count and the shard spec, so the rows sit on dim 0 rather
        than dim 2 -- that is the axis ``indexed_fill`` scatters a token into on its
        shard-local path (see :func:`_scatter_window_rows`).
        """
        height = batch * rows
        cfg = width_sharded_l1_config(height, width, device, tile_height=1)
        return ttnn.to_memory_config(
            ttnn.from_torch(
                torch.full((height, 1, 1, width), value),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            ),
            cfg,
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
        if is_csa:
            win_kv = _csa_window(cr, feat)
            win_gate = _csa_window(cr, feat)
            # Entry w pools window w-1's Ca with window w's Cb, so CSA also keeps the
            # previous window. ``-inf`` gates give window 0's absent Ca weight 0.
            prev_kv = _csa_window(cr, feat)
            prev_gate = _csa_window(cr, feat, _MASK_NEG)
        else:
            win_kv = _filled(cr, feat)
            win_gate = _filled(cr, feat)
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
    """Host-built additive decode mask ``[1, 1, 1, round_up(Skv, 32)]`` for position ``pos``.

    Mirrors the on-device mask in :meth:`DeepSeekV4Model._device_mask`: sliding
    columns mask slots with index ``> pos``; compressor columns mask windows with
    index ``>= (pos+1)//cr``. The tile-padding columns must also be negative:
    SDPA processes K's padded tile width, and zero padding here would add fake
    zero-valued KV logits to the softmax denominator.

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
    padded_width = ((width + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    if padded_width != width:
        invalid = torch.nn.functional.pad(invalid, (0, padded_width - width), value=True)
    mask = torch.zeros(1, 1, 1, padded_width, dtype=torch.float32)
    mask.masked_fill_(invalid.view(1, 1, 1, -1), _MASK_NEG)
    return ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def sdpa_causal_ok(sliding_window: int, layer_type: str, compress_rate: int | None, pos: int) -> bool:
    """Can this step's valid set be expressed as a single SDPA-decode ``cur_pos``?

    Sliding-only: the dense ring is written in order and wraps at ``W``. Valid slots
    are a prefix of ``min(pos, W-1)`` -- causal for every ``pos``. (Paged sliding
    uses absolute ``cur_pos`` plus ``sliding_window_size`` instead, inside
    :meth:`DeepSeekV4Attention.decode_static`.)

    CSA/HCA: before the first compressor window closes, the valid set is just the
    contiguous sliding prefix ``[0, pos]``. Afterwards, the KV axis is
    ``[sliding 0..W) | compressor 0..n_win_cap)`` and the
    valid set (see :func:`host_decode_mask`) is sliding slot ``i <= pos`` plus
    compressor window ``j < (pos+1)//cr``. Once the ring is full every sliding slot
    is valid, so the union is the contiguous prefix ``[0, W + (pos+1)//cr)`` -- which
    a single ``cur_pos`` describes exactly. Below that the set has a hole (slots
    ``pos+1 .. W-1`` are still unwritten) that no ``cur_pos`` can express, so those
    steps must keep the additive mask.
    """
    if layer_type == "sliding_attention":
        return True
    assert compress_rate is not None
    if (pos + 1) // compress_rate == 0:
        return True
    return pos + 1 >= sliding_window


def sdpa_causal_cur_pos(sliding_window: int, compress_rate: int | None, pos: int) -> int:
    """Inclusive last-valid index for causal SDPA-decode at ``pos``.

    Sliding-only: ``min(pos, W-1)`` on the ring. CSA/HCA: last valid index on the
    ``[sliding | compressor]`` axis. Only meaningful when :func:`sdpa_causal_ok`.
    Note the ``-1`` on the compressor formula: ``(pos+1)//cr`` is the *count* of
    closed windows, and ``cur_pos`` is inclusive. Dropping it (i.e. using
    ``W + pos//cr``) happens to agree only at window boundaries and otherwise
    exposes the still-open window, whose entry is unpooled -- a silent accuracy
    loss rather than an error.
    """
    if compress_rate is None:
        return min(pos, sliding_window - 1)
    if (pos + 1) // compress_rate == 0:
        return pos
    return sliding_window + (pos + 1) // compress_rate - 1


def decode_sdpa_bounds(
    sliding_window: int,
    layer_type: str,
    compress_rate: int | None,
    pos: int,
    max_seq: int,
    device: ttnn.MeshDevice,
    batch: int = 1,
) -> tuple[Optional[ttnn.Tensor], Optional[ttnn.Tensor]]:
    """``(mask, sdpa_cur_pos)`` for one decode step.

    Causal ``cur_pos`` whenever :func:`sdpa_causal_ok`, so
    :meth:`DeepSeekV4Attention._sdpa_decode` does not head-broadcast a mask row
    (a ``Repeat`` of the ``[1,1,1,Skv]`` additive mask across ``H``). Early CSA/HCA
    steps whose sliding region still has a hole keep the mask.
    """
    if sdpa_causal_ok(sliding_window, layer_type, compress_rate, pos):
        return None, int32_pos_tensor(sdpa_causal_cur_pos(sliding_window, compress_rate, pos), device, batch)
    return host_decode_mask(sliding_window, layer_type, compress_rate, pos, max_seq, device), None


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


def _packed_users(tokens: ttnn.Tensor) -> int:
    """Packed-token count (``B*S``) for a ``[1, 1, B*S, D]`` row, including a gathered replica.

    ``all_gather_for_matmul`` reports HEIGHT_SHARDED volume as ``num_cores * M``, so
    ``tokens.shape[-2]`` is not the user count after the decode gather. The shard height is.
    """
    mem = tokens.memory_config()
    if (
        tokens.is_sharded()
        and mem.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        and mem.shard_spec is not None
    ):
        return mem.shard_spec.shape[0]
    return tokens.shape[-2]


def _height_sharded_l1_config(
    num_users: int, width: int, device, layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT
) -> ttnn.MemoryConfig:
    """Height-sharded L1 config for a ``[1, B, 1, width]`` decode row: one core per user.

    ``paged_update_cache`` requires its (single-token) input to be height-sharded with the
    core count equal to the number of batch users -- it dispatches one user per core --
    shard width == the last dim, ROW_MAJOR orientation. ROW_MAJOR uses a 1-high shard
    (the contiguous token row the writer splices); TILE pads that to a 32-row tile.
    """
    shard_h = ttnn.TILE_SIZE if layout == ttnn.TILE_LAYOUT else 1
    grid = ttnn.num_cores_to_corerangeset(num_users, device.compute_with_storage_grid_size(), row_wise=True)
    shard_spec = ttnn.ShardSpec(grid, [shard_h, width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _one_row_per_user(x: ttnn.Tensor) -> ttnn.Tensor:
    """``[1, 1, B, F]`` -> ``[1, B, 1, F]`` ROW_MAJOR, height-sharded one user per core.

    ``paged_update_cache`` indexes by user on dim 1 and, for ROW_MAJOR input, copies the
    shard as a contiguous token row (it skips the TILE untilize). Packed projection
    rows therefore have to be spread back over one core each. A view at ``B == 1`` when
    the producer already holds that shard; a relayout otherwise. TILE input is untilized
    (via DRAM if the shard is 1-high, which cannot untilize in L1) rather than padded
    up to a 32-row tile.
    """
    x = ttnn.reshape(x, [1, x.shape[-2], 1, x.shape[-1]])
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    want = _height_sharded_l1_config(x.shape[1], x.shape[-1], x.device())
    return x if x.memory_config() == want else ttnn.to_memory_config(x, want)


def _rope_height_sharded_config(width: int, num_cores: int, device) -> ttnn.MemoryConfig:
    """Height-sharded L1 config: one tile-row (32 rows) per core over ``num_cores`` cores."""
    grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    shard_spec = ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _apply_rope(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    rot: ttnn.Tensor,
    rope_dim: int,
    head_dim: int | None = None,
) -> ttnn.Tensor:
    """Interleaved RoPE on each ``head_dim``-wide block of ``x`` ([.., D]).

    ``cos`` / ``sin`` are ``[1,1,L,rope_dim]`` tables (broadcast over batch/heads);
    ``rot`` is the ``[rope_dim, rope_dim]`` ``rotate_half`` matrix. Leading "nope"
    channels of each block pass through untouched. ``head_dim`` defaults to the
    last dim of ``x`` (one block). Packed heads (``D = H * head_dim``) are split
    on device.

    Delegates the whole calc to the fused ``ttnn.experimental.fused_partial_rope`` device
    op. Unsharded ``x`` is width-sharded in L1 first. ``cos`` / ``sin`` / ``trans_mat``
    are DRAM-interleaved (the reader streams each core's rope tiles). ROW_MAJOR ``x``
    is computed as 1x32 faces.

    ``rows`` is every leading dim multiplied out, not just ``x.shape[-2]``: the batched
    inputs here are ``[1, B, H, Dh]`` (SDPA-decode's head layout), whose ``B*H`` rows are
    contiguous because ``H`` is tile-aligned. The op counts rows the same way off the
    shard, so no reshape is needed to fold them onto dim -2.
    """
    _signpost("apply_rope start")
    d = x.shape[-1]
    hd = d if head_dim is None else head_dim
    assert d % hd == 0, f"last dim {d} is not a multiple of head_dim={hd}"

    # # The op reads one cos/sin tile-row per core, or a single tile-row broadcast across all
    # # rows on device (e.g. a shared decode position over heads). So cos/sin must cover either
    # # every input row or exactly one row.
    # assert cos.shape[-2] in (rows, 1), f"{cos.shape} not broadcastable to rows={rows}"

    # # cos/sin must already be DRAM-interleaved (the fused op's reader streams them from DRAM).
    # assert cos.memory_config().buffer_type == ttnn.BufferType.DRAM, "cos must be DRAM-interleaved"
    # assert sin.memory_config().buffer_type == ttnn.BufferType.DRAM, "sin must be DRAM-interleaved"

    # tile_height = ttnn.TILE_SIZE if x.layout == ttnn.TILE_LAYOUT else 1
    # if not x.is_sharded():
    #     x = ttnn.to_memory_config(x, width_sharded_l1_config(rows, d, device, tile_height=tile_height))

    out_sh = ttnn.experimental.fused_partial_rope(x, cos, sin, _trans_mat_for(rot), rope_dim, head_dim=hd)
    _signpost("apply_rope end")
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
def _sdpa_decode_output_config(
    batch: int, heads: int, head_dim: int, layout: ttnn.Layout, grid_size: ttnn.CoreCoord
) -> ttnn.MemoryConfig:
    """Native height-sharded output of ``sdpa_decode``: one reducer core per batch user.

    The writer has ``num_output_cores = B`` and places those reducers on the first
    ``B`` cores of the program grid in row-major order
    (``{idx % grid.x, idx / grid.x}``). Each core holds that user's full Q-head
    axis, so the shard is ``[H, Dh]`` (ROW_MAJOR) or ``[round_up(H, 32), Dh]``
    (TILE). Matching this spec lets the output CB alias the result buffer; any
    other grid or a width/block shard is either rejected or a reshard.
    """
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        shard_h = heads
    else:
        shard_h = ((heads + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    grid = ttnn.num_cores_to_corerangeset(batch, grid_size, row_wise=True)
    shard_spec = ttnn.ShardSpec(grid, [shard_h, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
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

    ``row`` has to arrive ROW_MAJOR, height-sharded one user per core
    (:func:`_height_sharded_l1_config`): it is written as it stands rather than
    resharded, since a copy here would be paid on every cache write, and it stays
    the caller's to free.
    """

    num_users, width = row.shape[1], row.shape[-1]
    expected = _height_sharded_l1_config(num_users, width, row.device(), layout=row.layout)
    assert (
        row.memory_config() == expected
    ), f"paged_update_cache needs its row one user per core: expected {expected}, got {row.memory_config()}"
    if paged is None:
        ttnn.experimental.paged_update_cache(cache, row, update_idxs_tensor=pos_tensor)
    else:
        ttnn.experimental.paged_update_cache(
            paged.pool,
            row,
            update_idxs_tensor=pos_tensor,
            page_table=paged.page_table,
            cache_position_modulo=paged.position_modulo,
        )


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
    ``index`` carries one row per user.
    """
    users, feat = row.shape[1], row.shape[-1]
    # Interleaved first: the incoming row is height-sharded one user per core, and a
    # reshape across that shard's dims is not a view.
    src = ttnn.reshape(ttnn.to_memory_config(row, ttnn.DRAM_MEMORY_CONFIG), [users, 1, 1, feat])
    _scatter_window_rows(cache, src, index)
    ttnn.deallocate(src)


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

    CSA windows are the same L1 WIDTH_SHARDED spec, so a device copy is a whole-buffer
    write into the persistent ``prev`` address. TILE DRAM windows (unused by CSA)
    still go through ``fill_cache``, which writes a single batch index per call.
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

        ``tokens`` is the block's packed-row hidden ``[1, 1, B, D]``, already gathered
        onto the decode activation grid when the caller used :meth:`DeepSeekV4Attention.decode_static`.

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

        ``win_*`` / ``prev_*`` are the persistent ROW_MAJOR L1 WIDTH_SHARDED
        ``[B*compress_rate, 1, 1, 2*Dh]`` buffers allocated by
        :func:`build_static_layer_cache`. The fused op consumes them in place.
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
        ``win_slot`` into the one-window L1 WIDTH_SHARDED buffers, and -- on the step
        that closes the window -- pool just that window (Ca/Cb overlap against the
        retained previous window) and append its single entry at row ``win_row``
        of the layer's KV axis (``combined_cache``, or ``paged``'s pool).

        ``tokens`` is the block's packed-row hidden ``[1, 1, B, D]``, already gathered
        onto the decode activation grid when the caller used :meth:`DeepSeekV4Attention.decode_static`.

        After pooling, the closing window becomes the ``prev_*`` the *next* window
        will overlap with. See :meth:`DeepSeekV4HCACompressor.decode_static`.
        """
        _signpost("CSA_START")
        feat = 2 * self.head_dim
        users = _packed_users(tokens)
        kv, gate = self._project(tokens)  # [1, 1, B, 2*Dh]
        kv = _one_row_per_user(ttnn.reshape(kv, [1, 1, users, feat]))
        gate = _one_row_per_user(ttnn.reshape(gate, [1, 1, users, feat]))
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
    ``qkv_tp_strategy`` controls the two small input projections (from the
    constructor, else ``attention.qkv_tp_strategy`` in the system profile).
    Galaxy32 uses ``replicated``: q_a and kv stay full-width on every rank (no
    all-gather), while ``q_b`` stays head-sharded TP4. ``balanced`` (N-shard then
    all-gather) and dedicated ranks remain available. Query heads and complete
    output groups are sharded across the mesh. Decode is ``M == 1``, so ``o_a`` is a
    batched ``matmul_decode`` over the local groups (the group-major permute is a
    no-op). ``o_b`` is row-parallel by default: it consumes those local groups and
    all-reduces the full-hidden partials. Column-parallel ``o_b`` remains available.

    ``use_prefetcher=True`` switches the decode projections that still fit the shared
    64-receiver GCB (q_b, batched o_a, row-parallel o_b, the compressor's kv/gate pair)
    onto DRISC-prefetched weights. Sequential o_a, if opted into, stays on the DRAM->L1
    copy: a private 32-core GCB on the same cores as the shared ring (and the pipeline
    socket at ``(0,0)``) collides with ``fused_hyperconnection`` static CBs. Each prefetched
    weight stays DRAM ND-sharded and the tensor prefetcher pushes it into the
    matmul's in1 buffer, instead of copying DRAM -> L1 before every call. Two
    things come with it:

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
        qkv_tp_strategy: Optional[str] = None,
        o_b_tp_strategy: str = "row",
        o_a_tp_strategy: str = "batched",
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
        if qkv_tp_strategy is None:
            qkv_tp_strategy = sys_cfg.attention.qkv_tp_strategy
        if qkv_tp_strategy not in (
            "balanced",
            "dedicated",
            "replicated",
        ):
            raise ValueError(f"qkv_tp_strategy must be balanced, dedicated, or replicated, got {qkv_tp_strategy!r}")
        self.qkv_tp_strategy = qkv_tp_strategy if tp_size > 1 else "replicated"
        self.dedicated_qkv_ranks = self.qkv_tp_strategy == "dedicated"
        self.balanced_qkv = self.qkv_tp_strategy == "balanced"
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

        def projection(name, weight=None, cache_suffix="", rectangle_b_grid=False):
            # A restricted matmul cannot consume a mesh-wide prefetch request:
            # pages sent to inactive ranks would never be acknowledged. q_a/kv
            # therefore use their ordinary L1 weight path under dedicated-rank TP.
            # Column-parallel o_b cuts N, so its B-core count no longer matches
            # the shared 64-receiver GCB. Balanced q_a/kv also cannot join that
            # ring (16- and 8-tile slabs vs a 32-tile page) -- they stream through
            # HC_FN_GCB instead, which is the same 64 receivers as hc.fn. Row-parallel
            # o_b only cuts K and stays on 64 cores, so it can share the decode buffer.
            restricted_projection = self.dedicated_qkv_ranks and name in ("q_a_proj", "kv_proj")
            local_receiver_grid = tp_size > 1 and (name == "o_b_proj" and not self.row_parallel_o_b)
            balanced_qkv = self.balanced_qkv and name in ("q_a_proj", "kv_proj")
            projection_uses_prefetcher = use_prefetcher and not (restricted_projection or local_receiver_grid)
            prefetch = {"use_prefetcher": projection_uses_prefetcher}
            if projection_uses_prefetcher:
                if balanced_qkv:
                    prefetch["global_cb"] = ensure_named_gcb(
                        prefetch_buffers,
                        HC_FN_GCB,
                        device,
                        hc_fn_ring_specs(),
                        weight_dtype,
                        num_pages=HC_FN_GCB_PAGES,
                    )
                    prefetch["global_cb_page_bytes"] = hc_fn_page_bytes(weight_dtype)
                elif name == "q_a_proj":
                    prefetch["global_cb"] = ensure_named_gcb(
                        prefetch_buffers,
                        Q_A_GCB,
                        device,
                        [DECODE_LAYOUTS["q_a_proj"]],
                        weight_dtype,
                    )
                    prefetch["global_cb_page_bytes"] = q_a_page_bytes(weight_dtype)
                elif name == "kv_proj":
                    prefetch["global_cb"] = ensure_named_gcb(
                        prefetch_buffers,
                        KV_GCB,
                        device,
                        [DECODE_LAYOUTS["kv_proj"]],
                        weight_dtype,
                    )
                    prefetch["global_cb_page_bytes"] = kv_page_bytes(weight_dtype)
                else:
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
                # features. Balanced q_a/KV keep a partial-K layout (64 B cores on
                # the local N) so LinearDecode folds each rank independently
                # before the mesh shard -- folding the global N first would mix
                # output ranges from different ranks.
                if balanced_qkv:
                    layout = balanced_qkv_layout(name, tp_size)
                else:
                    layout["N"] //= tp_size
                mapper = ttnn.ShardTensorToMesh(device, dim=-1)
                cache_name = f"{name}.tp{tp_size}.{self.qkv_tp_strategy}"
                if name in ("q_a_proj", "kv_proj"):
                    cache_name += f".k{layout['k_blocks']}n{layout['n_blocks']}"
            elif name == "o_b_proj" and self.row_parallel_o_b:
                layout["K"] //= tp_size
                mapper = ttnn.ShardTensorToMesh(device, dim=-2)
                cache_name = f"{name}.tp{tp_size}.row"
            elif tp_size > 1 and name in ("q_a_proj", "kv_proj") and not self.dedicated_qkv_ranks:
                # Full-width replica on every rank (galaxy32 ``replicated``). q_b stays
                # in the shard branch above.
                mapper = ttnn.ReplicateTensorToMesh(device)
                cache_name = f"{name}.tp{tp_size}.{self.qkv_tp_strategy}"
            if name == "q_a_proj" and not layout.get("partial_width_sharded", False):
                cache_name += ".full"
                # q_a's producer grid must be a subset of q_b's filled output-mcast
                # rectangle. The generic row-wise 32-core set can be ragged on Blackhole.
                rectangle_b_grid = True
            elif name == "kv_proj" and not layout.get("partial_width_sharded", False):
                cache_name += ".full"
            # Resident L1 for the q_a/kv pair when the profile asks for
            # it and they are *not* prefetched. Packed weights are already L1-resident;
            # the prefetcher streams into in1 and never holds an L1 tensor.
            resident = {}
            if (
                sys_cfg.attention.keep_qa_kv_weights_in_l1
                and name in ("q_a_proj", "kv_proj")
                and not projection_uses_prefetcher
                and not packed
            ):
                resident["keep_weights_in_l1"] = True
            return LinearDecode(
                weights[f"{name}.weight"] if weight is None else weight,
                device,
                cache.file(cache_name + cache_suffix),
                dtype=weight_dtype,
                mesh_mapper=mapper,
                **layout,
                **prefetch,
                **packed,
                **resident,
                rectangle_b_grid=rectangle_b_grid,
            )

        self.q_lora_rank = DECODE_LAYOUTS["q_a_proj"]["N"]
        self.q_a_proj = projection("q_a_proj")
        self.kv_proj = projection("kv_proj")
        # One replica of kv on core (0,0): the writer unicasts producer slices into that
        # dest bbox. Full-width only; skipped for the TP cuts that stay partial-K.
        if self.kv_proj._can_matmul_decode_rm_hs():
            self.kv_proj.set_output_core_grid(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
            )
        # q_a's RMSNorm rides in the q_a matmul epilogue when the op can take it.
        # The dest of that mcast must be a filled rectangle (NOC multicast) that also
        # holds q_b's B cores; the generic row-wise 64-core set can be ragged on Blackhole.
        self.fuse_q_a_norm = (
            self.packed_weights is None
            and not self.dedicated_qkv_ranks
            and not self.balanced_qkv
            and self.q_a_proj.can_fuse_rms_norm()
        )
        if self.fuse_q_a_norm:
            self.q_b_proj = projection("q_b_proj", cache_suffix=".rect", rectangle_b_grid=True)
        else:
            self.q_b_proj = projection("q_b_proj")
        self.o_b_proj = projection("o_b_proj")
        if self.fuse_q_a_norm:
            # Mcast a ROW_MAJOR HEIGHT_SHARDED replica of [M, q_lora] onto every q_b B
            # core so q_b can consume it as A without a DRAM round-trip.
            self.q_a_proj.set_output_core_grid(self.q_b_proj.b_core_grid())
            self.q_a_proj.enable_fused_rms_norm(self.eps, weights["q_a_norm.weight"])
        self.q_a_norm = (
            None
            if self.fuse_q_a_norm
            else DeepSeekV4RMSNorm(weights["q_a_norm.weight"], self.eps, device, cache.file("q_a_norm"), sharded=True)
        )
        self.fuse_q_b_norm = self.q_b_proj.can_fuse_rms_norm()
        if self.fuse_q_b_norm:
            self.q_b_proj.enable_fused_rms_norm(self.eps, 1.0, group_size=self.head_dim)
        self.fuse_kv_norm = (
            self.packed_weights is None
            and not self.dedicated_qkv_ranks
            and not self.balanced_qkv
            and self.kv_proj.can_fuse_rms_norm()
        )
        if self.fuse_kv_norm:
            self.kv_proj.enable_fused_rms_norm(self.eps, weights["kv_norm.weight"])
        self.kv_norm = (
            None
            if self.fuse_kv_norm
            else DeepSeekV4RMSNorm(weights["kv_norm.weight"], self.eps, device, cache.file("kv_norm"), sharded=True)
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

        On the prefetcher path it instead queues each projection configured for a GCB.
        Shared-ring weights (q_b, compressor, o_a/o_b, ...) use one FIFO. Full-width
        q_a uses its own 32-receiver FIFO; balanced q_a/kv instead share
        :data:`HC_FN_GCB` with the hyper-connections. They are still queued here in
        decode order so each ring stays in step. Column-parallel o_b and sequential
        o_a keep a local receiver grid and stay on the transient L1 path.

        Projections on the shared GCB use one FIFO, so they are queued here in the order
        ``decode`` calls them -- which puts the compressor between kv_proj and o_a_proj, since
        ``decode_static`` runs it after ``_qkv`` and before ``_attend`` reaches the output
        projections (see ``_grouped_output``). Nothing checks the shared FIFO order: a
        projection whose matmul runs out of turn pops its own page size off the head of another
        weight's slab, which is wrong results rather than an error.
        """
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

        The op's cheapest sharded output is height-sharded L1 on ``B`` cores (one
        reducer per user) with shard ``[H, Dh]``; see :func:`_sdpa_decode_output_config`.
        """
        h, dh = self.local_num_heads, self.head_dim
        # Decode ``q`` from :meth:`_qkv` is still packed ``[1, 1, B, H*Dh]``; the PCC
        # tests feed the op's head layout ``[1, B, H, Dh]`` directly.
        packed = q.shape[-1] != dh
        batch = q.shape[-2] if packed else q.shape[1]
        grid_size = self._sdpa_pcfg.compute_with_storage_grid_size
        out_mem = _sdpa_decode_output_config(batch, h, dh, q.layout, grid_size)
        # Reshard while the last dim still matches the producer: packed Q is
        # ``H*Dh`` wide, so the move uses one row of ``H*Dh`` per user (same bytes
        # as ``[H, Dh]`` for ROW_MAJOR). The head-axis view comes after, so SDPA
        # sees ``[1, B, H, Dh]``.
        if packed:
            shard_h = 1 if q.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.TILE_SIZE
            grid = ttnn.num_cores_to_corerangeset(batch, grid_size, row_wise=True)
            q_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [shard_h, q.shape[-1]], ttnn.ShardOrientation.ROW_MAJOR),
            )
        else:
            q_mem = out_mem
        if q.memory_config() != q_mem:
            q = ttnn.to_memory_config(q, q_mem)
        if packed:
            q = ttnn.experimental.view(q, [1, batch, h, dh])
        if cur_pos is not None:
            bounds = {"is_causal": True, "cur_pos_tensor": cur_pos}
        else:
            # The op compares logical head counts, so a head-independent ``[1,1,1,Skv]``
            # row has to be materialised across ``H``. Prefer ``cur_pos`` (above) so this
            # Repeat never runs once the valid set is a prefix.
            attn_mask = mask
            if mask.shape[-2] != self.local_num_heads:
                attn_mask = ttnn.repeat(mask, ttnn.Shape([1, 1, self.local_num_heads, 1]))
            bounds = {"is_causal": False, "attn_mask": attn_mask}
        common = dict(
            attention_sink=self.sdpa_sinks_tt,
            scale=self.scaling,
            program_config=self._sdpa_pcfg,
            compute_kernel_config=_HIFI4_SDPA,
            memory_config=out_mem,
            **bounds,
        )
        if paged is not None:
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                paged.pool,
                paged.pool,  # K == V (shared single KV head)
                paged.page_table,
                sliding_window_size=sliding_window,
                cache_position_modulo=paged.position_modulo,
                **common,
            )
        kv = ttnn.to_memory_config(kv, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            kv,
            kv,  # K == V (shared single KV head)
            **common,
        )  # [1, B, H, Dh] height-sharded L1

    def _grouped_output(self, attn: ttnn.Tensor) -> ttnn.Tensor:
        """``DeepseekV4GroupedLinear`` (o_a) + ``o_b_proj``.

        ``attn`` is SDPA-decode's ``[1, B, H, Dh]``; returns the block's hidden output
        back on packed rows, ``[1, 1, B, D]``. ``o_a`` is block-diagonal over
        ``o_groups``, and ``o_b_proj`` then mixes the groups back to hidden.

        Decode is a single token (``M == 1``), so ``[1, 1, H, Dh]`` is already
        group-major in memory: each group's ``K = H*Dh / g`` features are a
        contiguous slice. Batched ``matmul_decode`` reads that as ``[1, g, 1, K]``
        (a view) and returns ``[1, g, 1, N]``, which is the same bytes as o_b's
        packed ``[1, 1, 1, g*N]``.

        With TP, each rank owns a contiguous set of complete groups. ``o_a`` is
        consequently group-sharded and runs locally. In the default row-parallel
        mode, ``o_b`` consumes those local groups, computes full-N partials, and
        all-reduces them. Column mode instead gathers all groups, computes N/TP
        hidden features per rank, and gathers the final hidden state.
        """
        # SDPA-decode returns ROW_MAJOR data height-sharded by user. Do not reinterpret that
        # shard as group-major and reshard the view: the physical row is H*Dh wide while the
        # view claims each row is only K wide, so the sharded conversion uses the wrong row
        # stride. Materialize the logical [H, Dh] tensor in tiled interleaved memory before
        # folding heads into groups.
        if attn.layout == ttnn.ROW_MAJOR_LAYOUT:
            attn = ttnn.to_layout(ttnn.to_memory_config(attn, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
        _, m, h, dh = attn.shape
        groups = self.local_o_groups
        in_per_group = (h * dh) // groups
        assert not self.sequential_o_a, "decode o_a is batched (M == 1); sequential is not wired"
        assert m == 1, f"batched o_a is decode-only (M == 1), got M={m}"
        x = ttnn.reshape(attn, [1, groups, 1, in_per_group])
        y = self.o_a_proj(x)  # DRAM-interleaved [1, g, 1, N]
        y = ttnn.reshape(y, [1, 1, 1, groups * self.o_lora_rank])
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

        Shared tail of :meth:`decode` / :meth:`decode_static`: ``q`` packed
        ``[1, 1, B, H*Dh]`` (or already ``[1, B, H, Dh]``), the shared K==V ``kv``
        ``[B, 1, Skv, Dh]`` (or ``paged``'s block pool) and either ``sdpa_cur_pos`` or
        the additive ``mask`` ``[1,1,1,Skv]`` -> the block's hidden output on packed
        rows, ``[1,1,B,D]``. ``kv`` is the layer's persistent buffer, updated in place;
        the only per-path difference is where ``mask`` / ``sdpa_cur_pos`` come from
        (host-built for eager, device-generated for the traced path).
        """
        attn = self._sdpa_decode(
            q, kv, mask, cur_pos=sdpa_cur_pos, paged=paged, sliding_window=sliding_window
        )  # [1, B, H, Dh]
        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim)
        return self._grouped_output(attn)

    def _decode_activation_grid(self) -> ttnn.CoreRangeSet:
        """Core set that receives the decode all-gather replica of packed ``tokens``.

        Start from q_a's B grid (the larger of the two input projections) and grow to any
        compressor kv/gate grid that already contains it, so one multicast covers q_a, kv
        and the compressor pair. LinearDecode reuses a replica whose grid is a superset of
        its B cores.
        """
        grid = self.q_a_proj.b_core_grid()
        extras = [self.kv_proj.b_core_grid()]
        if self.compressor is not None:
            extras.extend((self.compressor.kv_proj.b_core_grid(), self.compressor.gate_proj.b_core_grid()))
        for other in extras:
            if other.num_cores() > grid.num_cores() and _core_grid_contains(other, grid):
                grid = other
        return grid

    def _qkv(self, tokens: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Project + RoPE the query and (shared) K=V for the packed-row ``tokens`` ``[1, 1, B, D]``.

        Returns ``q`` ``[1, 1, B, H_local*Dh]`` (packed; ``H_local == H / TP``) and the
        rotated, replicated ``kv`` ``[1, 1, B, Dh]``, still on packed rows
        (pre-compressor, pre-cache). Shared by the decode paths.

        ``q`` stays packed through RoPE; :meth:`_sdpa_decode` is the one op that
        wants a head axis, and it views ``[1, B, H, Dh]`` after the height-shard
        reshard. The projections and norms all run over the single tile-row the
        batch occupies, so a B-user step issues the same ops a one-user step does.
        """
        _profile(self.device)
        # ``tokens`` is already the decode-static all-gather: ROW_MAJOR HEIGHT_SHARDED with
        # full K on every core of :meth:`_decode_activation_grid`. q_a's B cores (and kv's,
        # a subset) already hold a replica; a partial-width weight cannot take this layout
        # (LinearDecode unreplicates).
        q_a_raw = self.q_a_proj(tokens, mesh_coords=self.q_projection_mesh_coords)
        if self.dedicated_qkv_ranks:
            q_a_raw = _replicate_from_tp_rank(q_a_raw, self.device, self.q_projection_rank, self.tp_size)
        elif self.balanced_qkv:
            q_a_raw = _gather_tp_width(q_a_raw, self.device)

        # ``None`` when the q_a matmul already normalized and mcast a replica onto
        # every q_b B core (see __init__).
        q_a = q_a_raw if self.q_a_norm is None else self.q_a_norm(q_a_raw)
        # The grouped epilogue needs q_b's one-row replicated-A path. A fused q_a mcast
        # already has that layout; otherwise replicate here. Unsupported q_b layouts
        # keep the tiled width-sharded activation and standalone per-head norm.
        q_b_input = self.q_b_proj.to_replicated_rm_hs_activation(q_a) if self.fuse_q_b_norm else q_a
        q = self.q_b_proj(q_b_input)  # [1, 1, B, H*Dh]
        if q_b_input is not q_a:
            ttnn.deallocate(q_b_input)
        assert self.fuse_q_b_norm, "q_b_norm must be fused"

        q = _apply_rope(q, cos, sin, self.rot, self.rope_dim, head_dim=self.head_dim)
        # kv_proj runs here rather than beside q_a_proj: one GCB is one FIFO, so a
        # prefetched matmul that runs out of turn pops another weight's page (see
        # ``prefetch_weights``). Reuse the same replicated activation; kv's B cores are a
        # subset of that grid. The caller still owns ``tokens`` (the compressor reads it).
        kv_raw = self.kv_proj(tokens, mesh_coords=self.kv_projection_mesh_coords)

        if self.dedicated_qkv_ranks:
            kv_raw = _replicate_from_tp_rank(kv_raw, self.device, self.kv_projection_rank, self.tp_size)
        elif self.balanced_qkv:
            kv_raw = _gather_tp_width(kv_raw, self.device)
        # ``None`` when the kv matmul normalized its own output (see __init__).
        kv = kv_raw if self.kv_norm is None else self.kv_norm(kv_raw)  # [1, 1, B, Dh]

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
        # One all-gather of the packed row onto the union of q_a / kv / compressor B cores.
        # matmul_decode's full-width hub mode reads that replica in place; a partial-width
        # weight unreplicates. The original width-sharded row is then free.
        gathered = ttnn.experimental.deepseek.all_gather_for_matmul(tokens, self._decode_activation_grid())
        if gathered is not tokens:
            ttnn.deallocate(tokens)
        tokens = gathered
        q, kv_new = self._qkv(tokens, cos, sin)  # q [1,1,B,H*Dh], kv_new [1,1,B,Dh]
        # kv_new = _one_row_per_user(kv_new)  # [1, B, 1, Dh] ROW_MAJOR, one core per user

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
                # Dense ring: causal over ``min(pos, W-1)`` when the caller passed
                # ``sdpa_cur_pos`` (see :func:`decode_sdpa_bounds`). The additive mask
                # remains as a fallback for callers that have not been switched over.
                out = self._attend(
                    q,
                    scache.sliding,
                    None if sdpa_cur_pos is not None else mask,
                    cos,
                    neg_sin,
                    sdpa_cur_pos=sdpa_cur_pos,
                )
            ttnn.deallocate(tokens)
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
        # ``q`` is packed ``[1, 1, B, H*Dh]`` (width-sharded over the head dim) and
        # nothing reads it until the SDPA below, while the compressor in between is the
        # step's L1 high-water mark. At a wide batch holding both at once is what leaves
        # an op's circular buffers nowhere to go, so park q in DRAM across the compressor
        # and bring it back before SDPA height-shards it. At batch 1 both fit and the
        # round trip is dead cost.
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
        ttnn.deallocate(tokens)
        if q_config is not None:
            q = ttnn.to_memory_config(q, q_config)
        out = self._attend(q, kv, mask, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos, paged=paged)
        return ttnn.reshape(out, [b, s, 1, d])
