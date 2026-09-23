from typing import Optional

import ttnn
import torch

from .common import DeepSeekV4Module, _HIFI4_SDPA, _MASK_NEG, _profile, _signpost, width_sharded_l1_config
from .decode_prefetch import (
    DECODE_LAYOUTS,
    KV_GCB,
    Q_A_GCB,
    ROUTER_GATE_GCB,
    check_decode_layout,
    decode_prefetch_page_bytes,
    ensure_named_gcb,
    kv_page_bytes,
    make_decode_prefetch_buffers,
    q_a_page_bytes,
    router_gate_page_bytes,
)
from .layers import (
    BatchedLinearDecode,
    DeepSeekV4RMSNorm,
    LinearDecode,
    _core_grid_contains,
)
from .paged_cache import PagedLayerView
from .system_config import active_system_config
from .weight_cache import WeightCache, _as_cache, _load_weight


def _decode_activation(layer: LinearDecode, x: ttnn.Tensor) -> ttnn.Tensor:
    """Match ``x`` to the ROW_MAJOR HEIGHT_SHARDED replica ``layer`` expects.

    Every projection this module builds is full-width hub mode (``use_rm_hs=True``), so
    the activation is a per-B-core replica of ``[M, K]`` -- handed back unchanged when
    the layer's own B grid already carries one, and otherwise reshared onto it. Returns
    that ``[M, K]`` ROW_MAJOR HEIGHT_SHARDED L1 replica, which the matmul reads as A.
    """
    if layer._is_replicated_rm_hs(x):
        a_grid = x.memory_config().shard_spec.grid
        b_grid = layer.b_core_grid()
        if a_grid == b_grid or _core_grid_contains(a_grid, b_grid):
            return x
    return layer.to_replicated_rm_hs_activation(x)


# ---------------------------------------------------------------------------- #
# DeepSeek-V4-Flash attention (decode, running KV cache)
#
# ttnn port of ``DeepseekV4Attention`` (and its CSA / HCA compressors) from
# ``modular_deepseek_v4.py``. Scope is *decode only*: each step appends the new
# token's K=V (and compressor projections) to the running cache and attends the
# tokens-so-far, via the fused ``paged_scaled_dot_product_attention_decode`` op.
#
# Letters: ``B`` users decoded per step, ``S`` query length (always 1 here), ``H`` /
# ``H_local`` attention heads (per TP rank), ``Dh`` head_dim, ``Rd`` qk_rope_head_dim,
# ``D`` hidden_size. V4 is shared-KV MQA (one KV head broadcast to all query heads) and
# lays each head out as ``[nope | rope]`` with interleaved RoPE on the trailing ``Rd``.
#
# A step decodes ``B`` users at once, all at the same absolute position, so the RoPE
# rows, the additive mask and the cache indices are shared -- what a batch stepped in
# lockstep from position 0 looks like. Users at *differing* positions would need
# per-user RoPE rows and masks, and are not supported.
#
# Two activation layouts appear throughout; which one a tensor is in matters because
# tiles are 32 rows tall:
#
#   * *packed rows* ``[1, 1, B, F]`` -- the B users on consecutive rows of one
#     tile-row. All per-token arithmetic (projections, norms, RoPE) runs here, so a
#     decode step costs one tile-row of work rather than B of them. This caps a step at
#     ``TILE_SIZE`` users.
#   * *per-user rows* ``[1, B, 1, F]`` / ``[B, 1, ..., F]`` -- one row per user, which
#     the KV-cache ops require (``paged_update_cache`` dispatches one user per core,
#     SDPA-decode indexes K/V by a leading batch). The surrounding block hands
#     ``hidden`` in as ``[B, S, 1, D]``.
#
# :func:`_pack_tokens` / :func:`_one_row_per_user` convert between them: view reshapes
# at ``B == 1``, relayouts above it.
#
# Two things bound ``B``, and the smaller one is not the one in the assert:
#
#   * the *layout* bound, ``TILE_SIZE`` (32), enforced by :func:`_pack_tokens`.
#   * an *L1* bound, well under that. Almost nothing here grows with ``B`` (the point of
#     the packed layout), except the query: SDPA-decode wants a head axis, so ``q`` is
#     width-sharded over ``B*H`` rows, ``B * H * Dh * 2`` bytes -- 64 KB per core at
#     ``B == 16``, ``head_dim 512``. Past a handful of users that crowds out SDPA-decode's
#     statically-allocated circular buffers and the op fails to build: measured on a
#     Blackhole grid at ``head_dim 512``, 8 users fit and 16 do not. It surfaces as
#     "statically allocated circular buffers ... clash with L1 buffers", not as a clean
#     batch-size error, so 8 is the supported ceiling until the query's residency is
#     reworked.
# ---------------------------------------------------------------------------- #
# KV / compressor cache (decode)
#
# Attention holds the stack's only cross-token state -- the hyper-connection streams,
# RMSNorms, routed/shared MoE and MLP are strictly per-token -- so a decode step only
# remembers, per decoder layer, the rotated sliding K=V entries (shared-KV MQA, K==V,
# capped to the ``sliding_window`` most recent tokens) and, for CSA / HCA, every source
# token's compressor projections (``kv`` / ``gate``). Those re-pool with the exact
# prefill pooling, so decode is bit-for-bit the same function of the tokens-so-far as a
# full prefill over them: no rolling-window / overlap / entry-count bookkeeping.
#
# That pool costs ``O(max_seq)`` rather than ``O(pos)``, so it runs only on the steps
# where it can change: a compressor emits an entry every ``compress_rate`` tokens and the
# additive block-bias exposes entries ``w < (pos+1)//compress_rate``, constant across the
# steps between two window closures. Pooling at each closure and reusing the result in
# between is therefore bit-identical to pooling every step, at ``1/compress_rate`` of the
# cost; callers drive the schedule via the ``pool`` flag (see
# ``DeepSeekV4Model._compressor_pool_due``). The pooled entries live in the layer's
# KV axis, after the sliding ring.
#
# The compressors themselves -- CSA's Ca/Cb overlap and HCA's single-window pooling, with
# the window-buffer writers they need -- live in :mod:`.attention_csa` and
# :mod:`.attention_hca`; this module keeps everything they share (the cache class, the
# packed-row / one-row-per-user layouts, RoPE, the in-place cache writers, the SDPA bounds
# and the block itself) and dispatches to them through :func:`_compressor_class`.
#
# Where a layer's KV lives depends on its type. Sliding and CSA layers keep it in one
# fixed-size DRAM-interleaved buffer, ``_StaticLayerCache.kv``, capped at the most
# entries the layer can ever attend: the ``sliding_window`` ring for sliding layers,
# and the ring plus ``CSA_MAX_COMPRESSED_ENTRIES`` compressed entries for CSA (which
# caps a CSA model's context at ``CSA_MAX_COMPRESSED_ENTRIES * compress_rate`` tokens).
# HCA layers read and write theirs through a paged block pool (see :mod:`.paged_cache`)
# and the active sessions' page table, following the GPT-OSS / tt-transformers paged-KV
# pattern (see the traced-decode banner below).
#
# TODO: the dense buffers are batch 1 only, and live in DRAM; move them to L1.
# ---------------------------------------------------------------------------- #
CSA_MAX_COMPRESSED_ENTRIES = 512
PAGED_KV_LAYER_TYPES = ("heavily_compressed_attention",)


def dense_kv_rows(layer_type: str, sliding_window: int) -> Optional[int]:
    """Rows of a layer's dense KV buffer, or ``None`` for a paged (HCA) layer."""
    if layer_type == "sliding_attention":
        return sliding_window
    if layer_type == "compressed_sparse_attention":
        return sliding_window + CSA_MAX_COMPRESSED_ENTRIES
    return None


def dense_kv_context_limit(layer_types, compress_rates: dict) -> Optional[int]:
    """Longest context (in tokens) the dense CSA buffers can hold for ``layer_types``,
    or ``None`` when there is no CSA layer."""
    if "compressed_sparse_attention" not in set(layer_types):
        return None
    return CSA_MAX_COMPRESSED_ENTRIES * compress_rates["compressed_sparse_attention"]


class _StaticLayerCache:
    """Fixed-size, in-place per-layer KV and compressor window buffers.

    Everything here is per-session state, written in place at the new token's slot (a
    device-tensor index) so the same trace serves every step:

      * ``kv`` -- sliding and CSA layers only: the layer's KV, TILE DRAM
        ``[1, 1, rows, Dh]`` with ``rows`` from :func:`dense_kv_rows`. HCA layers keep
        their KV in the shared block pool (:class:`~.paged_cache.PagedLayerView`), so
        this is ``None`` there.

      * ``win_kv`` / ``win_gate`` -- the compressor projections of the window
        *currently being filled*, at slot ``pos % compress_rate``. HCA keeps these as
        TILE DRAM ``[B, 1, compress_rate, Dh]`` for ``paged_update_cache``. CSA keeps
        them (and ``prev_*``) as ROW_MAJOR L1 WIDTH_SHARDED
        ``[B*compress_rate, 1, 1, 2*Dh]`` so ``csa_pool_window`` can consume them
        in place. Only one window is held, because pooling is incremental.
        ``None`` for sliding-only layers.
      * ``prev_kv`` / ``prev_gate`` -- CSA only: previous window's projections
        (same layout as CSA ``win_*``), because entry ``w`` also needs window
        ``w-1``'s Ca slice. Refreshed from ``win_*`` after each pool.
        ``prev_gate`` starts at ``_MASK_NEG`` so window 0's absent Ca half carries
        softmax weight 0. ``None`` for HCA and sliding-only layers.

    A CSA/HCA layer's KV axis holds both regions SDPA attends: the sliding ring
    in rows ``[0, window)`` and the pooled (normed, RoPE'd) compressed entries in rows
    ``[window, ...)``. The ring slot is ``pos % window`` and each pooled entry goes in
    at row ``window + w``, both by the same in-place write. That the sliding
    region comes *first* is also what makes the valid set a contiguous prefix once the
    ring is full, and hence causal SDPA possible.

    Built empty by :func:`build_static_layer_cache`; the prompt is written in by
    replaying decode one token at a time.
    """

    __slots__ = (
        "kv",
        "win_kv",
        "win_gate",
        "prev_kv",
        "prev_gate",
        "idx_win_kv",
        "idx_win_gate",
        "idx_prev_kv",
        "idx_prev_gate",
        "idx_key_cache",
    )

    def __init__(
        self,
        win_kv: Optional[ttnn.Tensor] = None,
        win_gate: Optional[ttnn.Tensor] = None,
        prev_kv: Optional[ttnn.Tensor] = None,
        prev_gate: Optional[ttnn.Tensor] = None,
        idx_win_kv: Optional[ttnn.Tensor] = None,
        idx_win_gate: Optional[ttnn.Tensor] = None,
        idx_prev_kv: Optional[ttnn.Tensor] = None,
        idx_prev_gate: Optional[ttnn.Tensor] = None,
        idx_key_cache: Optional[ttnn.Tensor] = None,
        kv: Optional[ttnn.Tensor] = None,
    ):
        """Store the pre-built buffers; each is ``None`` on a layer type that does not use it.

        Shapes and layouts are the class docstring's: ``win_kv`` / ``win_gate`` /
        ``prev_kv`` / ``prev_gate`` are the compressor windows (HCA TILE DRAM
        ``[B, 1, cr, Dh]``; CSA ROW_MAJOR L1 WIDTH_SHARDED ``[B*cr, 1, 1, 2*Dh]``).
        """
        self.kv = kv
        self.win_kv = win_kv
        self.win_gate = win_gate
        self.prev_kv = prev_kv
        self.prev_gate = prev_gate
        self.idx_win_kv = idx_win_kv
        self.idx_win_gate = idx_win_gate
        self.idx_prev_kv = idx_prev_kv
        self.idx_prev_gate = idx_prev_gate
        self.idx_key_cache = idx_key_cache


def build_static_layer_cache(
    device: ttnn.MeshDevice,
    layer_type: str,
    head_dim: int,
    max_seq: int,
    compress_rates: dict,
    sliding_window: int,
    batch: int = 1,
    index_head_dim: Optional[int] = None,
) -> _StaticLayerCache:
    """Allocate a layer's dense KV and compressor window buffers empty, for ``batch`` users.

    ``head_dim`` is ``Dh`` and ``max_seq`` sizes the indexer key cache
    (``max_seq // compress_rate`` rows). The buffers come back in the layouts the class
    docstring lists: TILE DRAM ``[1, 1, rows, Dh]`` for a sliding / CSA layer's ``kv``,
    TILE DRAM ``[batch, 1, rows, width]`` for HCA's window pair, ROW_MAJOR L1
    WIDTH_SHARDED ``[batch*compress_rate, 1, 1, 2*Dh]`` for CSA's. An HCA layer's KV
    lives in the paged block pool (see :mod:`.paged_cache`).
    """

    def _filled(rows: int, width: int, value: float = 0.0) -> ttnn.Tensor:
        """``value``-filled (0.0 by default) TILE DRAM ``[batch, 1, rows, width]`` bf16 cache."""
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
        32-column shard each (32 cores at CSA ``2*Dh == 1024``). Rows sit on dim 0
        (user-major packed ``B*cr``) so ``paged_update_cache`` can write a decode
        token as a linear row (see :func:`~.attention_csa._update_window_at`).
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

    kv = None
    rows = dense_kv_rows(layer_type, sliding_window)
    if rows is not None:
        if batch != 1:
            raise NotImplementedError(f"dense {layer_type} KV supports batch 1 only, got batch={batch}")
        kv = _filled(rows, head_dim)
    if layer_type == "sliding_attention":
        return _StaticLayerCache(kv=kv)
    win_kv = win_gate = prev_kv = prev_gate = None
    idx_win_kv = idx_win_gate = idx_prev_kv = idx_prev_gate = idx_key_cache = None
    cr = compress_rates[layer_type]
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
    if is_csa and index_head_dim:
        idx_feat = 2 * index_head_dim
        idx_win_kv = _csa_window(cr, idx_feat)
        idx_win_gate = _csa_window(cr, idx_feat)
        idx_prev_kv = _csa_window(cr, idx_feat)
        idx_prev_gate = _csa_window(cr, idx_feat, _MASK_NEG)
        n_win = max(max_seq // cr, 0)
        # Replicated on every TP rank. A sequence shard would make the cache
        # write's global window row illegal once ``start_pos`` passes the
        # local piece, and decode's query is one replicated token, not a
        # sequence shard the ring scorer expects. Pad so ``T - 32`` is
        # tile-aligned; top-k's valid length stops at closed windows.
        align = ttnn.TILE_SIZE
        t_alloc = max(align, ((n_win + align - 1) // align) * align)
        idx_key_cache = _filled(t_alloc, index_head_dim)
    return _StaticLayerCache(
        win_kv,
        win_gate,
        prev_kv,
        prev_gate,
        idx_win_kv,
        idx_win_gate,
        idx_prev_kv,
        idx_prev_gate,
        idx_key_cache,
        kv=kv,
    )


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
    """The cached per-tile ``rotate_half`` for ``rot`` ``[Rd, Rd]``, derived on first use.

    Returns the top-left ``[1, 1, TILE_SIZE, TILE_SIZE]`` tile, DRAM-interleaved (the fused
    op reads ``trans_mat`` from a DRAM source).

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


def _apply_rope(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    rot: ttnn.Tensor,
    rope_dim: int,
    head_dim: int | None = None,
) -> ttnn.Tensor:
    """Interleaved RoPE on each ``head_dim``-wide block of ``x`` ([.., D]).

    ``cos`` / ``sin`` are ``[1,1,L,rope_dim]`` DRAM tables (broadcast over batch/heads);
    ``rot`` is the ``[rope_dim, rope_dim]`` ``rotate_half`` matrix. Leading "nope"
    channels of each block pass through untouched. ``head_dim`` defaults to the
    last dim of ``x`` (one block). Packed heads (``D = H * head_dim``) are split
    on device. Returns ``x``'s shape, rotated on the trailing ``rope_dim`` of each block.

    Delegates the whole calc to the fused ``ttnn.experimental.fused_partial_rope`` device
    op, so ``x`` has to arrive width-sharded in L1 (ROW_MAJOR ``x`` is computed as 1x32
    faces). ``cos`` / ``sin`` / ``trans_mat`` are DRAM-interleaved (the reader streams
    each core's rope tiles), and the op reads one cos/sin tile-row per core or a single
    tile-row broadcast across all rows on device -- which is how one shared decode
    position serves every head.

    Batched inputs here are ``[1, B, H, Dh]`` (SDPA-decode's head layout), whose ``B*H``
    rows are contiguous because ``H`` is tile-aligned: the op counts rows the same way
    off the shard, so nothing has to be folded onto dim -2.
    """
    _signpost("apply_rope start")
    d = x.shape[-1]
    hd = d if head_dim is None else head_dim
    assert d % hd == 0, f"last dim {d} is not a multiple of head_dim={hd}"

    out_sh = ttnn.experimental.fused_partial_rope(x, cos, sin, _trans_mat_for(rot), rope_dim, head_dim=hd)
    _signpost("apply_rope end")
    return out_sh


# ---------------------------------------------------------------------------- #
# Traced-decode helpers (fixed-size, in-place KV cache via ``paged_update_cache``)
#
# A reusable ``ttnn`` trace requires fixed tensor shapes / addresses and no host
# round-trips inside the captured region, so the traced decode uses fixed-size DRAM
# buffers written *in place* every step at the new token's position (a device-tensor
# index, so the same trace serves every step). ``paged_update_cache`` is the canonical
# trace-safe in-place KV writer: it mutates the persistent cache buffer during capture,
# unlike ``ttnn.copy`` which is rejected mid-capture.
# ---------------------------------------------------------------------------- #
def _sdpa_decode_output_config(batch: int, heads: int, head_dim: int, grid_size: ttnn.CoreCoord) -> ttnn.MemoryConfig:
    """Native height-sharded output of ``sdpa_decode``: one reducer core per batch user.

    The writer has ``num_output_cores = B`` and places those reducers on the first
    ``B`` cores of the program grid in row-major order
    (``{idx % grid.x, idx / grid.x}``). Each core holds that user's full Q-head
    axis, so the shard is ``[H, Dh]`` (ROW_MAJOR). Matching this spec lets the output
    CB alias the result buffer; any other grid or a width/block shard is either
    rejected or a reshard.
    """
    grid = ttnn.num_cores_to_corerangeset(batch, grid_size, row_wise=True)
    shard_spec = ttnn.ShardSpec(grid, [heads, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _check_update_row(row: ttnn.Tensor) -> None:
    """``paged_update_cache`` takes its ``[1, B, 1, F]`` row ROW_MAJOR, height-sharded one
    user per core (:func:`_height_sharded_l1_config`). It is written as it stands rather
    than resharded, since a copy here would be paid on every cache write."""
    num_users, width = row.shape[1], row.shape[-1]
    expected = _height_sharded_l1_config(num_users, width, row.device(), layout=row.layout)
    assert (
        row.memory_config() == expected
    ), f"paged_update_cache needs its row one user per core: expected {expected}, got {row.memory_config()}"


def _update_cache_at(cache: ttnn.Tensor, row: ttnn.Tensor, pos_tensor: ttnn.Tensor) -> None:
    """In-place write ``row`` ``[1, B, 1, F]`` into a per-layer window buffer ``cache``
    at ``pos_tensor`` ``[B]`` (INT32). ``row`` stays the caller's to free."""
    _check_update_row(row)
    ttnn.experimental.paged_update_cache(cache, row, update_idxs_tensor=pos_tensor)


def _update_kv_at(kv: "PagedLayerView | ttnn.Tensor", row: ttnn.Tensor, pos_tensor: ttnn.Tensor) -> None:
    """In-place write ``row`` ``[1, B, 1, Dh]`` into a layer's KV at row ``pos_tensor``
    ``[B]`` (INT32): straight into a dense ``[1, 1, rows, Dh]`` buffer, or into
    ``kv.pool`` at that logical position through the active sessions' page table.

    ``kv.position_modulo`` wraps the logical position into a bounded capacity
    before the page-table lookup, which is what makes a sliding-window session need
    only ``window / block_size`` blocks; without it any position past that capacity
    resolves through the row's unmapped tail (see :mod:`.paged_cache`). ``row`` stays
    the caller's to free.
    """
    if isinstance(kv, ttnn.Tensor):
        _update_cache_at(kv, row, pos_tensor)
        return
    _check_update_row(row)
    ttnn.experimental.paged_update_cache(
        kv.pool,
        row,
        update_idxs_tensor=pos_tensor,
        page_table=kv.page_table,
        cache_position_modulo=kv.position_modulo,
    )


def _compressor_class(layer_type: str):
    """The compressor class for ``layer_type``, or ``None`` for a sliding-only layer.

    The two compressors live in :mod:`.attention_csa` / :mod:`.attention_hca`, which import this
    module's shared decode helpers (``_decode_activation``, ``_one_row_per_user``,
    ``_apply_rope``, ``_update_cache_at``, ``_compressor_projections``, ...). A module-level
    import here would therefore be a cycle, so it is deferred to this call: by the time a model
    is built every one of those helpers exists, whichever of the three modules was imported
    first.
    """
    if layer_type == "compressed_sparse_attention":
        from .attention_csa import DeepSeekV4CSACompressor

        return DeepSeekV4CSACompressor
    if layer_type == "heavily_compressed_attention":
        from .attention_hca import DeepSeekV4HCACompressor

        return DeepSeekV4HCACompressor
    return None


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
    *,
    feat: Optional[int] = None,
    layout_name: Optional[str] = None,
    weight_prefix: str = "compressor",
):
    """The compressor's ``(kv_proj, gate_proj)``, both projecting the block's ``hidden``.

    Returns two :class:`~.layers.LinearDecode` per compressor: a full-width hub-mode
    ``matmul_decode`` taking the decode all-gather replica as A (``[T, D]`` ROW_MAJOR
    HEIGHT_SHARDED, ``T`` = shard height) against a DRAM ND-sharded weight ``[D, N]``,
    with ``N = Dh`` for HCA and ``2*Dh`` for CSA's Ca/Cb pair -- which is why the layout
    is keyed by ``layer_type``.     The CSA Lightning Indexer reuses this helper at
    ``index_head_dim`` (``layout_name="indexer.kv_proj"``, ``N = 2*128``). Under the
    prefetcher CSA rides q_a's 32-receiver FIFO, the indexer's 8-core kv/gate pair
    rides the router gate's 8-receiver FIFO (same ``[4096, 256]`` cut), and HCA
    shares kv's 16-receiver FIFO, queued after those projections in the block's turn
    (see ``decode_prefetch``).
    """
    if feat is None:
        feat = config.head_dim * (2 if layer_type == "compressed_sparse_attention" else 1)
    layout_name = layout_name or layer_type
    layout = dict(check_decode_layout(layout_name, config.hidden_size, feat))
    if use_prefetcher and prefetch_buffers is None:
        prefetch_buffers = make_decode_prefetch_buffers(device, weight_dtype, num_prefetch_pages)
    prefetch = {"use_prefetcher": use_prefetcher}
    if use_prefetcher:
        # Same cut as q_a (32 cores) / kv (16 cores) / router gate (8 cores), so the
        # same rings. Queued after those projections in
        # :meth:`DeepSeekV4Attention.prefetch_weights`. Indexer kv/gate is the
        # router-gate cut (``n_blocks=8``) and cannot ride q_a's 32-receiver ring.
        if layout_name == "indexer.kv_proj":
            prefetch["global_cb"] = ensure_named_gcb(
                prefetch_buffers, ROUTER_GATE_GCB, device, [DECODE_LAYOUTS["router_gate"]], weight_dtype
            )
            prefetch["global_cb_page_bytes"] = router_gate_page_bytes(weight_dtype)
        elif layout_name == "compressed_sparse_attention" or layer_type == "compressed_sparse_attention":
            prefetch["global_cb"] = ensure_named_gcb(
                prefetch_buffers, Q_A_GCB, device, [DECODE_LAYOUTS["q_a_proj"]], weight_dtype
            )
            prefetch["global_cb_page_bytes"] = q_a_page_bytes(weight_dtype)
        else:
            prefetch["global_cb"] = ensure_named_gcb(
                prefetch_buffers, KV_GCB, device, [DECODE_LAYOUTS["kv_proj"]], weight_dtype
            )
            prefetch["global_cb_page_bytes"] = kv_page_bytes(weight_dtype)

    def projection(name):
        """``LinearDecode`` for ``{weight_prefix}.<name>.weight`` ``[D, N]``."""
        return LinearDecode(
            weights[f"{weight_prefix}.{name}.weight"],
            device,
            cache.file(f"{weight_prefix}.{name}.full"),
            dtype=weight_dtype,
            **layout,
            **prefetch,
            rectangle_b_grid=True,
            use_rm_hs=True,
        )

    kv_proj = projection("kv_proj")
    gate_proj = projection("gate_proj")
    assert kv_proj._can_matmul_decode_rm_hs() and gate_proj._can_matmul_decode_rm_hs(), (
        f"{layer_type} kv/gate must use ROW_MAJOR HEIGHT_SHARDED matmul_decode, "
        f"but kv partial={kv_proj.partial_width_sharded} gate partial={gate_proj.partial_width_sharded}"
    )
    return kv_proj, gate_proj


def _tp_cluster_axis(device: ttnn.MeshDevice) -> int:
    """Mesh axis of a 1xN (or flattened N-device) tensor-parallel group.

    ``1`` for a 2-D ``[1, TP]`` mesh, else ``0``: a flattened mesh keeps its ranks on
    axis 0.
    """
    shape = tuple(device.shape)
    return 1 if len(shape) == 2 and shape[1] > 1 else 0


class DeepSeekV4Attention(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4Attention`` (decode only, running KV cache).

    Construct from a ``config`` (the HF ``DeepseekV4Config`` or any object exposing the
    same attributes), the layer's torch ``weights`` (HF-named ``state_dict`` entries),
    and a device. :meth:`decode` / :meth:`decode_static` consume pre-built RoPE tables
    (see :func:`make_rope_table`); these are inputs because the rotary embedding is owned
    by the surrounding model in the reference, not by the attention block.

    ``tp_size > 1`` expects a 1xTP mesh and replicated hidden/KV inputs. q_a and kv
    stay full-width (replicated) on every rank, so neither needs an all-gather, while
    ``q_b`` is head-sharded across the ranks. Query heads and complete output groups are
    sharded across the mesh. Decode is ``M == 1``, so ``o_a`` is a batched
    ``matmul_decode`` over the local groups (the group-major permute is a no-op), and
    ``o_b`` is row-parallel: it consumes those local groups and all-reduces the
    full-hidden partials.

    ``use_prefetcher=True`` (what the model always passes) switches the decode projections
    that fit the shared 64-receiver GCB (q_b, batched o_a, row-parallel o_b) onto
    DRISC-prefetched weights. The compressor pair rides q_a's 32-core ring (CSA) or kv's
    16-core ring (HCA). Each prefetched weight stays DRAM ND-sharded and the tensor
    prefetcher pushes it into the matmul's in1 buffer, instead of copying DRAM -> L1 before
    every call. Two things come with it:

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
        tp_size: int = 1,
    ):
        """Build the projections, norms, sinks and compressor of layer ``layer_idx``.

        ``weights`` is the layer's HF-named torch ``state_dict`` (``q_a_proj``,
        ``q_b_proj``, ``kv_proj``, ``o_a_proj``, ``o_b_proj``, ``q_a_norm``, ``kv_norm``,
        ``sinks``, ...), each projection weight ``[K, N]`` (DRAM ND-sharded under the
        prefetcher); ``device`` is a 1xTP mesh when ``tp_size > 1``, with
        ``num_attention_heads`` and ``o_groups`` split evenly across the ranks.
        ``prefetch_buffers`` is the device-wide GCB set from
        :func:`~.decode_prefetch.make_decode_prefetch_buffers`.
        """
        # SDPA program config and the prefetch ring depth come from the system profile
        # unless the caller pinned them.
        sys_cfg = system_config or active_system_config()
        self.system_config = sys_cfg
        if num_prefetch_pages is None:
            num_prefetch_pages = sys_cfg.prefetcher.num_prefetch_pages
        self.use_prefetcher = use_prefetcher
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        if self.num_heads % tp_size:
            raise ValueError(f"num_attention_heads {self.num_heads} is not divisible by tp_size {tp_size}")
        if tp_size > 1 and device.get_num_devices() != tp_size:
            raise ValueError(
                f"tensor-parallel attention expects one device per TP rank, got tp_size={tp_size} "
                f"on a {device.get_num_devices()}-device mesh"
            )
        self.tp_size = tp_size
        self.local_num_heads = self.num_heads // tp_size
        # q_a and kv stay full-width (replicated) on every rank of the 1xTP stage, while
        # q_b stays head-sharded; o_b is row-parallel, reducing its full-hidden partials.
        self.qkv_tp_strategy = "replicated"
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

        if use_prefetcher and prefetch_buffers is None:
            prefetch_buffers = make_decode_prefetch_buffers(device, weight_dtype, num_prefetch_pages)

        def projection(name, weight=None, cache_suffix="", rectangle_b_grid=False):
            """``LinearDecode`` for ``weights[name].weight`` ``[K, N]`` DRAM ND-sharded.

            TP-sharded by the branches below (q_b head-sharded, o_b K-split, q_a/kv
            replicated). Every prefetched projection has a ring that matches its B-core
            count: q_a and kv get their own full-width rings, the rest share the
            64-receiver decode GCB.
            """
            prefetch = {"use_prefetcher": use_prefetcher}
            if use_prefetcher:
                if name == "q_a_proj":
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
            layout = dict(DECODE_LAYOUTS[name])
            mapper = None
            cache_name = name
            shard_projection = name == "q_b_proj"
            if shard_projection and tp_size > 1:
                # Cut the full [K, N] host tensor into contiguous query-head ranges.
                layout["N"] //= tp_size
                mapper = ttnn.ShardTensorToMesh(device, dim=-1)
                cache_name = f"{name}.tp{tp_size}.{self.qkv_tp_strategy}"
            elif name == "o_b_proj":
                layout["K"] //= tp_size
                mapper = ttnn.ShardTensorToMesh(device, dim=-2)
                cache_name = f"{name}.tp{tp_size}.row"
            elif tp_size > 1 and name in ("q_a_proj", "kv_proj"):
                # Full-width replica on every rank. q_b stays in the shard branch above.
                mapper = ttnn.ReplicateTensorToMesh(device)
                cache_name = f"{name}.tp{tp_size}.{self.qkv_tp_strategy}"
            if name == "q_a_proj":
                cache_name += ".full"
                # q_a's producer grid must be a subset of q_b's filled output-mcast
                # rectangle. The generic row-wise 32-core set can be ragged on Blackhole.
                rectangle_b_grid = True
            elif name == "kv_proj":
                cache_name += ".full"
            return LinearDecode(
                weights[f"{name}.weight"] if weight is None else weight,
                device,
                cache.file(cache_name + cache_suffix),
                dtype=weight_dtype,
                mesh_mapper=mapper,
                **layout,
                **prefetch,
                rectangle_b_grid=rectangle_b_grid,
                use_rm_hs=True,
            )

        self.q_a_proj = projection("q_a_proj")
        self.kv_proj = projection("kv_proj")
        # One replica of kv on core (0,0): the writer unicasts producer slices into that
        # dest bbox.
        self.kv_proj.set_output_core_grid(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        )
        # q_a's RMSNorm rides in the q_a matmul epilogue: q_lora is a whole number of
        # tiles and q_a is full-width hub mode, so the op can always take it. The dest of
        # that mcast must be a filled rectangle (NOC multicast) that also holds q_b's B
        # cores; the generic row-wise 64-core set can be ragged on Blackhole.
        self.fuse_q_a_norm = self.q_a_proj.can_fuse_rms_norm()
        if self.fuse_q_a_norm:
            self.q_b_proj = projection("q_b_proj", cache_suffix=".rect", rectangle_b_grid=True)
        else:
            self.q_b_proj = projection("q_b_proj")
        self.o_b_proj = projection("o_b_proj", cache_suffix=".rect", rectangle_b_grid=True)
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
        self.fuse_kv_norm = self.kv_proj.can_fuse_rms_norm()
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
        if tp_size > 1:
            o_a_layout = {
                **o_a_layout,
                "b_blocks": o_a_layout["b_blocks"] // tp_size,
                "n_blocks": o_a_layout["n_blocks"] * tp_size,
            }
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
            preprocess=lambda w: w.reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).contiguous(),
            **o_a_prefetch,
        )
        # Mcast the folded ``[1, g * o_lora_rank]`` result onto o_b's B-grid bounding box so
        # o_b can consume it as replicated ROW_MAJOR HEIGHT_SHARDED A. GCB receiver sets can
        # omit cores inside that box; NOC mcast still requires a filled rectangle.
        bbox = self.o_b_proj.b_core_grid().bounding_box()
        self.o_a_proj.set_output_core_grid(ttnn.CoreRangeSet({bbox}))

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
        # allocated and so have to fit under whatever L1 buffers are live at the call. At 4
        # the CB region overruns them and the program fails to build; the term above is
        # linear in ``cores_per_head - 1``, so dropping to 2 cuts that scratch CB to a
        # third of what 4 asks for, at the cost of splitting the KV reduction 2 ways
        # instead of 4.
        self._sdpa_pcfg = sys_cfg.attention.sdpa_program_config(device)

        # The rotate-half matrix must stay precise (a bf4 rotation would corrupt RoPE).
        self.rot = _load_weight(_interleaved_rotate_matrix(self.rope_dim), device, cache_file_name=cache.file("rot"))
        compressor_cls = _compressor_class(self.layer_type)
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
                tp_size=tp_size,
            )
            if compressor_cls is not None
            else None
        )
        self._sparse_sink = None
        self._sparse_full_heads = False
        indexer = getattr(self.compressor, "indexer", None)
        if indexer is not None:
            self._init_sparse_sink()

    def prefetch_weights(self, *, index_sparse: bool = False):
        """Stage this block's projection weights ahead of the :meth:`decode` that uses them.

        Queues every projection configured for a GCB. The weights stay DRAM ND-sharded
        ``[K, N]`` pages; FIFO order is part of the contract and
        is checked nowhere -- a matmul that runs out of turn pops another weight's page and
        computes wrong results rather than erroring -- so each ring is queued in the order
        :meth:`decode` consumes it. On the shared 64-receiver GCB that is q_b, then the
        indexer's q_b when scoring, then ``_attend``'s o_a before o_b. On q_a's private
        32-receiver ring it is q_a, then CSA's kv/gate. On kv's private 16-receiver ring it
        is kv, then HCA's kv/gate. The MoE shared expert's gate/up follow CSA on q_a's ring
        and its down follows o_b on the shared buffer (queued from
        :meth:`DeepSeekV4SparseMoeBlock.prefetch_weights`).

        One prefetcher socket serves every ring, and it does not read the next request
        while the current one is waiting for ring space. o_a and o_b are not popped until
        ``_attend``, so queueing them before the compressor leaves the compressor's readers
        parked in ``remote_cb_wait_front``. The compressor rings are therefore queued
        before those two, which does not change either ring's own order.

        ``index_sparse`` also stages the CSA indexer's score projections. Key compression
        is staged whenever the indexer is attached; the score pages are not, on the
        short-sequence trace that never pops them.
        """
        for proj in (self.q_a_proj, self.q_b_proj, self.kv_proj):
            if proj.use_prefetcher:
                proj.fetch_weights()
        if self.compressor is not None:
            self.compressor.prefetch_weights(index_sparse=index_sparse)
        # Indexer q_b shares q_b's 64-receiver ring. It is consumed after attention's
        # q_b and before o_a, so it has to sit between them in that ring's FIFO.
        indexer = getattr(self.compressor, "indexer", None)
        if index_sparse and indexer is not None and indexer.q_b_proj.use_prefetcher:
            indexer.q_b_proj.fetch_weights()
        for proj in (self.o_a_proj, self.o_b_proj):
            if proj.use_prefetcher:
                proj.fetch_weights()

    def _sdpa_decode(
        self,
        q: ttnn.Tensor,
        kv: "PagedLayerView | ttnn.Tensor",
        mask: ttnn.Tensor | None,
        cur_pos: ttnn.Tensor | None = None,
        sliding_window: int | None = None,
    ) -> ttnn.Tensor:
        """Single-token (``S == 1``) attention over the batch via the fused SDPA-decode op.

        The decode path's only attention primitive: fuses the scale, the masking, the
        per-head sink, and both matmuls into one device op.

        ``q`` is packed ``[1, 1, B, H_local*Dh]`` WIDTH_SHARDED L1 from :meth:`_qkv`;
        ``all_gather_for_matmul`` stitches the width shards and multicasts a ROW_MAJOR
        HEIGHT_SHARDED replica onto the SDPA reducer cores, and a view then exposes the
        op's head layout ``[1, B, H_local, Dh]``, which is also what it emits -- so no
        head/seq transposes around the call. K == V (MQA, one KV head) is either the
        layer's dense ``[1, 1, rows, Dh]`` buffer or its block pool ``kv.pool``, read
        through the active sessions' page table.

        Two mutually exclusive ways to bound the KV axis (the op rejects an ``attn_mask``
        in causal mode, so this is a real branch):

        * ``cur_pos`` ``[B]`` INT32 -- causal. The kernel derives its chunk range from the
          position and never reads or computes chunks past it, so cost tracks the *actual*
          position rather than the ``max_seq``-sized axis. Requires a contiguous-prefix
          valid set, and is exact mid-chunk because the kernel generates a partial mask for
          the final chunk.
        * ``mask`` ``[1, 1, 1, Skv]`` additive (``0`` valid / ``_MASK_NEG`` masked) -- the
          fallback for steps whose valid set has a hole. The mask is *data*, not control
          flow, so the kernel always walks the whole axis. It has to carry Q's (padded)
          head count, hence the broadcast across ``H`` that the causal path avoids; its
          leading dim stays 1, so the op broadcasts it over the batch, which the users of
          a step can share because they all sit at the same position.

        A bounded paged ring (``kv.position_modulo``) also passes ``sliding_window_size`` so
        the kernel attends the last ``window`` positions rather than the whole (wrapped)
        capacity.

        Under tensor parallelism Q and the per-head sink are sharded on the head axis while
        the shared MQA KV cache, positions, page table and mask are replicated, so each rank
        runs SDPA for ``H / TP`` heads independently and no collective is needed here. The
        result stays head-sharded through output RoPE and the group-local ``o_a``;
        :meth:`_grouped_output` gathers those groups before the global ``o_b`` mix, then the
        N/TP ``o_b`` partials are all-reduced back to a replicated hidden state.

        The op's cheapest sharded output is height-sharded L1 on ``B`` cores (one reducer
        per user) with shard ``[H_local, Dh]``; see :func:`_sdpa_decode_output_config`.
        """
        h, dh = self.local_num_heads, self.head_dim
        batch = _packed_users(q)
        grid_size = self._sdpa_pcfg.compute_with_storage_grid_size
        out_mem = _sdpa_decode_output_config(batch, h, dh, grid_size)
        gathered = ttnn.experimental.deepseek.all_gather_for_matmul(q, out_mem.shard_spec.grid)
        if gathered is not q:
            ttnn.deallocate(q)
        q = ttnn.experimental.view(gathered, [1, batch, h, dh])
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
        if isinstance(kv, ttnn.Tensor):
            return ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                kv,
                kv,  # K == V (shared single KV head)
                attention_sink=self.sdpa_sinks_tt,
                scale=self.scaling,
                program_config=self._sdpa_pcfg,
                compute_kernel_config=_HIFI4_SDPA,
                memory_config=out_mem,
                **bounds,
            )  # [1, B, H, Dh] height-sharded L1
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv.pool,
            kv.pool,  # K == V (shared single KV head)
            kv.page_table,
            sliding_window_size=sliding_window,
            cache_position_modulo=kv.position_modulo,
            attention_sink=self.sdpa_sinks_tt,
            scale=self.scaling,
            program_config=self._sdpa_pcfg,
            compute_kernel_config=_HIFI4_SDPA,
            memory_config=out_mem,
            **bounds,
        )  # [1, B, H, Dh] height-sharded L1

    def _grouped_output(self, attn: ttnn.Tensor) -> ttnn.Tensor:
        """``DeepseekV4GroupedLinear`` (o_a) + ``o_b_proj``.

        ``attn`` is SDPA-decode's ``[1, B, H, Dh]``; returns the block's hidden output
        back on packed rows, ``[1, 1, B, D]``. ``o_a`` is block-diagonal over
        ``o_groups``, and ``o_b_proj`` then mixes the groups back to hidden.

        Decode is a single token (``M == 1``), so ``[1, 1, H, Dh]`` is already
        group-major in memory: each group's ``K = H*Dh / g`` features are a
        contiguous slice. ``all_gather_for_matmul`` multicasts that ``[g, K]``
        replica from the (single-user) SDPA core onto ``o_a``'s weight grid so
        batched ``matmul_decode`` can consume it as ROW_MAJOR HEIGHT_SHARDED A.
        A view then names the group batch ``[1, g, num_cores, K]``.

        With TP, each rank owns a contiguous set of complete groups. ``o_a`` is
        consequently group-sharded and runs locally, and the row-parallel ``o_b``
        consumes those local groups, computes full-N partials and all-reduces them.
        """
        _, m, h, dh = attn.shape
        groups = self.local_o_groups
        in_per_group = (h * dh) // groups
        assert m == 1, f"batched o_a is decode-only (M == 1), got M={m}"
        oa_grid = self.o_a_proj.b_core_grid()
        oa_dest = ttnn.CoreRangeSet({oa_grid.bounding_box()})
        gathered = ttnn.experimental.deepseek.all_gather_for_matmul(attn, oa_dest)
        if gathered is not attn:
            ttnn.deallocate(attn)
        attn = ttnn.experimental.view(gathered, [1, groups, oa_dest.num_cores(), in_per_group])
        y = self.o_a_proj(attn)
        y = ttnn.experimental.view(y, [1, 1, y.shape[-2], groups * self.o_lora_rank])
        output = self.o_b_proj(_decode_activation(self.o_b_proj, y))
        if self.tp_size > 1:
            gathered = ttnn.experimental.deepseek.width_sharded_all_reduce(
                output,
                cluster_axis=_tp_cluster_axis(self.device),
                num_links=2,
                topology=ttnn.Topology.Linear,
            )
            ttnn.deallocate(output)
            output = gathered
        return output

    def _attend(
        self,
        q: ttnn.Tensor,
        kv: "PagedLayerView | ttnn.Tensor",
        mask: ttnn.Tensor | None,
        cos: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        sdpa_cur_pos: ttnn.Tensor | None = None,
        sliding_window: int | None = None,
    ) -> ttnn.Tensor:
        """Fused SDPA-decode + output RoPE + grouped output projection.

        Tail of :meth:`decode_static`: ``q`` packed ``[1, 1, B, H*Dh]`` WIDTH_SHARDED L1,
        the layer's KV (dense buffer or block pool) as the shared K==V, and either ``sdpa_cur_pos`` or the
        additive ``mask`` ``[1,1,1,Skv]`` -> the block's hidden output on packed rows,
        ``[1,1,B,D]``.
        """
        attn = self._sdpa_decode(q, kv, mask, cur_pos=sdpa_cur_pos, sliding_window=sliding_window)  # [1, B, H, Dh]
        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim)
        return self._grouped_output(attn)

    def _init_sparse_sink(self) -> None:
        """``[1, 1, 1, H]`` ROW_MAJOR DRAM sink for ``sparse_sdpa``.

        The kernel multiplies ``scale`` into the sink, and the reference leaves it
        unscaled, so this is ``model_sink / scale`` -- the same pre-division the
        fused SDPA-decode sink uses. TP ranks with fewer than 32 local heads gather
        to the full head count before the op, and then the sink is replicated;
        otherwise each rank keeps its head shard.
        """
        sinks = (self.sinks_torch.reshape(1, 1, 1, self.num_heads) / self.scaling).to(torch.bfloat16)
        self._sparse_full_heads = self.tp_size > 1 and (
            self.local_num_heads < ttnn.TILE_SIZE or self.local_num_heads % ttnn.TILE_SIZE != 0
        )
        mapper = None
        if self.tp_size > 1:
            mapper = (
                ttnn.ReplicateTensorToMesh(self.device)
                if self._sparse_full_heads
                else ttnn.ShardTensorToMesh(self.device, dim=3)
            )
        self._sparse_sink = ttnn.from_torch(
            sinks,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def _sparse_k_chunk(self, width: int) -> int:
        """Smallest tile-multiple chunk that divides the sparse index row.

        ``sparse_sdpa`` places its circular buffers on every core. A 128-wide
        chunk at head dim 512 does not fit above the prefetcher rings, so the
        chunk stays at 32 whenever the index row allows it.
        """
        for chunk in (32, 64, 128):
            if width % chunk == 0:
                return chunk
        raise ValueError(f"sparse index width {width} is not divisible by 32")

    def _sparse_attend(
        self,
        q: ttnn.Tensor,
        kv: ttnn.Tensor,
        indices: ttnn.Tensor,
        cos: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """``sparse_sdpa`` over the sliding ring plus the indexer's compressed picks.

        ``q`` is packed ``[1, 1, B, H_local*Dh]``. ``kv`` is a dense combined cache
        ``[B, 1, Skv, Dh]`` TILE (K == V). ``indices`` is uint32 ROW_MAJOR
        ``[B, 1, 1, window + k]``. Returns the same packed hidden ``[1, 1, B, D]``
        :meth:`_attend` does. One ``sparse_sdpa`` call scores a single cache slot, so
        the batch is unrolled; ``B`` is fixed for the captured trace.
        """
        h, dh = self.local_num_heads, self.head_dim
        batch = _packed_users(q)
        grid_size = self._sdpa_pcfg.compute_with_storage_grid_size
        out_mem = _sdpa_decode_output_config(batch, h, dh, grid_size)
        gathered = ttnn.experimental.deepseek.all_gather_for_matmul(q, out_mem.shard_spec.grid)
        if gathered is not q:
            ttnn.deallocate(q)
        q = ttnn.experimental.view(gathered, [1, batch, h, dh])
        q_dram = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        # The gather replica is L1 on the SDPA cores. Leave it there and
        # sparse_sdpa's full-grid circular buffers overlap it.
        if q_dram is not q and q_dram.buffer_address() != q.buffer_address():
            ttnn.deallocate(q)
        q = q_dram
        q = ttnn.permute(q, (0, 2, 1, 3))  # [1, H_local, B, Dh]
        if self._sparse_full_heads:
            q = ttnn.all_gather(
                q,
                dim=1,
                cluster_axis=_tp_cluster_axis(self.device),
                num_links=1,
                topology=ttnn.Topology.Linear,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
        kv_rm = ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT)
        heads = q.shape[1]
        width = indices.shape[-1]
        k_chunk = self._sparse_k_chunk(width)
        use_slot = batch > 1 or kv.shape[0] > 1
        rows = []
        for user in range(batch):
            q_user = ttnn.slice(q, [0, 0, user, 0], [1, heads, user + 1, dh])
            idx_user = ttnn.slice(indices, [user, 0, 0, 0], [user + 1, 1, indices.shape[2], width])
            attended = ttnn.transformer.sparse_sdpa(
                q_user,
                kv_rm,
                idx_user,
                v_dim=dh,
                kv_format=ttnn.transformer.SparseKVFormat.BF16,
                scale=self.scaling,
                k_chunk_size=k_chunk,
                attention_sink=self._sparse_sink,
                cache_batch_idx=user if use_slot else None,
            )
            ttnn.deallocate(q_user)
            ttnn.deallocate(idx_user)
            if self._sparse_full_heads:
                # mesh_partition wants TILE. The height shard below is [H_local, Dh]
                # with H_local=16, which is not a tile, so untilize before sharding
                # — the same ROW_MAJOR layout sdpa_decode hands to RoPE.
                tiled = ttnn.to_layout(attended, ttnn.TILE_LAYOUT)
                ttnn.deallocate(attended)
                local = ttnn.mesh_partition(
                    tiled,
                    dim=1,
                    cluster_axis=_tp_cluster_axis(self.device),
                )
                ttnn.deallocate(tiled)
                attended = ttnn.to_layout(local, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(local)
            rows.append(ttnn.permute(attended, (0, 2, 1, 3)))  # [1, 1, H_local, Dh]
            ttnn.deallocate(attended)
        ttnn.deallocate(q)
        ttnn.deallocate(kv_rm)
        attn = rows[0] if batch == 1 else ttnn.concat(rows, dim=1)
        attn = ttnn.to_memory_config(attn, out_mem)
        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim)
        return self._grouped_output(attn)

    def _decode_activation_grid(self) -> ttnn.CoreRangeSet:
        """Core set that receives the ``[1, 1, B, D]`` decode all-gather replica of ``tokens``.

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

    def _qkv(
        self, tokens: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Project + RoPE the query and (shared) K=V for the packed-row ``tokens`` ``[1, 1, B, D]``.

        Returns ``q`` ``[1, 1, B, H_local*Dh]`` (packed; ``H_local == H / TP``), the
        rotated, replicated ``kv`` ``[1, 1, B, Dh]``, and the normalized q_a latent
        ``[1, 1, B, q_lora]`` the CSA indexer scores from. All three are still on
        packed rows (pre-compressor, pre-cache). ``cos`` / ``sin`` are the
        ``[1,1,L,Rd]`` RoPE tables. Shared by the decode paths.

        ``q`` stays packed through RoPE; :meth:`_sdpa_decode` gathers it onto the
        SDPA cores and views ``[1, B, H_local, Dh]``. The projections and norms all run
        over the single tile-row the batch occupies, so a B-user step issues the same ops
        a one-user step does.
        """
        _profile(self.device)
        # ``tokens`` is the decode-static all-gather: ROW_MAJOR HEIGHT_SHARDED with full K on
        # every core of :meth:`_decode_activation_grid`, so q_a's B cores (and kv's, a subset)
        # already hold a replica; a partial-width weight cannot take this layout.
        q_a_raw = self.q_a_proj(_decode_activation(self.q_a_proj, tokens))

        # ``None`` when the q_a matmul already normalized and mcast a replica onto
        # every q_b B core (see __init__).
        q_a = q_a_raw if self.q_a_norm is None else self.q_a_norm(q_a_raw)
        # The grouped epilogue needs q_b's one-row replicated-A path. A fused q_a mcast
        # already has that layout; otherwise replicate here. Unsupported q_b layouts
        # keep the tiled width-sharded activation and standalone per-head norm.
        q_b_input = _decode_activation(self.q_b_proj, q_a)
        q = self.q_b_proj(q_b_input)  # [1, 1, B, H*Dh]
        if q_b_input is not q_a:
            ttnn.deallocate(q_b_input)
        assert self.fuse_q_b_norm, "q_b_norm must be fused"

        q = _apply_rope(q, cos, sin, self.rot, self.rope_dim, head_dim=self.head_dim)
        # kv_proj runs here rather than beside q_a_proj: one GCB is one FIFO, so a
        # prefetched matmul that runs out of turn pops another weight's page (see
        # ``prefetch_weights``). Reuse the same replicated activation; kv's B cores are a
        # subset of that grid. The caller still owns ``tokens`` (the compressor reads it).
        kv_raw = self.kv_proj(_decode_activation(self.kv_proj, tokens))
        # ``None`` when the kv matmul normalized its own output (see __init__).
        kv = kv_raw if self.kv_norm is None else self.kv_norm(kv_raw)  # [1, 1, B, Dh]

        kv = _apply_rope(kv, cos, sin, self.rot, self.rope_dim)
        return q, kv, q_a

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
        paged: PagedLayerView | None,
        pool_compressor: bool = True,
        sdpa_cur_pos: ttnn.Tensor | None = None,
        win_slot: ttnn.Tensor | None = None,
        win_row: ttnn.Tensor | None = None,
        index_sparse: bool = False,
    ) -> ttnn.Tensor:
        """Single-token decode attention; an alias of :meth:`decode_static`, with the same
        arguments and shapes, for callers stepping the block outside a trace."""
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
            paged,
            pool_compressor=pool_compressor,
            sdpa_cur_pos=sdpa_cur_pos,
            win_slot=win_slot,
            win_row=win_row,
            index_sparse=index_sparse,
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
        paged: PagedLayerView | None,
        pool_compressor: bool = True,
        sdpa_cur_pos: ttnn.Tensor | None = None,
        win_slot: ttnn.Tensor | None = None,
        win_row: ttnn.Tensor | None = None,
        index_sparse: bool = False,
    ) -> ttnn.Tensor:
        """Trace-safe single-token decode against the layer's KV.

        ``hidden`` is ``[B, S, 1, D]`` (``S == 1``: the block decodes all ``B`` users in one
        step, every one of them at the same absolute position), the RoPE rows ``cos`` /
        ``sin`` / ``neg_sin`` are ``[1, 1, L, Rd]``, ``mask`` the additive
        ``[1, 1, 1, Skv]`` row or ``None`` under causal SDPA, ``sliding_pos`` /
        ``compress_pos`` / ``win_slot`` / ``win_row`` INT32 ``[B]`` device index vectors
        (``win_row`` is read only when pooling), and the return is the block's hidden
        output ``[B, S, 1, D]``. One shared position means one ``mask``, one RoPE row and
        one (identical) entry per user in the vectors, since the cache and SDPA ops index
        per user.

        ``paged`` is an HCA layer's KV: a shared block pool read through the active
        sessions' page table, which is what lets several sessions share one captured trace
        (see :mod:`.paged_cache`). Sliding and CSA layers pass ``None`` and use the dense
        ``scache.kv`` instead, which is swapped per session outside the trace along with
        the compressor's window buffers.

        ``pool_compressor`` says whether this step closes (and so pools) a compressor
        window; sliding layers ignore it. On CSA/HCA layers ``win_slot`` is this token's
        slot in the window buffer (``pos % compress_rate``) and, when pooling, ``win_row``
        is the KV-axis row the new entry lands in (``sliding_window + w``) and
        ``cos_win`` / ``sin_win`` are window ``w``'s single RoPE row ``[1, 1, 1, Rd]``.

        ``sdpa_cur_pos`` ``[B]`` INT32, when set, replaces ``mask`` with causal-mode SDPA
        bounded by that position (see :meth:`_sdpa_decode`). Sliding layers require it:
        ``min(pos, sliding_window - 1)``, the last valid row of the ring.

        ``index_sparse`` selects the CSA lightning-indexer trace: score compressed keys
        and attend with ``sparse_sdpa``. The indexer is off in the system config (see
        the TODO below).
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
        q, kv_new, q_a = self._qkv(tokens, cos, sin)  # q [1,1,B,H*Dh], kv_new [1,1,B,Dh]
        use_indexer = (
            index_sparse
            and self.layer_type == "compressed_sparse_attention"
            and getattr(self.compressor, "indexer", None) is not None
        )

        is_paged = self.layer_type in PAGED_KV_LAYER_TYPES
        assert (paged is not None) == is_paged, f"{self.layer_type} KV must be {'paged' if is_paged else 'dense'}"
        kv = paged if is_paged else scache.kv

        if self.compressor is None:
            ttnn.deallocate(q_a)
            # The KV axis is the sliding ring alone, written at slot ``pos % window``. Once
            # the ring is full every row is valid and attention is order-independent (RoPE
            # is applied before the write), so a causal bound of ``min(pos, window - 1)``
            # covers exactly the valid rows.
            assert sdpa_cur_pos is not None, "sliding layers need sdpa_cur_pos = min(pos, sliding_window - 1)"
            _update_kv_at(kv, kv_new, sliding_pos)
            ttnn.deallocate(kv_new)
            out = self._attend(q, kv, None, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos)
            ttnn.deallocate(tokens)
            return ttnn.reshape(out, [b, s, 1, d])

        # One KV axis holds both regions, so there is no per-step concat: the ring slot
        # ``pos % window`` lands in the prefix and each pooled entry is appended after
        # it at row ``window + w``. Both indices are pre-wrapped, so no
        # ``cache_position_modulo`` is needed.
        if not use_indexer:
            ttnn.deallocate(q_a)
            q_a = None
        else:
            # The fused q_a epilogue multicasts one [M, q_lora] replica onto every q_b
            # core, so the tensor height is cores*M. Spilling that as-is makes the
            # indexer's q_b read M=64. Collapse to the single packed row first.
            if self.q_b_proj._is_replicated_rm_hs(q_a):
                spilled_qa = self.q_b_proj._unreplicate_rm_hs_activation(q_a)
            else:
                spilled_qa = ttnn.to_memory_config(q_a, ttnn.DRAM_MEMORY_CONFIG)
            if spilled_qa is not q_a:
                ttnn.deallocate(q_a)
            q_a = spilled_qa
        _update_kv_at(kv, kv_new, sliding_pos)
        # One row per user is a whole tile of L1 each -- worth handing back before the
        # compressor and SDPA below ask for their own.
        ttnn.deallocate(kv_new)
        # ``q`` is packed and nothing reads it until the SDPA below, while the compressor in
        # between is the step's L1 high-water mark. A wide batch, or the indexer's second
        # compressor pair on top of the prefetch rings, leaves the gate matmul's circular
        # buffers nowhere to go, so park q in DRAM across the compressor and bring it back
        # before SDPA height-shards it.
        indexer_attached = getattr(self.compressor, "indexer", None) is not None
        q_config = q.memory_config() if b > 1 or indexer_attached else None
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
        )
        indices = None
        if use_indexer:
            indices = self.compressor.indexer.score_and_select(
                tokens, q_a, cos, sin, scache.idx_key_cache, compress_pos
            )
            ttnn.deallocate(q_a)
            # weights_proj has no prefetch ring, so its matmul keeps a width-sharded
            # L1 copy. That copy sits on cores sparse_sdpa also uses for circular
            # buffers; drop it before the attend. The next score copies it again.
            held = self.compressor.indexer.weights_proj.l1_weights
            if held is not None and held.is_allocated():
                ttnn.deallocate(held)
            self.compressor.indexer.weights_proj.l1_weights = None
        ttnn.deallocate(tokens)
        if q_config is not None:
            q = ttnn.to_memory_config(q, q_config)
        if use_indexer:
            # TODO: the indexer is off in the system config; turn it back on once this
            # path is validated against the dense CSA ``scache.kv``.
            out = self._sparse_attend(q, kv, indices, cos, neg_sin)
            ttnn.deallocate(indices)
        else:
            out = self._attend(q, kv, mask, cos, neg_sin, sdpa_cur_pos=sdpa_cur_pos)
        return ttnn.reshape(out, [b, s, 1, d])
