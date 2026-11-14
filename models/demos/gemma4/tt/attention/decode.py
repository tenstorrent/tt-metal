# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Decode-mode attention forward pass for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (no transformation matrices).
"""

import os

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.custom_ops.grid import (
    SDPA_DECODE_TILE_BYTES,
    plan_packed_sdpa,
    sdpa_decode_cb_budget_bytes,
    sdpa_decode_cb_bytes,
    sdpa_decode_cores_per_head,
    sdpa_decode_page_table_bytes,
    sdpa_tile_bytes,
)
from models.demos.gemma4.custom_ops.paged_update_rows import PagedUpdateRows
from models.demos.gemma4.custom_ops.paged_update_rows.op import ENV as PACKED_KV_WRITE_OP_ENV
from models.demos.gemma4.custom_ops.paged_update_rows.op import enabled as packed_kv_write_op_enabled
from models.demos.gemma4.custom_ops.qkv_norm_rope import QkvNormRope
from models.demos.gemma4.tt.ccl import ccl_allgather, oneshot_matmul_kwargs
from models.demos.gemma4.tt.compute_config import sdpa_fp32_dest_acc_en, sdpa_math_fidelity
from models.demos.gemma4.tt.kv_pair_shard import KV_PAIR_SHARD_ENV, kv_pair_shard_enabled
from models.demos.gemma4.tt.mesh_orientation import decode_flag
from models.demos.gemma4.tt.sdpa_pair_split import LOGIT_CLAMP, STATS_L_COL, STATS_M_COL, log_once, pair_split_enabled
from models.demos.gemma4.tt.verify_multipos import TILE as MULTIPOS_TILE
from models.demos.gemma4.tt.verify_multipos import log_once as multipos_log_once
from models.demos.gemma4.tt.verify_multipos import multipos_layer_reason
from models.demos.gemma4.tt.verify_prefix_lse import TILE as PREFIX_TILE
from models.demos.gemma4.tt.verify_prefix_lse import log_once as prefix_log_once
from models.demos.gemma4.tt.verify_prefix_lse import prefix_lse_layer_reason

from .operations import (
    _paged_fill_cache,
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    apply_rope_decode_peruser,
    concat_heads,
    effective_block_size,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights

# Cache for the L1 height-sharded ``MemoryConfig`` that
# ``nlp_create_qkv_heads_decode`` produces. The spec only depends on shape
# constants (B, num_heads_local, head_dim, qkv_dim, kv_replicated/global flags)
# — all fixed for a given model+batch. By caching across calls we avoid the
# per-layer probe decode-split that exists ONLY to discover this spec.
# Populated on the first (un-traced compile) call; inside trace capture the
# probe is skipped entirely.
_Q_SHARDED_MEM_CACHE: dict = {}

# GEMMA4_QKV_NORM_ROPE_FUSED=1: the per-head q/k/v norms and the q/k RoPE as one generic op on the height-sharded
# head tensors (custom_ops/qkv_norm_rope). The TILE [1, 1, 1, head_dim] gamma rows it takes are built with the
# attention weights (AttentionWeights.q_norm_tile / k_norm_tile, tt/attention/weights.py). The earlier module-level
# cache keyed by id(weights.q_norm_weight) handed a layer the tiles of a freed, same-id tensor from an earlier
# attention instance (dead device memory: the attention unit test failed by cache length, i.e. by allocation
# layout) -- never key device tensors by Python id.


def _gamma_tiles(weights: AttentionWeights, head_dim: int):
    # built with the weights (tt/attention/weights.py); a per-call reshape + tilize here would allocate under trace
    assert weights.q_norm_tile is not None and weights.k_norm_tile is not None, "AttentionWeights without norm tiles"
    return weights.q_norm_tile, weights.k_norm_tile


def _sdpa_max_cores_per_head_kwargs() -> dict:
    """GEMMA4_SDPA_MAX_CORES_PER_HEAD=N pins the decode SDPA per-(row, head) core count. The op
    splits its cores per padded row, so each decode graph (bucket) reduces a row in a different
    order; pinning N to the widest graph's split (grid / (32 * kv heads per device)) makes every
    graph reduce identically, at the cost of a lone row's attention speed at long context.
    Unset (default) keeps the op's own cap (tenstorrent/tt-metal#59300)."""
    v = os.environ.get("GEMMA4_SDPA_MAX_CORES_PER_HEAD", "")
    return {"max_cores_per_head_batch": int(v)} if v else {}


SPLITFREE_ENV = "GEMMA4_SDPA_DECODE_SPLITFREE"
_SPLITFREE_WARNED = set()


def splitfree_sdpa_enabled(config) -> bool:
    """GEMMA4_SDPA_DECODE_SPLITFREE (``decode_flag``: the env when set, else the build profile, default off):
    "1" = the split-independent paged decode SDPA (custom_ops/sdpa_decode_splitfree) on the global layers,
    "2" = on the sliding layers as well. Its output for a row is bitwise identical across the decode graphs of
    a build (unit and chunk are build constants), so the graphs can use the cores they have; the pin above
    stays as it is for the stock op."""
    mode = decode_flag(SPLITFREE_ENV)
    if mode == "2":
        return True
    return mode == "1" and not config.is_sliding


def _splitfree_refused(reason: str):
    """One warning per reason: the stock op serves the call."""
    if reason not in _SPLITFREE_WARNED:
        _SPLITFREE_WARNED.add(reason)
        logger.warning(f"{SPLITFREE_ENV}: the stock paged decode SDPA serves this call: {reason}")


def packed_write_takes_fallback_loop(kv_staging, embed_idx, is_kv_shared):
    """True when ``packed_decode_forward`` writes the KV rows through the per-position
    ``paged_update_cache`` loop, which needs the probed height-sharded layout.

    The loop-free staging fill runs only with BOTH a staging buffer and an embed
    index; a caller that keeps the staging buffer attached but writes by position
    (``kv_write_idxs``, no embed index) still takes the loop, so the probe must not
    be skipped on the presence of staging alone (serving records hit that case).
    """
    return (not is_kv_shared) and (kv_staging is None or embed_idx is None)


_ROWS_OP_WARNED = set()


def packed_rows_op_reason(
    kv_cache, kv_write_idxs, page_table, batch, packed_p, nkv_local, head_dim, config, modulo=None
):
    """Why the packed verify's KV write takes the per-position loop instead of ``custom_ops/paged_update_rows``:
    ``"off"`` unless ``GEMMA4_PACKED_KV_WRITE_OP=1`` (the served default: the loop, until the device A/B passes),
    else the op's ``unsupported_reason`` (warned once per reason), else ``None`` (the op serves)."""
    if not packed_kv_write_op_enabled():
        return "off"
    k_cache_w, _ = kv_cache
    reason = PagedUpdateRows.unsupported_reason(
        k_cache_w,
        kv_write_idxs,
        page_table,
        batch=batch,
        packed_p=packed_p,
        num_kv_heads=nkv_local,
        head_dim=head_dim,
        block_size=effective_block_size(k_cache_w, head_dim, nkv_local),
        modulo=modulo if modulo is not None else config.cache_position_modulo,
    )
    if reason is not None and reason not in _ROWS_OP_WARNED:
        _ROWS_OP_WARNED.add(reason)
        logger.warning(f"{PACKED_KV_WRITE_OP_ENV}: the per-position paged_update_cache loop serves this call: {reason}")
    return reason


def packed_kv_write_rows(k_cache_w, v_cache_w, tt_k, tt_v, kv_write_idxs, page_table, batch, nkv_local, eff_bs, modulo):
    """The packed verify's K / V rows written by ``PagedUpdateRows``: one program per tensor (2 per layer) in
    place of the loop below (6 P per layer), the same bytes. ``tt_k / tt_v [1, nkv, S_pad, head_dim]`` bf16 TILE
    (the prefill-style split after the norm and RoPE; consumed)."""
    for cache, src in ((k_cache_w, tt_k), (v_cache_w, tt_v)):
        PagedUpdateRows.op(
            cache,
            src,
            kv_write_idxs,
            page_table,
            batch=batch,
            num_kv_heads=nkv_local,
            block_size=eff_bs,
            cache_position_modulo=modulo,
        )
    ttnn.deallocate(tt_k)
    ttnn.deallocate(tt_v)


def packed_kv_write_loop(
    k_cache_w,
    v_cache_w,
    tt_k,
    tt_v,
    kv_write_idxs,
    page_table,
    B,
    P,
    nkv_local,
    head_dim,
    eff_bs,
    q_sharded_mem,
    modulo,
):
    """The per-position ``paged_update_cache`` loop of the packed verify (the served default): per packed position
    p the K and V rows of every record row are sliced out, resharded one user per core and written by the stock
    op, which serializes the read-modify-write of a tile row that several positions share (one batched call over
    the P positions would race on it). ``tt_k / tt_v [1, nkv, S_pad, head_dim]`` bf16 TILE (consumed)."""
    # Convert the prefill-style [1, nkv, B*P, hd] to decode layout
    # [1, B*P, nkv, hd] (DRAM — the full tensors stay resident across the
    # loop; only the per-p reshard occupies L1).
    tt_k_bp = ttnn.permute(tt_k, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_v_bp = ttnn.permute(tt_v, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tt_k)
    ttnn.deallocate(tt_v)
    # The prefill-style split may leave the B*P sequence rows tile-padded to a
    # multiple of 32 (ttnn-version dependent — same guard as the Q pack at ⑥).
    # Slice back to the logical B*P rows so the (1,B,P,nkv,hd) reshape is
    # volume-exact (padding is always at the tail). The slice can alias the
    # parent's DRAM storage, so keep tt_k_bp/tt_v_bp alive until the end-of-loop
    # cleanup rather than freeing them here.
    k_src, v_src = tt_k_bp, tt_v_bp
    if tt_k_bp.shape[1] != B * P:
        k_src = ttnn.slice(tt_k_bp, [0, 0, 0, 0], [1, B * P, nkv_local, head_dim])
        v_src = ttnn.slice(tt_v_bp, [0, 0, 0, 0], [1, B * P, nkv_local, head_dim])
    tt_k_view = ttnn.reshape(k_src, (1, B, P, nkv_local, head_dim))
    tt_v_view = ttnn.reshape(v_src, (1, B, P, nkv_local, head_dim))
    for p in range(P):
        k_p = ttnn.slice(tt_k_view, [0, 0, p, 0, 0], [1, B, p + 1, nkv_local, head_dim])
        k_p = ttnn.reshape(k_p, (1, B, nkv_local, head_dim))
        v_p = ttnn.slice(tt_v_view, [0, 0, p, 0, 0], [1, B, p + 1, nkv_local, head_dim])
        v_p = ttnn.reshape(v_p, (1, B, nkv_local, head_dim))
        k_p = ttnn.to_memory_config(k_p, q_sharded_mem)
        v_p = ttnn.to_memory_config(v_p, q_sharded_mem)
        if page_table is not None:
            # Bounded ring: wrap absolute write positions into the ring
            # (same contract as decode_forward's paged_modulo_kwargs).
            _mod = {"cache_position_modulo": modulo} if modulo is not None else {}
            ttnn.experimental.paged_update_cache(
                k_cache_w,
                k_p,
                update_idxs_tensor=kv_write_idxs[p],
                page_table=page_table,
                block_size=eff_bs,
                num_kv_heads=nkv_local,
                **_mod,
            )
            ttnn.experimental.paged_update_cache(
                v_cache_w,
                v_p,
                update_idxs_tensor=kv_write_idxs[p],
                page_table=page_table,
                block_size=eff_bs,
                num_kv_heads=nkv_local,
                **_mod,
            )
        else:
            ttnn.experimental.paged_update_cache(k_cache_w, k_p, update_idxs_tensor=kv_write_idxs[p])
            ttnn.experimental.paged_update_cache(v_cache_w, v_p, update_idxs_tensor=kv_write_idxs[p])
        ttnn.deallocate(k_p)
        ttnn.deallocate(v_p)
    ttnn.deallocate(tt_k_bp)
    ttnn.deallocate(tt_v_bp)


KV_FUSED_UPDATE_ENV = "GEMMA4_KV_FUSED_UPDATE"
# Blackhole mesh shapes where the fused K+V cache write is the default (measured: see kv_fused_update_default).
KV_FUSED_UPDATE_DEFAULT_SHAPES = ((1, 4), (1, 8))
KV_FUSED_STATS = {"fused": 0}  # fused-program calls (unit-test hook)
_KV_FUSED_NOTED = set()


def kv_fused_update_default(mesh_device) -> bool:
    """Whether the decode K and V cache writes run as ONE ``paged_fused_update_cache`` program per layer (item 11 of
    the gemma4_12b_qb2 port) instead of two ``paged_update_cache`` programs.

    ``GEMMA4_KV_FUSED_UPDATE=1|0`` overrides. Policy when unset: on for the Blackhole mesh shapes in
    ``KV_FUSED_UPDATE_DEFAULT_SHAPES`` (those where the served A/B measured a win with the PCC gates intact: the
    12B 1x4 QB2 block, and the (1,8) Galaxy DP4 engine with the six port knobs together on the 12B and the 31B,
    see ``default_ccl_oneshot``), off everywhere else (Wormhole, the lane / fractured meshes; the P150x8 Loudbox
    shares the (1,8) shape and is unmeasured). The fused program needs V on cores disjoint from K's, which the composite head path
    gives for free and ``QkvHeadsNormRope`` emits on request; the ``QkvNormRope`` form (the MoE and lane builds) keeps
    K and V on one grid and stays on two programs regardless (``kv_fused_update_reason``)."""
    env = os.environ.get(KV_FUSED_UPDATE_ENV, "").strip().lower()
    if env:
        return env in ("1", "true", "yes", "on")
    shape = tuple(int(s) for s in mesh_device.shape) if hasattr(mesh_device, "shape") else ()
    return is_blackhole() and shape in KV_FUSED_UPDATE_DEFAULT_SHAPES


def kv_fused_update_reason(k_cache, v_cache, config, num_local_kv_heads, *, batch, page_table, sequential, qkv_path):
    """Why the two-program K / V cache write serves a decode call: None when the fused program can.

    ``paged_fused_update_cache`` has no ``block_size`` / ``num_kv_heads`` / ``cache_position_modulo`` overrides: it
    reads the block size and the head count off the cache tensor and wants the input's head_dim equal to the cache's,
    so a cache allocated for another layer type's view (vLLM's hybrid kv-cache groups sharing one buffer between the
    sliding and the full layers) or a bounded sliding ring keeps the two ``paged_update_cache`` programs. The
    speculative verify's per-position loop (one single-core slice per candidate), the non-paged legacy cache and the
    ``QkvNormRope`` head path (K and V on one grid; the fused program places one kernel set per input on that
    input's shard cores) stay on two programs as well.
    """
    if page_table is None:
        return "non-paged cache (legacy path)"
    if sequential and int(batch) > 1:
        return "speculative verify writes each candidate position with its own program"
    if qkv_path == "qkv_norm_rope":
        return "QkvNormRope keeps K and V on one shard grid"
    if getattr(config, "cache_position_modulo", None) is not None:
        return "bounded sliding ring (cache_position_modulo)"
    head_dim = int(config.head_dim)
    eff_bs = effective_block_size(k_cache, head_dim, num_local_kv_heads)
    k_shape = tuple(int(s) for s in k_cache.padded_shape)
    if eff_bs != k_shape[2] or int(num_local_kv_heads) != k_shape[1] or head_dim != k_shape[-1]:
        return (
            f"cache view {k_shape} differs from the layer's (block {eff_bs}, {num_local_kv_heads} heads x {head_dim})"
        )
    if tuple(int(s) for s in v_cache.padded_shape) != k_shape or v_cache.dtype != k_cache.dtype:
        return "K and V caches differ in shape or dtype"
    return None


def _note_kv_fused(config, qkv_path, batch, reason):
    """One INFO line per (layer type, head path, batch, reason) the fused-write decision takes."""
    key = (bool(config.is_sliding), int(config.head_dim), qkv_path, int(batch), reason)
    if key in _KV_FUSED_NOTED:
        return
    _KV_FUSED_NOTED.add(key)
    layer = "sliding" if config.is_sliding else "full"
    if reason is None:
        logger.info(f"{KV_FUSED_UPDATE_ENV}: fused K+V cache write on the {layer} layers ({qkv_path}, batch {batch})")
    else:
        logger.info(
            f"{KV_FUSED_UPDATE_ENV}: two-program K / V write on the {layer} layers ({qkv_path}, batch {batch}): {reason}"
        )


def _kv_grids_disjoint(tt_k, tt_v) -> bool:
    """Whether K's and V's shard grids share no core (the fused write's placement rule)."""
    try:
        from models.demos.gemma4.custom_ops.qkv_heads_norm_rope.op import shard_cores

        k_cores = {(c.x, c.y) for c in shard_cores(tt_k.memory_config().shard_spec.grid)}
        v_cores = {(c.x, c.y) for c in shard_cores(tt_v.memory_config().shard_spec.grid)}
    except Exception:  # noqa: BLE001 - an interleaved tensor has no shard spec
        return False
    return bool(k_cores) and bool(v_cores) and not (k_cores & v_cores)


# ── GEMMA4_SDPA_PAIR_SPLIT: a global layer's decode attention over the two chips of a KV-head pair ─────────────────
# tt/sdpa_pair_split.py holds the knob, the layouts and a torch emulation of pair_merge; the pair QKV / o_proj and the
# partner selector are AttentionWeights.pair (tt/attention/weights.py).


def pair_split_active(weights, page_table, kv_cache, is_kv_shared, sequential_kv_write) -> bool:
    """Whether a decode call runs on the pair view ``weights.pair``: built for the layer (a qualifying global layer,
    knob on at the build) and a plain paged decode step. The speculative verify (``sequential_kv_write``), KV-shared
    layers (the MTP drafter) and a non-paged cache keep the 4-head path (logged once per reason). Only static
    arguments decide, never a device value, so warmup, trace capture and replay take the same branch."""
    if getattr(weights, "pair", None) is None or not pair_split_enabled():
        return False
    reason = None
    sharded = kv_pair_shard_enabled()  # the KV is split over the pair: only the pair view can read it
    if page_table is None or kv_cache is None:
        reason = "no paged KV cache"
    elif is_kv_shared and not sharded:
        reason = "KV-shared layer"
    elif sequential_kv_write and not sharded:
        # (under the shard the per-row write loop goes through the write table and the pair view reads)
        reason = "speculative verify (per-candidate KV write)"
    if reason is not None:
        log_once(f"a decode call keeps the 4-head attention: {reason}")
        return False
    return True


def pair_split_sdpa(
    tt_q, k_cache, v_cache, cache_pos, page_table, config, weights, program_config, mesh_config, ccl_manager
):
    """One chip's pair-split decode attention merged with its partner's: ``SdpaDecodeLse(pair_split=True)`` over
    this chip's half (``c % 2``) of the row's key chunks, with the stock call's program config and compute config,
    then ``pair_merge``. ``tt_q [1, B, G, D]`` bf16 TILE interleaved (the G q heads of the chip's KV group). Returns
    ``w_self * O_self [1, B, G, D]``, or None when the op refuses the call (logged once; the caller then runs the
    stock op over every key on both chips of the pair and halves it).

    ``GEMMA4_KV_PAIR_SHARD``: the chip holds only its half of every logical block (``tt/kv_pair_shard.py``), so the
    op runs ``half_block=True`` over the logical table (each chip deriving its position from the row's) and a
    refusal raises: no chip holds every key, the stock fallback would attend half of them."""
    from models.demos.gemma4.custom_ops.sdpa_decode_lse import SdpaDecodeLse

    kwargs = dict(
        scale=1.0,
        num_kv_heads=1,
        block_size=effective_block_size(k_cache, config.head_dim, 1),
        cache_position_modulo=config.cache_position_modulo,
        program_config=program_config,
        compute_kernel_config=None,  # the stock call passes none: the op default (HiFi2, fp32 dest off)
    )
    # pair_split=True: the half-block mode runs on the pair split's meshes too (one program per column parity)
    reason = SdpaDecodeLse.unsupported_reason(
        tt_q, k_cache, v_cache, cache_pos, page_table, memory_config=ttnn.DRAM_MEMORY_CONFIG, pair_split=True, **kwargs
    )
    half_block = kv_pair_shard_enabled()
    if reason is not None:
        if half_block:
            raise ValueError(f"{KV_PAIR_SHARD_ENV}: sdpa_decode_lse refuses a pair-sharded global decode: {reason}")
        log_once(f"sdpa_decode_lse refuses the call; the stock op attends every key on both chips: {reason}")
        return None
    out, stats = SdpaDecodeLse.op(
        tt_q,
        k_cache,
        v_cache,
        cache_pos,
        page_table,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        stats_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pair_split=not half_block,
        half_block=half_block,
        **kwargs,
    )
    return pair_merge(out, stats, weights.pair_selector, mesh_config, ccl_manager, scale=kwargs["scale"])


def pair_merge(out, stats, selector, mesh_config, ccl_manager, scale=1.0):
    """``w_self * out`` with ``w_self = sigmoid(lse_self - lse_partner)``, partner = column ``c ^ 1`` of the row.
    Device-only and trace-safe: fixed shapes per batch, no host read, the selector a build-time constant.

    ``out [1, B, H, D]`` bf16 (this chip's half-normalised output), ``stats [1, B, PNH, 64]`` bf16 (m at column 0, l
    at column 32; both consumed). The LSE is formed in fp32 from the two bf16 columns (``lse = scale * m + ln l``;
    bf16 would round it by up to 0.125 at |lse| ~ 30), flattened to one ``[1, 1, B*H, 1]`` column, all-gathered over
    the row on a fresh dim 1 (``[1, tp, B*H, 1]``), multiplied by the chip's selector row ``e_c - e_{c^1}``
    (``[1, tp, 1, 1]``, exact) and summed over dim 1 (the accurate fp32 SFPU reduction): ``lse_self - lse_partner``.
    The difference is clamped to +-LOGIT_CLAMP (an empty half sits at ~-3e4, whose exp overflows inside the sigmoid),
    ``w = sigmoid(.)``. Rows the op skipped (cur_pos = -1) hold garbage in both outputs, as with the stock op; every
    step is elementwise or reduces over the device dim only, so it stays in those rows."""
    B, H = int(out.shape[1]), int(out.shape[2])
    stats32 = ttnn.typecast(stats, ttnn.float32)
    ttnn.deallocate(stats)
    m = ttnn.slice(stats32, [0, 0, 0, STATS_M_COL], [1, B, H, STATS_M_COL + 1])
    l = ttnn.slice(stats32, [0, 0, 0, STATS_L_COL], [1, B, H, STATS_L_COL + 1])
    ln_l = ttnn.log(l)
    if scale != 1.0:
        m_scaled = ttnn.multiply(m, float(scale))
        ttnn.deallocate(m)
        m = m_scaled
    lse = ttnn.add(m, ln_l)  # [1, B, H, 1] fp32
    for t in (m, l, ln_l, stats32):
        ttnn.deallocate(t)
    lse_rows = ttnn.reshape(lse, (1, 1, B * H, 1))  # B*H rows on ceil(B*H / 32) tiles instead of B
    gathered = ccl_allgather(lse_rows, mesh_config, ccl_manager, dim=1)  # [1, tp, B*H, 1]; consumes lse_rows
    picked = ttnn.multiply(gathered, selector)
    ttnn.deallocate(gathered)
    diff = ttnn.sum(picked, dim=1, keepdim=True)  # [1, 1, B*H, 1] = lse_self - lse_partner
    ttnn.deallocate(picked)
    clamped = ttnn.clamp(diff, min=-LOGIT_CLAMP, max=LOGIT_CLAMP)
    ttnn.deallocate(diff)
    w32 = ttnn.sigmoid(clamped)
    ttnn.deallocate(clamped)
    w_rows = ttnn.typecast(w32, ttnn.bfloat16)
    ttnn.deallocate(w32)
    w = ttnn.reshape(w_rows, (1, B, H, 1))
    merged = ttnn.multiply(out, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # w broadcast over head_dim
    ttnn.deallocate(out)
    ttnn.deallocate(w)
    return merged


def _q_sharded_mem_key(B, qkv_dim, config, weights, tp):
    """Hashable key for the q_sharded_mem cache. Captures every input that
    affects ``nlp_create_qkv_heads_decode``'s output shard spec."""
    return (
        int(B),
        int(qkv_dim),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
        bool(weights.is_global),
        bool(weights.kv_replicated),
        int(tp),
    )


def decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    position_idx,
    token_index,
    page_table=None,
    ccl_manager=None,
    is_kv_shared=False,
    position_idx_cache=None,
    sequential_kv_write=False,
    rope_presliced=False,
    rope_prepared=None,
    sdpa_scratch=None,
    kv_write_table=None,
):
    """
    Single-token decode attention, fully on device.

    Args:
        hidden_states: [1, 1, batch, hidden_size] on device
        cos_cache: [max_seq_len, head_dim] 2D cache for embedding lookup, or [1,1,max_seq_len,head_dim] 4D
        sin_cache: same format as cos_cache
        rope_presliced: if True, cos_cache/sin_cache are already position-gathered
            [1, 1, batch_pad, head_dim] tensors (one row per user), shared across all
            layers of this layer_type by the model. The per-layer ttnn.embedding slice
            is skipped and Q+K are RoPE'd in a single fused call. (Consumed in the
            fused-RoPE branch.)
        weights: AttentionWeights container
        kv_cache: [k_cache, v_cache] TT tensors (for shared layers, this is the source layer's cache)
        config: Gemma4AttentionConfig
        mesh_config: MeshConfig
        mesh_device: TT device
        position_idx: [batch] tensor of current positions for KV cache update + RoPE embedding lookup
        token_index: int position for legacy RoPE slicing (unused when cos_cache is 2D)
        page_table: optional paged attention table
        ccl_manager: optional CCL manager for TP > 1
        is_kv_shared: if True, skip K/V projection and cache update (use source layer's KV cache)
        sdpa_scratch: the model's persistent fp32 scratch of the split-independent decode SDPA
            (``GEMMA4_SDPA_DECODE_SPLITFREE``, custom_ops/sdpa_decode_splitfree); None keeps the stock op
        kv_write_table: ``GEMMA4_KV_PAIR_SHARD`` -- the per-chip write table (``tt/kv_pair_shard.py``) the cache
            writes go through instead of ``page_table`` (which the attention reads through), so a chip writes only
            its own half of every logical block
    """
    tp = mesh_config.tp if mesh_config else 1
    # GEMMA4_SDPA_PAIR_SPLIT: a qualifying global layer decodes on its pair view (pair QKV / o_proj): each chip runs
    # the whole KV group's q heads, twice the 4-head layout's, so the local q-head count derives from tp / 2.
    pair = pair_split_active(weights, page_table, kv_cache, is_kv_shared, sequential_kv_write)
    if pair:
        weights = weights.pair
    head_tp = tp // 2 if pair else tp

    # 1. Fused QKV projection (GEMMA4_QKV_L1_OUT=1: straight into L1, where the head split reads it anyway --
    # drops the DRAM -> L1 copy of the Blackhole workaround in split_qkv_heads_decode)
    xqkv = apply_qkv_projection(
        hidden_states,
        weights,
        memory_config=ttnn.L1_MEMORY_CONFIG if decode_flag("GEMMA4_QKV_L1_OUT") == "1" else None,
    )

    batch_rows = int(xqkv.shape[2])
    fused_rope_ok = (
        not is_kv_shared
        and rope_presliced
        and len(cos_cache.shape) == 4
        and int(cos_cache.shape[1]) == 1
        and not (sequential_kv_write and batch_rows > 1)
    )
    # GEMMA4_QKV_HEADS_FUSED=1 (with GEMMA4_QKV_L1_OUT=1): the head split, the per-head norms and the q/k RoPE as
    # one program straight from the L1 projection output (custom_ops/qkv_heads_norm_rope), in place of
    # nlp_create_qkv_heads_decode + qkv_norm_rope; the head_dim-256 (sliding) layers, as the fresh-output form.
    heads_fused = (
        decode_flag("GEMMA4_QKV_HEADS_FUSED") == "1"
        and fused_rope_ok
        and int(config.head_dim) % 64 == 0
        and not xqkv.is_sharded()
        and xqkv.memory_config().buffer_type == ttnn.BufferType.L1
    )
    # GEMMA4_KV_FUSED_UPDATE: one paged_fused_update_cache program for K and V, decided before the head split since
    # V then has to come out on cores disjoint from K's (kv_fused_v_mem: V's layout when the fused write serves).
    kv_fused, kv_fused_v_mem = False, None
    if kv_cache is not None and not is_kv_shared and kv_fused_update_default(mesh_device):
        # the head path the else-branch below picks (same predicate as its ``fused_qkv``)
        _qkv_norm_rope_path = (
            not heads_fused
            and decode_flag("GEMMA4_QKV_NORM_ROPE_FUSED") == "1"
            and fused_rope_ok
            and (int(config.head_dim) <= 256 or QkvNormRope.in_place())
        )
        _qkv_path = "heads_fused" if heads_fused else ("qkv_norm_rope" if _qkv_norm_rope_path else "composite")
        _nkv_local = 1 if weights.kv_replicated else config.num_key_value_heads // tp
        _reason = kv_fused_update_reason(
            kv_cache[0],
            kv_cache[1],
            config,
            _nkv_local,
            batch=batch_rows,
            page_table=page_table,
            sequential=sequential_kv_write,
            qkv_path=_qkv_path,
        )
        if _reason is None:
            from models.demos.gemma4.custom_ops.qkv_heads_norm_rope.op import heads_v_disjoint_memory_config

            kv_fused_v_mem = heads_v_disjoint_memory_config(mesh_device, batch_rows, int(config.head_dim))
            if kv_fused_v_mem is None:
                _reason = "the grid holds fewer than 2 x batch cores"
        kv_fused = _reason is None
        _note_kv_fused(config, _qkv_path, batch_rows, _reason)
    if heads_fused:
        from models.demos.gemma4.custom_ops.qkv_heads_norm_rope import QkvHeadsNormRope

        gq, gk = _gamma_tiles(weights, config.head_dim)
        tt_q, tt_k, tt_v = QkvHeadsNormRope.op(
            xqkv,
            config.num_attention_heads // head_tp,
            1 if weights.kv_replicated else config.num_key_value_heads // tp,
            config.head_dim,
            gq,
            gk,
            cos_cache,
            sin_cache,
            config.rms_norm_eps,
            v_memory_config=kv_fused_v_mem,
        )
        xqkv.deallocate(True)
        q_sharded_mem = tt_q.memory_config()
        fused_qkv = True
    else:
        # 2. Split into Q, K, V heads
        tt_q, tt_k, tt_v = split_qkv_heads_decode(
            xqkv, config, weights.is_global, tp=head_tp, kv_replicated=weights.kv_replicated
        )

        # 3. Per-head norms (move to DRAM for rms_norm, restore sharded for RoPE)
        q_sharded_mem = tt_q.memory_config()
        fused_qkv = (
            decode_flag("GEMMA4_QKV_NORM_ROPE_FUSED") == "1"
            and fused_rope_ok
            # With fresh (not in-place) outputs the op's six live [32, 512] shards per core push the lowest live
            # L1 buffer under the static circular-buffer region of the following paged_update_cache program on
            # the head_dim-512 global layers at 32 users ("statically allocated dataflow buffers clash with L1
            # buffers"), so that form serves the sliding layers only.
            and (int(config.head_dim) <= 256 or QkvNormRope.in_place())
        )
    if heads_fused:
        pass  # split, norms and RoPE done above
    elif fused_qkv:
        # norms + RoPE on the sharded head tensors, outputs in the same sharded layout (custom_ops/qkv_norm_rope)
        gq, gk = _gamma_tiles(weights, config.head_dim)
        nq, nk, nv = QkvNormRope.op(tt_q, tt_k, tt_v, gq, gk, cos_cache, sin_cache, config.rms_norm_eps)
        for t, o in ((tt_q, nq), (tt_k, nk), (tt_v, nv)):
            if o is not t:  # fresh-output form: the input shards are dead
                t.deallocate(True)
        # k / v stay sharded for the paged cache update below (freed right after it); q stays height-sharded into
        # SDPA (same result as the DRAM form: tests/unit/test_attention_qkv_stages.py).
        tt_q, tt_k, tt_v = nq, nk, nv
    else:
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)

    if fused_qkv:
        pass
    elif is_kv_shared:
        # KV-shared layer: discard own K/V, use source layer's KV cache directly
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    else:
        tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
        tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
        # Do not K→V clone (resync): that produced unicode garbage on LB 12B.
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    # 4. RoPE — use on-device embedding lookup for trace compatibility
    # use_embedding_rope: cos/sin are per-position [1,1,batch_pad,head_dim] tensors.
    #   - rope_presliced: gathered once per layer_type by the model and shared across
    #     all layers (no per-layer ttnn.embedding here).
    #   - 2D cache: gather the position rows with ttnn.embedding right here (legacy
    #     per-layer path, kept for the it-assistant drafter / direct callers).
    use_embedding_rope = rope_presliced or len(cos_cache.shape) == 2
    if fused_qkv:
        pass  # rotated by the fused op above
    elif use_embedding_rope:
        if rope_presliced:
            cos_pos, sin_pos = cos_cache, sin_cache  # [1, 1, batch_pad, head_dim], shared
        else:
            # Gather position-specific cos/sin via ttnn.embedding (fully on-device, trace-safe)
            # position_idx: [1, 32] uint32 padded tensor for embedding lookup
            cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT))
            sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT))
        # RoPE. batch=1 uses the fused single-position rotary_embedding (one core
        # but cheap, no slice/tilize churn). batch>1 needs per-user positions,
        # which that op can't express, so fall back to the manual elementwise
        # q*cos + rotate_half(q)*sin (numerically equivalent — isolation PCC
        # ~0.99999 vs the fused op and the HF reference — but a few ops costlier).
        batch = tt_q.shape[1]
        # The fused op and the elementwise form agree to PCC ~0.99999, not bitwise. With decode
        # graphs per batch size (GEMMA4_DECODE_WARMUP_BATCHES) a request decoded in the batch-1
        # graph then sees different Q/K than the same request in a wider graph, which breaks
        # seeded reproducibility (tt-metal#59300). GEMMA4_ROPE_PERUSER_B1=1 keeps every graph on
        # the elementwise form; off by default until measured.
        peruser_at_b1 = os.environ.get("GEMMA4_ROPE_PERUSER_B1", "0") == "1"
        if batch > 1 or peruser_at_b1:
            # ``rope_prepared`` (per step, per layer type, from the model) keeps the transposed/sliced cos/sin
            # and their head-repeated forms across the layers of one type: only the first layer pays the ops.
            base = rope_prepared.get("base") if rope_prepared is not None else None
            if base is None:
                cos_b = ttnn.transpose(cos_pos, 1, 2)[:, :batch, :, :]  # [1, batch, 1, head_dim]
                sin_b = ttnn.transpose(sin_pos, 1, 2)[:, :batch, :, :]
                if rope_prepared is not None:
                    rope_prepared["base"] = (cos_b, sin_b)
            else:
                cos_b, sin_b = base

        def _rope(t):
            if batch == 1 and not peruser_at_b1:
                return apply_rope(t, cos_pos, sin_pos, token_index=0)
            return apply_rope_decode_peruser(t, cos_b, sin_b, prepared=rope_prepared)

        # Rotate Q (and K, unless this is a KV-shared layer) with the shared
        # cos/sin. A concat(Q,K)->rope->split "fusion" was tried to collapse the
        # two rotary_embedding calls into one, but at decode batch=1 under metal
        # trace replay it regressed throughput (~3%): host dispatch is already
        # free under replay, so it only added concat+split device kernels while
        # removing one tiny rope kernel. Keep separate rotations.
        tt_q = _rope(tt_q)
        if not is_kv_shared:
            tt_k = _rope(tt_k)
    else:
        # Legacy path: full 4D cache with Python int token_index
        tt_q = apply_rope(tt_q, cos_cache, sin_cache, token_index=token_index)
        if not is_kv_shared:
            tt_k = apply_rope(tt_k, cos_cache, sin_cache, token_index=token_index)

    # 5. KV cache update — skip for KV-shared layers (source layer already updated the cache)
    # Use position_idx_cache (int32) for cache ops when position_idx is uint32 (embedding lookup format)
    cache_pos = position_idx_cache if position_idx_cache is not None else position_idx
    # Bounded sliding-window cache: when set, the op wraps absolute positions into a
    # circular buffer of ``cache_position_modulo`` tokens before the page_table lookup.
    # ``None`` = legacy unbounded behavior (set on full-attention layers or when the
    # bounded-cache mode isn't wired up). Empty kwargs dict on the legacy path keeps
    # kernel signatures and program-cache keys for existing callers byte-identical.
    paged_modulo_kwargs = (
        {"cache_position_modulo": config.cache_position_modulo} if config.cache_position_modulo is not None else {}
    )
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        if not is_kv_shared:
            if not fused_qkv:
                # After HF-style RoPE, tensors may be in DRAM. Move to HEIGHT_SHARDED for cache update. (The fused
                # op's k / v already are; the earlier `and not fused_qkv` on this branch skipped the whole cache
                # update for the fused path, so no decode token's K / V ever reached the cache: decode gate 12/64.)
                tt_k = ttnn.to_memory_config(tt_k, q_sharded_mem)
                # the fused K+V write wants V apart from K: the same reshard, onto the next ``batch`` cores
                tt_v = ttnn.to_memory_config(tt_v, kv_fused_v_mem if kv_fused else q_sharded_mem)

            if page_table is not None:
                # Per-device kv-head count of the layer's input view. When the cache
                # was allocated for a different layer type under HMA cross-group
                # sharing (Gemma4-26B-A4B sliding kv=8 / full kv=2 on multi-device
                # TP) cache.padded_shape[1] disagrees with what the kernel needs to
                # write — see paged_update_cache num_kv_heads kwarg. Mirrors
                # split_qkv_heads_decode's local head count.
                num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
                eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
                batch = tt_k.shape[1]
                write_table = kv_write_table if kv_write_table is not None else page_table
                if sequential_kv_write and batch > 1:
                    # Speculative VERIFY: the B candidates sit at consecutive
                    # positions that share ONE paged block (all batch rows of the
                    # page table point to the same physical blocks). A single
                    # batched paged_update_cache then has multiple batch entries
                    # read-modify-write the SAME block tile concurrently and race
                    # (only some writes survive -> corrupt KV). Writing each
                    # position with its own ordered op serializes the block RMW.
                    # Matmuls + SDPA still run once on the full batch, so the
                    # verify speedup is preserved (only these tiny writes loop).
                    # paged_update_cache wants a sharded update tensor with
                    # num_shards == num_users. Slice each candidate to a batch=1
                    # [1,1,nkv,hd] tensor (from DRAM, where slicing is clean) and
                    # reshard onto a single core. q_sharded_mem shards one user per
                    # core (shard shape = [32, head_dim], head_dim shared by Q/K),
                    # so a 1-core config with that same shard shape is exactly the
                    # per-user layout the op expects.
                    _shard_shape = list(q_sharded_mem.shard_spec.shape)
                    _one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
                    single_user_mem = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(_one_core, _shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
                    )
                    k_seq = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
                    v_seq = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
                    nkv, hd = k_seq.shape[2], k_seq.shape[3]
                    for b in range(batch):
                        kb = ttnn.slice(k_seq, [0, b, 0, 0], [1, b + 1, nkv, hd])
                        vb = ttnn.slice(v_seq, [0, b, 0, 0], [1, b + 1, nkv, hd])
                        kb = ttnn.to_memory_config(kb, single_user_mem)
                        vb = ttnn.to_memory_config(vb, single_user_mem)
                        pos_b = ttnn.slice(cache_pos, [b], [b + 1])
                        pt_b = ttnn.slice(write_table, [b, 0], [b + 1, write_table.shape[1]])
                        ttnn.experimental.paged_update_cache(
                            k_cache,
                            kb,
                            update_idxs_tensor=pos_b,
                            page_table=pt_b,
                            block_size=eff_bs,
                            num_kv_heads=num_local_kv_heads,
                            **paged_modulo_kwargs,
                        )
                        ttnn.experimental.paged_update_cache(
                            v_cache,
                            vb,
                            update_idxs_tensor=pos_b,
                            page_table=pt_b,
                            block_size=eff_bs,
                            num_kv_heads=num_local_kv_heads,
                            **paged_modulo_kwargs,
                        )
                        for t in (kb, vb, pos_b, pt_b):
                            t.deallocate(True)
                    k_seq.deallocate(True)
                    v_seq.deallocate(True)
                elif kv_fused and _kv_grids_disjoint(tt_k, tt_v):
                    # one program writes both caches (K's kernels on K's shard cores, V's on V's); the block
                    # size and head count come off the cache, which kv_fused_update_reason checked equal to the
                    # layer's view
                    ttnn.experimental.paged_fused_update_cache(
                        k_cache, tt_k, v_cache, tt_v, update_idxs_tensor=cache_pos, page_table=write_table
                    )
                    KV_FUSED_STATS["fused"] += 1
                else:
                    ttnn.experimental.paged_update_cache(
                        k_cache,
                        tt_k,
                        update_idxs_tensor=cache_pos,
                        page_table=write_table,
                        block_size=eff_bs,
                        num_kv_heads=num_local_kv_heads,
                        **paged_modulo_kwargs,
                    )
                    ttnn.experimental.paged_update_cache(
                        v_cache,
                        tt_v,
                        update_idxs_tensor=cache_pos,
                        page_table=write_table,
                        block_size=eff_bs,
                        num_kv_heads=num_local_kv_heads,
                        **paged_modulo_kwargs,
                    )
            else:
                ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=cache_pos)
                ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=cache_pos)
            if fused_qkv:
                # the fused op's k / v shards are dead once written: free their L1 before SDPA
                tt_k.deallocate(True)
                tt_v.deallocate(True)
    else:
        k_cache = tt_k
        v_cache = tt_v

    # 6. SDPA (scale=1.0)
    sliding_window = config.sliding_window if config.is_sliding else None

    # Always pass an SDPAProgramConfig so num_cores_per_head stays within
    # MAX_TREE_REDUCTION_ROUNDS=6 (=> 64 cores/head). With program_config=None,
    # the SDPA op falls back to the full device grid, which exceeds 64 cores
    # on Blackhole (>=110 cores) when num_kv_heads is small. The struct's
    # default max_cores_per_head_batch=16 caps the per-head reduction tree.
    # Batched Q is height-sharded row-major across the device grid (one user per
    # core), so the SDPA grid must cover those cores — an 8-wide grid would miss
    # users on columns 8..10 of an 11-wide Blackhole grid and corrupt the output.
    batch_size = tt_q.shape[1]
    device_grid = mesh_device.compute_with_storage_grid_size()
    if config.head_dim >= 512 and batch_size == 1 and k_cache.dtype != ttnn.bfloat8_b:
        # Single-user global layers over a bf16 cache keep the (8,4) grid: only the bfp8 cache was measured on the
        # whole grid. The cores per head are min(grid, max_cores_per_head_batch) on both grids (16 by default, 32
        # under GEMMA4_SDPA_MAX_CORES_PER_HEAD=32), so the grid only spreads the same cores; one BH chip, 4 Q / 1 KV
        # heads, bfp8 cache, 32 cores per head: 0.493 -> 0.453 ms per global layer at 131K, 0.175 -> 0.164 at 32K,
        # same output (2026-11-10).
        sdpa_grid = ttnn.CoreCoord(8, 4)
    else:
        # Sliding layers, and all batched decode: use the full device compute grid.
        sdpa_grid = ttnn.CoreCoord(device_grid.x, device_grid.y)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_grid,
        q_chunk_size=32,
        # GEMMA4_SDPA_DECODE_K_CHUNK: K chunk of the decode SDPA (default 64; a multiple of 32, bounded by L1 at
        # head_dim 512) for the knob sweep; the page block (64) stays the paged read unit either way.
        k_chunk_size=int(os.environ.get("GEMMA4_SDPA_DECODE_K_CHUNK", "64")),
        exp_approx_mode=False,
        **_sdpa_max_cores_per_head_kwargs(),
    )

    tt_sdpa = None
    pair_merged = False
    if pair:
        # GEMMA4_SDPA_PAIR_SPLIT: this chip's half of the key chunks for the whole KV group, merged with the partner's
        # through their LSEs (pair_split_sdpa); it takes precedence over the split-independent op on these layers
        if tt_q.is_sharded():  # the fused head ops leave q height-sharded in L1; the op reads interleaved tiles
            q_dram = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
            tt_q.deallocate(True)
            tt_q = q_dram
        tt_sdpa = pair_split_sdpa(
            tt_q,
            k_cache,
            v_cache,
            cache_pos,
            page_table,
            config,
            weights,
            sdpa_program_config,
            mesh_config,
            ccl_manager,
        )
        pair_merged = tt_sdpa is not None
    if not pair and page_table is not None and sdpa_scratch is not None and splitfree_sdpa_enabled(config):
        # GEMMA4_SDPA_DECODE_SPLITFREE: the custom op (custom_ops/sdpa_decode_splitfree): fixed KV units per
        # (row, group) on a static core mapping, fp32 partials in the model's scratch, a fixed-order combine --
        # the row's bits do not depend on the padded batch or the cores (tt-metal#59300), and a lone row's units
        # sit on consecutive cores whatever the padded batch (64 cores at 256K). Out of contract (the wrapper
        # says why), the stock op serves.
        from models.demos.gemma4.custom_ops.sdpa_decode_splitfree import SdpaDecodeSplitfree

        sdpa_num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
        if tt_q.is_sharded():  # the fused head ops leave q height-sharded in L1; the op reads interleaved tiles
            q_dram = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
            tt_q.deallocate(True)
            tt_q = q_dram
        splitfree_kwargs = dict(
            num_kv_heads=sdpa_num_local_kv_heads,
            block_size=effective_block_size(k_cache, config.head_dim, sdpa_num_local_kv_heads),
            sliding_window=sliding_window,
        )
        splitfree_reason = SdpaDecodeSplitfree.unsupported_reason(
            tt_q, k_cache, v_cache, cache_pos, page_table, sdpa_scratch, **splitfree_kwargs
        )
        if splitfree_reason is None:
            tt_sdpa = SdpaDecodeSplitfree.op(
                tt_q,
                k_cache,
                v_cache,
                cache_pos,
                page_table,
                sdpa_scratch,
                scale=1.0,
                cache_position_modulo=config.cache_position_modulo,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **splitfree_kwargs,
            )
        else:
            _splitfree_refused(splitfree_reason)
    if tt_sdpa is not None:
        pass  # the split-independent op (or the pair split) served the call
    elif page_table is not None:
        sdpa_num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
        tt_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=cache_pos,
            page_table_tensor=page_table,
            scale=1.0,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
            # Tell SDPA the layer's view of the cache when the buffer was allocated
            # for a different layer type under HMA cross-group sharing — same
            # rationale as the num_kv_heads override on paged_update_cache.
            paged_cache_geometry=ttnn.PagedCacheGeometryOverride(
                block_size=effective_block_size(k_cache, config.head_dim, sdpa_num_local_kv_heads),
                num_kv_heads=sdpa_num_local_kv_heads,
            ),
            **paged_modulo_kwargs,
        )
    else:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=cache_pos,
            scale=1.0,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
        )
    tt_q.deallocate(True)
    if pair and not pair_merged:
        # the refused pair call: both chips of the pair attended every key of the group, so each contributes half
        halved = ttnn.multiply(tt_sdpa, 0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_sdpa.deallocate(True)
        tt_sdpa = halved

    # 7. Concat heads + output projection + allreduce (the pair view: the group's heads into the pair o_proj; the
    # pair's two partial products and the tp / 2 pairs add up in the same row all-reduce)
    num_local_heads = config.num_attention_heads // head_tp
    tt_out = concat_heads(
        tt_sdpa,
        is_decode_mode=True,
        num_heads=num_local_heads,
        head_dim=config.head_dim,
        mesh_device=mesh_device,
        o_proj=weights.o_proj,
    )
    # GEMMA4_CCL_ONESHOT=1: the DRAM-sharded o_proj keeps its partial sums L1 width-sharded (in the payload dtype)
    # and the all-reduce is the one-shot op on slot "o" (tt/ccl.py); otherwise the kwargs are empty and the
    # interleaved output takes the composite all-reduce as before.
    rows = int(tt_out.shape[-2])
    tt_out = apply_output_projection(tt_out, weights, **oneshot_matmul_kwargs(mesh_config, ccl_manager))
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size, slot="o", rows=rows)

    return tt_out


# ── Packed-query verify decode ─────────────────────────────────────────────
# Verifies P speculative positions per user in ONE attention pass by packing
# the P positions into the query-heads dim (H_local*P packed heads) and
# running a single non-causal SDPA with an additive mask that bakes in the
# per-position causal upper bound (and the sliding-window lower bound on
# sliding layers). Ported from the gemma4_cody packed-verify path.


def _packed_sdpa_grid(config, mesh_device):
    """SDPA grid for the packed verify: the (8,4) grid for global layers (head_dim 512, the planner sizes its L1 per
    core against that grid), the full device grid for sliding layers. The single-token decode's batch-1 global layers
    take the whole grid over a bfp8 cache; the packed verify has not been measured there."""
    if config.head_dim >= 512:
        return ttnn.CoreCoord(8, 4)
    device_grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreCoord(device_grid.x, device_grid.y)


def _verify_head_splits(
    B,
    H_local,
    nkv_local,
    P,
    head_dim,
    grid=None,
    max_cores_per_head=16,
    k_chunk=64,
    kv_tile_bytes=SDPA_DECODE_TILE_BYTES,
    extra_bytes=0,
    budget_bytes=None,
):
    """How many QUERY-HEAD-wise sub-ops to split the packed-verify SDPA into.

    The packed verify folds ``H_local*P`` query rows onto ``nkv_local`` KV
    groups; the per-core SDPA-decode CBs (Q, QK scores, and the flash
    cross-core reduction buffer) scale with the packed query-head tile count
    ``PNHt = H_local*P/32``. Global-attention layers (head_dim=512 ⇒ DHt 16,
    single local KV head at high TP) can overflow the 1.5 MB L1 even at the
    default k_chunk; we split the packed-head dim across ``n`` full-grid ops +
    a concat, carrying ``H_local/n`` heads each. Sliding layers (head_dim 256)
    fit and are left as one op.

    Enabled ONLY when ``nkv_local == 1``: then every query head maps to the
    lone KV head, so each sub-op can SHARE the full (paged) K/V cache +
    page_table with NO cache slicing — any head subset is GQA-correct. With >1
    local KV head a head split would need per-op cache slicing; such configs
    are left unsplit.

    ``GEMMA4_PV_SDPA_HEAD_SPLITS`` overrides the count (clamped to the largest
    valid split <= the request). ``max_cores_per_head``, ``k_chunk``,
    ``kv_tile_bytes`` (the cache format's tile), ``extra_bytes`` (the untiled
    page-table CB) and ``budget_bytes`` are the packed SDPA plan's
    (``plan_packed_sdpa``), so the split and the config read the same CB model."""
    if nkv_local != 1:
        return 1
    # Valid splits: whole heads per op AND 32-tile-aligned query-row slices.
    valid = [d for d in range(1, H_local + 1) if H_local % d == 0 and ((H_local // d) * P) % 32 == 0]
    if not valid:
        return 1
    env = os.environ.get("GEMMA4_PV_SDPA_HEAD_SPLITS")
    if env:
        want = max(1, min(int(env), H_local))
        return max(d for d in valid if d <= want)
    if (H_local * P) * head_dim <= 8192:  # not heavy (sliding) — one op fits
        return 1
    # sdpa_decode_program_factory's core allocation (num_kv_heads == 1) and CB
    # bytes (custom_ops.grid), with the packed sdpa_program_config's grid,
    # max_cores_per_head_batch and k_chunk, against the planner's budget (15/16
    # of the L1 above the unreserved base the CBs start at).
    if grid is None:
        grid = 32 if head_dim >= 512 else 64
    if budget_bytes is None:
        budget_bytes = sdpa_decode_cb_budget_bytes()
    cores_per_head = sdpa_decode_cores_per_head(grid, B, 1, max_cores_per_head)
    DHt, Sk = head_dim // 32, max(1, k_chunk // 32)

    def per_core_bytes(n):
        return sdpa_decode_cb_bytes((H_local * P) // (32 * n), DHt, Sk, cores_per_head, kv_tile_bytes, extra_bytes)

    for d in valid:  # ascending ⇒ smallest split that fits the budget
        if per_core_bytes(d) <= budget_bytes:
            return d
    return valid[-1]


def packed_verify_mask_span(table_cols, block_size, mask_cols):
    """The keys a paged packed verify attends: ``table_cols * block_size``, which must equal the mask width.

    sdpa_decode checks a mask against K only for unpaged calls; paged, it attends the table width times the block
    size and reads every mask row tile past the first at THAT pitch (``mask_tile_id = start + row * PSt``), so a
    table narrower than the mask shifts the mask of packed rows 32 and up by the difference (the 31B at TP4:
    a 4096-column sliding table under a 263168-key mask moved three of eight local heads 1024 keys ahead).
    Returns the span; raises on a mismatch."""
    table_cols, block_size, mask_cols = int(table_cols), int(block_size), int(mask_cols)
    span = table_cols * block_size
    if span != mask_cols:
        raise ValueError(
            f"packed verify: the page table spans {table_cols} x {block_size} = {span} keys but the mask is "
            f"{mask_cols} wide; the kernel attends the table span and reads the mask's row tiles at that pitch, so "
            f"the two must be equal (serving_verify_table_widths sizes the table from the mask)"
        )
    return span


def _packed_verify_sdpa(
    q_packed, k, v, layer_page_table, attn_mask, scale, pc, n_splits, B, H_local, P, head_dim, block_size, num_kv_heads
):
    """Run the packed-verify decode-SDPA, optionally head-split into
    ``n_splits`` full-grid ops + a concat (see ``_verify_head_splits``).
    Splitting SHARES the (paged) K/V cache + page_table across sub-ops unsliced
    — valid only for a single local KV head — and slices only Q and the
    additive mask on the packed query-head dim. Does NOT free its inputs.
    Returns [1, B, H_local*P, head_dim]."""

    # Single-device (TP=1) only: the packed layout folds P candidates into the
    # query-heads dim, so with all heads on one device the SDPA decode's
    # padded-heads-per-core (PNHt) can be odd (e.g. 3). ttnn asserts
    # MUL_BCAST_GRANULARITY = min(PNHt * Sk_chunk_t, dst_size) is a power of 2; with
    # the op-default dst_size=8 and Sk_chunk_t=2 that leaves PNHt*2=6 and aborts.
    # Match the op's default compute config (HiFi2) but flip fp32_dest_acc_en=True so
    # dst_size=4 ⇒ min(2*PNHt, 4) is always 2 or 4 (power of 2), no k_chunk/L1
    # increase, and fp32 accumulation is strictly higher precision. On multi-device
    # meshes heads split so PNHt is already power-of-2 — leave the op default (None)
    # there so the packed path stays identical to upstream.
    if layer_page_table is not None and attn_mask is not None:
        packed_verify_mask_span(layer_page_table.shape[-1], block_size, attn_mask.shape[-1])
    _dev = k.device()
    _num_dev = getattr(_dev, "get_num_devices", lambda: 1)()
    # Single-device ONLY, as originally shipped. Forcing fp32_dest_acc on a
    # MESH was tried against the long-S_k greedy drift (128k packed verify
    # diverges at ~tok 41 with op defaults) and made it far worse -- 0.06/5
    # accepted, divergent from token 0 -- so the mesh keeps the op default and
    # the drift is handled by gating packed verify per ISL tier instead.
    compute_kernel_config = (
        ttnn.init_device_compute_kernel_config(
            _dev.arch(),
            math_fidelity=sdpa_math_fidelity(ttnn.MathFidelity.HiFi2),
            math_approx_mode=True,
            fp32_dest_acc_en=sdpa_fp32_dest_acc_en(True),
            packer_l1_acc=False,
        )
        if _num_dev == 1
        else None
    )

    def _call(q, m):
        if layer_page_table is not None:
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                k,
                v,
                page_table_tensor=layer_page_table,
                is_causal=False,
                attn_mask=m,
                scale=scale,
                sliding_window_size=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=pc,
                compute_kernel_config=compute_kernel_config,
                paged_cache_geometry=ttnn.PagedCacheGeometryOverride(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                ),
            )
        return ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=m,
            scale=scale,
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            compute_kernel_config=compute_kernel_config,
        )

    if n_splits <= 1:
        return _call(q_packed, attn_mask)
    s_k = attn_mask.shape[3] if attn_mask is not None else None
    rows_per = (H_local // n_splits) * P  # packed query rows per sub-op (32-aligned)
    parts = []
    for s in range(n_splits):
        r0, r1 = s * rows_per, (s + 1) * rows_per
        q_i = ttnn.slice(q_packed, [0, 0, r0, 0], [1, B, r1, head_dim])
        m_i = ttnn.slice(attn_mask, [0, 0, r0, 0], [B, 1, r1, s_k]) if attn_mask is not None else None
        out_i = _call(q_i, m_i)
        ttnn.deallocate(q_i)
        if m_i is not None:
            ttnn.deallocate(m_i)
        parts.append(out_i)
    out = ttnn.concat(parts, dim=2)  # [1, B, H_local*P, head_dim]
    for p in parts:
        ttnn.deallocate(p)
    return out


# ── GEMMA4_SPEC_VERIFY_MULTIPOS: a global layer's verify on the causal multi-position paged SDPA ─────────────────────
# tt/verify_multipos.py holds the knob, the planner, the layouts (torch emulations of the ops below) and the argument
# why "attend [0, c_b + p]" is the packed mask's rows; tt/spec_decode.py plans the record and uploads the positions.


def multipos_verify_reason(multipos, config, weights, kv_cache, kv_modulo, B, P, tp):
    """None when this packed verify call runs the multi-position SDPA with ``multipos`` (``{"plan", "cur_pos",
    "page_table"}``), else why not: the layer's static check (``multipos_layer_reason``) and the record's plan
    matching the call (rows, candidates, heads, the page table's ``B * G`` rows). Static values only (trace-safe)."""
    plan = multipos["plan"]
    h_local = config.num_attention_heads // tp
    reason = multipos_layer_reason(
        is_sliding=config.is_sliding,
        cache_position_modulo=config.cache_position_modulo,
        kv_modulo=kv_modulo,
        nkv_local=1 if weights.kv_replicated else config.num_key_value_heads // tp,
        h_local=h_local,
        head_dim=config.head_dim,
        cache_shape=tuple(kv_cache[0].padded_shape),
    )
    if reason is None and not plan.fits:
        reason = plan.describe()
    if reason is None and (plan.batch, plan.packed_p, plan.h_local, plan.head_dim) != (B, P, h_local, config.head_dim):
        reason = (
            f"the record's plan is for {plan.batch} rows x {plan.packed_p} candidates, {plan.h_local} heads of "
            f"{plan.head_dim}; the call has {B} x {P}, {h_local} of {config.head_dim}"
        )
    if reason is None and int(multipos["page_table"].shape[0]) != plan.sdpa_batch:
        reason = f"a {int(multipos['page_table'].shape[0])}-row page table for {plan.sdpa_batch} batch groups"
    return reason


def _free_source(src, out):
    """Free ``src`` once ``out`` was made from it, unless ttnn handed back the same buffer (a no-op layout change, pad
    or slice, or a reshape view)."""
    if out is not src and int(out.buffer_address()) != int(src.buffer_address()):
        ttnn.deallocate(src)


def _multipos_verify_rows(tt_q, k_cache, v_cache, multipos, B, P, H_local, head_dim):
    """The global layer's packed verify attention as ONE causal multi-position paged SDPA
    (``spec_multi_pos_tiles = Tg``): each record row's KV is read once per group of ``Tg`` candidates, up to the
    group's last position, instead of the rung's whole span under a mask.

    ``tt_q [1, H_local, B*P (tile-padded), head_dim]`` (the prefill-style split after RoPE, row ``b*P + p``; consumed)
    -> ROW_MAJOR, logical rows -> permute ``[1, B*P, H_local, D]`` -> zero rows ``H_local..31`` -> reshape
    ``[1, B*G, Tg*32, D]`` (candidate ``p = g*Tg + j`` of row ``b`` = row tile ``j`` of group ``b*G + g``) -> TILE
    DRAM bf16; the op against ``multipos["page_table"]`` ``[B*G, cols]`` and ``multipos["cur_pos"]`` ``[B*P]``
    (entry ``b*P + p = c_b + p``); the output ``[1, B*G, Tg*32, D]`` -> ROW_MAJOR -> ``[1, B*P, 32, D]`` -> rows
    ``0..H_local-1``. ``verify_multipos.multipos_q_layout`` / ``multipos_out_rows`` are these steps in torch.
    Returns ``[1, B*P, H_local, head_dim]`` ROW_MAJOR in L1, the rows the o_proj tail takes."""
    plan = multipos["plan"]
    tg, groups = plan.tg, plan.groups
    rows = B * P
    dram = ttnn.DRAM_MEMORY_CONFIG
    q = ttnn.to_layout(tt_q, ttnn.ROW_MAJOR_LAYOUT)
    _free_source(tt_q, q)
    if q.shape[2] != rows:
        q_rows = ttnn.slice(q, [0, 0, 0, 0], [1, H_local, rows, head_dim])
        _free_source(q, q_rows)
        q = q_rows
    q_c = ttnn.permute(q, (0, 2, 1, 3))  # [1, B*P, H_local, D]: candidate-major, heads within a candidate
    _free_source(q, q_c)
    if H_local < MULTIPOS_TILE:
        q_pad = ttnn.pad(q_c, [(0, 0), (0, 0), (0, MULTIPOS_TILE - H_local), (0, 0)], value=0.0)
        _free_source(q_c, q_pad)
        q_c = q_pad
    q_g = ttnn.reshape(q_c, (1, B * groups, tg * MULTIPOS_TILE, head_dim))
    q_mp = ttnn.to_layout(q_g, ttnn.TILE_LAYOUT, memory_config=dram)
    _free_source(q_g, q_mp)
    if q_mp.dtype != ttnn.bfloat16:  # the op takes a bf16 Q
        q_bf16 = ttnn.typecast(q_mp, ttnn.bfloat16)
        ttnn.deallocate(q_mp)
        q_mp = q_bf16
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_mp,
        k_cache,
        v_cache,
        page_table_tensor=multipos["page_table"],
        cur_pos_tensor=multipos["cur_pos"],
        is_causal=True,
        scale=1.0,
        memory_config=dram,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(int(plan.grid[0]), int(plan.grid[1])),
            q_chunk_size=32,
            k_chunk_size=plan.k_chunk,
            exp_approx_mode=False,
            max_cores_per_head_batch=plan.max_cores_per_head_batch,
        ),
        spec_multi_pos_tiles=tg,
    )
    ttnn.deallocate(q_mp)
    out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    _free_source(out, out_rm)
    out_c = ttnn.reshape(out_rm, (1, rows, MULTIPOS_TILE, head_dim))  # candidate b*P + p, its heads in rows 0..
    sdpa_rows = ttnn.slice(out_c, [0, 0, 0, 0], [1, rows, H_local, head_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
    _free_source(out_c, sdpa_rows)
    return sdpa_rows


def _packed_verify_out(tt_sdpa, weights, config, mesh_config, ccl_manager):
    """The packed verify's tail from the attention rows ``[1, B*P, H_local, head_dim]`` ROW_MAJOR (L1): concat heads,
    o_proj and the row all-reduce. Returns ``[1, 1, B*P, hidden_size]``."""
    l1 = ttnn.L1_MEMORY_CONFIG
    tt_sdpa = ttnn.to_layout(tt_sdpa, ttnn.TILE_LAYOUT, memory_config=l1)

    # Decode-layout concat: transpose batch/heads then nlp_concat_heads.
    tt_sdpa = ttnn.transpose(tt_sdpa, 1, 2)
    tt_out = ttnn.experimental.nlp_concat_heads(tt_sdpa, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tt_sdpa)
    # one-shot all-reduce (slot "o") when the B*P packed rows fit one row tile; above it the o_proj runs its
    # prefill config and the interleaved output takes the composite all-reduce (see the single-token path)
    rows = int(tt_out.shape[-2])
    tt_out = apply_output_projection(tt_out, weights, **oneshot_matmul_kwargs(mesh_config, ccl_manager))
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size, slot="o", rows=rows)
    return tt_out


# ── GEMMA4_SPEC_VERIFY_PREFIX_LSE: a global layer's verify as a causal prefix decode + the candidates' own keys ──────
# tt/verify_prefix_lse.py holds the knob, the gate, the planner and torch emulations of the ops below; tt/spec_decode.py
# plans the record, builds the constant fresh mask and uploads the prefix positions and biases per replay.


def prefix_lse_verify_reason(prefix_lse, config, weights, kv_cache, kv_modulo, B, P, tp, is_kv_shared=False):
    """None when this packed verify call runs the prefix + fresh split with ``prefix_lse`` (``{"plan", "cur_pos",
    "bias", "fresh_mask"}``), else why not: the layer's static check (``prefix_lse_layer_reason``) and the record's
    plan matching the call (rows, candidates, heads, head dim, the cache's effective block: a buffer shared with the
    sliding layers is read through the geometry override, as ``pair_split_sdpa`` reads it) and, for a pair plan, the
    layer's pair view (``weights.pair``, a GEMMA4_SDPA_PAIR_SPLIT build). Static values only (trace-safe)."""
    plan = prefix_lse["plan"]
    h_local = config.num_attention_heads // (tp // 2 if plan.pair else tp)
    reason = prefix_lse_layer_reason(
        is_sliding=config.is_sliding,
        cache_position_modulo=config.cache_position_modulo,
        kv_modulo=kv_modulo,
        nkv_local=1 if weights.kv_replicated else config.num_key_value_heads // tp,
        head_dim=config.head_dim,
        cache_shape=tuple(kv_cache[0].padded_shape),
        is_kv_shared=is_kv_shared,
    )
    if reason is None and not plan.fits:
        reason = plan.describe()
    if reason is None and plan.pair and getattr(weights, "pair", None) is None:
        reason = "the record plans the pair split but the layer has no pair view (GEMMA4_SDPA_PAIR_SPLIT build)"
    if reason is None:
        block = effective_block_size(kv_cache[0], config.head_dim, 1)
        if block != plan.geometry.block_size or tuple(kv_cache[1].padded_shape) != tuple(kv_cache[0].padded_shape):
            reason = (
                f"the record's plan reads blocks of {plan.geometry.block_size}; the layer's K / V caches "
                f"{tuple(kv_cache[0].padded_shape)} / {tuple(kv_cache[1].padded_shape)} give {block}"
            )
    if reason is None and (plan.batch, plan.packed_p, plan.h_local, plan.head_dim) != (
        B,
        P,
        h_local,
        int(config.head_dim),
    ):
        reason = (
            f"the record's plan is for {plan.batch} rows x {plan.packed_p} candidates, {plan.h_local} heads of "
            f"{plan.head_dim}; the call has {B} x {P}, {h_local} of {config.head_dim}"
        )
    return reason


def _fresh_kv_rows(t, B, P, head_dim, cache_dtype):
    """The candidates' K or V per record row for the fresh part: ``t [1, 1, B*P (tile-padded), D]`` (the prefill-style
    split after its norm, and RoPE for K; NOT consumed, the KV write takes it next) -> ROW_MAJOR, logical rows ->
    ``[1, B, P, D]`` -> zero rows ``P..31`` -> ``[1, B, 32, D]`` bf16 TILE in DRAM
    (``verify_prefix_lse.fresh_kv_rows``). Over a bfloat8_b cache the rows go through that format and back, so the fresh
    keys are the bytes the cache now holds (the masked verify and the plain decode read the candidates from the
    cache): bfloat8_b shares an exponent over 16 consecutive elements of one tile row, here and in the cache one
    position's 16 head-dim entries, and a cache shared with the sliding layers (read through the geometry override)
    only moves the whole tile a position lands in."""
    dram = ttnn.DRAM_MEMORY_CONFIG
    rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    if int(rm.buffer_address()) == int(t.buffer_address()):  # never free the caller's K / V
        rm = ttnn.clone(rm, memory_config=dram)
    if rm.shape[2] != B * P:
        rows = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, B * P, head_dim])
        _free_source(rm, rows)
        rm = rows
    per_row = ttnn.reshape(rm, (1, B, P, head_dim))
    padded = ttnn.pad(per_row, [(0, 0), (0, 0), (0, PREFIX_TILE - P), (0, 0)], value=0.0)
    _free_source(per_row, padded)
    out = ttnn.to_layout(padded, ttnn.TILE_LAYOUT, memory_config=dram)
    _free_source(padded, out)
    if out.dtype != ttnn.bfloat16:
        out_bf16 = ttnn.typecast(out, ttnn.bfloat16)
        ttnn.deallocate(out)
        out = out_bf16
    if cache_dtype is not None and cache_dtype != ttnn.bfloat16:
        rounded = ttnn.typecast(out, cache_dtype)
        ttnn.deallocate(out)
        out = ttnn.typecast(rounded, ttnn.bfloat16)
        ttnn.deallocate(rounded)
    return out


def _prefix_two_way_weights(lse1, lse2, l2):
    """``(w1, c2)`` of the prefix and the fresh part (``verify_prefix_lse.merge_prefix_fresh_torch``): ``d =
    clamp(lse1 - lse2)``, ``w1 = sigmoid(d)``, ``c2 = sigmoid(-d) / l2`` (the fresh weight over its own normaliser).
    ``lse1 / lse2 / l2 [1, B, R, 1]`` fp32; consumes ``lse1`` and ``lse2``."""
    diff = ttnn.subtract(lse1, lse2)
    ttnn.deallocate(lse1)
    ttnn.deallocate(lse2)
    clamped = ttnn.clamp(diff, min=-LOGIT_CLAMP, max=LOGIT_CLAMP)
    ttnn.deallocate(diff)
    w1 = ttnn.sigmoid(clamped)
    negated = ttnn.multiply(clamped, -1.0)
    ttnn.deallocate(clamped)
    w2 = ttnn.sigmoid(negated)
    ttnn.deallocate(negated)
    c2 = ttnn.div(w2, l2)
    ttnn.deallocate(w2)
    return w1, c2


def _prefix_pair_weights(lse_s, lse_f, l2, selector, mesh_config, ccl_manager):
    """``(w_self, c_fresh)`` of the pair split's three-way merge (``verify_prefix_lse.pair_prefix_merge_torch``): this
    chip's half ``lse_s`` (bias included), the partner's (the row all-gather as in ``pair_merge``, times
    ``relu(-selector) = e_{c^1}`` and summed over the device dim: the partner's value exactly, so an empty half's
    sentinel never costs the other two terms precision), and the fresh ``lse_f``: ``top`` their max, ``a_i =
    exp(lse_i - top)``, ``S = (a_self + a_partner) + a_fresh``, ``w_self = a_self / S``, ``c_fresh = 0.5 * a_fresh /
    (S * l2)`` (both chips of the pair add half of the identical fresh term). ``[1, B, R, 1]`` fp32; consumes ``lse_s``
    and ``lse_f``."""
    B, R = int(lse_s.shape[1]), int(lse_s.shape[2])
    rows = ttnn.reshape(lse_s, (1, 1, B * R, 1))
    if int(rows.buffer_address()) == int(lse_s.buffer_address()):
        rows = ttnn.clone(rows)  # the gather frees its input and lse_s is read below
    gathered = ccl_allgather(rows, mesh_config, ccl_manager, dim=1)  # [1, tp, B*R, 1]; consumes rows
    negated = ttnn.multiply(selector, -1.0)
    partner = ttnn.relu(negated)  # e_{c^1}: the selector row is e_c - e_{c^1}
    ttnn.deallocate(negated)
    picked = ttnn.multiply(gathered, partner)
    ttnn.deallocate(gathered)
    ttnn.deallocate(partner)
    lse_p_rows = ttnn.sum(picked, dim=1, keepdim=True)  # [1, 1, B*R, 1]: one non-zero term, the partner's lse
    ttnn.deallocate(picked)
    lse_p = ttnn.reshape(lse_p_rows, (1, B, R, 1))
    _free_source(lse_p_rows, lse_p)
    top_sp = ttnn.maximum(lse_s, lse_p)
    top = ttnn.maximum(top_sp, lse_f)
    ttnn.deallocate(top_sp)
    exps = []
    for lse in (lse_s, lse_p, lse_f):
        shifted = ttnn.subtract(lse, top)
        ttnn.deallocate(lse)
        exps.append(ttnn.exp(shifted, fast_and_approximate_mode=False))
        ttnn.deallocate(shifted)
    ttnn.deallocate(top)
    a_s, a_p, a_f = exps
    sum_sp = ttnn.add(a_s, a_p)
    ttnn.deallocate(a_p)
    total = ttnn.add(sum_sp, a_f)
    ttnn.deallocate(sum_sp)
    w_s = ttnn.div(a_s, total)
    ttnn.deallocate(a_s)
    norm = ttnn.multiply(total, l2)
    ttnn.deallocate(total)
    c_f_full = ttnn.div(a_f, norm)
    ttnn.deallocate(a_f)
    ttnn.deallocate(norm)
    c_f = ttnn.multiply(c_f_full, 0.5)
    ttnn.deallocate(c_f_full)
    return w_s, c_f


def _prefix_fresh_merge(q, out1, stats, k_fresh, v_fresh, fresh_mask, bias, compute_kernel_config, pair=None):
    """One prefix call's rows merged with the fresh part (``verify_prefix_lse.fresh_attention_torch`` and
    ``merge_prefix_fresh_torch`` op for op). ``q [1, B, R, D]`` bf16 TILE (not consumed), ``out1 [1, B, R, D]`` and
    ``stats [1, B, PNH, 64]`` the prefix call's outputs (consumed), ``k_fresh / v_fresh [1, B, 32, D]`` bf16 (keys
    ``P..31`` zero), ``fresh_mask [1, B, R, 32]`` and ``bias [1, B, R, 1]`` fp32. The fresh scores, softmax
    statistics and the whole merge stay fp32; the scale is 1.0 (Gemma's), as the packed verify's. ``pair``:
    ``(selector, mesh_config, ccl_manager)`` when the prefix call ran pair-split (``_prefix_pair_weights``). Returns
    ``[1, B, R, D]`` bf16 TILE."""
    f32 = ttnn.float32
    B, R = int(q.shape[1]), int(q.shape[2])
    # fresh part: scores against the candidates' own keys, the constant causal mask, its own (m, l), unnormalised P @ V
    scores = ttnn.matmul(q, k_fresh, transpose_b=True, dtype=f32, compute_kernel_config=compute_kernel_config)
    masked = ttnn.add(scores, fresh_mask)
    ttnn.deallocate(scores)
    m2 = ttnn.max(masked, dim=-1, keepdim=True)  # [1, B, R, 1]
    shifted = ttnn.subtract(masked, m2)
    ttnn.deallocate(masked)
    e = ttnn.exp(shifted, fast_and_approximate_mode=False)
    ttnn.deallocate(shifted)
    l2 = ttnn.sum(e, dim=-1, keepdim=True)  # [1, B, R, 1]
    o2 = ttnn.matmul(e, v_fresh, dtype=f32, compute_kernel_config=compute_kernel_config)  # [1, B, R, D]
    ttnn.deallocate(e)
    ln_l2 = ttnn.log(l2)
    lse2 = ttnn.add(m2, ln_l2)
    ttnn.deallocate(m2)
    ttnn.deallocate(ln_l2)
    # prefix statistics: m (raw max) at column 0, l at column 32, bf16 -> fp32 before the log (see pair_merge)
    stats32 = ttnn.typecast(stats, f32)
    ttnn.deallocate(stats)
    m1 = ttnn.slice(stats32, [0, 0, 0, STATS_M_COL], [1, B, R, STATS_M_COL + 1])
    l1 = ttnn.slice(stats32, [0, 0, 0, STATS_L_COL], [1, B, R, STATS_L_COL + 1])
    ttnn.deallocate(stats32)
    ln_l1 = ttnn.log(l1)
    lse1 = ttnn.add(m1, ln_l1)
    for t in (m1, l1, ln_l1):
        ttnn.deallocate(t)
    lse1_b = ttnn.add(lse1, bias)  # EMPTY_PREFIX_LSE on a row without prefix
    ttnn.deallocate(lse1)
    if pair is None:
        w1, c2 = _prefix_two_way_weights(lse1_b, lse2, l2)
    else:
        w1, c2 = _prefix_pair_weights(lse1_b, lse2, l2, *pair)
    ttnn.deallocate(l2)
    o1 = ttnn.typecast(out1, f32)
    ttnn.deallocate(out1)
    part1 = ttnn.multiply(o1, w1)
    ttnn.deallocate(o1)
    ttnn.deallocate(w1)
    part2 = ttnn.multiply(o2, c2)
    ttnn.deallocate(o2)
    ttnn.deallocate(c2)
    merged = ttnn.add(part1, part2)
    ttnn.deallocate(part1)
    ttnn.deallocate(part2)
    out = ttnn.typecast(merged, ttnn.bfloat16)
    ttnn.deallocate(merged)
    return out


def _prefix_lse_verify_rows(
    tt_q, k_fresh, v_fresh, k_cache, v_cache, page_table, prefix_lse, B, P, H_local, head_dim, mesh_device, pair=None
):
    """The global layer's packed verify attention as the prefix + fresh split (tt/verify_prefix_lse.py).

    ``tt_q [1, H_local, B*P (tile-padded), D]`` (after RoPE; consumed) -> ROW_MAJOR, logical rows -> head-major
    ``[1, B, H_local*P, D]`` (row ``h*P + p``, the masked verify's pack) -> per prefix call ``s`` (``plan.splits``
    calls of ``R = (H_local / splits) * P`` rows, whole heads) a TILE DRAM bf16 slice ->
    ``SdpaDecodeLse`` causal at ``prefix_lse["cur_pos"]`` (``[B]`` int32, ``max(c_b - 1, 0)``) over the layer's verify
    table, then ``_prefix_fresh_merge`` with the fresh K / V -> concat over the calls -> ``[1, B*P, H_local, D]``
    ROW_MAJOR in L1, the rows the o_proj tail takes (``verify_prefix_lse.prefix_q_rows`` / ``prefix_out_rows``).
    ``pair`` (``(selector, mesh_config, ccl_manager)``, the pair view): each prefix call runs ``pair_split=True`` and
    merges three ways with the partner's half."""
    from models.demos.gemma4.custom_ops.sdpa_decode_lse import SdpaDecodeLse

    plan = prefix_lse["plan"]
    splits = int(plan.splits)
    rows = (H_local // splits) * P
    dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    q = ttnn.to_layout(tt_q, ttnn.ROW_MAJOR_LAYOUT)
    _free_source(tt_q, q)
    if q.shape[2] != B * P:
        q_rows = ttnn.slice(q, [0, 0, 0, 0], [1, H_local, B * P, head_dim])
        _free_source(q, q_rows)
        q = q_rows
    q5 = ttnn.reshape(q, (1, H_local, B, P, head_dim))
    q_perm = ttnn.permute(q5, (0, 2, 1, 3, 4))  # [1, B, H_local, P, D]
    _free_source(q5, q_perm)
    q_rm = ttnn.reshape(q_perm, (1, B, H_local * P, head_dim))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(int(plan.grid[0]), int(plan.grid[1])),
        q_chunk_size=32,
        k_chunk_size=int(plan.k_chunk),
        exp_approx_mode=False,
        max_cores_per_head_batch=int(plan.max_cores_per_head),
    )
    fresh_cc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    kwargs = dict(
        scale=1.0,
        num_kv_heads=1,
        block_size=effective_block_size(k_cache, head_dim, 1),
        cache_position_modulo=None,
        program_config=program_config,
        compute_kernel_config=None,  # the stock call's default (HiFi2, fp32 dest off), as the plain decode's
        # GEMMA4_KV_PAIR_SHARD: the chip holds half of every logical block, the half-block mode reads it
        pair_split=pair is not None and not kv_pair_shard_enabled(),
        half_block=pair is not None and kv_pair_shard_enabled(),
    )
    parts = []
    for s in range(splits):
        q_s = q_rm if splits == 1 else ttnn.slice(q_rm, [0, 0, s * rows, 0], [1, B, (s + 1) * rows, head_dim])
        q_t = ttnn.to_layout(q_s, ttnn.TILE_LAYOUT, memory_config=dram)
        if q_s is not q_rm:
            _free_source(q_s, q_t)
        if q_t.dtype != ttnn.bfloat16:  # the op takes a bf16 Q
            q_bf16 = ttnn.typecast(q_t, ttnn.bfloat16)
            ttnn.deallocate(q_t)
            q_t = q_bf16
        reason = SdpaDecodeLse.unsupported_reason(
            q_t, k_cache, v_cache, prefix_lse["cur_pos"], page_table, memory_config=dram, **kwargs
        )
        if reason is not None:
            raise ValueError(f"packed verify: sdpa_decode_lse refuses the prefix call ({plan.describe()}): {reason}")
        out1, stats = SdpaDecodeLse.op(
            q_t,
            k_cache,
            v_cache,
            prefix_lse["cur_pos"],
            page_table,
            memory_config=dram,
            stats_memory_config=dram,
            **kwargs,
        )
        parts.append(
            _prefix_fresh_merge(
                q_t, out1, stats, k_fresh, v_fresh, prefix_lse["fresh_mask"], prefix_lse["bias"], fresh_cc, pair=pair
            )
        )
        ttnn.deallocate(q_t)
    ttnn.deallocate(q_rm)
    if splits == 1:
        merged = ttnn.to_layout(parts[0], ttnn.ROW_MAJOR_LAYOUT, memory_config=l1)
        _free_source(parts[0], merged)
    else:
        rm_parts = []
        for part in parts:
            part_rm = ttnn.to_layout(part, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
            _free_source(part, part_rm)
            rm_parts.append(part_rm)
        merged = ttnn.concat(rm_parts, dim=2, memory_config=l1)  # [1, B, H_local*P, D], head-major again
        for part_rm in rm_parts:
            ttnn.deallocate(part_rm)
    if merged.shape[2] != H_local * P:
        merged_rows = ttnn.slice(merged, [0, 0, 0, 0], [1, B, H_local * P, head_dim])
        _free_source(merged, merged_rows)
        merged = merged_rows
    out5 = ttnn.reshape(merged, (1, B, H_local, P, head_dim))
    out_perm = ttnn.permute(out5, (0, 1, 3, 2, 4))  # [1, B, P, H_local, D]
    _free_source(out5, out_perm)
    return ttnn.reshape(out_perm, (1, B * P, H_local, head_dim))


def _packed_fill_kv_loopfree_embed(cache, staging, new_seq, embed_idx, hot_pt):
    """Loop-free packed KV-cache write via PERSISTENT STAGING and a
    transpose-free ``ttnn.embedding`` row-gather merge — replaces the
    per-position ``paged_update_cache`` loop, and never reads the committed
    cache on the hot path.

    ``staging`` is this layer's resident hot-block copy
    ``[1, nkv, S2, head_dim]`` (S2 = max_batch * PV_HOT_BLOCKS * block_size),
    slot-indexed: slot ``s`` owns seq ``[s*BLK*bs, (s+1)*BLK*bs)`` and already
    holds the committed prefix of its hot block(s) — seeded at the first
    verify, maintained step-to-step by the spec-decode driver. ``new_seq`` is
    the freshly projected/normed/RoPE'd K (or V) ``[1, nkv, n_new, head_dim]``
    (user-major / position-minor rows, n_new tile-aligned — caller pads).

    Per step (all fixed-shape ⇒ trace-safe; cost scales with S2, never with
    max_num_blocks):
      1. merge: ``ttnn.embedding`` row-gathers the flattened
         ``concat([staging, new_seq])`` — committed positions copy from staging
         (identity, or shifted one block on a rollover), the P new positions
         pull from ``new_seq``. ``embed_idx`` ([1, nkv*S2] u32) bakes the
         per-head row offset in: ``embed_idx[h*S2 + j] = h*(S2+n_new) +
         merge_idx[j]``.
      2. ``assign`` the merged hot blocks back into ``staging`` (persists for
         the next step — no cache read needed next time).
      3. one ``paged_fill_cache`` writes each slot's hot block(s) to its
         physical page(s) named by ``hot_pt`` (-1 = skip ⇒ idle slots / the
         unused spill block are left untouched).
    """
    dram = ttnn.DRAM_MEMORY_CONFIG
    nkv = staging.shape[1]
    S2 = staging.shape[2]
    head_dim = staging.shape[3]
    src_seq = S2 + new_seq.shape[2]

    # ① merge resident staging with this step's new K/V — no committed-cache read.
    new_dram = ttnn.to_memory_config(new_seq, dram)
    src = ttnn.concat([staging, new_dram], dim=2, memory_config=dram)  # [1, nkv, src_seq, hd] TILE
    ttnn.deallocate(new_dram)
    # Per-head row-gather merge. Folding all nkv heads into one row axis
    # (``reshape(src, (nkv*src_seq, hd))`` + a single ``embedding``) mis-orders
    # the 2nd+ head's tiles in TILE layout, corrupting cache head>=1 whenever a
    # device holds more than one local KV head (``nkv_local >= 2`` — i.e. TP
    # configs with ``num_kv_heads // tp >= 2``). Gathering each head's
    # ``[src_seq, hd]`` block on its own keeps every head tile-contiguous.
    # ``embed_idx`` bakes a per-head row offset ``h*src_seq``; head 0's slice is
    # the bare merge_idx, valid for every head's local row-gather.
    base_idx = ttnn.slice(embed_idx, [0, 0], [1, S2])  # head-0 merge_idx (no head offset)
    parts = []
    for h in range(nkv):
        src_h = ttnn.slice(src, [0, h, 0, 0], [1, h + 1, src_seq, head_dim])
        src_h2d = ttnn.reshape(src_h, (src_seq, head_dim))
        merged_h = ttnn.embedding(base_idx, src_h2d, layout=ttnn.TILE_LAYOUT)  # [1, S2, hd]
        parts.append(ttnn.reshape(merged_h, (1, 1, S2, head_dim)))
    ttnn.deallocate(src)
    merged = ttnn.concat(parts, dim=1) if nkv > 1 else parts[0]
    merged = ttnn.to_memory_config(merged, dram)  # [1, nkv, S2, hd]

    # ② persist updated hot blocks for next step, then ③ one fill launch.
    ttnn.assign(merged, staging)
    _paged_fill_cache(cache, merged, hot_pt, batch_idx=0)
    ttnn.deallocate(merged)


def packed_decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    position_idx,
    kv_write_idxs,
    attn_mask,
    packed_p,
    page_table=None,
    ccl_manager=None,
    is_kv_shared=False,
    rope_packed=None,
    kv_staging=None,
    embed_idx=None,
    hot_pt=None,
    kv_modulo=None,
    multipos=None,
    prefix_lse=None,
    kv_write_table=None,
):
    """Packed multi-token decode attention — P query positions/slot in one pass.

    Hoists QKV projection, prefill-style split, per-head norm, and RoPE OUT of
    any per-position loop: those run once on the full B*P tensor. The KV write
    is loop-free when staging is provided (one paged_fill_cache per K/V), else
    a per-position paged_update_cache fallback.

    Args:
        hidden_states: [1, 1, B*P, hidden_size]. Rows user-major / position-
            minor — row u*P+p is slot u's p-th packed token.
        cos_cache, sin_cache: 2D RoPE caches [max_seq_len, head_dim].
        position_idx: [1, B*P] uint32 — RoPE position per row (cur_pos_u + p).
        kv_write_idxs: list of P int32 [B] device tensors — the cache write
            position for each packed position p (fallback per-p loop only).
        attn_mask: [B, 1, H_local*P, S_k] head-major additive mask baking in
            the per-position causal upper bound and (sliding layers) the
            window lower bound. S_k must be a multiple of k_chunk (64).
            None on a global layer the multi-position SDPA serves.
        packed_p: P, the number of packed positions per slot.
        rope_packed: optional pre-gathered (cos_bp, sin_bp) for this layer
            type — identical for all layers of a type, so the caller gathers
            once per type per step.
        kv_staging: optional [k_staging, v_staging] persistent hot-block
            buffers (loop-free write path; see _packed_fill_kv_loopfree_embed).
        embed_idx: [1, nkv_local*S2] uint32 merge gather index (loop-free path).
        hot_pt: [1, max_batch*PV_HOT_BLOCKS] int32 physical fill pages, -1=skip.
        kv_modulo: the per-position KV write's ``cache_position_modulo`` when
            ``page_table`` is a ring over an unbounded pool (the serving
            records' sliding span ring); None = the layer's own
            ``config.cache_position_modulo``.
        multipos: GEMMA4_SPEC_VERIFY_MULTIPOS (global layers of the serving
            records): ``{"plan": MultiposPlan, "cur_pos": [B*P] int32,
            "page_table": [B*G, cols] int32}``; the attention runs the causal
            multi-position SDPA (``_multipos_verify_rows``) when
            ``multipos_verify_reason`` allows, else the masked SDPA below.
        prefix_lse: GEMMA4_SPEC_VERIFY_PREFIX_LSE (global layers of the
            serving records): ``{"plan": PrefixLsePlan, "cur_pos": [B] int32,
            "bias": [1, B, R, 1] fp32, "fresh_mask": [1, B, R, 32] fp32}``;
            the attention runs a causal decode over each row's prefix plus the
            candidates' own K / V, merged through their LSEs
            (``_prefix_lse_verify_rows``), when ``prefix_lse_verify_reason``
            allows, else the masked SDPA below.

    Returns:
        [1, 1, B*P, hidden_size] — attention output for every packed position.
    """
    P = packed_p
    tp = mesh_config.tp if mesh_config else 1
    B = hidden_states.shape[2] // P
    H_local = config.num_attention_heads // tp
    head_dim = config.head_dim
    nkv_local = 1 if weights.kv_replicated else config.num_key_value_heads // tp
    # Bounded sliding is supported: the caller provides ring-aware inputs (the
    # sliding mask spans RING slots, hot_pt names pages in the ring pool with a
    # wrapped spill block, and page_table is this layer's own pool). The SDPA
    # here is position-free -- causality/window live entirely in the explicit
    # additive mask -- and the staging fill writes physical pages, so neither
    # needs cache_position_modulo.
    if kv_cache is None:
        raise ValueError("packed_decode_forward requires a KV cache (it attends through the paged cache)")
    # GEMMA4_SPEC_VERIFY_PREFIX_LSE: decided on static values only, before anything runs (the fresh part needs the K / V
    # this call computes, taken before the KV write)
    prefix_on = False
    if prefix_lse is not None:
        prefix_reason = prefix_lse_verify_reason(
            prefix_lse, config, weights, kv_cache, kv_modulo, B, P, tp, is_kv_shared=is_kv_shared
        )
        prefix_on = prefix_reason is None
        if not prefix_on:
            if attn_mask is None:
                raise ValueError(
                    f"packed verify: no mask was built and the prefix + fresh split refuses: {prefix_reason}"
                )
            prefix_log_once(f"a global layer keeps the masked packed verify: {prefix_reason}")
    # the pair view (GEMMA4_SDPA_PAIR_SPLIT build): the QKV of the chip's whole KV group -- the same Q, K and V on both
    # chips of a pair, K / V those of the 4-head layout, so the KV write below is unchanged -- and the group's o_proj
    if kv_write_table is not None and not (prefix_on and prefix_lse["plan"].pair):
        why = prefix_reason if not prefix_on else "the prefix plan has no pair view"
        raise ValueError(
            f"{KV_PAIR_SHARD_ENV}: a pair-sharded global layer's verify reads only through the pair-view prefix + "
            f"fresh split (GEMMA4_SPEC_VERIFY_PREFIX_LSE): {why}"
        )
    prefix_pair = None
    head_tp = tp
    if prefix_on and prefix_lse["plan"].pair:
        weights = weights.pair
        prefix_pair = (weights.pair_selector, mesh_config, ccl_manager)
        head_tp = tp // 2
        H_local = config.num_attention_heads // head_tp
    write_modulo = kv_modulo if kv_modulo is not None else config.cache_position_modulo
    l1 = ttnn.L1_MEMORY_CONFIG

    # ── ① QKV projection (one call on the full B*P, output kept on L1) ──────
    xqkv = apply_qkv_projection(hidden_states, weights, memory_config=l1)
    qkv_dim = xqkv.shape[-1]

    # ── ② L1 height-sharded MemoryConfig for the fallback paged_update_cache ─
    # ``paged_update_cache`` needs the layout ``nlp_create_qkv_heads_decode``
    # emits; the spec depends only on shape constants, so probe once and cache.
    cache_key = _q_sharded_mem_key(B, qkv_dim, config, weights, head_tp)
    q_sharded_mem = _Q_SHARDED_MEM_CACHE.get(cache_key)
    # The loop-free custom op (GEMMA4_PACKED_KV_WRITE_OP=1) needs no sharded layout: the probe runs only when the
    # loop serves the write.
    # GEMMA4_KV_PAIR_SHARD: the writes go through the per-chip write table (the reads through page_table)
    write_table = kv_write_table if kv_write_table is not None else page_table
    rows_op_reason = (
        packed_rows_op_reason(kv_cache, kv_write_idxs, write_table, B, P, nkv_local, head_dim, config, write_modulo)
        if packed_write_takes_fallback_loop(kv_staging, embed_idx, is_kv_shared)
        else "staging"
    )
    if (
        q_sharded_mem is None
        and rows_op_reason is not None
        and packed_write_takes_fallback_loop(kv_staging, embed_idx, is_kv_shared)
    ):
        probe = ttnn.slice(xqkv, [0, 0, 0, 0], [1, 1, B, qkv_dim])
        q_probe, k_probe, v_probe = split_qkv_heads_decode(
            probe, config, weights.is_global, tp=head_tp, kv_replicated=weights.kv_replicated
        )
        q_sharded_mem = q_probe.memory_config()
        _Q_SHARDED_MEM_CACHE[cache_key] = q_sharded_mem
        for t in (q_probe, k_probe, v_probe, probe):
            ttnn.deallocate(t)

    # ── ②b Prefill-style split → L1 Q/K/V on B*P ────────────────────────────
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=head_tp, kv_replicated=weights.kv_replicated, memory_config=l1
    )
    # Q: [1, H_local, B*P, head_dim], K/V: [1, nkv_local, B*P, head_dim]
    ttnn.deallocate(xqkv)

    # ── ③ Per-head norms (one call each on B*P, kept on L1) ─────────────────
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=l1)
    if not is_kv_shared:
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True, memory_config=l1)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False, memory_config=l1)

    # ── ④ RoPE on B*P, prefill mode (kernel iterates along S=B*P) ───────────
    if rope_packed is not None:
        cos_bp, sin_bp = rope_packed
        owns_rope = False
    else:
        cos_bp = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT))
        sin_bp = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT))
        owns_rope = True
    tt_q = apply_rope(tt_q, cos_bp, sin_bp, token_index=None, memory_config=l1)
    if not is_kv_shared:
        tt_k = apply_rope(tt_k, cos_bp, sin_bp, token_index=None, memory_config=l1)
    if owns_rope:
        ttnn.deallocate(cos_bp)
        ttnn.deallocate(sin_bp)

    if is_kv_shared:
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_v)

    # ── ④' GEMMA4_SPEC_VERIFY_PREFIX_LSE: the candidates' own K / V per record row for the fresh part, taken here
    # because the KV write below consumes tt_k / tt_v ──────────────────────────────────────────────────────────────
    k_fresh = v_fresh = None
    if prefix_on:
        k_fresh = _fresh_kv_rows(tt_k, B, P, head_dim, kv_cache[0].dtype)
        v_fresh = _fresh_kv_rows(tt_v, B, P, head_dim, kv_cache[1].dtype)

    # ── ⑤ KV write — loop-free persistent staging (primary) ────────────────
    if not is_kv_shared and kv_staging is not None and embed_idx is not None:
        if kv_write_table is not None:
            raise ValueError(f"{KV_PAIR_SHARD_ENV}: the staging KV write names physical pages, not per-chip halves")
        # Park Q off L1 (idle until the SDPA pack ⑥); merge intermediates live
        # in DRAM and Q never re-enters L1 before the SDPA call.
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        k_cache_w, v_cache_w = kv_cache
        k_stg, v_stg = kv_staging
        if k_stg.shape[1] != k_cache_w.shape[1]:
            raise ValueError("staging nkv must match cache nkv (HMA-shared caches unsupported)")
        # Pad new rows up to a tile boundary so the merge concat + flatten
        # reshape stay metadata-only; embed_idx is built against the padded
        # src_seq, so padded rows are never gathered.
        n_new = tt_k.shape[2]
        pad = (-n_new) % 32
        if pad:
            k_pad = ttnn.pad(tt_k, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            v_pad = ttnn.pad(tt_v, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            ttnn.deallocate(tt_k)
            ttnn.deallocate(tt_v)
            tt_k, tt_v = k_pad, v_pad
        _packed_fill_kv_loopfree_embed(k_cache_w, k_stg, tt_k, embed_idx, hot_pt)
        _packed_fill_kv_loopfree_embed(v_cache_w, v_stg, tt_v, embed_idx, hot_pt)
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_v)

    # ── ⑤ FALLBACK: the per-position paged_update_cache loop (packed_kv_write_loop, 6 P programs per layer),
    # or with GEMMA4_PACKED_KV_WRITE_OP=1 the loop-free custom op writing the same bytes (2 per layer) ─────
    elif not is_kv_shared:
        tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
        k_cache_w, v_cache_w = kv_cache
        eff_bs = effective_block_size(k_cache_w, head_dim, nkv_local)
        if rows_op_reason is None:
            packed_kv_write_rows(
                k_cache_w,
                v_cache_w,
                tt_k,
                tt_v,
                kv_write_idxs,
                write_table,
                B,
                nkv_local,
                eff_bs,
                write_modulo,
            )
        else:
            packed_kv_write_loop(
                k_cache_w,
                v_cache_w,
                tt_k,
                tt_v,
                kv_write_idxs,
                write_table,
                B,
                P,
                nkv_local,
                head_dim,
                eff_bs,
                q_sharded_mem,
                write_modulo,
            )

    k_cache_use, v_cache_use = kv_cache

    # ── ⑥'' GEMMA4_SPEC_VERIFY_PREFIX_LSE: the keys [0, c_b - 1] as one causal decode per record row (read only up to
    # the row) + the candidates' own keys from k_fresh / v_fresh, merged through their LSEs ────────────────────────────
    if prefix_on:
        tt_sdpa = _prefix_lse_verify_rows(
            tt_q,
            k_fresh,
            v_fresh,
            k_cache_use,
            v_cache_use,
            page_table,
            prefix_lse,
            B,
            P,
            H_local,
            head_dim,
            mesh_device,
            pair=prefix_pair,
        )
        ttnn.deallocate(k_fresh)
        ttnn.deallocate(v_fresh)
        return _packed_verify_out(tt_sdpa, weights, config, mesh_config, ccl_manager)

    # ── ⑥' GEMMA4_SPEC_VERIFY_MULTIPOS: a global layer's causal multi-position SDPA (the K/V of the P candidates is
    # in the cache already, step ⑤) — the keys [0, c_b + p] the mask below would leave, read only up to the row ─────
    if multipos is not None:
        mp_reason = multipos_verify_reason(multipos, config, weights, kv_cache, kv_modulo, B, P, tp)
        if mp_reason is None:
            tt_sdpa = _multipos_verify_rows(tt_q, k_cache_use, v_cache_use, multipos, B, P, H_local, head_dim)
            return _packed_verify_out(tt_sdpa, weights, config, mesh_config, ccl_manager)
        if attn_mask is None:
            raise ValueError(f"packed verify: no mask was built and the multi-position SDPA refuses: {mp_reason}")
        multipos_log_once(f"a global layer keeps the masked packed verify: {mp_reason}")

    # ── ⑥ Head-major pack Q → [1, B, H_local*P, head_dim] → SDPA ────────────
    # ROW_MAJOR on purpose: P (< 32) never lands on a tile axis, so the
    # split/merge reshapes are free views and the rank-5 permute is one strided
    # copy; in TILE the same reshapes re-tile AND pad P→32. The head-major
    # order (h*P+p) is load-bearing: SDPA maps packed query head i → KV head
    # i // group, so a KV group must be a contiguous head block.
    tt_q = ttnn.to_layout(tt_q, ttnn.ROW_MAJOR_LAYOUT)
    # The prefill-style split / RoPE may keep the B*P sequence rows tile-padded
    # to 32 in the logical shape (ttnn-version dependent). Slice back to the
    # logical B*P rows so the head-major pack reshape is volume-exact (padding
    # is always at the tail, so the first B*P rows hold the real positions).
    if tt_q.shape[2] != B * P:
        tt_q_unpad = ttnn.slice(tt_q, [0, 0, 0, 0], [1, H_local, B * P, head_dim])
        ttnn.deallocate(tt_q)
        tt_q = tt_q_unpad
    tt_q = ttnn.reshape(tt_q, (1, H_local, B, P, head_dim))
    tt_q = ttnn.permute(tt_q, (0, 2, 1, 3, 4))  # [1, B, H_local, P, hd]
    tt_q = ttnn.reshape(tt_q, (1, B, H_local * P, head_dim))
    q_packed = ttnn.to_layout(tt_q, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tt_q)

    # The K chunk and the cores per head come from the CB model of the op (custom_ops.grid.plan_packed_sdpa):
    # every Q-shaped CB scales with the packed query rows (H_local * P, in tiles) and the cross-core reduction
    # staging with the cores per head, so a geometry that stacks more local heads (the 31B at TP4: 8 local q
    # heads on the head_dim-512 global layers) overflowed the per-core L1 at the op defaults. GEMMA4_PACKED_SDPA_
    # K_CHUNK / _Q_CHUNK pin the chunks (GEMMA4_PV_K_CHUNK is the earlier name of the K knob); unset, derived. A
    # wider K chunk than 64 is not on the ladder: k_chunk 256 TT_THROWs at program build and 128 buys nothing for
    # the long-S_k greedy drift (tested; the ISL tier gate in spec_decode._fused_packed_enabled handles that).
    _grid = _packed_sdpa_grid(config, mesh_device)
    _plan = plan_packed_sdpa(
        H_local,
        nkv_local,
        P,
        head_dim,
        B,
        _grid.x * _grid.y,
        arch=getattr(mesh_device, "arch", lambda: None)(),
        kv_tile_bytes=sdpa_tile_bytes(getattr(k_cache_use, "dtype", None)),
        extra_bytes=sdpa_decode_page_table_bytes(page_table.shape[-1]) if page_table is not None else 0,
    )
    if not _plan.fits and not getattr(packed_decode_forward, "_overflow_logged", False):
        packed_decode_forward._overflow_logged = True
        logger.warning(
            f"Gemma4 packed verify: no SDPA config on the ladder fits the per-core L1 on the {_grid.x}x{_grid.y} "
            f"grid: {_plan.describe()}"
        )
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=_grid,
        q_chunk_size=_plan.q_chunk,
        k_chunk_size=_plan.k_chunk,
        exp_approx_mode=False,
        max_cores_per_head_batch=_plan.max_cores_per_head_batch,
    )
    n_sdpa_splits = _verify_head_splits(
        B,
        H_local,
        nkv_local,
        P,
        head_dim,
        grid=_grid.x * _grid.y,
        max_cores_per_head=_plan.max_cores_per_head_batch,
        k_chunk=_plan.k_chunk,
        kv_tile_bytes=_plan.kv_tile_bytes,
        extra_bytes=_plan.extra_bytes,
        budget_bytes=_plan.budget_bytes,
    )
    eff_bs_sdpa = effective_block_size(k_cache_use, head_dim, nkv_local)
    tt_sdpa = _packed_verify_sdpa(
        q_packed,
        k_cache_use,
        v_cache_use,
        page_table,
        attn_mask,
        1.0,
        sdpa_program_config,
        n_sdpa_splits,
        B,
        H_local,
        P,
        head_dim,
        eff_bs_sdpa,
        nkv_local,
    )
    ttnn.deallocate(q_packed)

    # ── ⑦ Unpack head-major SDPA output → concat heads + o_proj + AR ────────
    tt_sdpa = ttnn.to_layout(tt_sdpa, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1)  # DRAM TILE → L1 RM
    # Same tile-padding guard as the Q pack: the SDPA output's H_local*P packed
    # query rows may come back padded to 32; slice to the logical extent so the
    # unpack reshape is volume-exact.
    if tt_sdpa.shape[2] != H_local * P:
        tt_sdpa_unpad = ttnn.slice(tt_sdpa, [0, 0, 0, 0], [1, B, H_local * P, head_dim])
        ttnn.deallocate(tt_sdpa)
        tt_sdpa = tt_sdpa_unpad
    tt_sdpa = ttnn.reshape(tt_sdpa, (1, B, H_local, P, head_dim))
    tt_sdpa = ttnn.permute(tt_sdpa, (0, 1, 3, 2, 4))  # [1, B, P, H_local, head_dim]
    tt_sdpa = ttnn.reshape(tt_sdpa, (1, B * P, H_local, head_dim))
    return _packed_verify_out(tt_sdpa, weights, config, mesh_config, ccl_manager)
