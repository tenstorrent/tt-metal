# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared attention operations for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (not the llama variant).
No Meta-format weight conversion needed. No transformation matrices needed.

Handles:
- Per-head RMSNorm (q_norm, k_norm, v_norm) via reshape trick
- Partial RoPE for global layers (split, rotate, concat)
- K=V tying (fused Q+K+K weight, standard nlp_create_qkv_heads split)
- No bias on any projection
- scaling=1.0 (no 1/sqrt(d_k))
"""

import os

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce

from .weights import AttentionWeights

# Non-chunked prefill SDPA silently returns WRONG results once the Q sequence
# exceeds 32768 (2^15) — generation degrades to garbage. Above this length we
# fall back to chunked SDPA (chunk Q, attend the full K prefix from the paged
# cache). Overridable via env for validation (e.g. force chunking at small seq).
PREFILL_SDPA_MAX_SEQ = int(os.environ.get("GEMMA4_PREFILL_SDPA_MAX_SEQ", "32768"))
# Q chunk size for chunked prefill. Must be <= 32768 and a multiple of the SDPA
# q/k_chunk_size (128) and the page block size. 8192 keeps L1 bounded for the
# global layers' head_dim=512.
PREFILL_CHUNK_SIZE = int(os.environ.get("GEMMA4_PREFILL_CHUNK_SIZE", "8192"))
# Sliding-layer chunked prefill stride. The chunked SDPA op is causal-only (no
# sliding window), and the non-chunked op requires Q.s == K.s, so sliding layers
# are chunked with an OVERLAPPING window: each chunk runs SDPA over a slice of
# (stride + sliding_window) positions and keeps only the last ``stride`` rows.
# slice length = stride + sliding_window must stay <= 32768, so 30720 + 1024 =
# 31744 < 32768. Must be a multiple of TILE (32).
PREFILL_SLIDING_CHUNK_SIZE = int(os.environ.get("GEMMA4_PREFILL_SLIDING_CHUNK_SIZE", "30720"))


def apply_qkv_projection(hidden_states, weights: AttentionWeights, memory_config=None):
    """Fused QKV matmul (no bias for Gemma4).

    ``memory_config`` lets the packed-verify decode keep the projection output
    resident on L1; ``None`` keeps the op default (DRAM) for existing callers.
    """
    return ttnn.linear(hidden_states, weights.wqkv, memory_config=memory_config)


def split_qkv_heads_decode(xqkv_fused, config, is_global: bool, tp: int = 1, kv_replicated: bool = False):
    """
    Split fused QKV into separate head tensors for decode mode.
    When TP > 1, uses local head counts (global / tp).
    When kv_replicated (num_kv_heads < TP), each device has 1 KV head (GQA-assigned).
    """
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
    # Workaround for Blackhole bug in nlp_create_qkv_heads_decode interleaved
    # reader kernel: with DRAM input the kernel zeros odd-indexed Q rows due
    # to a NoC DRAM-read alignment-match violation (see tt-metal #16667). Move
    # the fused QKV from DRAM to L1 before the split — the L1 path uses a
    # different code path that's not affected. No-op on Wormhole (correctness
    # preserved; small extra L1 copy in exchange for arch-portable behavior).
    if xqkv_fused.memory_config().buffer_type == ttnn.BufferType.DRAM:
        xqkv_fused = ttnn.to_memory_config(xqkv_fused, ttnn.L1_MEMORY_CONFIG)
    return ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )


def split_qkv_heads_prefill(
    xqkv_fused, config, is_global: bool, tp: int = 1, kv_replicated: bool = False, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """
    Split fused QKV into separate head tensors for prefill mode.
    When TP > 1, uses local head counts (global / tp).
    When kv_replicated (num_kv_heads < TP), each device has 1 KV head (GQA-assigned).

    ``memory_config`` defaults to DRAM (true prefill: seq_len can be thousands of
    tokens and would not fit L1). The packed-verify decode caller overrides it to
    L1 so the split output — and the downstream activation stream — stays
    resident on L1 (the op only emits sharded output for sharded input, so an L1
    interleaved input yields L1 interleaved output).
    """
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        transpose_k_heads=False,
        memory_config=memory_config,
    )


def apply_per_head_norm(tensor, weight, eps, with_scale=True, memory_config=None):
    """
    Apply RMSNorm per-head on the head_dim dimension.

    Input: [1, num_heads, S, head_dim] or batched prefill [B, num_heads, S, head_dim]
    Process: reshape to [1, 1, num_heads*S, head_dim] (or B*num_heads*S for batch) -> rms_norm -> reshape back

    ``memory_config`` is forwarded to ``rms_norm``; pass L1 to keep the normed
    activation resident on L1 (packed-verify decode path). ``None`` keeps the
    op's default (follows the input's layout).
    """
    orig_shape = tensor.shape
    head_dim = orig_shape[-1]
    if len(orig_shape) == 4 and orig_shape[0] > 1:
        batch, num_heads, seq_len, _ = orig_shape
        flat = ttnn.reshape(tensor, (1, 1, batch * num_heads * seq_len, head_dim))
    else:
        num_heads = orig_shape[1]
        seq_or_batch = orig_shape[2]
        flat = ttnn.reshape(tensor, (1, 1, num_heads * seq_or_batch, head_dim))
    if with_scale and weight is not None:
        normed = ttnn.rms_norm(flat, weight=weight, epsilon=eps, memory_config=memory_config)
    else:
        normed = ttnn.rms_norm(flat, epsilon=eps, memory_config=memory_config)

    return ttnn.reshape(normed, orig_shape)


def apply_rope(tensor, cos_cache, sin_cache, token_index=None, memory_config=None):
    """
    Apply HF-style rotary position embedding.

    Uses ttnn.experimental.rotary_embedding (not llama variant).
    No transformation matrix needed. Position slicing is internal.

    Args:
        tensor: [1, heads, S, head_dim] (prefill) or [1, batch, heads, head_dim] (decode)
        cos_cache: [1, 1, max_seq_len, head_dim] - full cos cache
        sin_cache: [1, 1, max_seq_len, head_dim] - full sin cache
        token_index: int or None. If int (decode), slices into cache at that position.
                     If None (prefill), applies to full sequence.
        memory_config: forwarded to the rotary op; the packed-verify decode passes
                     L1 to keep the rotated tensor resident on L1.

    Note: rotary_embedding pads dim 2 to TILE_HEIGHT (32) in decode mode.
    We reshape+slice to restore the original logical shape, following the
    tt_transformers _hf_rope_decode pattern.
    """
    orig_shape = tensor.shape
    result = ttnn.experimental.rotary_embedding(tensor, cos_cache, sin_cache, token_index, memory_config=memory_config)

    # In decode mode (token_index provided), dim 2 gets padded to 32.
    # Reshape to indicate logical vs padded size, then slice back.
    if token_index is not None and result.shape[2] != orig_shape[2]:
        result = ttnn.reshape(
            result,
            (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]),
            (orig_shape[0], orig_shape[1], 32, orig_shape[3]),
        )
        result = result[:, :, : orig_shape[2]]

    return result


def _rotate_half(x):
    """NeoX-style rotate_half: cat(-x[..., d/2:], x[..., :d/2])."""
    hd = x.shape[-1]
    x1 = x[..., : hd // 2]
    x2 = x[..., hd // 2 :]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def apply_rope_decode_peruser(tensor, cos_b, sin_b):
    """Per-user decode RoPE: ``q*cos + rotate_half(q)*sin``.

    Batched, multi-core, and convention-matched to Gemma4's HF NeoX cos/sin
    caches (verified equivalent to the legacy single-position
    ``ttnn.experimental.rotary_embedding`` at PCC ~0.99999 for both sliding and
    global/partial-rotary layers). Unlike that op (one position via token_index,
    i.e. batch=1) and unlike ``rotary_embedding_hf`` (which mis-rotates Gemma4's
    global head_dim=512 partial-rotary cache, PCC ~0.72), this handles per-user
    positions:     ``tensor`` is ``[1, batch, heads, head_dim]`` (batch in dim1, heads in dim2).
    cos_b/sin_b are ``[1, batch, 1, head_dim]`` (one position per user). binary_ng
    can't broadcast over the heads (Y/height) dim within a tile, so the cos/sin
    are materialized over heads with ttnn.repeat before the same-shape multiply.
    """
    heads = tensor.shape[2]
    if cos_b.shape[2] != heads:
        cos_b = ttnn.repeat(cos_b, ttnn.Shape([1, 1, heads, 1]))
        sin_b = ttnn.repeat(sin_b, ttnn.Shape([1, 1, heads, 1]))
    return ttnn.add(ttnn.mul(tensor, cos_b), ttnn.mul(_rotate_half(tensor), sin_b))


def prefill_sdpa_program_config(head_dim, seq_len):
    """Tuned SDPAProgramConfig for the non-chunked prefill path (seq <= 32768).

    The op defaults to small auto-picked q/k chunks; explicit larger chunks cut
    the per-chunk launch + softmax-reduction overhead that dominates the
    flash-attention inner loop at these sequence lengths. head_dim=512 (global)
    layers need the smaller (8,4) grid + 128 chunks to fit L1 (q=k=256 overflows
    there); head_dim<=256 (sliding) layers run the full (8,8) grid with larger
    chunks. Chunks are clamped to divide seq_len (buckets are powers of two, as
    are the chunk sizes, so min() stays a divisor). Sizes overridable for sweeps
    via GEMMA4_PREFILL_SDPA_QCHUNK / _KCHUNK.
    """
    if head_dim >= 512:
        grid = ttnn.CoreCoord(8, 4)
        dq, dk = 128, 128
    else:
        # head_dim<=256: q=k=512 (~3.5 MB) and even q=k=256 (~1.58 MB) overflow
        # L1 (max 1.57 MB); q=256/k=128 keeps the static CBs in budget while
        # still giving wider Q parallelism than the default.
        grid = ttnn.CoreCoord(8, 8)
        dq, dk = 256, 128
    q_chunk = int(os.environ.get("GEMMA4_PREFILL_SDPA_QCHUNK", dq))
    k_chunk = int(os.environ.get("GEMMA4_PREFILL_SDPA_KCHUNK", dk))
    # Chunk sizes must be a multiple of 32 and not exceed the (padded) seq_len.
    q_chunk = max(32, min(q_chunk, seq_len))
    k_chunk = max(32, min(k_chunk, seq_len))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


def chunked_prefill_sdpa(tt_q, k_cache, v_cache, page_table, user_id, head_dim, scale=1.0, base_offset=0):
    """Chunked causal prefill SDPA over a paged KV cache.

    Splits the Q sequence into chunks of ``PREFILL_CHUNK_SIZE`` (<=32768) and runs
    ``chunked_scaled_dot_product_attention`` per chunk against the full paged K/V
    (already filled with this user's prompt). This avoids the non-chunked SDPA's
    32768-token Q correctness cliff. Causal-only (no sliding window): use for
    full-attention layers.

    Args:
        tt_q: [1, num_local_heads, seq_len, head_dim] (TILE layout), RoPE'd.
        k_cache, v_cache: paged caches [max_blocks, num_local_kv_heads, block, head_dim].
        page_table: int32 [batch, num_pages] (row ``user_id`` is this user's blocks).
        user_id: which page_table row maps this user's logical->physical blocks.
        head_dim: layer head_dim (512 for global layers; sizes the SDPA grid).
        base_offset: absolute position (in the full sequence) of ``tt_q``'s first
            row. Non-zero for generator-level multi-chunk prefill: chunk N's Q sits
            at ``N*chunk_size`` and must attend the full prior prefix already in the
            paged cache. Added to each internal Q-chunk offset so the op's causal
            mask covers ``[0, base_offset + local_end)``. Must be a multiple of the
            program's ``q_chunk_size`` (128); generator chunk sizes are >=128 powers
            of two so this holds.
    """
    seq_len = tt_q.shape[-2]
    nh = tt_q.shape[1]
    # head_dim=512 needs more L1/core, so use a smaller grid + 128 chunks (the
    # validated config); sliding-size head_dim uses the full grid.
    if head_dim >= 512:
        grid = ttnn.CoreCoord(8, 4)
    else:
        grid = ttnn.CoreCoord(8, 8)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    # HiFi4 + FP32 dest-acc: restore the softmax-reduce precision #47311 removed.
    # Matches the non-chunked prefill SDPA so long-context (>32768) prefill keeps
    # the same accumulation precision as the short-seq path.
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        tt_q.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # Page table row for this user: [1, num_pages], int32, ROW_MAJOR.
    num_pages = page_table.shape[-1]
    if page_table.shape[0] > 1:
        user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, num_pages])
        owns_user_pt = True
    else:
        user_pt = page_table
        owns_user_pt = False

    outs = []
    start = 0
    while start < seq_len:
        clen = min(PREFILL_CHUNK_SIZE, seq_len - start)
        q_chunk = ttnn.slice(tt_q, [0, 0, start, 0], [1, nh, start + clen, head_dim])
        # chunked SDPA q_chunk_size=128 needs the Q chunk length to be a multiple
        # of 128; pad the (tile-aligned) tail chunk up and slice the result back.
        pad = (-clen) % 128
        if pad:
            q_unpadded = q_chunk
            q_chunk = ttnn.pad(q_unpadded, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            q_unpadded.deallocate(True)
        out_chunk = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_chunk,
            k_cache,
            v_cache,
            user_pt,
            chunk_start_idx=base_offset + start,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        q_chunk.deallocate(True)
        if pad:
            out_padded = out_chunk
            out_chunk = ttnn.slice(out_padded, [0, 0, 0, 0], [1, nh, clen, head_dim])
            out_padded.deallocate(True)
        outs.append(out_chunk)
        start += clen

    if owns_user_pt:
        user_pt.deallocate(True)
    if len(outs) == 1:
        return outs[0]
    out = ttnn.concat(outs, dim=2)
    for chunk in outs:
        chunk.deallocate(True)
    return out


def chunked_prefill_sdpa_sliding(tt_q, tt_k, tt_v, sliding_window, head_dim, scale=1.0):
    """Chunked causal+sliding-window prefill SDPA for sliding-window layers.

    The chunked SDPA op is causal-only and the non-chunked op requires Q.s==K.s
    (square mask) with the 32768 cliff on that shared length. A sliding-window
    query at position p only depends on K/V in (p - window, p], so we process Q
    in strides of ``PREFILL_SLIDING_CHUNK_SIZE`` and, for each stride, run the
    normal causal+sliding SDPA over an OVERLAPPING slice that also covers the
    preceding ``sliding_window`` positions (Q and K both = that slice, so it's a
    square ≤32768 case). Only the last ``stride`` output rows are kept — their
    full window lies inside the slice; the leading ``window`` rows were already
    emitted by the previous stride.

    Args:
        tt_q, tt_k, tt_v: [1, heads/kv_heads, seq_len, head_dim] (TILE), RoPE'd.
        sliding_window: window size W (Gemma4 sliding layers: 1024, tile-aligned).
        head_dim: layer head_dim (256 for sliding layers).
    Returns:
        [1, num_heads, seq_len, head_dim] attention output (TILE layout).
    """
    seq_len = tt_q.shape[-2]
    nh = tt_q.shape[1]
    nkv = tt_k.shape[1]
    # History must be tile-aligned for ttnn.slice on TILE layout. Gemma4's window
    # (1024) is already a multiple of 32; round up otherwise. Including one extra
    # older key is harmless — sliding_window_size masks it out.
    hist = ((sliding_window + 31) // 32) * 32
    stride = PREFILL_SLIDING_CHUNK_SIZE
    # HiFi4 + FP32 dest-acc: restore the softmax-reduce precision #47311 removed,
    # matching the non-chunked prefill SDPA on the long-context (>32768) path.
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        tt_q.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    outs = []
    start = 0
    while start < seq_len:
        out_len = min(stride, seq_len - start)
        slice_start = max(0, start - hist)
        slice_end = start + out_len
        q_slice = ttnn.slice(tt_q, [0, 0, slice_start, 0], [1, nh, slice_end, head_dim])
        k_slice = ttnn.slice(tt_k, [0, 0, slice_start, 0], [1, nkv, slice_end, head_dim])
        v_slice = ttnn.slice(tt_v, [0, 0, slice_start, 0], [1, nkv, slice_end, head_dim])
        o = ttnn.transformer.scaled_dot_product_attention(
            q_slice,
            k_slice,
            v_slice,
            is_causal=True,
            scale=scale,
            sliding_window_size=sliding_window,
            compute_kernel_config=compute_kernel_config,
        )
        q_slice.deallocate(True)
        k_slice.deallocate(True)
        v_slice.deallocate(True)
        # Keep only the queries [start, slice_end); drop the leading history rows.
        drop = start - slice_start
        if drop:
            o_full = o
            o = ttnn.slice(o_full, [0, 0, drop, 0], [1, nh, slice_end - slice_start, head_dim])
            o_full.deallocate(True)
        outs.append(o)
        start += out_len

    if len(outs) == 1:
        return outs[0]
    out = ttnn.concat(outs, dim=2)
    for chunk in outs:
        chunk.deallocate(True)
    return out


def concat_heads(tensor, is_decode_mode: bool, num_heads: int = None, head_dim: int = None, mesh_device=None):
    """Concatenate attention heads back to hidden dimension.

    Decode uses ``nlp_concat_heads_decode`` (multi-core, width-sharded output)
    instead of ``transpose + nlp_concat_heads``. The decode op needs a
    height-sharded input ([1, batch, heads_padded_to_32, head_dim] sharded with
    one core per user), so the SDPA output (DRAM, [1, batch, heads, head_dim]) is
    resharded first across ``batch`` cores. The old path ran the concat on a
    single core (~30 us/layer) and needed a separate transpose; the decode op
    drops the transpose and spreads work across cores. Output is converted back
    to DRAM interleaved so the downstream o_proj matmul is unchanged.
    """
    if is_decode_mode:
        if num_heads is None or head_dim is None:
            raise ValueError("decode concat_heads requires num_heads and head_dim")
        batch = tensor.shape[1]
        # One core per user (batch), arranged as a contiguous rectangle. A plain
        # num_cores_to_corerangeset spills into a non-rectangular set once
        # batch > grid width, which the height-sharded mem config rejects
        # ("bad optional access"); num_to_corerange forces a grid-width-aligned
        # rectangle (e.g. batch=8→8x1, 16→8x2, 32→8x4 on an 8-wide grid) that
        # the kernel accepts.
        from models.tt_transformers.tt.model_config import num_to_corerange

        compute_grid = mesh_device.compute_with_storage_grid_size() if mesh_device is not None else None
        physical_grid_x = compute_grid.x if compute_grid is not None else 8
        grid_y = compute_grid.y if compute_grid is not None else 8
        grid_x = min(batch, physical_grid_x)
        if batch >= grid_x and batch % grid_x != 0:
            grid_x = max(x for x in range(grid_x, 0, -1) if batch % x == 0 and batch // x <= grid_y)
        core_grid = ttnn.CoreRangeSet({num_to_corerange(batch, grid_x=grid_x, grid_y=grid_y)})
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tensor_sh = ttnn.to_memory_config(tensor, shard_cfg)
        # Output is [1, 1, B(padded to 32), num_heads*head_dim] width-sharded.
        out = ttnn.experimental.nlp_concat_heads_decode(tensor_sh, num_heads=num_heads)
        tensor_sh.deallocate(True)
        out_sh = out
        out = ttnn.sharded_to_interleaved(out_sh, ttnn.DRAM_MEMORY_CONFIG)
        out_sh.deallocate(True)
        # Drop the batch padding (B is padded to 32 by the op) so downstream sees
        # [1, 1, batch, hidden_local] just like the old transpose+concat path.
        if out.shape[2] != batch:
            out_padded = out
            out = out_padded[:, :, :batch, :]
            out_padded.deallocate(True)
        return out
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights):
    """Apply output projection (no bias for Gemma4)."""
    out = ttnn.linear(tensor, weights.o_proj)
    tensor.deallocate(True)
    return out


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """Apply tensor-parallel allreduce if TP > 1."""
    return ccl_allreduce(tensor, mesh_config, ccl_manager)


def effective_block_size(k_cache, head_dim: int, num_kv_heads: int) -> int:
    """Block-size override to pass to paged-cache ops when the cache
    buffer's *allocation* view doesn't match this layer's view.

    vLLM's hybrid kv-cache-groups manager runs a per-block-byte unifier
    and can share one physical buffer between layers of different
    attention types — for Gemma4 that's sliding (head_dim=256, kv=N)
    and full (head_dim=512, kv=M). Per-block elements are preserved
    across HMA-shared views, so a layer recovers its effective
    block_size by inverting the invariant
    ``input_kv * eff_bs * input_hd == cache_kv * cache_bs * cache_hd``:

      eff_bs = cache_kv * cache_bs * cache_hd / (input_kv * input_hd)

    Earlier versions of this helper omitted the ``cache_kv / input_kv``
    factor, which was a no-op for Gemma4-E2B / E4B (sliding and full
    share num_kv_heads) but produced the wrong block_size on
    Gemma4-26B-A4B (kv=8 / 2) and 31B (kv=16 / 4) at small TP, tripping
    the paged_fill_cache / paged_update_cache byte-count check.

    When the cache happens to have been allocated *with* this layer's
    own view, the override is a no-op for the kernel; passing it
    unconditionally keeps the paged_{fill,update}_cache and
    paged_scaled_dot_product_attention_decode call sites symmetric.
    """
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be > 0, got {num_kv_heads}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {head_dim}")

    cache_num_heads = k_cache.padded_shape[1]
    cache_block_size = k_cache.padded_shape[2]
    cache_head_dim = k_cache.padded_shape[-1]

    numerator = cache_num_heads * cache_block_size * cache_head_dim
    denominator = num_kv_heads * head_dim
    if numerator % denominator != 0:
        raise ValueError(
            "KV-cache layout is incompatible with the requested layer view: "
            f"({cache_num_heads} * {cache_block_size} * {cache_head_dim}) is not exactly divisible by "
            f"({num_kv_heads} * {head_dim})"
        )

    return numerator // denominator
