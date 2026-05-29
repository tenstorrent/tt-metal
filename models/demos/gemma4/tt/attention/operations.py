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

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """Fused QKV matmul (no bias for Gemma4)."""
    return ttnn.linear(hidden_states, weights.wqkv)


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


def split_qkv_heads_prefill(xqkv_fused, config, is_global: bool, tp: int = 1, kv_replicated: bool = False):
    """
    Split fused QKV into separate head tensors for prefill mode.
    When TP > 1, uses local head counts (global / tp).
    When kv_replicated (num_kv_heads < TP), each device has 1 KV head (GQA-assigned).
    """
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_per_head_norm(tensor, weight, eps, with_scale=True):
    """
    Apply RMSNorm per-head on the head_dim dimension.

    Input: [1, num_heads, S, head_dim] or batched prefill [B, num_heads, S, head_dim]
    Process: reshape to [1, 1, num_heads*S, head_dim] (or B*num_heads*S for batch) -> rms_norm -> reshape back
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
        normed = ttnn.rms_norm(flat, weight=weight, epsilon=eps)
    else:
        normed = ttnn.rms_norm(flat, epsilon=eps)

    return ttnn.reshape(normed, orig_shape)


def apply_rope(tensor, cos_cache, sin_cache, token_index=None):
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

    Note: rotary_embedding pads dim 2 to TILE_HEIGHT (32) in decode mode.
    We reshape+slice to restore the original logical shape, following the
    tt_transformers _hf_rope_decode pattern.
    """
    orig_shape = tensor.shape
    result = ttnn.experimental.rotary_embedding(tensor, cos_cache, sin_cache, token_index)

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


def concat_heads(tensor, is_decode_mode: bool):
    """Concatenate attention heads back to hidden dimension."""
    if is_decode_mode:
        tensor = ttnn.transpose(tensor, 1, 2)
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_rope_hf_manual(x, cos, sin):
    """Apply HF rotary position embedding manually (slice + rotate_half + fma).

    Computes ``x * cos + rotate_half(x) * sin`` with
    ``rotate_half(x) = cat([-x2, x1])`` split at ``head_dim // 2`` — the exact HF
    convention (``transformers`` ``apply_rotary_pos_emb``). ``cos``/``sin`` are
    ``[1, batch, 1, head_dim]`` and broadcast over the heads dim.

    This exists because ``ttnn.experimental.rotary_embedding_hf`` is numerically
    incorrect for ``head_dim == 512`` (verified: PCC ~0.76 at 512 vs ~1.0 at 256),
    which is exactly the Gemma4 global-attention head dim. The manual path is
    layout-agnostic (works interleaved) and correct for any head_dim and per-user
    positions, at the cost of a few extra elementwise ops.
    """
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
    return ttnn.add(ttnn.mul(x, cos), ttnn.mul(rotated, sin))


def _batch_to_corerange(batch: int, grid_size: ttnn.CoreCoord) -> ttnn.CoreRangeSet:
    """Map a decode batch size to a single rectangular CoreRange of exactly ``batch`` cores.

    ``nlp_concat_heads_decode`` requires ``num_cores == num_users`` (one user per
    core). Using a single contiguous rectangle anchored at (0, 0) keeps the op on
    its fast path and avoids the ``on_subcoregrids`` requirement that a fragmented
    CoreRangeSet would trigger (the SDPA output is resharded here from DRAM, so the
    rectangle need not match Q/K's row-major decode grid).

    Picks the widest rectangle that tiles ``batch`` exactly and fits the device
    grid: e.g. on an 8- or 11-wide grid, batch {1, 8, 16, 32} map to 1x1, 8x1,
    8x2, 8x4 respectively.
    """
    grid_x, grid_y = grid_size.x, grid_size.y
    width = min(batch, grid_x)
    while width >= 1 and (batch % width != 0 or batch // width > grid_y):
        width -= 1
    if width < 1:
        raise ValueError(
            f"batch={batch} cannot form a single rectangle on a {grid_x}x{grid_y} grid; "
            "supported batch sizes are {1, 8, 16, 32}"
        )
    height = batch // width
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))
    return ttnn.CoreRangeSet({core_range})


def concat_heads_decode(tensor, num_local_heads: int, batch: int, mesh_device):
    """Concatenate attention heads for batched decode (one user per core).

    The SDPA decode output arrives in DRAM as ``[1, batch, num_local_heads, head_dim]``.
    Reshard it height-sharded onto a single-rect CoreRange of ``batch`` cores, then
    use ``nlp_concat_heads_decode`` to fold heads back into the (local) hidden dim.
    """
    head_dim = tensor.shape[-1]
    grid_size = mesh_device.compute_with_storage_grid_size()
    batch_grid = _batch_to_corerange(batch, grid_size)
    padded_heads = ((num_local_heads + 31) // 32) * 32
    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tensor = ttnn.to_memory_config(tensor, height_sharded_mem_config)
    return ttnn.experimental.nlp_concat_heads_decode(tensor, num_heads=num_local_heads)


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
    cache_num_heads = k_cache.padded_shape[1]
    cache_block_size = k_cache.padded_shape[2]
    cache_head_dim = k_cache.padded_shape[-1]
    return (cache_num_heads * cache_block_size * cache_head_dim) // (num_kv_heads * head_dim)
