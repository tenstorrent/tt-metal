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
        # ("bad optional access"); num_to_corerange forces an 8-wide rectangle
        # (batch=8→8x1, 16→8x2, 32→8x4) that the kernel accepts.
        from models.tt_transformers.tt.model_config import num_to_corerange

        grid_y = mesh_device.compute_with_storage_grid_size().y if mesh_device is not None else 8
        core_grid = ttnn.CoreRangeSet({num_to_corerange(batch, grid_x=8, grid_y=grid_y)})
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
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        # Drop the batch padding (B is padded to 32 by the op) so downstream sees
        # [1, 1, batch, hidden_local] just like the old transpose+concat path.
        if out.shape[2] != batch:
            out = out[:, :, :batch, :]
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
