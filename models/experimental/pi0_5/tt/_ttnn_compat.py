# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fallbacks for pi0.5 fused TTNN ops when running against a main-branch build."""

from __future__ import annotations


import os

import ttnn


def nlp_create_qkv_heads_rope(
    xqkv,
    cos,
    sin,
    num_heads: int,
    num_kv_heads: int,
    *,
    memory_config=None,
):
    mem = memory_config or ttnn.L1_MEMORY_CONFIG
    # The fused op is now tile-aware (it validates Ht == 1 against the operand's own tile height,
    # sizes every CB from its tensor's tile, preserves the input page config on its outputs, and
    # hashes the tile dims), so it is used at ANY tile height. One dispatch instead of three.
    if hasattr(ttnn.experimental, "nlp_create_qkv_heads_rope"):
        return ttnn.experimental.nlp_create_qkv_heads_rope(xqkv, cos, sin, num_heads, num_kv_heads, memory_config=mem)
    # Fallback: nlp_create_qkv_heads requires a sharded (non-width-sharded) output when its input is
    # sharded. The DECODE_ALL path feeds a WIDTH-SHARDED qkv; convert it to interleaved first so the
    # interleaved-in -> interleaved-out path is taken and the requested `mem` config is honored.
    if xqkv.is_sharded():
        xqkv = ttnn.sharded_to_interleaved(xqkv, mem)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        xqkv,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=mem,
    )
    q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=mem)
    k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=mem)
    return q, k, v


def concat_heads_matmul(attn_out, weight, *, memory_config=None, program_config=None, dtype=None):
    mem = memory_config or ttnn.L1_MEMORY_CONFIG
    out_dtype = dtype or ttnn.bfloat16
    if hasattr(ttnn.experimental, "concat_heads_matmul"):
        kwargs = {"memory_config": mem}
        if program_config is not None:
            kwargs["program_config"] = program_config
        return ttnn.experimental.concat_heads_matmul(attn_out, weight, **kwargs)
    heads = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=mem)
    kwargs = {"dtype": out_dtype, "memory_config": mem}
    if program_config is not None:
        kwargs["program_config"] = program_config
    out = ttnn.linear(heads, weight, **kwargs)
    ttnn.deallocate(heads)
    return out


def concat_heads_matmul_decode(
    attn_out,
    weight,
    *,
    output_dtype=None,
    compute_kernel_config=None,
    reshard_cores=None,
    residual=None,
    gate=None,
):
    if hasattr(ttnn.experimental, "concat_heads_matmul_decode"):
        return ttnn.experimental.concat_heads_matmul_decode(
            attn_out,
            weight,
            output_dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
            reshard_cores=reshard_cores,
            residual=residual,
            gate=gate,
        )
    out = concat_heads_matmul(
        attn_out,
        weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=output_dtype or ttnn.bfloat16,
    )
    if residual is not None and gate is not None:
        out = ttnn.addcmul(residual, gate, out, memory_config=ttnn.L1_MEMORY_CONFIG)
    return out


def kv_sdpa(
    q,
    k,
    v,
    *,
    attn_mask=None,
    scale=None,
    past_k=None,
    past_v=None,
    compute_kernel_config=None,
    max_kv_chunk_tiles=None,
    kv_splits=None,
    prefix_valid_tiles=None,
):
    # kv_sdpa now has a FUSED mask path (cb_mask_in + a use_provided_mask compile-time gate), so a masked
    # call no longer has to detour through the general SDPA. It carries two geometry requirements the op
    # validates: the mask's tile height must match q's (the mask is added to the [Sq_h x 32] score tiles),
    # and the prefix's tile height must be 32 so prefix tile g lines up with mask column-tile g. The pi0.5
    # config satisfies both at either tile height. Anything else (e.g. a caller with no past, or a 16-row
    # prefix) still takes the promote-to-32 general-SDPA fallback below rather than silently dropping the
    # mask, which would change numerics wherever it is non-zero (the keep_padded phantom -1e4 band).
    # ---- STATUS: the fused mask path is NOT correct in-model yet; it is OPT-IN. -------------------
    # It passes every unit check at TILE_HEIGHT=32 (whole-tile mask == tile skipping 0.999848; a
    # half-masked tile lands strictly between open and blocked; mask composed with prefix_valid_tiles
    # 0.999716) and the in-model single-layer PCC gate (0.9999 at both tile heights) -- yet LIBERO fails
    # 6/6 at BOTH tile heights with it enabled, running clean and producing wrong actions, where the
    # promote-to-32 fallback below scores 40/40. So something the model does is not covered by any of
    # those checks. Prime suspect: the in-model chunk geometry. With max_kv_chunk_tiles=64 and DHt=8 the
    # cap is 8 tiles, and LIBERO's prefix_Kt_eff of 17 (17 valid tiles, prime) forces
    # prefix_Sk_chunk_t == 1 / prefix_num_chunks == 17 -- a 1-tile-per-chunk regime that also stresses
    # add_block_inplace's pop/reserve/push cycle on cb_qk_im, which only lands back on the same physical
    # tiles when the chunk fills the whole (single-buffered) CB.
    # Note the single-layer PCC gate is near-useless here: with all-real inputs the pi0.5 mask blocks
    # only the phantom suffix tail, so masked and unmasked agree to six decimals.
    _FUSED_KV_MASK = os.environ.get("PI05_FUSED_KV_MASK", "0") == "1"

    def _tile_h(t):
        return int(t.get_tile().tile_shape[0])

    can_fuse_mask = attn_mask is None or (
        _FUSED_KV_MASK and past_k is not None and _tile_h(past_k) == 32 and _tile_h(attn_mask) == _tile_h(q)
    )
    if hasattr(ttnn, "kv_sdpa") and can_fuse_mask:
        kwargs = {"attn_mask": attn_mask, "scale": scale}
        if past_k is not None:
            kwargs["past_k"] = past_k
            kwargs["past_v"] = past_v
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
        if max_kv_chunk_tiles is not None:
            kwargs["max_kv_chunk_tiles"] = max_kv_chunk_tiles
        if kv_splits is not None:
            kwargs["kv_splits"] = kv_splits
        if prefix_valid_tiles:
            kwargs["prefix_valid_tiles"] = list(prefix_valid_tiles)
        return ttnn.kv_sdpa(q, k, v, **kwargs)

    # ---- Masked fallback at a tiny tile: run the attention at the PREFIX's tile height -------------
    # kv_sdpa reads a 32x32 prefix and a 16x32 suffix as SEPARATE operands, but the general SDPA needs
    # one concatenated K/V and imposes two constraints the tiny-tile geometry violates:
    #   "All TILE-layout concat inputs must share the same tile shape (got 16x32 vs 32x32)"
    #   "Inputs to SDPA must have the same tile size"
    # So promote q / suffix-k / suffix-v / mask to the prefix tile, run SDPA there (exactly the geometry
    # the TILE_HEIGHT=32 path already validates), then bring the output back down. Only the suffix-sized
    # tensors are retiled -- the 1024-row prefix is reused as-is, which is what keeps this affordable.
    q_tile_in = tuple(q.get_tile().tile_shape)
    q_rows = q.shape[-2]
    promoted = False
    if past_k is not None and past_v is not None:
        p_tile = tuple(past_k.get_tile().tile_shape)
        if q_tile_in != p_tile:
            tgt = ttnn.Tile((int(p_tile[0]), int(p_tile[1])))
            q = ttnn.tilize(q, tile=tgt, dtype=q.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
            k = ttnn.tilize(k, tile=tgt, dtype=k.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
            v = ttnn.tilize(v, tile=tgt, dtype=v.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
            if attn_mask is not None and tuple(attn_mask.get_tile().tile_shape) != p_tile:
                # SDPA requires the mask in DRAM ("When mask is provided to SDPA, it must be in DRAM").
                attn_mask = ttnn.tilize(
                    attn_mask, tile=tgt, dtype=attn_mask.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            promoted = True
        k = ttnn.concat([past_k, k], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.concat([past_v, v], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

    kwargs = {
        "attn_mask": attn_mask,
        "is_causal": False,
        "scale": scale,
        "memory_config": ttnn.L1_MEMORY_CONFIG,
    }
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config
    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, **kwargs)
    if promoted:
        # Back to the model tile, trimming the rows the promotion padded (those q rows are independent,
        # so whatever they computed is simply discarded).
        out = ttnn.tilize(
            out,
            tile=ttnn.Tile((int(q_tile_in[0]), int(q_tile_in[1]))),
            dtype=out.dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if out.shape[-2] != q_rows:
            out = ttnn.slice(out, [0, 0, 0, 0], [out.shape[0], out.shape[1], q_rows, out.shape[-1]])
    return out


def decode_all_supported() -> bool:
    """True when the ops the decode_all denoise path actually dispatches are present.

    The tiny-tile block projects through ``ttnn.experimental.matmul_decode`` (tiny-tile inputA,
    32x32 inputB). The older fused path used ``ttnn.matmul_decode`` + ``ttnn.gate_up_matmul_decode``;
    either is sufficient, so accept a build that exposes one family or the other.
    """
    return hasattr(ttnn.experimental, "matmul_decode") or (
        hasattr(ttnn, "matmul_decode") and hasattr(ttnn, "gate_up_matmul_decode")
    )
