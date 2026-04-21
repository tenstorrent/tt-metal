# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """
    Apply QKV projection and add bias.

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        weights: Attention weights container

    Returns:
        Fused QKV tensor [batch, seq_len, total_qkv_dim]
    """
    xqkv_fused = ttnn.matmul(hidden_states, weights.wqkv, dtype=ttnn.bfloat16)
    ttnn.add(xqkv_fused, weights.wqkv_bias, output_tensor=xqkv_fused)
    return xqkv_fused


def split_qkv_heads_decode(xqkv_fused, num_heads: int, num_kv_heads: int):
    """
    Split QKV into separate head tensors for decode mode.

    Args:
        xqkv_fused: Fused QKV tensor
        num_heads: Number of Q heads
        num_kv_heads: Number of K/V heads

    Returns:
        Tuple (Q, K, V) with shapes [1, num_heads, 1, head_dim]
    """
    return ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )


def split_qkv_heads_prefill(xqkv_fused, num_heads: int, num_kv_heads: int):
    """
    Split QKV into separate head tensors for prefill mode.

    Args:
        xqkv_fused: Fused QKV tensor
        num_heads: Number of Q heads
        num_kv_heads: Number of K/V heads

    Returns:
        Tuple (Q, K, V) with shapes [1, num_heads, seq_len, head_dim]
    """
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_rope(tensor, rope_mats, transformation_mat, is_decode_mode: bool):
    """
    Apply rotary position embedding (RoPE).

    Args:
        tensor: Input tensor (Q or K)
        rope_mats: Tuple of (cos, sin) matrices
        transformation_mat: Transformation matrix for the mode
        is_decode_mode: Whether in decode mode

    Returns:
        Tensor with RoPE applied
    """
    return ttnn.experimental.rotary_embedding_llama(
        tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=is_decode_mode
    )


def concat_heads(tensor, is_decode_mode: bool):
    """
    Concatenate attention heads back to hidden dimension.

    Args:
        tensor: Attention output tensor with separate heads
        is_decode_mode: Whether in decode mode

    Returns:
        Tensor with concatenated heads [batch, seq_len, hidden_size]
    """
    if is_decode_mode:
        tensor = ttnn.transpose(tensor, 1, 2)
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype):
    """
    Apply output projection and bias.

    Args:
        tensor: Attention output tensor
        weights: Attention weights container
        activation_dtype: Target dtype for output

    Returns:
        Output tensor after projection
    """
    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)
    out = ttnn.matmul(tensor, weights.o_proj, dtype=activation_dtype)
    tensor.deallocate(True)
    ttnn.add(out, weights.o_proj_bias, output_tensor=out)
    return out


# Per-shape tuned params for the fused attention o_proj matmul + reduce-scatter.
# Values came from a 180-config (S=128) / 504-config (S=1024) sweep on TG 4x8
# with 5 warmup + 10 measured iters per config. The kernel is
# ttnn.experimental.minimal_matmul_strided_reduce_scatter_async and the shape
# is [M=seq_len, K=512, N=3072] per device (after TP=8 sharding of hidden_size).
#
# Sweep covered:
#     mm_grid.y ∈ {2,3,4,5,6,7}           (MM/RS core split on the 8×8 grid)
#     M_block   ∈ divisors of M_tiles/grid.y
#     K_block   ∈ {2,4,8}  (divisors of K_tiles=16)
#     N_block   ∈ {2,3,4,6} (≤ N_tiles_per_core=12)
#     chunk_width_in_mm_blocks ∈ {1,2,4}
#     subblock_h/w with subblock_h*subblock_w ≤ 4 (fp32_dest_acc_en=True)
#
# Winners below beat the non-fused MM+RS chain by:
#   - S=128:  attention-block total 3714.83 us → 3693.54 us (−21 us/layer)
#   - S=1024: fused-op kernel 279 us vs non-fused ~450-500 us equivalent
_FUSED_MM_RS_CONFIGS = {
    # M_tiles  : (grid.y, M_block, K_block, N_block, chunk_w, subblock_h, subblock_w)
    4: (2, 2, 8, 6, 2, 1, 1),  # S=128:  winner y2_mb2_kb8_nb6_cw2 (sbh×sbw not swept separately; 1×1 safe)
    32: (4, 8, 8, 6, 2, 1, 2),  # S=1024: winner y4_mb8_kb8_nb6_cw2, subblock sweep picked sbw=2 (−3.6%)
}
# Fallback for unswept shapes — conservative: favour RS workers, small blocks.
_FUSED_MM_RS_FALLBACK = (2, 2, 8, 6, 2, 1, 1)


def apply_output_projection_fused_rs(tensor, weights: AttentionWeights, mesh_config, ccl_manager):
    """Attention output projection + TP reduce-scatter fused into one device op.

    Replaces the sequential `apply_output_projection` + first half of
    `mesh_config.allreduce` (the reduce-scatter). The trailing all-gather and
    padding-trim stay as separate ops in `apply_allgather_and_slice`.

    Internally uses `ttnn.experimental.minimal_matmul_strided_reduce_scatter_async`,
    which overlaps MM compute on one half of the core grid with the RS ring
    traffic on the other half — MM signals RS via on-chip semaphore per output
    block so RS starts forwarding as soon as the first block lands in DRAM.

    Dtypes match the non-fused baseline:
      - activation cast to bf8_b before MM (DRAM-bandwidth saver, matches baseline)
      - weight is bf8_b (as loaded, row-parallel sharded on K across TP=8)
      - MM math: LoFi (same as `ttnn.matmul` default)
      - fused-op output defaults to input dtype (bf8_b); we cast back to bf16
        afterwards so downstream AllGather / residual-add / LayerNorm see the
        same bf16 signature as the baseline path.

    Returns the reduce-scattered tensor of shape
    [1, 1, S, padded_hidden_total / TP] in bf16.

    Only valid for prefill (TP > 1); the decoder path uses a different pattern.
    """
    TILE = 32
    K = tensor.shape[-1]
    N = weights.o_proj.shape[-1]
    assert K % TILE == 0 and N % TILE == 0, f"K={K}, N={N} must be tile-aligned"

    M_tiles = (tensor.shape[-2] + TILE - 1) // TILE
    K_tiles = K // TILE
    N_tiles = N // TILE

    grid_y, m_block, k_block, n_block, chunk_width, subblock_h, subblock_w = _FUSED_MM_RS_CONFIGS.get(
        M_tiles, _FUSED_MM_RS_FALLBACK
    )

    mm_core_grid = ttnn.CoreCoord(8, grid_y)
    Nt_per_core = N_tiles // mm_core_grid.x
    Mt_per_core = max(1, math.ceil(M_tiles / grid_y))

    # Clamp blocks to what the current shape actually supports — defensive in
    # case _FUSED_MM_RS_CONFIGS is extended for a shape where these don't fit.
    m_block = min(m_block, Mt_per_core)
    k_block = min(k_block, K_tiles)
    n_block = min(n_block, Nt_per_core)

    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)

    mm_config = ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=k_block,
        N_block_size=n_block,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=mm_core_grid,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        ccl_manager.mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    mm_out, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
        tensor,
        weights.o_proj,
        3,  # scatter on last (hidden) dim
        ccl_manager.get_rs_ping_pong_semaphore(),
        ttnn.CoreCoord(0, grid_y),  # RS cores start just below the MM rows
        compute_kernel_config=compute_config,
        num_links=ccl_manager.num_links,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ccl_manager.topology,
        cluster_axis=mesh_config.tp_axis,
        bias=weights.o_proj_bias,
        config=mm_config,
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        chunk_width_in_mm_blocks=chunk_width,
    )

    tensor.deallocate(True)
    mm_out.deallocate(True)

    # The fused op inherits output dtype from the input (bf8_b). Non-fused
    # path produced bf16 via explicit `ttnn.matmul(..., dtype=bf16)`. Cast up
    # so downstream AllGather / add / LayerNorm see bf16, same as baseline.
    rs_out_bf16 = ttnn.typecast(rs_out, ttnn.bfloat16)
    rs_out.deallocate(True)
    return rs_out_bf16


def apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size: int):
    """Complete the attention-output chain after `apply_output_projection_fused_rs`.

    Runs the all-gather (TP) on the scattered output, then drops the padding
    columns that `weights.py` adds for tile alignment of the CCL ops.
    """
    gathered = mesh_config.allgather(rs_out, ccl_manager, axis=mesh_config.tp_axis)
    rs_out.deallocate(True)

    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    if padded_local_hidden != local_hidden:
        shape = gathered.shape
        sliced = ttnn.slice(
            gathered,
            starts=[0, 0, 0, 0],
            ends=[shape[0], shape[1], shape[2], hidden_size],
            steps=[1, 1, 1, 1],
        )
        gathered.deallocate(True)
        return sliced
    return gathered


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """
    Apply tensor parallel allreduce if needed.

    Args:
        tensor: Input tensor
        mesh_config: Mesh configuration
        ccl_manager: Communication manager
        batch_size: Batch size for final reshape
        seq_len: Sequence length for final reshape
        hidden_size: Hidden size for final reshape

    Returns:
        Tensor after allreduce (if TP > 1) or original tensor
    """
    if mesh_config.tp > 1:
        tensor_allreduced = mesh_config.allreduce(tensor, ccl_manager, pad_size=0, axis=mesh_config.tp_axis)
        tensor.deallocate(True)
        tensor = tensor_allreduced

        # Remove padding added in weights.py for tile-aligned CCL operations.
        # If local_hidden was padded (e.g., 360 -> 384), we need to slice back to original hidden_size.
        local_hidden = hidden_size // mesh_config.tp
        padded_local_hidden = ((local_hidden + 31) // 32) * 32
        if padded_local_hidden != local_hidden:
            # Slice from padded_hidden back to hidden_size on the last dimension.
            # Works for both decode [1, 1, batch, padded_hidden] and prefill [1, batch, seq_len, padded_hidden].
            shape = tensor.shape
            tensor_sliced = ttnn.slice(
                tensor,
                starts=[0, 0, 0, 0],
                ends=[shape[0], shape[1], shape[2], hidden_size],
                steps=[1, 1, 1, 1],
            )
            tensor.deallocate(True)
            tensor = tensor_sliced
    return tensor


def get_mesh_coords(mesh_shape: list[int], row: int = None, col: int = None) -> list:
    """
    Get mesh coordinates for a given mesh shape and optional row and column indices.

    This is used to specify which devices should execute paged cache operations
    when the KV cache is replicated but users are sharded across rows.

    Args:
        mesh_shape: Shape of the mesh as [num_rows, num_cols]
        row: Optional row index to filter (None = all rows)
        col: Optional column index to filter (None = all columns)

    Returns:
        List of ttnn.MeshCoordinate objects for the specified row/column
    """
    if row is not None:
        assert 0 <= row < mesh_shape[0], f"Row index {row} out of bounds for mesh shape {mesh_shape}"
    if col is not None:
        assert 0 <= col < mesh_shape[1], f"Column index {col} out of bounds for mesh shape {mesh_shape}"

    row_select = range(mesh_shape[0]) if row is None else [row]
    col_select = range(mesh_shape[1]) if col is None else [col]
    return [ttnn.MeshCoordinate(r, c) for r in row_select for c in col_select]
