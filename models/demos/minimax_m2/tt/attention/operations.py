# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """
    Apply QKV projection.

    MiniMax-M2 has no biases on q/k/v projections (bias=False in HF), so this is
    a plain matmul.

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        weights: Attention weights container

    Returns:
        Fused QKV tensor [batch, seq_len, total_qkv_dim]
    """
    xqkv_fused = ttnn.linear(hidden_states, weights.wqkv, dtype=ttnn.bfloat16)
    return xqkv_fused


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
    Apply rotary position embedding (RoPE), supporting PARTIAL rotary.

    MiniMax-M2 rotates only the first ``rotary_dim`` (64) dims of each 128-wide
    head and passes the rest through unchanged. We detect the rotary width from
    the cos matrix (``rope_mats[0]`` is built at rotary_dim width in
    model.create_rope_setup). When rotary_dim == head_dim this is a plain full
    rotation (the rotary_embedding_llama op always rotates its full last dim, so
    we slice the head into rotate / pass-through parts ourselves).

    Args:
        tensor: Input tensor (Q or K), shape [..., head_dim]
        rope_mats: Tuple of (cos, sin) matrices, last dim = rotary_dim
        transformation_mat: Transformation matrix for the mode
        is_decode_mode: Whether in decode mode

    Returns:
        Tensor with RoPE applied
    """
    rotary_dim = rope_mats[0].shape[-1]
    head_dim = tensor.shape[-1]

    if rotary_dim >= head_dim:
        return ttnn.experimental.rotary_embedding_llama(
            tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=is_decode_mode
        )

    # Partial rotary: split [..., :rotary_dim] (rotate) and [..., rotary_dim:] (pass through).
    shape = tensor.shape
    t_rot = ttnn.slice(tensor, [0, 0, 0, 0], [shape[0], shape[1], shape[2], rotary_dim])
    t_pass = ttnn.slice(tensor, [0, 0, 0, rotary_dim], [shape[0], shape[1], shape[2], head_dim])
    t_rot = ttnn.experimental.rotary_embedding_llama(
        t_rot, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=is_decode_mode
    )
    out = ttnn.concat([t_rot, t_pass], dim=-1)
    t_rot.deallocate(True)
    t_pass.deallocate(True)
    return out


def distributed_rms_norm(x, weight, normalized_size: int, eps: float, mesh_config, ccl_manager):
    """Full-width RMSNorm whose feature dim is sharded across the TP axis.

    MiniMax-M2's q_norm/k_norm normalise over the *entire* Q (6144) / K (1024)
    projection output, but that output is column-parallel sharded across TP, so
    each device only holds ``normalized_size / tp`` features. We compute the
    per-token sum-of-squares locally and share ONLY that ``[.., 1]`` partial
    across the TP axis (cheap), then finish the norm locally. ``weight`` is the
    matching column-parallel shard of the gain vector.

    NOTE: the exact ttnn reduction / broadcast calls below (sum keepdim over the
    last dim, [.,1]*[.,N] broadcast multiply, all-gather of a width-1 partial)
    need validation on first hardware run — the math is standard distributed
    RMSNorm; only the op-level API surface is unverified offline.
    """
    sq = ttnn.mul(x, x)
    local_ss = ttnn.sum(sq, dim=-1, keepdim=True)  # [1, 1, M, 1] local partial sum of squares
    sq.deallocate(True)

    if mesh_config.tp > 1:
        # Gather the per-device partials along TP and sum -> full sum over normalized_size.
        gathered = mesh_config.allgather(local_ss, ccl_manager, axis=mesh_config.tp_axis, dim=3)  # [1,1,M,tp]
        local_ss.deallocate(True)
        total_ss = ttnn.sum(gathered, dim=-1, keepdim=True)  # [1, 1, M, 1]
        gathered.deallocate(True)
    else:
        total_ss = local_ss

    mean_sq = ttnn.multiply(total_ss, 1.0 / normalized_size)
    total_ss.deallocate(True)
    inv_rms = ttnn.rsqrt(ttnn.add(mean_sq, eps))  # [1, 1, M, 1]
    mean_sq.deallocate(True)

    x_norm = ttnn.multiply(x, inv_rms)  # broadcast [.., 1] over the feature dim
    inv_rms.deallocate(True)
    x_norm = ttnn.multiply(x_norm, weight)
    return x_norm


def apply_qk_norm(xqkv_fused, weights: AttentionWeights, config, mesh_config, ccl_manager):
    """Apply MiniMax-M2 full-width QK-norm to the fused QKV tensor, in place of the
    HF "norm the flat q/k projection before reshaping to heads" step.

    Slices the per-device fused tensor into its local Q / K / V parts, runs the
    distributed RMSNorm on Q and K (V is untouched), and re-concatenates so the
    downstream head-split is unchanged.
    """
    head_dim = config.head_dim
    q_local = mesh_config.shard_size(config.num_heads) * head_dim
    k_local = mesh_config.shard_size(config.num_kv_heads) * head_dim
    total = xqkv_fused.shape[-1]

    q = ttnn.slice(xqkv_fused, [0, 0, 0, 0], [xqkv_fused.shape[0], xqkv_fused.shape[1], xqkv_fused.shape[2], q_local])
    k = ttnn.slice(
        xqkv_fused,
        [0, 0, 0, q_local],
        [xqkv_fused.shape[0], xqkv_fused.shape[1], xqkv_fused.shape[2], q_local + k_local],
    )
    v = ttnn.slice(
        xqkv_fused, [0, 0, 0, q_local + k_local], [xqkv_fused.shape[0], xqkv_fused.shape[1], xqkv_fused.shape[2], total]
    )

    # normalized_size is the GLOBAL width (across all TP shards): num_heads*head_dim for Q.
    q = distributed_rms_norm(
        q, weights.q_norm, config.num_heads * head_dim, config.rms_norm_eps, mesh_config, ccl_manager
    )
    k = distributed_rms_norm(
        k, weights.k_norm, config.num_kv_heads * head_dim, config.rms_norm_eps, mesh_config, ccl_manager
    )

    out = ttnn.concat([q, k, v], dim=-1)
    q.deallocate(True)
    k.deallocate(True)
    v.deallocate(True)
    return out


def concat_heads(tensor):
    """
    Concatenate attention heads back to hidden dimension (prefill).

    Args:
        tensor: Attention output tensor with separate heads

    Returns:
        Tensor with concatenated heads [batch, seq_len, hidden_size]
    """
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype):
    """
    Apply output projection.

    MiniMax-M2 has no bias on o_proj (bias=False in HF), so this is a plain matmul.

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
    return out


# Per-M_tiles tuned config for the fused attention o_proj matmul + reduce-scatter.
# Tuple layout: (grid_y, M_block, K_block, N_block, chunk_width, subblock_h, subblock_w, num_workers).
# S=1024 (M_tiles=32): min=256us on Tracy (2026-04-23 sweep); see commit for methodology.
_FUSED_MM_RS_CONFIGS = {
    32: (4, 4, 4, 6, 2, 2, 2, 3),  # S=1024
}


def is_shape_fused_mm_rs_supported(tensor) -> bool:
    m_tiles = (tensor.shape[-2] + 31) // 32
    return m_tiles in _FUSED_MM_RS_CONFIGS


def apply_output_projection_fused_rs(tensor, weights: AttentionWeights, mesh_config, ccl_manager):
    """Attention output projection + TP reduce-scatter fused into one device op.

    Replaces the sequential `apply_output_projection` + first half of
    `mesh_config.allreduce` (the reduce-scatter). The trailing all-gather and
    padding-trim stay as separate ops in `apply_allgather_and_slice`.

    Internally uses `ttnn.experimental.minimal_matmul_strided_reduce_scatter_async`,
    which overlaps MM compute on one half of the core grid with the RS ring
    traffic on the other half — MM signals RS via on-chip semaphore per output
    block so RS starts forwarding as soon as the first block lands in DRAM.

    Dtypes (end-to-end bf8_b through the TP allreduce):
      - activation cast to bf8_b before MM (DRAM-bandwidth saver)
      - weight is bf8_b (as loaded, row-parallel sharded on K across TP=8)
      - MM math: LoFi (same as `ttnn.matmul` default, via compute_kernel_config)
      - fused-op output inherits input dtype → bf8_b
      - downstream all-gather + padding-slice operate on bf8_b

    Returns the reduce-scattered tensor of shape
    [1, 1, S, padded_hidden_total / TP] in bf8_b.

    Only valid for prefill (TP > 1); the decoder path uses a different pattern.
    """
    TILE = 32
    K = tensor.shape[-1]
    N = weights.o_proj.shape[-1]
    assert K % TILE == 0 and N % TILE == 0, f"K={K}, N={N} must be tile-aligned"

    M_tiles = (tensor.shape[-2] + TILE - 1) // TILE
    K_tiles = K // TILE
    N_tiles = N // TILE

    if M_tiles not in _FUSED_MM_RS_CONFIGS:
        raise ValueError(
            f"No tuned fused o_proj MM+RS config for M_tiles={M_tiles}; "
            f"tuned shapes: {sorted(_FUSED_MM_RS_CONFIGS)}. "
            f"Use apply_output_projection + apply_allreduce for untuned shapes."
        )
    grid_y, m_block, k_block, n_block, chunk_width, subblock_h, subblock_w, num_workers = _FUSED_MM_RS_CONFIGS[M_tiles]

    mm_core_grid = ttnn.CoreCoord(8, grid_y)
    Nt_per_core = N_tiles // mm_core_grid.x
    Mt_per_core = max(1, math.ceil(M_tiles / grid_y))

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
        bias=None,  # MiniMax-M2 has no o_proj bias
        config=mm_config,
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        chunk_width_in_mm_blocks=chunk_width,
        num_workers_per_link=num_workers,
    )

    tensor.deallocate(True)
    mm_out.deallocate(True)
    return rs_out


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
        # ``mesh_config.allreduce`` now frees its input internally between
        # reduce_scatter and all_gather to keep peak DRAM within bounds for
        # long-context prefill; don't deallocate again here.
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
