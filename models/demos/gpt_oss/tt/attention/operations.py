# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import ttnn

from .weights import AttentionWeights


def _env_int(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def _fused_attn_out_enabled():
    return os.environ.get("GPT_OSS_FUSED_ATTN_OUT", "0") == "1"


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


def apply_output_projection_fused_rs(tensor, weights: AttentionWeights, mesh_config, ccl_manager):
    """Fused output projection matmul + reduce scatter (prefill, TP>1 only).

    Replaces the sequential `apply_output_projection` + `mesh_config.allreduce` (RS half)
    with a single `minimal_matmul_strided_reduce_scatter_async` call. The trailing
    all-gather + padding slice stay in `apply_allreduce_after_fused_rs`.

    Returns the reduce-scatter output tensor [1, 1, S, padded_hidden/TP].
    """
    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)

    TILE = 32
    K = tensor.shape[-1]
    N = weights.o_proj.shape[-1]
    assert K % TILE == 0 and N % TILE == 0, f"K={K}, N={N} must be tile-aligned"

    # Tunable via env for sweeps. Defaults came out of the S=128 parameter
    # sweep on TG (num_links=4, 4x8 mesh): winner was
    #   mm_grid=(8,2), M_block=2, K_block=8, N_block=6, chunk_width=4
    # at 105.8 us avg (vs 173.9 us for non-fused MM+bias+RS).
    mm_grid_x = _env_int("GPT_OSS_FMM_GRID_X", 8)
    mm_grid_y = _env_int("GPT_OSS_FMM_GRID_Y", 2)
    mm_core_grid = ttnn.CoreCoord(mm_grid_x, mm_grid_y)

    M_tiles = (tensor.shape[-2] + TILE - 1) // TILE
    K_tiles = K // TILE
    N_tiles = N // TILE
    Nt_per_core = N_tiles // mm_grid_x
    Mt_per_core = max(1, math.ceil(M_tiles / mm_grid_y))

    M_block = _env_int("GPT_OSS_FMM_MBLOCK", min(Mt_per_core, 2))
    K_block = _env_int("GPT_OSS_FMM_KBLOCK", min(8, K_tiles))
    N_block = _env_int("GPT_OSS_FMM_NBLOCK", min(6, Nt_per_core))
    subblock_h = _env_int("GPT_OSS_FMM_SBH", 1)
    subblock_w = _env_int("GPT_OSS_FMM_SBW", 1)
    chunk_width = _env_int("GPT_OSS_FMM_CHUNK", 4)
    num_workers_env = os.environ.get("GPT_OSS_FMM_WORKERS", "")
    num_workers = int(num_workers_env) if num_workers_env else None
    num_buffers_env = os.environ.get("GPT_OSS_FMM_BUFFERS", "")
    num_buffers = int(num_buffers_env) if num_buffers_env else None

    config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block,
        K_block_size=K_block,
        N_block_size=N_block,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=mm_core_grid,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        ccl_manager.mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    mm_out, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
        tensor,
        weights.o_proj,
        3,
        ccl_manager.get_rs_ping_pong_semaphore(),
        ttnn.CoreCoord(0, mm_grid_y),
        compute_kernel_config=compute_config,
        num_links=ccl_manager.num_links,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ccl_manager.topology,
        cluster_axis=mesh_config.tp_axis,
        bias=weights.o_proj_bias,
        config=config,
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        chunk_width_in_mm_blocks=chunk_width,
        num_workers_per_link=num_workers,
        num_buffers_per_channel=num_buffers,
    )

    tensor.deallocate(True)
    mm_out.deallocate(True)
    return rs_out


def apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size: int):
    """Complete the attention-output path after a fused MM+RS: AG, then drop padding."""
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
