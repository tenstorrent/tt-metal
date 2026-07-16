# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS attention primitive ops. Mirrors ``minimax_m3/tt/attention/operations.py`` and
``gpt_oss/tt/attention/operations.py`` with the gpt-oss deltas: QKV/O projections carry bias,
RoPE is FULL rotary (rotary_dim == head_dim, no partial-rotary slice/concat), and there is no
QK-norm.
"""

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """
    Apply QKV projection and add the fused QKV bias.

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        weights: Attention weights container

    Returns:
        Fused QKV tensor [batch, seq_len, total_qkv_dim]
    """
    xqkv_fused = ttnn.linear(hidden_states, weights.wqkv, bias=weights.wqkv_bias, dtype=ttnn.bfloat16)
    return xqkv_fused


def split_qkv_heads_prefill(xqkv_fused, num_heads: int, num_kv_heads: int):
    """
    Split fused QKV into separate head tensors for prefill mode (GQA: num_heads Q, num_kv_heads K/V).

    Args:
        xqkv_fused: Fused QKV tensor
        num_heads: Number of (local) Q heads
        num_kv_heads: Number of (local) K/V heads

    Returns:
        Tuple (Q, K, V) with shapes [1, num_heads, seq_len, head_dim] / [1, num_kv_heads, seq_len, head_dim]
    """
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_rope(tensor, rope_mats, transformation_mat, is_decode_mode: bool = False):
    """
    Apply rotary position embedding (RoPE) — FULL rotary for gpt-oss (rotary_dim == head_dim, 64).

    The YaRN scaling (rope_theta 150000, factor 32, orig_max_pos 4096) is baked into the cos/sin
    matrices (``rope_mats``) at build time, so this op is a plain full rotation of the whole head.

    Args:
        tensor: Input tensor (Q or K), shape [1, n_heads, seq_len, head_dim]
        rope_mats: Tuple/list of (cos, sin) matrices, last dim = head_dim
        transformation_mat: Transformation matrix for the mode
        is_decode_mode: Whether in decode mode (False for prefill)

    Returns:
        Tensor with RoPE applied
    """
    return ttnn.experimental.rotary_embedding_llama(
        tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=is_decode_mode
    )


def concat_heads(tensor):
    """
    Concatenate attention heads back to the hidden dimension (prefill).

    Args:
        tensor: Attention output tensor with separate heads [1, n_heads, seq_len, head_dim]

    Returns:
        Tensor with concatenated heads [1, 1, seq_len, hidden_size]
    """
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype):
    """
    Apply output projection and add the o_proj bias.

    Args:
        tensor: Attention output tensor [1, 1, seq_len, local_hidden]
        weights: Attention weights container
        activation_dtype: Target dtype for output

    Returns:
        Output tensor after projection (+ bias)
    """
    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)
    out = ttnn.matmul(tensor, weights.o_proj, dtype=activation_dtype)
    tensor.deallocate(True)
    ttnn.add(out, weights.o_proj_bias, output_tensor=out)
    return out


# Per-M_tiles tuned config for the fused attention o_proj matmul + reduce-scatter.
# Tuple layout: (grid_y, M_block, K_block, N_block, chunk_width, subblock_h, subblock_w, num_workers).
_FUSED_MM_RS_CONFIGS = {
    32: (4, 4, 4, 6, 2, 2, 2, 3),  # S=1024
}


def is_shape_fused_mm_rs_supported(tensor) -> bool:
    # The async fused matmul+reduce_scatter (minimal_matmul_strided_reduce_scatter_async) RACES on
    # Blackhole (semaphore/overlap sync bug) -> non-deterministic garbage. It is only validated on
    # Wormhole, so gate the fused path off on Blackhole and fall back to the correct non-fused
    # matmul+allreduce path there. Remove this gate once the fused-op sync is fixed for Blackhole.
    if "blackhole" in ttnn.get_arch_name():
        return False
    m_tiles = (tensor.shape[-2] + 31) // 32
    return m_tiles in _FUSED_MM_RS_CONFIGS


def apply_output_projection_fused_rs(tensor, weights: AttentionWeights, mesh_config, ccl_manager):
    """Attention output projection + TP reduce-scatter fused into one device op.

    Replaces the sequential `apply_output_projection` + first half of `mesh_config.allreduce`
    (the reduce-scatter). The trailing all-gather and padding-trim stay as separate ops in
    `apply_allgather_and_slice`. Internally uses
    `ttnn.experimental.minimal_matmul_strided_reduce_scatter_async` (Ring topology only).

    Returns the reduce-scattered tensor [1, 1, S, padded_hidden_total / TP] in bf8_b.
    Only valid for prefill (TP > 1).
    """
    TILE = 32
    K = tensor.shape[-1]
    N = weights.o_proj.shape[-1]
    assert K % TILE == 0 and N % TILE == 0, f"K={K}, N={N} must be tile-aligned"

    M_tiles = (tensor.shape[-2] + TILE - 1) // TILE
    N_tiles = N // TILE

    if M_tiles not in _FUSED_MM_RS_CONFIGS:
        raise ValueError(
            f"No tuned fused o_proj MM+RS config for M_tiles={M_tiles}; "
            f"tuned shapes: {sorted(_FUSED_MM_RS_CONFIGS)}. "
            f"Use apply_output_projection + apply_allreduce for untuned shapes."
        )
    grid_y, m_block, k_block, n_block, chunk_width, subblock_h, subblock_w, num_workers = _FUSED_MM_RS_CONFIGS[M_tiles]

    mm_core_grid = ttnn.CoreCoord(8, grid_y)

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
        num_workers_per_link=num_workers,
    )

    tensor.deallocate(True)
    mm_out.deallocate(True)
    return rs_out


def apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size: int):
    """Complete the attention-output chain after `apply_output_projection_fused_rs`.

    Runs the all-gather (TP) on the scattered output, then drops the padding columns that
    `weights.py` adds for tile alignment of the CCL ops.
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
    Apply tensor-parallel allreduce if needed (TP > 1), then strip the o_proj tile-alignment padding.

    Args:
        tensor: Input tensor
        mesh_config: Mesh configuration
        ccl_manager: Communication manager
        hidden_size: Hidden size (for padding removal)

    Returns:
        Tensor after allreduce (if TP > 1) or the original tensor
    """
    if mesh_config.tp > 1:
        tensor_allreduced = mesh_config.allreduce(tensor, ccl_manager, pad_size=0, axis=mesh_config.tp_axis)
        # `mesh_config.allreduce` frees its input internally between reduce_scatter and all_gather;
        # don't deallocate again here.
        tensor = tensor_allreduced

        # Remove padding added in weights.py for tile-aligned CCL operations (e.g. 360 -> 384).
        local_hidden = hidden_size // mesh_config.tp
        padded_local_hidden = ((local_hidden + 31) // 32) * 32
        if padded_local_hidden != local_hidden:
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
