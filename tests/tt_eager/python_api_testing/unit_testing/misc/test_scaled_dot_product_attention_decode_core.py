# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
    comp_and_get_pcc,
)
import ttnn
from loguru import logger
import pytest
from enum import Enum
from typing import Optional
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_blackhole
import math
import numpy as np
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
)


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


def get_chunk_size(max_start_pos, s):
    if max_start_pos <= 32:
        chunk_size = 32
    elif max_start_pos <= 64:
        chunk_size = 32
    elif max_start_pos <= 128:
        chunk_size = 32
    elif max_start_pos <= 1024:
        chunk_size = 128
    else:
        chunk_size = 512
    # find maximum power of 2 divisor of s
    for i in range(1, s):
        if s % (2 ** (i + 1)) != 0:
            break
    chunk_size = min(chunk_size, 2**i)
    return chunk_size


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def get_min_pcc(q_dtype: ttnn.DataType, kv_dtype: ttnn.DataType, num_parallel_cores: int):
    if num_parallel_cores == 1:
        min_pcc = 0.90
    else:
        min_pcc = 0.99
        if q_dtype == ttnn.bfloat8_b:
            min_pcc = 0.98
        min_pcc = 0.91 if kv_dtype == ttnn.bfloat4_b else min_pcc
    return min_pcc


def run_reference_op(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, padded_seq_len: int, scale, attn_mask: torch.Tensor
):
    """
    Run the reference scaled dot product attention operation.
    Args:
        Q: Query tensor
        K: Key tensor
        V: Value tensor
        padded_seq_len: Padded sequence length
        scale: Scale factor
        attn_mask: Attention mask
        is_causal: Whether to apply causal masking
    Returns:
        out: Output tensor
    """
    b, nh, _, _ = Q.shape  # b, nh, 1, d
    _, nkv, _, _ = K.shape

    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
    K_slice = K[:, :, :padded_seq_len, :]  # b, nkv, S, d
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    V_slice = V[:, :, :padded_seq_len, :]  # b, nkv, S, d
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S
    out = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    out = out.squeeze(2).unsqueeze(0)
    return out


def create_qo_memory_config(
    is_sharded: bool, q_shape: tuple, q_layout=ttnn.TILE_LAYOUT, start_core=ttnn.CoreCoord(0, 0), sub_core_grids=None
):
    """
    Create the memory config for the query or output tensor

    Args:
        is_sharded: Whether query or output tensor is sharded on L1 or in DRAM
        q_shape: Shape of the query or output tensor
        q_layout: TILE LAYOUT or ROW MAJOR LAYOUT of query or output tensor
        sub_core_grids: The sub core grids to shard the query or output tensor on.
    """
    _, b, nh, d = q_shape
    if is_sharded:
        # If Q Layout is TILE LAYOUT we pad to 32 Q heads
        nh = nearest_pow_2(nearest_n(nh, n=32)) if q_layout == ttnn.TILE_LAYOUT else nh
        if sub_core_grids is None:
            shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
        else:
            shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(start_core, b, sub_core_grids, row_wise=True)
        shard_spec = ttnn.ShardSpec(shard_grid, (nh, d), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    else:
        return ttnn.DRAM_MEMORY_CONFIG


def create_cur_pos_tensor(
    batch_size: int,
    seq_len: int,
    pos_id: int,
    pos_type: str,
    is_sharded: bool,
    sub_core_grids: ttnn.CoreRangeSet,
    use_cur_pos_tensor: bool,
    device: ttnn.MeshDevice,
) -> tuple[ttnn.Tensor, torch.Tensor]:
    """
    Create the current position tensors for the attention cache.

    Position Types:
        - random: Randomly sample a position for each batch.
        - linear: Sample a position for each batch in a linear fashion.
        - constant: Use a constant position for all batches.
        - drop_users: Drop every other user.
    Args:
        batch_size: The number of batches.
        seq_len: The full sequence length.
        pos_id: The position id. If pos_type is "constant", this is the constant position id.
        pos_type: The type of position to create. Can be "random", "linear", "constant", "half_seq_len", or "drop_users".
        is_sharded: Whether to shard/replicatethe current position on the L1 of sub core grids.
        sub_core_grids: The sub core grids to shard the current position on.
        use_cur_pos_tensor: Whether to create a ttnn tensor for the current position
    """
    if pos_type == "random":
        cur_pos = torch.randint(0, seq_len, (batch_size,), dtype=torch.int32)
    elif pos_type == "linear":
        cur_pos = torch.linspace(0, seq_len - 1, batch_size, dtype=torch.int32)
    elif pos_type == "constant":
        cur_pos = torch.full((batch_size,), pos_id, dtype=torch.int32)
    elif pos_type == "drop_users":
        cur_pos = torch.tensor([100 if b % 2 == 0 else -1 for b in range(batch_size)], dtype=torch.int32)
    elif pos_type == "half_seq_len":
        cur_pos = torch.tensor([seq_len // 2 for _ in range(batch_size)], dtype=torch.int32)
    else:
        raise ValueError(f"Invalid position type: {pos_type}")
    cur_pos = cur_pos.repeat(sub_core_grids.num_cores(), 1) if is_sharded else cur_pos
    cur_pos_shard_spec = (
        ttnn.ShardSpec(sub_core_grids, (batch_size,), ttnn.ShardOrientation.ROW_MAJOR) if is_sharded else None
    )
    cur_pos_memory_config = (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cur_pos_shard_spec)
        if is_sharded
        else ttnn.DRAM_MEMORY_CONFIG
    )
    tt_cur_pos = (
        ttnn.as_tensor(
            cur_pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=cur_pos_memory_config, device=device
        )
        if use_cur_pos_tensor
        else None
    )
    cur_pos_pt = cur_pos.tolist()
    return tt_cur_pos, cur_pos_pt


def create_page_table_tensor(
    batch_size: int,
    config: PagedAttentionConfig,
    is_sharded: bool,
    sub_core_grids: ttnn.CoreRangeSet,
    device: ttnn.MeshDevice,
) -> tuple[ttnn.Tensor, torch.Tensor]:
    """
    Setup the page-related tensors for the attention cache.
    Args:
        batch_size: The number of batches.
        config: PagedAttentionConfig object containing configuration parameters.
        is_sharded: Whether to shard the page table.
        sub_core_grids: The sub core grids to shard the page table on.
    Returns:
        tt_page_table: The TTNN page table tensor.
        page_table: The torch page table tensor.
    """
    block_size, max_num_blocks = config.block_size, config.max_num_blocks
    assert (
        max_num_blocks % batch_size == 0
    ), f"max_num_blocks {max_num_blocks} must be divisible by batch_size {batch_size}."

    # Torch page table tensor
    page_table = torch.randperm(max_num_blocks, dtype=torch.int32)
    page_table = page_table.reshape(batch_size, max_num_blocks // batch_size)
    page_table = page_table.repeat(sub_core_grids.num_cores(), 1) if is_sharded else page_table

    # Memory config / Layout / Shard spec for the TTNN page table tensor
    page_table_dtype = ttnn.uint16 if is_sharded else ttnn.int32
    page_table_layout = ttnn.ROW_MAJOR_LAYOUT
    page_table_shard_spec = (
        ttnn.ShardSpec(sub_core_grids, (batch_size, max_num_blocks // batch_size), ttnn.ShardOrientation.ROW_MAJOR)
        if is_sharded
        else None
    )
    page_table_memory_config = (
        ttnn.DRAM_MEMORY_CONFIG
        if not is_sharded
        else ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, page_table_shard_spec)
    )
    # TTNN page table tensor
    tt_page_table = ttnn.as_tensor(
        page_table,
        dtype=page_table_dtype,
        layout=page_table_layout,
        memory_config=page_table_memory_config,
        device=device,
    )
    return tt_page_table, page_table


def to_paged_cache(
    cache: torch.Tensor,
    mapping: torch.Tensor,
    config: PagedAttentionConfig,
) -> torch.Tensor:
    """
    Convert a cache tensor to a paged cache using the provided mapping.
    Args:
        cache: The original cache tensor.
        mapping: The mapping tensor that defines how to convert the cache.
        config: PagedAttentionConfig object containing configuration parameters.
    Returns:
        paged_cache: The converted paged cache tensor.
    """
    batch_size, nh, seq_len, dim = cache.shape

    block_size, max_num_blocks = config.block_size, config.max_num_blocks
    assert (
        max_num_blocks % batch_size == 0
    ), f"max_num_blocks {max_num_blocks} must be divisible by batch_size {batch_size}."
    assert seq_len == block_size * (
        max_num_blocks // batch_size
    ), f"Sequence length {seq_len} must equal effective paged seq_len {block_size * (max_num_blocks // batch_size)}."

    paged_cache = cache.reshape(batch_size, nh, -1, block_size, dim)  # (B, H, num_blocks // B, block_size, D)
    paged_cache = paged_cache.transpose(1, 2)  # (B, num_blocks // B, H, block_size, D)
    paged_cache = paged_cache.reshape(max_num_blocks, nh, block_size, dim)  # (num_blocks, H, block_size, D)

    """
    Get the reverse mapping to reorder the paged cache,
    so that paged cache + mapping = original cache
    and paged_cache = original_cache + inverse mapping

    For example:
        cache = [0, 1, 2, 3]
        mapping = [1, 3, 0, 2]
        inverse_mapping (argsort) = [2, 0, 3, 1]
    Then,
        paged_cache = cache[inverse_mapping] = [2, 0, 3, 1]
        paged_cache[mapping] = cache = [0, 1, 2, 3]
    """

    inverse_mapping = torch.argsort(mapping.view(-1))
    paged_cache = paged_cache[inverse_mapping]

    return paged_cache


def from_paged_cache(
    paged_cache: torch.Tensor,
    mapping: torch.Tensor,
    config: PagedAttentionConfig,
) -> torch.Tensor:
    """
    Convert a paged cache back to the original cache format using the provided mapping.
    Args:
        paged_cache: The paged cache tensor.
        mapping: The mapping tensor that defines how to convert the paged cache.
        config: PagedAttentionConfig object containing configuration parameters.
    Returns:
        cache: The converted cache tensor.
    """
    max_num_blocks, nh, block_size, dim = paged_cache.shape  # (max_num_blocks, H, block_size, D)
    assert (
        block_size == config.block_size
    ), f"block_size {block_size} must match the paged attention config block size {config.block_size}."
    assert (
        max_num_blocks == config.max_num_blocks
    ), f"max_num_blocks {max_num_blocks} must match the paged attention config max_num_blocks {config.max_num_blocks}."

    batch, num_blocks_per_batch = mapping.shape

    # Use the mapping to get the original order, paged_cache + mapping = original cache
    cache = paged_cache[mapping.view(-1)]

    # Reshape the paged cache back to the original cache format (make the cache contiguous)
    cache = cache.reshape(batch, num_blocks_per_batch, nh, block_size, dim)  # (B, num_blocks // B, H, block_size, D)
    cache = cache.transpose(1, 2)  # (B, H, num_blocks // B, block_size, D)
    cache = cache.reshape(batch, nh, -1, dim)  # (B, H, seq_len, D)

    return cache


def create_attention_mask(
    b: int, nh: int, seq_len: int, cur_pos_list: int, sliding_window_size: Optional[int], mask_type: str
):
    """
    Create attention mask for the given mask type.

    - causal: mask positions before current position
    - non_causal: mask positions after current position
    - sliding_window: mask positions outside sliding window

    Args:
        b: batch size
        nh: number of heads
        seq_len: sequence length
        cur_pos_list: list of current positions for each batch
        sliding_window_size: sized of the sliding window to perform attention on
        mask_type: type of mask to create

    Returns:
        attn_mask: [b, 1, nh, seq_len] mask with -inf for positions outside window
    """
    assert len(cur_pos_list) == b, "Length of cur_pos_list should be equal to batch size"
    if mask_type == "causal":
        attn_mask = torch.zeros((b, nh, 1, seq_len))
        for i in range(b):
            cur_pos = cur_pos_list[i]
            attn_mask[i, :, :, cur_pos + 1 :] = torch.finfo(torch.float32).min
    elif mask_type == "non_causal":
        attn_mask = torch.bernoulli(torch.full((batch, nh, 1, seq_len), 0.25))
        attn_mask = attn_mask * torch.finfo(torch.float32).min
    elif mask_type == "sliding_window":
        assert (
            sliding_window_size is not None and sliding_window_size > 0
        ), f"Mask type requested is sliding_window but provided invalid sliding_window_size {sliding_window_size}"
        attn_mask = torch.zeros((b, nh, 1, seq_len))
        for i in range(b):
            cur_pos = cur_pos_list[i]
            window_end = cur_pos + 1  # exclusive
            window_start = max(0, window_end - sliding_window_size)
            if window_start > 0:
                attn_mask[i, :, :, :window_start] = torch.finfo(torch.float32).min
            if cur_pos < seq_len:
                attn_mask[i, :, :, cur_pos : cur_pos + 1 + 1 :] = torch.finfo(torch.float32).min
    else:
        raise ValueError(f"Invalid mask type: {mask_type}. Only causal, non-causal and sliding window are supported")
    return attn_mask


def run_scaled_dot_product_attention_decode(
    device: ttnn.MeshDevice,
    b: int,
    s: int,
    nh: int,
    nkv: int,
    d: int,
    q_dtype: ttnn.DataType,
    q_layout: ttnn.TensorMemoryLayout,
    kv_dtype: ttnn.DataType,
    q_chunk_size=None,
    k_chunk_size=None,
    grid_size: tuple[int, int] = (8, 8),
    sliding_window_size: Optional[int] = None,
    use_cur_pos_tensor: bool = False,
    cur_pos_type: str = "half_seq_len",
    cur_pos_id: Optional[int] = None,
    cur_pos_is_sharded: bool = False,
    page_table_is_sharded: bool = False,
    sharded_in: bool = False,
    sharded_out: bool = False,
    start_core: ttnn.CoreCoord = ttnn.CoreCoord(0, 0),
    sub_core_grids: ttnn.CoreRangeSet = None,
    is_causal: bool = True,
    use_paged_attention: bool = False,
    is_single_iter: bool = True,
    is_multi_pos: bool = False,
    block_size: int = 32,
    mask_type: str = "causal",
    num_repeat_iters: int = 1,
    step_size: int = 1,
):
    # Log the test parameters
    logger.debug(f"Running Scaled Dot Product Attention Decode with parameters: ")
    logger.debug(f"Batch: {b}")
    logger.debug(f"Sequence Length: {s}")
    logger.debug(f"Number of Heads (Q): {nh}")
    logger.debug(f"Number of Heads (KV): {nkv}")
    logger.debug(f"Dimensionality: {d}")
    logger.debug(f"Query Data Type: {q_dtype}")
    logger.debug(f"Query Memory Layout: {q_layout}")
    logger.debug(f"Key-Value Data Type: {kv_dtype}")
    logger.debug(f"Is Input Sharded: {sharded_in}")
    logger.debug(f"Is Output Sharded: {sharded_out}")
    logger.debug(f"Q chunk size: {q_chunk_size}")
    logger.debug(f"K chunk size: {k_chunk_size}")
    logger.debug(f"Is Causal Attention: {k_chunk_size}")
    logger.debug(f"Is Paged Attention: {use_paged_attention}")
    logger.debug(f"Is Single Iter: {is_single_iter}")
    logger.debug(f"Block Size: {block_size}")
    logger.debug(f"Sliding Window Size: {sliding_window_size}")
    logger.debug(f"Number of repeated iterations of same position: {num_repeat_iters}")
    compute_grid_size = device.compute_with_storage_grid_size()
    grid_height, grid_width = grid_size
    num_grid_cores = math.prod(grid_size)
    num_cores = sub_core_grids.num_cores() if sub_core_grids is not None else num_grid_cores
    compute_sub_core_grids = (
        ttnn.num_cores_to_corerangeset_in_subcoregrids(start_core, num_grid_cores, sub_core_grids, row_wise=True)
        if sub_core_grids is not None
        else None
    )
    if sub_core_grids is not None:
        assert (
            compute_sub_core_grids.num_cores() == num_cores
        ), f"Number of cores {num_cores} must be equal to number of compute sub core grids {compute_sub_core_grids.num_cores()}"
    assert (
        grid_height <= compute_grid_size.x
    ), f"Invalid grid height on X dimension ({grid_height} > {compute_grid_size.x})"
    assert (
        grid_width <= compute_grid_size.y
    ), f"Invalid grid height on Y dimension ({grid_width} > {compute_grid_size.y})"

    # Set min pcc based on q_dtype, kv_dtype, and number of cores per batch
    min_pcc = get_min_pcc(q_dtype, kv_dtype, num_cores // b)

    # Prepare KV
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    # Paged Attention Setup
    paged_attention_cfg = None
    tt_page_table = None
    if use_paged_attention:
        assert s % block_size == 0, f"Sequence length must be a multiple of {block_size} for paged attention."
        max_num_blocks = s // block_size * b
        paged_attention_cfg = PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )
        tt_page_table, page_table = create_page_table_tensor(
            b, paged_attention_cfg, page_table_is_sharded, sub_core_grids, device
        )

        # Paged K and V tensors
        K_paged = to_paged_cache(K, page_table, paged_attention_cfg)
        V_paged = to_paged_cache(V, page_table, paged_attention_cfg)

        # Unshuffled K and V tensors
        K_unshuffled = from_paged_cache(K_paged, page_table, paged_attention_cfg)
        V_unshuffled = from_paged_cache(V_paged, page_table, paged_attention_cfg)

        assert torch.allclose(K, K_unshuffled)
        assert torch.allclose(V, V_unshuffled)

    tt_K = ttnn.as_tensor(
        K_paged if use_paged_attention else K,
        device=device,
        dtype=kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_V = ttnn.as_tensor(
        V_paged if use_paged_attention else V,
        device=device,
        dtype=kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Default SDPA Program Config
    q_chunk_size = nearest_pow_2(nearest_n(nh, n=32)) if q_chunk_size is None else q_chunk_size
    k_chunk_size = get_chunk_size(cur_pos_id + 1, s) if k_chunk_size is None else k_chunk_size
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        sub_core_grids=compute_sub_core_grids,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    # Compute Kernel Config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Prepare scale
    scale = d**-0.5
    # Prepare current position tensor
    tt_cur_pos, cur_pos = create_cur_pos_tensor(
        b, s, cur_pos_id, cur_pos_type, cur_pos_is_sharded, sub_core_grids, use_cur_pos_tensor, device
    )

    # Defining TTNN OP
    # Choose which attention op to use (paged flash decode or normal flash decode)
    run_sdpa_decode = (
        ttnn.transformer.paged_scaled_dot_product_attention_decode
        if use_paged_attention
        else ttnn.transformer.scaled_dot_product_attention_decode
    )

    failures = []
    all_pcc_values = []
    all_pos_pass = True
    tolerance = 0.95
    max_cur_pos = max(cur_pos)
    max_num_iters = 1 if is_single_iter else num_repeat_iters

    # Iterate over seq len
    assert (
        is_multi_pos and step_size > 0
    ), f"Step size must be greater than 0 for multi-position testing, got {step_size}"
    pos_range = range(max_cur_pos, s, step_size)

    for pos_idx in pos_range:
        all_iters_pass = True
        prev_pcc = None

        # We randomize Q on each decode iteration
        Q = fa_rand(1, b, nh, d)

        # Input Q and Output memory configs
        q_memory_config = create_qo_memory_config(sharded_in, Q.shape, q_layout, start_core, sub_core_grids)
        out_memory_config = create_qo_memory_config(sharded_out, Q.shape, ttnn.TILE_LAYOUT, start_core, sub_core_grids)

        # Stress test for max_num_iters iterations for each decode iteration
        # Only runs once if is_single_iter = True
        for iters in range(max_num_iters):
            # Create TTNN Query tensor
            tt_Q = ttnn.as_tensor(Q, device=device, dtype=q_dtype, layout=q_layout, memory_config=q_memory_config)

            # Update Program Configs
            k_chunk_size = get_chunk_size(pos_idx + 1, s)
            padded_seq_len = nearest_n(pos_idx + 1, n=k_chunk_size) if is_causal else s

            sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=q_chunk_size,
                k_chunk_size=k_chunk_size,
                exp_approx_mode=False,
            )

            # Update Attention Mask
            attn_mask = create_attention_mask(b, nh, padded_seq_len, cur_pos, sliding_window_size, mask_type)
            tt_attn_mask = ttnn.as_tensor(
                attn_mask,
                device=device,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # SDPA Decode Args
            sdpa_kwargs = {
                "cur_pos": cur_pos,  # cur_pos_ids is not used if cur_pos_tensor is used OR attention is non-causal
                "cur_pos_tensor": tt_cur_pos,
                "page_table_tensor": tt_page_table,
                "sliding_window_size": sliding_window_size,
                "attn_mask": None if is_causal else tt_attn_mask,
                "scale": scale,
                "program_config": sdpa_program_config,
                "compute_kernel_config": compute_kernel_config,
                "memory_config": out_memory_config,
            }
            sdpa_kwargs.pop("cur_pos_tensor") if not use_cur_pos_tensor else sdpa_kwargs.pop("cur_pos")

            # ==== Run TTNN Scaled Dot Product Attention Decode ====
            tt_out = run_sdpa_decode(tt_Q, tt_K, tt_V, **sdpa_kwargs)
            tt_out = ttnn.to_torch(tt_out)[:, :, :nh, :]

            # ==== Run Torch Scaled Dot Product Attention Decode ====
            ref_out = run_reference_op(Q, K, V, padded_seq_len, scale, attn_mask)

            # ==== Perform PCC Validation Check ====
            out_pass, out_pcc, pcc_val = comp_and_get_pcc(ref_out, tt_out, min_pcc)
            logger.debug(
                f"TTNN vs PyTorch Output PCC Check: {out_pcc} for current position idx {pos_idx} for iters = {iters}"
            )

            # Check for ND PCC (all iterations with same Q must have same PCC)
            all_iters_pass = all_iters_pass and out_pass
            if prev_pcc is not None:
                assert out_pcc == prev_pcc, f"Iteration {iters}: pcc changed from {prev_pcc} to {out_pcc}"
            prev_pcc = pcc_val

            # Store failures
            if not out_pass:
                failures.append(
                    {
                        "current_position_idx": pos_idx,
                        "pcc_value": pcc_val,
                        "repeat_iteration": iters,
                        "tt_output": tt_out,
                        "torch_output": ref_out,
                    }
                )

            # Store PCC values for each iteration
            all_pcc_values.append(pcc_val)

        # Advance current position if is_multi_pos is True
        if is_multi_pos:
            # Update cur pos tensors
            if tt_cur_pos is not None:
                for i in range(step_size):
                    ttnn.plus_one(tt_cur_pos)
            cur_pos = [pos + step_size for pos in cur_pos]
        else:
            break

    if len(failures) > 0:
        for f in failures:
            logger.debug(f"PCC Check FAILED with the following configuration:")
            logger.debug(f"max_cur_pos = {f['max_cur_pos']}")
            logger.debug(f"repeat_iteration = {f['repeat_iteration']}")
            logger.debug(f"tt_output = {f['tt_output']}")
            logger.debug(f"torch_output = {f['torch_output']}")
    else:
        logger.debug("All decode iterations and iterations PASSED")

    def tolerance_pcc_pass(pcc_values, min_pcc, tolerance):
        pcc_values = torch.as_tensor(pcc_values)
        return (pcc_values >= min_pcc).float().mean().item() >= tolerance

    assert tolerance_pcc_pass(
        all_pcc_values, min_pcc, tolerance
    ), f"At least {tolerance*100}% of the PCC values must be greater than or equal to {min_pcc}"


@pytest.mark.parametrize(
    "kv_dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.bfloat8_b, ttnn.bfloat16],
        [ttnn.bfloat16, ttnn.bfloat16],
    ],
    ids=[
        "bfp8_cache_bfp8_query",
        "bfp8_cache_bfp16_query",
        "bfp16_cache_bfp16_query",
    ],
)
@pytest.mark.parametrize("q_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("q_chunk_size, k_chunk_size", [(0, 0), (32, 32), (128, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("start_core", [ttnn.CoreCoord(1, 0)])
@pytest.mark.parametrize("sub_core_grids", [None])
@pytest.mark.parametrize("use_paged_attention", [True, False])
@pytest.mark.parametrize("cur_pos_is_sharded", [False])
@pytest.mark.parametrize("page_table_is_sharded", [False])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window_size", [None])
@pytest.mark.parametrize("num_repeat_iters", [1])
@pytest.mark.parametrize(
    "is_causal, mask_type, use_cur_pos_tensor, cur_pos_type, cur_pos_id, is_multi_pos, step_size, sliding_window_size",
    [
        (True, "causal", True, "constant", 0, True, 31, 0),
        (True, "causal", True, "half_seq_len", -1, False, 0, 0),
        (True, "causal", True, "random", None, False, 0, 0),
        (True, "causal", True, "linear", None, False, 0, 0),
        (True, "causal", True, "drop_users", None, False, 0, 0),
        (False, "non_causal", False, "half_seq_len", None, False, 0, 0),
        (True, "sliding_window", False, "constant", 0, True, 31, 64),
        (True, "sliding_window", False, "constant", 0, True, 31, 128),
        (True, "sliding_window", False, "constant", 0, True, 31, 256),
        (True, "sliding_window", False, "constant", 0, True, 31, 512),
        (True, "sliding_window", False, "constant", 0, True, 31, 1024),
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, is_single_iter, sharded_in, sharded_out",
    [
        [32, 32, 8, 8192, 128, (8, 8), True, True, False],
        # [32, 32, 8, 4224, 128, (8, 8), True, True, False ],
        # [32, 8, 1, 32768, 128, (8, 6), True, True, False ],
        # [8, 8, 1, 32768, 128, (8, 6), True, True, False ],
        # [8, 16, 4, 4096, 128, (8, 2), True, True, False ],
        # [4, 32, 8, 8192, 128, (8, 8), True, True, False ],
        # [4, 16, 4, 32768, 128, (8, 8),True, True, False ],
        # [1, 8, 1, 128 * 1024, 128, (8, 4), True, True, False],
        # [1, 32, 8, 128 * 1024, 128, (8, 1), True, True, False],
        # [1, 4, 2, 128 * 1024, 128, (8, 8), True, True, False]
    ],
)
def test_sdpa_decode_core(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    q_dtype,
    q_layout,
    kv_dtype,
    q_chunk_size,
    k_chunk_size,
    grid_size,
    sliding_window_size,
    use_cur_pos_tensor,
    cur_pos_type,
    cur_pos_id,
    cur_pos_is_sharded,
    page_table_is_sharded,
    sharded_in,
    sharded_out,
    start_core,
    sub_core_grids,
    is_causal,
    use_paged_attention,
    is_single_iter,
    is_multi_pos,
    block_size,
    mask_type,
    num_repeat_iters,
    step_size,
):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("For Grouped Query Attention (nkv > 1)we require q_dtype to be bfloat16")
    run_scaled_dot_product_attention_decode(
        device,
        b,
        s,
        nh,
        nkv,
        d,
        q_dtype,
        q_layout,
        kv_dtype,
        q_chunk_size,
        k_chunk_size,
        grid_size,
        sliding_window_size,
        use_cur_pos_tensor,
        cur_pos_type,
        cur_pos_id,
        cur_pos_is_sharded,
        page_table_is_sharded,
        sharded_in,
        sharded_out,
        start_core,
        sub_core_grids,
        is_causal,
        use_paged_attention,
        is_single_iter,
        is_multi_pos,
        block_size,
        mask_type,
        num_repeat_iters,
        step_size,
    )


# @pytest.mark.parametrize(
#     "kv_dtype, q_dtype",
#     [
#         [ttnn.bfloat8_b, ttnn.bfloat16],
#     ],
# )
# @pytest.mark.parametrize("sharded_in, sharded_out", [
#     (True, False),
#     (True, True),
#     (False, False),
# ])
# @pytest.mark.parametrize("cur_pos_is_sharded", [True, False])
# @pytest.mark.parametrize("page_table_is_sharded", [True, False])
# @pytest.mark.parametrize("use_paged_attention", [True])
# @pytest.mark.parametrize("is_single_iter", [True])
# @pytest.mark.parametrize("block_size", [32])
# @pytest.mark.parametrize("mask_type", ["causal"])
# @pytest.mark.parametrize("num_repeat_iters", [1])
# @pytest.mark.parametrize("sliding_window_size", [None])
# @pytest.mark.parametrize("cur_pos_tensor", [True])
# @pytest.mark.parametrize("cur_pos_type", ["half_seq_len"])
# @pytest.mark.parametrize("cur_pos_id", [-1])
# @pytest.mark.parametrize("q_chunk_size", [None])
# @pytest.mark.parametrize("k_chunk_size", [None])
# @pytest.mark.parametrize("grid_size", [(8, 8)])
# @pytest.mark.parametrize(
#     "start_core, sub_core_grids",
#     [
#         (
#             ttnn.CoreCoord(1, 0),
#             ttnn.CoreRangeSet(
#                 [
#                     ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
#                     ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
#                 ]
#             ),
#         ),
#         (
#             ttnn.CoreCoord(0, 0),
#             ttnn.CoreRangeSet(
#                 [
#                     ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7)),
#                 ]
#             ),
#         )
#     ],
#     ids=[
#         "glx_subcoregrid",
#         'n150_subcoregrid',
#     ]
# )
# def test_sdpa_decode_sharded(
#     device,
#     b,
#     nh,
#     nkv,
#     s,
#     d,
#     grid_size,
#     single_iter,
#     q_dtype,
#     q_layout,
#     kv_dtype,
#     q_chunk_size,
#     k_chunk_size,
#     cur_pos_tensor,
#     cur_pos_type,
#     cur_pos_id,
#     cur_pos_is_sharded,
#     page_table_is_sharded,
#     sharded_in,
#     sharded_out,
#     start_core,
#     sub_core_grids,
#     is_causal,
#     use_paged_attention,
#     is_single_iter,
#     block_size,
#     mask_type,
#     num_repeat_iters,
#     sliding_window_size
# ):
#     run_scaled_dot_product_attention_decode(
#         device,
#         b,
#         s,
#         nh,
#         nkv,
#         d,
#         q_dtype,
#         q_layout,
#         kv_dtype,
#         q_chunk_size,
#         k_chunk_size,
#         grid_size,
#         use_cur_pos_tensor,
#         cur_pos_type,
#         cur_pos_id,
#         cur_pos_is_sharded,
#         page_table_is_sharded,
#         sharded_in,
#         sharded_out,
#         start_core,
#         sub_core_grids,
#         is_causal,
#         use_paged_attention,
#         single_iter,
#         block_size,
#         mask_type,
#         num_repeat_iters,
#         sliding_window_size,
#         pos_step,
#     )


# test_sdpa_decode
# - run_test_sdpa_decode_multi_pos
# - run_test_sdpa_decode_single_iter
# test_sdpa_decode_non_causal
# - run_test_sdpa_decode_single_iter
# test_sdpa_decode_ignore_users
# - run_test_sdpa_decode_single_iter
# test_sdpa_decode_paged_attention
# - run_test_sdpa_decode_paged_attention
# test_sdpa_decode_sharded
# - run_test_sdpa_decode_single_iter
# test_sdpa_decode_sharded_on_subcoregrids
# - run_test_sdpa_decode_single_iter
# - run_test_sdpa_decode_paged_attention_single_iter
# test_sdpa_decode_program_cache
# - run_test_sdpa_decode_single_iter
# test_sdpa_decode_ndpcc
# - run_test_sdpa_decode_ndpcc
