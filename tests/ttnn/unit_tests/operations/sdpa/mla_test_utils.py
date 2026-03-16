# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.common.utility_functions import nearest_y

import ttnn
from loguru import logger
import pytest

from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
)


def scaled_dot_product_attention_reference(Q, K, V, start_indices, padded_layer_len, scale, is_causal=True):
    b, nh, _, _ = Q.shape
    _, nkv, _, _ = K.shape

    attn_mask = None
    if is_causal:
        attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
        for i in range(b):
            start_idx = start_indices[i]
            attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min
    else:
        assert False, "Non-causal attention is not supported in this function."

    Q_slice = Q[:, :nh, :, :]
    K_slice = K[:, :nkv, :padded_layer_len, :]
    K_slice = torch.cat([K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_slice = V[:, :, :padded_layer_len, :]
    V_slice = torch.cat([V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    attn_mask_slice = attn_mask[:, :nh, :, :]
    out = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )

    return out


def scaled_dot_product_attention_reference_prefill(Q, K, V, scale, is_causal=True):
    """
    Full-sequence causal SDPA reference.
    Q: (B, nh, S, d_qk), K/V: (B, nkv, S, d)
    """
    _, nh, _, _ = Q.shape
    _, nkv, _, _ = V.shape
    # Expand KV to match Q heads
    head_rep = nh // nkv
    K_exp = K.repeat_interleave(head_rep, dim=1)
    V_exp = V.repeat_interleave(head_rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K_exp, V_exp, attn_mask=None, scale=scale, is_causal=is_causal
    )


def page_table_setup(batch_size: int, config: PagedAttentionConfig) -> torch.Tensor:
    """
    Setup the page-related tensors for the attention cache.
    Args:
        batch_size: The number of batches.
        config: PagedAttentionConfig object containing configuration parameters.
    Returns:
        page_table: The page table tensor.
    """
    block_size, max_num_blocks = config.block_size, config.max_num_blocks
    assert (
        max_num_blocks % batch_size == 0
    ), f"max_num_blocks {max_num_blocks} must be divisible by batch_size {batch_size}."

    page_table = torch.randperm(max_num_blocks, dtype=torch.int32)
    page_table = page_table.reshape(batch_size, max_num_blocks // batch_size)

    return page_table


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

    paged_cache = cache.reshape(batch_size, nh, -1, block_size, dim)
    paged_cache = paged_cache.transpose(1, 2)
    paged_cache = paged_cache.reshape(max_num_blocks, nh, block_size, dim)
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
    max_num_blocks, nh, block_size, dim = paged_cache.shape
    assert (
        block_size == config.block_size
    ), f"block_size {block_size} must match the paged attention config block size {config.block_size}."
    assert (
        max_num_blocks == config.max_num_blocks
    ), f"max_num_blocks {max_num_blocks} must match the paged attention config max_num_blocks {config.max_num_blocks}."

    batch, num_blocks_per_batch = mapping.shape

    # Use the mapping to get the original order, paged_cache + mapping = original cache
    cache = paged_cache[mapping.view(-1)]
    cache = cache.reshape(batch, num_blocks_per_batch, nh, block_size, dim)
    cache = cache.transpose(1, 2)
    cache = cache.reshape(batch, nh, -1, dim)
    return cache


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    power = math.ceil(math.log2(x))
    return 1 << power


def run_flash_mla_decode_impl(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_num_cores,
    q_dtype,
    q_mem_config,
    dtype,
    use_paged_attention=False,
    block_size=ttnn.TILE_SIZE,
    reuse_k=False,
    max_cores_per_head_batch=16,
):
    # Can't run too many iters, or run out of L1
    num_iters = 3

    # Log the test parameters
    logger.debug(f"Running FlashMLA Decode with parameters: ")
    logger.debug(f"Batch: {batch}")
    logger.debug(f"Sequence Length: {seq_len}")
    logger.debug(f"Number of Heads (Q): {nh}")
    logger.debug(f"Number of Heads (KV): {nkv}")
    logger.debug(f"KV LoRA Rank: {kv_lora_rank}")
    logger.debug(f"Dimensionality of RoPE: {d_rope}")
    logger.debug(f"Number of Cores for Q Sharding: {q_num_cores}")
    logger.debug(f"Query Data Type: {q_dtype}")
    logger.debug(f"Key-Value Data Type: {dtype}")

    # Paged attention configuration
    paged_attention_cfg = None
    if use_paged_attention:
        assert seq_len % block_size == 0, f"Sequence length must be a multiple of {block_size=} for paged attention."

        max_num_blocks = seq_len // block_size * batch
        paged_attention_cfg = PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, 1, kv_lora_rank + d_rope).float()
    k = torch.randn(batch, nkv, seq_len, kv_lora_rank + d_rope).float()
    v = k[..., :kv_lora_rank]

    # When Q memory config is provided, it is expected that Q is sharded such that
    # each worker core has its own local Q shard
    if q_mem_config is not None:
        q_heads_parallel_factor = math.ceil(nh / ttnn.TILE_SIZE)
        assert (
            q_heads_parallel_factor > 1
        ), f"Custom Q memory config requires q_heads_parallel_factor > 1, got {q_heads_parallel_factor}."
        heads_per_vbatch = nh // q_heads_parallel_factor
        q_for_tt = (
            q.permute(2, 0, 1, 3)
            .reshape(1, batch, q_heads_parallel_factor, heads_per_vbatch, -1)
            .repeat_interleave(max_cores_per_head_batch, dim=2)
            .reshape(1, 1, -1, q.shape[-1])
        )
    else:
        q_for_tt = q.permute(2, 0, 1, 3)

    ######################
    ### TT Setup
    #######################

    # Page-related setup
    tt_k_torch = k
    tt_v_torch = v
    tt_page_table = None
    if paged_attention_cfg:
        page_table = page_table_setup(batch, paged_attention_cfg)
        tt_k_torch = to_paged_cache(k, page_table, paged_attention_cfg)
        tt_k_torch_og = from_paged_cache(tt_k_torch, page_table, paged_attention_cfg)
        assert torch.all(tt_k_torch_og == k), "Paged cache conversion for K failed."

        tt_v_torch = to_paged_cache(v, page_table, paged_attention_cfg)
        tt_v_torch_og = from_paged_cache(tt_v_torch, page_table, paged_attention_cfg)
        assert torch.all(tt_v_torch_og == v), "Paged cache conversion for V failed."

        tt_page_table = ttnn.from_torch(
            page_table,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    q_chunk_size = 0
    k_chunk_size = 128

    scale = (kv_lora_rank + d_rope) ** -0.5

    max_start_idx = seq_len // 2
    start_indices = np.linspace(0, max_start_idx, batch, dtype=np.int32).tolist() if batch > 1 else [max_start_idx]
    padded_layer_len = nearest_y(max_start_idx + 1, k_chunk_size)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=max_cores_per_head_batch,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Set up input tensors
    if q_num_cores < 1:
        q_mem_config = ttnn.DRAM_MEMORY_CONFIG
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        num_cores_x, num_cores_y = device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y
        if q_num_cores > num_cores_x * num_cores_y:
            pytest.skip(
                f"Skipping test with q_num_cores {q_num_cores} > device compute grid size {num_cores_x * num_cores_y}."
            )

        if nkv == 1:
            q_num_cores = min(batch * nh, q_num_cores)
        else:
            q_num_cores = min(batch, q_num_cores)

        block_height = nearest_y(np.prod(q.shape[:-1]) // q_num_cores, ttnn.TILE_SIZE)

        q_core_grid = ttnn.num_cores_to_corerangeset(
            q_num_cores, device.compute_with_storage_grid_size(), row_wise=True
        )
        if q_mem_config is None:
            q_mem_config = ttnn.create_sharded_memory_config(
                shape=(block_height, q.shape[-1]),
                core_grid=q_core_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            out_mem_config = ttnn.create_sharded_memory_config(
                shape=(block_height, v.shape[-1]),
                core_grid=q_core_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    # GQA only supports DRAM memory config for output
    if nkv > 1:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    tt_q = ttnn.from_torch(
        q_for_tt,
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem_config,
    )

    tt_k = ttnn.from_torch(
        tt_k_torch,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_v = (
        ttnn.from_torch(
            tt_v_torch,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if not reuse_k
        else None
    )

    tt_start_indices = ttnn.from_torch(
        torch.tensor(start_indices),
        device=device,
        dtype=ttnn.int32,
    )

    ##########################
    ### FlashMLA Decode
    ##########################
    logger.info(
        f"Running FlashMLA Decode with TT Q shape: {tt_q.shape}, TT K shape: {tt_k.shape}, head_dim_v: {kv_lora_rank}"
    )

    def run_op():
        if tt_page_table:
            tt_out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                tt_q,
                tt_k,
                tt_v,
                head_dim_v=kv_lora_rank,
                page_table_tensor=tt_page_table,
                cur_pos_tensor=tt_start_indices,
                scale=scale,
                program_config=sdpa_program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=out_mem_config,
            )
        else:
            tt_out = ttnn.transformer.flash_multi_latent_attention_decode(
                tt_q,
                tt_k,
                tt_v,
                head_dim_v=kv_lora_rank,
                cur_pos_tensor=tt_start_indices,
                scale=scale,
                program_config=sdpa_program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=out_mem_config,
            )

        return tt_out

    with device.cache_entries_counter.measure():
        tt_outs = []
        for i in range(num_iters):  # Check for program cache
            logger.debug(f"Running FlashMLA Decode operation iteration {i + 1}/{num_iters}")
            tt_out = run_op()
            tt_outs.append(tt_out)

            # Increment start indices for the next iteration
            ttnn.plus_one(tt_start_indices)

        ########################
        ### Validation
        ########################
        outs = []
        for _ in range(num_iters):
            out_t = scaled_dot_product_attention_reference(
                q,
                k,
                v,
                start_indices,
                padded_layer_len,
                scale,
            )
            outs.append(out_t)

            start_indices = [x + 1 for x in start_indices]

        pcc_threshold = 0.999
        if dtype == ttnn.bfloat4_b:
            pcc_threshold = 0.91
        if dtype == ttnn.bfloat8_b:
            pcc_threshold = 0.98

    for i, (tt_out, out_t) in enumerate(zip(tt_outs, outs)):
        tt_out_torch = ttnn.to_torch(tt_out)[..., :nh, :].permute(1, 2, 0, 3)
        out_pass, out_pcc = comp_pcc(tt_out_torch, out_t, pcc_threshold)
        logger.debug(f"Output PCC: {out_pcc}")

        assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"

    # FlashMLA + PlusOne
    assert (
        device.cache_entries_counter.total == 2
    ), f"Expected 2 program cache entries, got {device.cache_entries_counter.total}."


def run_flash_mla_prefill_impl(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    use_paged_attention=False,
    block_size=ttnn.TILE_SIZE,
):
    # Log the test parameters
    logger.debug(f"Running FlashMLA Prefill with parameters: ")
    logger.debug(f"Batch: {batch}")
    logger.debug(f"Sequence Length: {seq_len}")
    logger.debug(f"Number of Heads (Q): {nh}")
    logger.debug(f"Number of Heads (KV): {nkv}")
    logger.debug(f"KV LoRA Rank: {kv_lora_rank}")
    logger.debug(f"Dimensionality of RoPE: {d_rope}")
    logger.debug(f"Query Data Type: {q_dtype}")
    logger.debug(f"Key-Value Data Type: {dtype}")

    # Paged attention configuration
    paged_attention_cfg = None
    if use_paged_attention:
        assert seq_len % block_size == 0, f"Sequence length must be a multiple of {block_size=} for paged attention."

        max_num_blocks = seq_len // block_size * batch
        paged_attention_cfg = PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, seq_len, kv_lora_rank + d_rope).float()
    k = torch.randn(batch, nkv, seq_len, kv_lora_rank + d_rope).float()
    v = k[..., :kv_lora_rank]

    ######################
    ### TT Setup
    #######################

    # Page-related setup
    tt_k_torch = k
    tt_page_table = None
    if paged_attention_cfg:
        page_table = page_table_setup(batch, paged_attention_cfg)
        tt_k_torch = to_paged_cache(k, page_table, paged_attention_cfg)
        tt_k_torch_og = from_paged_cache(tt_k_torch, page_table, paged_attention_cfg)
        assert torch.all(tt_k_torch_og == k), "Paged cache conversion for K failed."

        tt_page_table = ttnn.from_torch(
            page_table,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    q_chunk_size = padded_num_heads
    k_chunk_size = 128

    scale = (kv_lora_rank + d_rope) ** -0.5

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_q = ttnn.from_torch(
        q,
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_k = ttnn.from_torch(
        tt_k_torch,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ##########################
    ### FlashMLA Prefill
    ##########################
    if tt_page_table:
        tt_out = ttnn.transformer.chunked_flash_mla_prefill(
            tt_q,
            tt_k,
            kv_lora_rank,
            tt_page_table,
            chunk_start_idx=0,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        tt_out = ttnn.transformer.flash_mla_prefill(
            tt_q,
            tt_k,
            head_dim_v=kv_lora_rank,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            attn_mask=None,
            is_causal=True,
        )
    tt_back = ttnn.to_torch(tt_out)
    tt_out_torch = tt_back[:, :nh, :seq_len, :]

    ########################
    ### Validation
    ########################
    out_t = scaled_dot_product_attention_reference_prefill(
        q,
        k,
        v,
        scale,
        is_causal=True,
    )

    pcc_threshold = 0.99
    if dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.98

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_t, pcc_threshold)
    logger.debug(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"
