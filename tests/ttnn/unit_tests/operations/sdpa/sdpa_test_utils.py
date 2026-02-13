# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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
import math
import numpy as np


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power
    # if (2**math.log2(x) == x):
    #     return x
    # return 2**(int(x).bit_length())


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


def create_sliding_window_mask(b, nh, seq_len, cur_pos_list, sliding_window_size):
    """
    Create attention mask for sliding window attention.

    Args:
        b: batch size
        nh: number of heads
        seq_len: sequence length
        cur_pos_list: list of current positions for each batch
        sliding_window_size: sliding window size

    Returns:
        attn_mask: [b, nh, 1, seq_len] mask with -inf for positions outside window
    """
    attn_mask = torch.zeros((b, nh, 1, seq_len))

    for i in range(b):
        cur_pos = cur_pos_list[i]

        # Calculate sliding window bounds
        window_end = cur_pos + 1  # exclusive
        window_start = max(0, window_end - sliding_window_size)

        # Mask positions before sliding window start
        if window_start > 0:
            attn_mask[i, :, :, :window_start] = torch.finfo(torch.float32).min

        # Mask positions after current position (causal)
        if cur_pos + 1 < seq_len:
            attn_mask[i, :, :, cur_pos + 1 :] = torch.finfo(torch.float32).min

    return attn_mask


def flash_attention_loop(q, K, V, mask, scale, k_chunk_size):
    seqlen = K.shape[-2]
    padded_num_heads = q.shape[-2]
    Tc = seqlen // k_chunk_size
    O = torch.zeros_like(q)
    l = torch.zeros([1, 1, padded_num_heads, 1])
    m = torch.ones([1, 1, padded_num_heads, 1]) * torch.finfo(torch.float32).min
    for t in range(Tc):
        K_chunk = K[:, :, t * k_chunk_size : (t + 1) * k_chunk_size, :]
        V_chunk = V[:, :, t * k_chunk_size : (t + 1) * k_chunk_size, :]
        mask_chunk = mask[:, :, :, t * k_chunk_size : (t + 1) * k_chunk_size]

        attn = torch.matmul(q, K_chunk.transpose(-2, -1)) * scale + mask_chunk
        m_old = m
        m = torch.max(m_old, torch.max(attn, dim=-1, keepdim=True)[0])
        P = torch.exp(attn - m)
        l = torch.exp(m_old - m) * l + torch.sum(P, dim=-1, keepdim=True)
        O = torch.matmul(P, V_chunk) + torch.matmul(torch.eye(padded_num_heads) * torch.exp(m_old - m), O)
    return O, m, l


def scaled_dot_product_attention_simulated(
    tt_Q,
    tt_K,
    tt_V,
    tt_attn_mask,
    is_causal,
    scale,
    program_config,
    valid_seq_len,
    compute_kernel_config,
    output_mem_config,
):
    # inputs
    tt_Q = ttnn.to_torch(tt_Q).to(torch.float32)
    tt_K = ttnn.to_torch(tt_K).to(torch.float32)
    tt_V = ttnn.to_torch(tt_V).to(torch.float32)
    tt_attn_mask = ttnn.to_torch(tt_attn_mask).to(torch.float32)

    # shapes
    k_chunk_size = program_config.k_chunk_size
    batch = tt_Q.shape[-3]
    head_dim = tt_Q.shape[-1]
    padded_num_heads = tt_Q.shape[-2]
    seqlen = tt_K.shape[-2]
    core_grid = program_config.compute_with_storage_grid_size
    num_cores = core_grid.x * core_grid.y

    # split to cores
    num_cores_per_batch = num_cores // batch
    num_active_cores = num_cores_per_batch * batch
    active_cores = [[i + k * num_cores_per_batch for i in range(num_cores_per_batch)] for k in range(batch)]

    # sequence length assignment
    assert valid_seq_len % k_chunk_size == 0
    num_chunks = valid_seq_len // k_chunk_size
    chunks_per_core = math.ceil(num_chunks // num_cores_per_batch)
    chunk_assignment = [[i * chunks_per_core, (i + 1) * chunks_per_core] for i in range(num_cores_per_batch)]
    chunk_assignment[-1][-1] += num_chunks % num_cores_per_batch

    # loop over batches
    output_tensor = torch.zeros_like(tt_Q)
    for b, batch_cores in enumerate(active_cores):
        O_intermed = []
        m_intermed = []
        l_intermed = []
        for i, core in enumerate(batch_cores):
            chunk_start, chunk_end = chunk_assignment[i]
            if chunk_start == chunk_end:
                continue
            O, m, l = flash_attention_loop(
                tt_Q[:, [b]],
                tt_K[:, [b], chunk_start * k_chunk_size : chunk_end * k_chunk_size, :],
                tt_V[:, [b], chunk_start * k_chunk_size : chunk_end * k_chunk_size, :],
                tt_attn_mask[:, [b], :, chunk_start * k_chunk_size : chunk_end * k_chunk_size],
                scale,
                k_chunk_size,
            )
            O_intermed.append(O)
            m_intermed.append(m)
            l_intermed.append(l)
        O, m, l = O_intermed[0], m_intermed[0], l_intermed[0]
        for O_2, m_2, l_2 in zip(O_intermed[1:], m_intermed[1:], l_intermed[1:]):
            O_1, m_1, l_1 = O, m, l
            m = torch.max(m_1, m_2)
            l = torch.exp(m_2 - m) * l_2 + torch.exp(m_1 - m) * l_1
            O = torch.matmul(torch.eye(padded_num_heads) * torch.exp(m_2 - m), O_2) + torch.matmul(
                torch.eye(padded_num_heads) * torch.exp(m_1 - m), O_1
            )
        output_tensor[:, b] = torch.matmul(torch.eye(padded_num_heads) * 1 / l, O)
    return output_tensor


def run_test_sdpa_decode_multi_pos(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    dtype,
    grid_size,
    q_dtype=ttnn.bfloat16,
    cur_pos_tensor=False,
    sharded_in=False,
    sharded_out=False,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    num_parallel_cores = grid_size[0] * grid_size[1] // b
    if num_parallel_cores == 1:
        min_pcc = 0.90
    else:
        min_pcc = 0.99
        if q_dtype == ttnn.bfloat8_b:
            min_pcc = 0.98
        min_pcc = 0.91 if dtype == ttnn.bfloat4_b else min_pcc

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)

    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    max_start_idx = 0

    while max_start_idx < s:
        scale = d**-0.5
        start_indices = np.linspace(0, max_start_idx, b, dtype=np.int32).tolist() if b > 1 else [max_start_idx]

        k_chunk_size = get_chunk_size(max_start_idx + 1, s)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,  # device.compute_with_storage_grid_size(),
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size)

        # Test various sequence lengths
        logger.info(f"Testing with sequence length: {max_start_idx}")
        logger.info(f"Using chunk size: {k_chunk_size}")
        logger.info(f"Using padded layer length: {padded_layer_len}")
        logger.info(f"Using padded num heads: {padded_num_heads}")

        attn_mask = torch.zeros((b, padded_num_heads, 1, padded_layer_len))
        for i in range(b):
            start_idx = start_indices[i]
            attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min

        Q = fa_rand(1, b, padded_num_heads, d)

        tt_Q = ttnn.as_tensor(
            Q[:, :, :nh],
            device=device,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=height_sharded_memcfg if sharded_in else dram_memcfg,
        )
        if cur_pos_tensor:
            start_indices_tt = ttnn.Tensor(torch.tensor(start_indices), ttnn.int32).to(device)

            tt_back = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                cur_pos_tensor=start_indices_tt,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )
        else:
            tt_back = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                cur_pos=start_indices,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )

        tt_back = ttnn.to_torch(tt_back)

        tt_back = tt_back[:, :, :nh, :]

        Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
        K_slice = K[:, :, :padded_layer_len, :]  # b, nkv, S, d
        K_slice = torch.cat(
            [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
        )  # b, nh, S, d
        V_slice = V[:, :, :padded_layer_len, :]  # b, nkv, S, d
        V_slice = torch.cat(
            [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
        )  # b, nh, S, d
        attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S

        expect = torch.nn.functional.scaled_dot_product_attention(
            Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
        )  # b, nh, 1, d
        expect = expect.squeeze(2).unsqueeze(0)

        out_pass, out_pcc = comp_pcc(expect, tt_back, min_pcc)

        logger.debug(f"python vs pytorch: {out_pcc}")

        assert out_pass

        max_start_idx += 71 if max_start_idx < 4096 else 3001


def run_test_sdpa_decode_single_iter(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    dtype,
    grid_size,
    q_dtype=ttnn.bfloat16,
    cur_pos_tensor=False,
    sharded_in=False,
    sharded_out=False,
    start_indices=None,
    causal=True,
    start_core=ttnn.CoreCoord(0, 0),
    sub_core_grids=None,
    override_q_chunk_size=None,
    override_k_chunk_size=None,
    sliding_window_size=None,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if sub_core_grids is None:
        if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
            pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    else:
        unharvested_grid_size = (7, 10)
        if unharvested_grid_size[0] > compute_grid_size.x or unharvested_grid_size[1] > compute_grid_size.y:
            pytest.skip(f"Need {unharvested_grid_size} grid size to run this test but core grid is {compute_grid_size}")
        if grid_size[0] * grid_size[1] > sub_core_grids.num_cores():
            pytest.skip(
                f"Need {grid_size[0]*grid_size[1]} grid size to run this test but core grid is {sub_core_grids.num_cores()}"
            )
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    num_parallel_cores = grid_size[0] * grid_size[1] // b
    if num_parallel_cores == 1:
        min_pcc = 0.90
    else:
        min_pcc = 0.99
        if q_dtype == ttnn.bfloat8_b:
            min_pcc = 0.98
        min_pcc = 0.91 if dtype == ttnn.bfloat4_b else min_pcc

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    if sub_core_grids is None:
        shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
        compute_sub_core_grids = None
    else:
        shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(start_core, b, sub_core_grids, row_wise=True)
        compute_sub_core_grids = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            start_core, grid_size[0] * grid_size[1], sub_core_grids, row_wise=True
        )
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)

    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_indices = [s // 2 for _ in range(b)] if start_indices is None else start_indices
    max_start_idx = max(start_indices)
    scale = d**-0.5

    q_chunk_size = padded_num_heads if override_q_chunk_size is None else override_q_chunk_size
    k_chunk_size = get_chunk_size(max_start_idx + 1, s) if override_k_chunk_size is None else override_k_chunk_size

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        sub_core_grids=compute_sub_core_grids,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size) if causal else s

    # Test various sequence lengths
    logger.debug(f"Testing with sequence length: {max_start_idx if causal else s}")
    logger.debug(f"Using chunk size: {k_chunk_size}")
    logger.debug(f"Using padded layer length: {padded_layer_len}")
    logger.debug(f"Using padded num heads: {padded_num_heads}")

    if causal:
        if sliding_window_size is not None:
            # Use sliding window mask
            attn_mask = create_sliding_window_mask(b, nh, padded_layer_len, start_indices, sliding_window_size)
        else:
            # Use regular causal mask
            attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
            for i in range(b):
                start_idx = start_indices[i]
                attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min
    else:
        attn_mask = torch.bernoulli(
            torch.full(
                (b, nh, 1, padded_layer_len),
                0.25,
            )
        )
        attn_mask = attn_mask * torch.finfo(torch.float32).min

    Q = fa_rand(1, b, nh, d)

    tt_Q = ttnn.as_tensor(
        Q[:, :, :nh],
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=height_sharded_memcfg if sharded_in else dram_memcfg,
    )
    if causal:
        if cur_pos_tensor:
            start_indices_tt = ttnn.Tensor(torch.tensor(start_indices), ttnn.int32).to(device)
            tt_back = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                cur_pos_tensor=start_indices_tt,
                scale=scale,
                sliding_window_size=sliding_window_size,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )
        else:
            tt_back = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                cur_pos=start_indices,
                scale=scale,
                sliding_window_size=sliding_window_size,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )
    else:
        tt_mask = ttnn.as_tensor(
            attn_mask.transpose(1, 2).contiguous(),
            device=device,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dram_memcfg,
        )
        tt_back = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_Q,
            tt_K,
            tt_V,
            is_causal=False,
            attn_mask=tt_mask,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
        )

    tt_back = ttnn.to_torch(tt_back)
    tt_back = tt_back[:, :, :nh, :]

    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
    K_slice = K[:, :, :padded_layer_len, :]  # b, nkv, S, d
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, S, d
    V_slice = V[:, :, :padded_layer_len, :]  # b, nkv, S, d
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, S, d
    attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S
    expect = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    expect = expect.squeeze(2).unsqueeze(0)

    non_skip_indices = torch.tensor(start_indices) != -1
    out_pass, out_pcc = comp_pcc(expect[:, non_skip_indices], tt_back[:, non_skip_indices], min_pcc)

    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


def run_test_sdpa_decode_paged_attention(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    kv_dtype,
    grid_size,
    q_dtype,
    cur_pos_tensor,
    block_size,
    sharded_in=True,
    sharded_out=True,
    sliding_window_size=None,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    # Paged cache attributes
    max_num_blocks_per_seq = s // block_size
    assert max_num_blocks_per_seq * block_size == s
    max_num_blocks = b * s // block_size
    assert max_num_blocks * block_size == b * s

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    def to_paged_cache(cache, batch, num_kv, max_num_blocks_per_seq, block_size, head_dim, max_seq_len):
        return (
            cache.reshape(batch, num_kv, max_num_blocks_per_seq, block_size, head_dim)
            .transpose(1, 2)
            .reshape(batch * max_num_blocks_per_seq, num_kv, block_size, head_dim)
        )

    def to_contiguous_cache(paged_cache, batch, num_kv, max_num_blocks_per_seq, block_size, head_dim, max_seq_len):
        return (
            paged_cache.reshape(batch, max_num_blocks_per_seq, num_kv, block_size, head_dim)
            .transpose(1, 2)
            .reshape(batch, num_kv, max_seq_len, head_dim)
        )

    # Create paged K and V based on block size\
    paged_k = to_paged_cache(K, b, nkv, max_num_blocks_per_seq, block_size, d, s)
    paged_v = to_paged_cache(V, b, nkv, max_num_blocks_per_seq, block_size, d, s)

    # We need a random permutation to shuffle pages in the cache
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(b, max_num_blocks_per_seq)

    paged_k_shuffled = paged_k[permutation]
    paged_v_shuffled = paged_v[permutation]

    paged_k_unshuffled = paged_k_shuffled[reverse_permutation]
    paged_v_unshuffled = paged_v_shuffled[reverse_permutation]

    # Check that page/shuffle/unshuffle/unpage logic is correct
    K_back = to_contiguous_cache(paged_k_unshuffled, b, nkv, max_num_blocks_per_seq, block_size, d, s)
    V_back = to_contiguous_cache(paged_v_unshuffled, b, nkv, max_num_blocks_per_seq, block_size, d, s)

    assert torch.allclose(K, K_back)
    assert torch.allclose(V, V_back)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))

    num_parallel_cores = grid_size[0] * grid_size[1] // b
    if num_parallel_cores == 1:
        min_pcc = 0.90
    else:
        min_pcc = 0.98  # TODO: Investigate why PCC drops below 0.99 for certain decode positions
        if q_dtype == ttnn.bfloat8_b:
            min_pcc = 0.98
        min_pcc = 0.91 if kv_dtype == ttnn.bfloat4_b else min_pcc

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)

    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    tt_K = ttnn.as_tensor(
        paged_k_shuffled, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
    )
    tt_V = ttnn.as_tensor(
        paged_v_shuffled, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
    )
    tt_page_table = ttnn.Tensor(page_table, ttnn.int32).to(device)

    max_start_idx = 0
    causal = True

    while max_start_idx < s or not causal:
        scale = d**-0.5
        start_indices = np.linspace(max(max_start_idx - b, 0), max_start_idx, b, dtype=np.int32).tolist()

        # Test when page_table does not contain blocks for full sequence length
        k_chunk_size = get_chunk_size(max_start_idx + 1, s)
        padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size) if causal else s

        tt_page_table = ttnn.Tensor(page_table, ttnn.int32).to(device)

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,  # device.compute_with_storage_grid_size(),
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        # Test various sequence lengths
        logger.debug(
            f"Testing {'causal' if causal else 'non-causal'} with sequence length: {max_start_idx if causal else s}"
        )
        logger.info(f"Using chunk size: {k_chunk_size}")
        logger.info(f"Using padded layer length: {padded_layer_len}")
        logger.info(f"Using padded num heads: {padded_num_heads}")

        if causal:
            if sliding_window_size is not None:
                # Use sliding window mask
                attn_mask = create_sliding_window_mask(b, nh, padded_layer_len, start_indices, sliding_window_size)
            else:
                # Use regular causal mask
                attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
                for i in range(b):
                    start_idx = start_indices[i]
                    attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min
        else:
            attn_mask = torch.bernoulli(
                torch.full(
                    (b, nh, 1, padded_layer_len),
                    0.25,
                )
            )
            attn_mask = attn_mask * torch.finfo(torch.float32).min

        Q = fa_rand(1, b, nh, d)

        tt_Q = ttnn.as_tensor(
            Q[:, :, :nh],
            device=device,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=height_sharded_memcfg if sharded_in else dram_memcfg,
        )

        start_indices_tt = ttnn.Tensor(torch.tensor(start_indices), ttnn.int32).to(device)

        if causal:
            tt_back = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                tt_page_table,
                cur_pos_tensor=start_indices_tt,
                scale=scale,
                sliding_window_size=sliding_window_size,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )
        else:
            tt_mask = ttnn.as_tensor(
                attn_mask.transpose(1, 2).contiguous(),
                device=device,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=dram_memcfg,
            )
            tt_back = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                tt_page_table,
                is_causal=False,
                attn_mask=tt_mask,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )

        tt_back = ttnn.to_torch(tt_back)

        tt_back = tt_back[:, :, :nh, :]

        Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
        K_slice = K[:, :, :padded_layer_len, :]  # b, nkv, S, d
        K_slice = torch.cat(
            [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
        )  # b, nh, S, d
        V_slice = V[:, :, :padded_layer_len, :]  # b, nkv, S, d
        V_slice = torch.cat(
            [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
        )  # b, nh, S, d
        attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S

        expect = torch.nn.functional.scaled_dot_product_attention(
            Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
        )  # b, nh, 1, d
        expect = expect.squeeze(2).unsqueeze(0)

        out_pass, out_pcc = comp_pcc(expect, tt_back, min_pcc)

        logger.debug(f"python vs pytorch: {out_pcc}")

        assert out_pass

        max_start_idx += 31 if max_start_idx < 4096 else 3001

        if not causal:
            # only run one iteration for non-causal
            break
        if max_start_idx >= s:
            # run last iteration to test non-causal
            causal = False


def run_test_sdpa_decode_paged_attention_single_iter(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    kv_dtype,
    grid_size,
    q_dtype,
    cur_pos,
    block_size,
    q_chunk_size,
    k_chunk_size,
    sharded_in=True,
    sharded_out=True,
    start_core=ttnn.CoreCoord(0, 0),
    sub_core_grids=None,
    q_layout=ttnn.TILE_LAYOUT,
    is_cur_pos_sharded=False,
    is_page_table_sharded=False,
):
    torch.manual_seed(1234)
    compute_grid_size = device.compute_with_storage_grid_size()
    if sub_core_grids is None:
        if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
            pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    else:
        if grid_size[0] * grid_size[1] > sub_core_grids.num_cores():
            pytest.skip(
                f"Need {grid_size[0]*grid_size[1]} grid size to run this test but core grid is {sub_core_grids.num_cores()}"
            )

    # Paged cache attributes
    max_num_blocks_per_seq = s // block_size
    assert max_num_blocks_per_seq * block_size == s
    max_num_blocks = b * s // block_size
    assert max_num_blocks * block_size == b * s

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    def to_paged_cache(cache, batch, num_kv, max_num_blocks_per_seq, block_size, head_dim, max_seq_len):
        return (
            cache.reshape(batch, num_kv, max_num_blocks_per_seq, block_size, head_dim)
            .transpose(1, 2)
            .reshape(batch * max_num_blocks_per_seq, num_kv, block_size, head_dim)
        )

    def to_contiguous_cache(paged_cache, batch, num_kv, max_num_blocks_per_seq, block_size, head_dim, max_seq_len):
        return (
            paged_cache.reshape(batch, max_num_blocks_per_seq, num_kv, block_size, head_dim)
            .transpose(1, 2)
            .reshape(batch, num_kv, max_seq_len, head_dim)
        )

    # Create paged K and V based on block size\
    paged_k = to_paged_cache(K, b, nkv, max_num_blocks_per_seq, block_size, d, s)
    paged_v = to_paged_cache(V, b, nkv, max_num_blocks_per_seq, block_size, d, s)

    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(b, max_num_blocks_per_seq)

    paged_k_shuffled = paged_k[permutation]
    paged_v_shuffled = paged_v[permutation]

    paged_k_unshuffled = paged_k_shuffled[reverse_permutation]
    paged_v_unshuffled = paged_v_shuffled[reverse_permutation]

    # Check that page/shuffle/unshuffle/unpage logic is correct
    K_back = to_contiguous_cache(paged_k_unshuffled, b, nkv, max_num_blocks_per_seq, block_size, d, s)
    V_back = to_contiguous_cache(paged_v_unshuffled, b, nkv, max_num_blocks_per_seq, block_size, d, s)

    assert torch.allclose(K, K_back)
    assert torch.allclose(V, V_back)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))

    min_pcc = 0.99

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    if sub_core_grids is None:
        shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
        compute_sub_core_grids = None
    else:
        shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(start_core, b, sub_core_grids, row_wise=True)
        compute_sub_core_grids = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            start_core, grid_size[0] * grid_size[1], sub_core_grids, row_wise=True
        )

    shard_spec = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)
    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    if q_layout == ttnn.ROW_MAJOR_LAYOUT:
        shard_spec_rm = ttnn.ShardSpec(shard_grid, (nh, d), ttnn.ShardOrientation.ROW_MAJOR)
        Q_height_sharded_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec_rm
        )
    else:
        Q_height_sharded_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
        )

    tt_K = ttnn.as_tensor(
        paged_k_shuffled, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
    )
    tt_V = ttnn.as_tensor(
        paged_v_shuffled, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
    )

    # We need a random permutation to shuffle pages in the cache
    if is_page_table_sharded:
        page_table = page_table.repeat(compute_sub_core_grids.num_cores(), 1)
        page_table_shard_spec = ttnn.ShardSpec(
            compute_sub_core_grids, (b, max_num_blocks_per_seq), ttnn.ShardOrientation.ROW_MAJOR
        )
        page_table_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, page_table_shard_spec
        )
        tt_page_table = ttnn.as_tensor(
            page_table,
            device=device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=page_table_memory_config,
        )
    else:
        tt_page_table = ttnn.Tensor(page_table, ttnn.int32).to(device)

    scale = d**-0.5
    start_indices = [cur_pos] * b

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        sub_core_grids=compute_sub_core_grids,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    Q = fa_rand(1, b, nh, d)

    tt_Q = ttnn.as_tensor(
        Q[:, :, :nh],
        device=device,
        dtype=q_dtype,
        layout=q_layout,
        memory_config=Q_height_sharded_memcfg if sharded_in else dram_memcfg,
    )

    if is_cur_pos_sharded:
        cur_pos_core_grids = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 0)),
            ]
        )
        cur_pos_shard_spec = ttnn.ShardSpec(cur_pos_core_grids, (1, b), ttnn.ShardOrientation.ROW_MAJOR)
        cur_pos_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cur_pos_shard_spec
        )
        start_indices_pt = torch.tensor([start_indices] * cur_pos_core_grids.num_cores())
        start_indices_tt = ttnn.Tensor(start_indices_pt, ttnn.int32).to(device, mem_config=cur_pos_memory_config)
    else:
        start_indices_pt = torch.tensor(start_indices)
        start_indices_tt = ttnn.Tensor(start_indices_pt, ttnn.int32).to(device)

    tt_back = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        tt_page_table,
        cur_pos_tensor=start_indices_tt,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
    )

    tt_back = ttnn.to_torch(tt_back)
    tt_back = tt_back[:, :, :nh, :]

    # PyTorch reference
    # Test when page_table does not contain blocks for full sequence length
    padded_layer_len = cur_pos
    if k_chunk_size > 0:
        padded_layer_len = nearest_n(cur_pos + 1, k_chunk_size)

    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
    K_slice = K[:, :, :padded_layer_len, :]  # b, nkv, S, d
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, S, d
    V_slice = V[:, :, :padded_layer_len, :]  # b, nkv, S, d
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, S, d

    attn_mask = torch.zeros((b, padded_num_heads, 1, padded_layer_len))
    for i in range(b):
        start_idx = start_indices[i]
        attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min

    attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S

    expect = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    expect = expect.squeeze(2).unsqueeze(0)

    out_pass, out_pcc = comp_pcc(expect, tt_back, min_pcc)

    logger.debug(f"python vs pytorch: {out_pcc}")

    assert out_pass


def run_test_sdpa_decode_ndpcc(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype=ttnn.bfloat16):
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_idx = 0

    failed_start_pos = []

    while start_idx < 32000:
        Q = fa_rand(1, b, padded_num_heads, d)
        prev_pcc = None

        scale = d**-0.5

        k_chunk_size = get_chunk_size(start_idx + 1, s)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,  # device.compute_with_storage_grid_size(),
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        padded_layer_len = nearest_n(start_idx + 1, n=k_chunk_size)

        # Test various sequence lengths
        logger.info(f"Testing with sequence length: {start_idx}")
        logger.info(f"Using chunk size: {k_chunk_size}")
        logger.info(f"Using padded layer length: {padded_layer_len}")
        logger.info(f"Using padded num heads: {padded_num_heads}")

        attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
        # Assume all users are at same position
        attn_mask[:, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min

        Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
        K_slice = K[:, :, :padded_layer_len, :]
        V_slice = V[:, :, :padded_layer_len, :]
        attn_mask_slice = attn_mask[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S

        expect = torch.nn.functional.scaled_dot_product_attention(
            Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
        )  # b, nh, 1, d
        expect = expect.squeeze(2).unsqueeze(0)

        all_out_pass = True

        for i in range(500):
            tt_Q = ttnn.as_tensor(
                Q[:, :, :nh],
                device=device,
                dtype=q_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=dram_memcfg,  # height_sharded_memcfg
            )

            tt_back = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                cur_pos=[start_idx for _ in range(b)],
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=dram_memcfg,  # height_sharded_memcfg,
            )

            tt_back = ttnn.to_torch(tt_back)

            tt_back = tt_back[:, :, :nh, :]

            out_pass, out_pcc, pcc_val = comp_and_get_pcc(expect, tt_back, 0.98)

            logger.debug(f"python vs pytorch: {out_pcc}")

            if prev_pcc is not None:
                assert out_pcc == prev_pcc, f"Iteration {i}: pcc changed from {prev_pcc} to {out_pcc}"

            if not out_pass:
                all_out_pass = False
                logger.debug(f"PCC Failed at iteration {i}")

            prev_pcc = out_pcc

        if not all_out_pass:
            failed_start_pos.append(start_idx)

        start_idx += 200  # if start_idx < 4096 else 3001

    logger.info(f"PCC failed Start Pos: {failed_start_pos}")


###############################################################################
# Functions extracted from test_sdpa_decode_sink.py
###############################################################################


def flash_decode_sdpa(Q, K_cache, V_cache, sink, sm_scale, sliding_window=0, block_size=128):
    """
    Flash Decode implementation for autoregressive generation.

    Args:
        Q: Query tensor [batch, 1, n_heads, q_mult, d_head] - only the new token
        K_cache: Key cache [batch, seq_len, n_heads, d_head] - all previous tokens
        sink: Attention sink tensor [batch, 1, n_heads, 1, 1] - used for scaling
        V_cache: Value cache [batch, seq_len, n_heads, d_head] - all previous tokens
        sm_scale: Scaling factor for attention scores
        sliding_window: Sliding window size (0 = no window)
        block_size: Block size for tiled computation

    Returns:
        attn: Attention output [batch, 1, n_heads * q_mult, d_head]
    """
    batch_size, q_len, n_heads, q_mult, d_head = Q.shape
    _, seq_len, _, _ = K_cache.shape

    assert q_len == 1, "Flash decode expects single query token"
    assert K_cache.shape == (batch_size, seq_len, n_heads, d_head)
    assert V_cache.shape == (batch_size, seq_len, n_heads, d_head)

    # Expand K and V cache to match Q's q_mult dimension
    K_cache = K_cache[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)  # [batch, seq_len, n_heads, q_mult, d_head]
    V_cache = V_cache[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)  # [batch, seq_len, n_heads, q_mult, d_head]

    # Initialize output and normalization tensors
    O = torch.zeros_like(Q).squeeze(1)  # [batch, n_heads, q_mult, d_head]
    l = torch.zeros(batch_size, n_heads, q_mult, device=Q.device, dtype=Q.dtype)  # row sums
    m = torch.full((batch_size, n_heads, q_mult), -float("inf"), device=Q.device, dtype=Q.dtype)  # row maxes

    Q = Q.squeeze(1)  # [batch, n_heads, q_mult, d_head]

    # Apply sliding window mask bounds
    start_idx = 0
    if sliding_window > 0:
        start_idx = max(0, seq_len - sliding_window)

    # Process in blocks for memory efficiency
    for block_start in range(start_idx, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)

        # Extract block
        K_block = K_cache[:, block_start:block_end, :, :, :]  # [batch, block_len, n_heads, q_mult, d_head]
        V_block = V_cache[:, block_start:block_end, :, :, :]  # [batch, block_len, n_heads, q_mult, d_head]

        # Compute attention scores for this block
        # Q: [batch, n_heads, q_mult, d_head]
        # K_block: [batch, block_len, n_heads, q_mult, d_head]
        S = torch.einsum("bhmd,bkhmd->bhmk", Q, K_block) * sm_scale  # [batch, n_heads, q_mult, block_len]

        # Apply causal mask - since we're at position seq_len, all previous tokens are visible
        # No additional masking needed for causality in decode phase

        # Online softmax update
        m_new = torch.maximum(m[:, :, :, None], S.max(dim=-1, keepdim=True)[0])  # [batch, n_heads, q_mult, 1]

        # Compute exponentials with numerically stable softmax
        alpha = torch.exp(m[:, :, :, None] - m_new)  # [batch, n_heads, q_mult, 1]
        exp_S = torch.exp(S - m_new)  # [batch, n_heads, q_mult, block_len]

        # Update row sum
        l_new = alpha.squeeze(-1) * l + exp_S.sum(dim=-1)  # [batch, n_heads, q_mult]

        # Update output
        O = alpha.squeeze(-1)[:, :, :, None] * O + torch.einsum("bhmk,bkhmd->bhmd", exp_S, V_block)

        # Update running statistics
        l = l_new
        m = m_new.squeeze(-1)

    if sink is not None:
        ##### Handle Attention Sink #####
        sink = sink.reshape(batch_size, n_heads, q_mult, 1)  # [batch, n_heads, q_mult, 1]

        # max(m, sink)
        m_new = torch.maximum(m[:, :, :, None], sink)  # [batch, n_heads, q_mult, 1]

        # exp(m - m_new)
        alpha = torch.exp(m[:, :, :, None] - m_new)  # [batch, n_heads, q_mult, 1]

        # sub_exp -> exp(sink - m_new)
        exp_sink = torch.exp(sink - m_new)  # [batch, n_heads, q_mult, 1]

        # l_new -> l * alpha + sub_exp, ie, re-scale l + exp_sink
        l_new = alpha.squeeze(-1) * l + exp_sink.sum(dim=-1)  # [batch, n_heads, q_mult]

        # O -> O * alpha
        O = alpha.squeeze(-1)[:, :, :, None] * O

        l = l_new
        ##### Done Handling Attention Sink #####

    # Final normalization
    O = O / l[:, :, :, None]

    # Reshape output to [batch, 1, n_heads * q_mult, d_head]
    O = O.reshape(batch_size, 1, n_heads * q_mult, d_head)

    return O


def run_sdpa_decode_sink_impl(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    dim,
    q_dtype,
    dtype,
):
    num_iters = 1

    # Log the test parameters
    logger.debug(f"Running SDPA Decode with parameters: ")
    logger.debug(f"Batch: {batch}")
    logger.debug(f"Sequence Length: {seq_len}")
    logger.debug(f"Number of Heads (Q): {nh}")
    logger.debug(f"Number of Heads (KV): {nkv}")
    logger.debug(f"Dim: {dim}")
    logger.debug(f"Query Data Type: {q_dtype}")
    logger.debug(f"Key-Value Data Type: {dtype}")

    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, seq_len, dim).float()  # (B, H, S, D)
    k = torch.randn(batch, nkv, seq_len, dim).float()  # (B, H, S, D)
    v = torch.randn(batch, nkv, seq_len, dim).float()  # (B, H, S, D)

    # TODO: Remove -inf after ttnn supports attention sink
    sink = torch.randn(1, 1, nh, 1, 1).repeat(batch, 1, 1, 1, 1)  # (batch, 1, H, 1, 1)
    sink *= 4.0  # Closer to real distribution

    ref_q = q.permute(0, 2, 1, 3).view(batch, seq_len, nkv, nh // nkv, dim)
    ref_k = k.permute(0, 2, 1, 3)
    ref_v = v.permute(0, 2, 1, 3)

    tt_q_in = q[:, :, -1:, :].permute(2, 0, 1, 3)  # (D, B, H, S) -> (S, B, H, D)

    ######################
    ### TT Setup
    #######################
    q_chunk_size = 0  # Not used in decode
    k_chunk_size = 128

    scale = dim**-0.5
    start_indices = batch * [seq_len - 1]

    # Setup sink for TTNN
    tt_sink_in = sink[:1, ...].reshape(nh, 1)
    tt_sink_in = torch.nn.functional.pad(tt_sink_in, (0, ttnn.TILE_SIZE - 1), "constant", 0)
    tt_sink_in /= scale  # Important!! GPT-OSS expects sink to not be scaled

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

    # Set up input tensors
    q_mem_config = ttnn.DRAM_MEMORY_CONFIG
    out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    tt_q = ttnn.from_torch(
        tt_q_in,
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem_config,
    )
    tt_k = ttnn.from_torch(
        k,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_v = ttnn.from_torch(
        v,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_sink = ttnn.from_torch(
        tt_sink_in,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_start_indices = ttnn.from_torch(
        torch.tensor(start_indices),
        device=device,
        dtype=ttnn.int32,
    )

    ##########################
    ### SDPA Decode
    ##########################
    logger.info(f"Running SDPA Decode with TT Q shape: {tt_q.shape}, TT K shape: {tt_k.shape}, dtype: {dtype}")

    def run_op():
        torch_out = flash_decode_sdpa(
            ref_q[:, -1:, ...],
            ref_k,
            ref_v,
            sink,
            scale,
            sliding_window=0,
        )

        tt_out = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            cur_pos_tensor=tt_start_indices,
            scale=scale,
            attention_sink=tt_sink,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=out_mem_config,
        )

        return torch_out, tt_out

    outs = []
    for i in range(num_iters):  # Check for program cache
        logger.debug(f"Running SDPA Decode operation iteration {i + 1}/{num_iters}")
        torch_out, tt_out = run_op()
        outs.append((torch_out, tt_out))

        # Increment start indices for the next iteration
        ttnn.plus_one(tt_start_indices)
        start_indices = [x + 1 for x in start_indices]

    ########################
    ### Validation
    ########################
    pcc_threshold = 0.999
    if dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.98
    if dtype == ttnn.bfloat8_b:
        pcc_threshold = 0.999

    for i, (torch_out, tt_out) in enumerate(outs):
        tt_out = ttnn.to_torch(tt_out)[..., :nh, :]  # (S, B, H, D)

        out_pass, out_pcc = comp_pcc(torch_out, tt_out, pcc_threshold)
        logger.debug(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"

    # Check program cache entries
    num_program_cache_entries = device.num_program_cache_entries()

    # SDPA + PlusOne
    expected_num_program_cache_entries = 2
    assert (
        num_program_cache_entries == expected_num_program_cache_entries
    ), f"Expected {expected_num_program_cache_entries} program cache entries, got {num_program_cache_entries}."
