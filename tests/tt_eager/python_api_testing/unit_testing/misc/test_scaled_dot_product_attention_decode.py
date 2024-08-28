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
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0
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
    return ttnn.experimental.tensor.CoreRange(
        ttnn.experimental.tensor.CoreCoord(0, 0),
        ttnn.experimental.tensor.CoreCoord(num_x - 1, num_y - 1),
    )


def get_chunk_size(s):
    if s <= 32:
        return 32
    if s <= 64:
        return 32
    if s <= 128:
        return 32
    if s <= 256:
        return 256
    if s <= 2048:
        return 512
    return 512


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


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
        min_pcc = 0.93 if dtype == ttnn.bfloat4_b else min_pcc

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.types.MemoryConfig(ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM)

    shard_grid = ttnn.experimental.tensor.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, (padded_num_heads, d), ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
    )

    height_sharded_memcfg = ttnn.types.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    K = fa_rand(nkv, b, s, d)
    V = fa_rand(nkv, b, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    max_start_idx = 0

    while max_start_idx < s:
        scale = d**-0.5
        start_indices = np.linspace(0, max_start_idx, b, dtype=np.int32).tolist()

        k_chunk_size = get_chunk_size(max_start_idx + 1)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,  # device.compute_with_storage_grid_size(),
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
        )

        padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size)

        # Test various sequence lengths
        logger.info(f"Testing with sequence length: {max_start_idx}")
        logger.info(f"Using chunk size: {k_chunk_size}")
        logger.info(f"Using padded layer length: {padded_layer_len}")
        logger.info(f"Using padded num heads: {padded_num_heads}")

        attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
        for i in range(b):
            start_idx = start_indices[i]
            attn_mask[:, i, :, start_idx + 1 :] = torch.finfo(torch.float32).min

        Q = fa_rand(1, b, padded_num_heads, d)

        tt_Q = ttnn.as_tensor(
            Q,
            device=device,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=height_sharded_memcfg if sharded_in else dram_memcfg,
        )
        if cur_pos_tensor:
            start_indices_tt = ttnn.Tensor(torch.tensor(start_indices), ttnn.uint32).to(device)

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
                start_indices,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
            )

        tt_back = ttnn.to_torch(tt_back)

        tt_back = tt_back[:, :, :nh, :]

        Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
        K_slice = K[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
        V_slice = V[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
        attn_mask_slice = attn_mask[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S

        expect = torch.nn.functional.scaled_dot_product_attention(
            Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
        )  # b, nh, 1, d
        expect = expect.squeeze().unsqueeze(0)

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
        min_pcc = 0.93 if dtype == ttnn.bfloat4_b else min_pcc

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.types.MemoryConfig(ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM)

    shard_grid = ttnn.experimental.tensor.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, (padded_num_heads, d), ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
    )

    height_sharded_memcfg = ttnn.types.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    K = fa_rand(nkv, b, s, d)
    V = fa_rand(nkv, b, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_indices = [s // 2 for _ in range(b)] if start_indices is None else start_indices
    max_start_idx = max(start_indices)
    scale = d**-0.5

    k_chunk_size = get_chunk_size(max_start_idx + 1)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
    )

    padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size)

    # Test various sequence lengths
    logger.debug(f"Testing with sequence length: {max_start_idx}")
    logger.debug(f"Using chunk size: {k_chunk_size}")
    logger.debug(f"Using padded layer length: {padded_layer_len}")
    logger.debug(f"Using padded num heads: {padded_num_heads}")

    attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
    for i in range(b):
        start_idx = start_indices[i]
        attn_mask[:, i, :, start_idx + 1 :] = torch.finfo(torch.float32).min

    Q = fa_rand(1, b, padded_num_heads, d)

    tt_Q = ttnn.as_tensor(
        Q,
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=height_sharded_memcfg if sharded_in else dram_memcfg,
    )
    if cur_pos_tensor:
        start_indices_tt = ttnn.Tensor(torch.tensor(start_indices), ttnn.uint32).to(device)
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
            start_indices,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
        )

    tt_back = ttnn.to_torch(tt_back)
    tt_back = tt_back[:, :, :nh, :]

    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
    K_slice = K[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
    V_slice = V[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
    attn_mask_slice = attn_mask[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S
    expect = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    expect = expect.squeeze().unsqueeze(0)

    out_pass, out_pcc = comp_pcc(expect, tt_back, min_pcc)

    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.skip("Skipping due to potential nd pcc issue #9370")
@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.bfloat8_b, ttnn.bfloat16],
        [ttnn.bfloat4_b, ttnn.bfloat16],
    ],
    ids=[
        "all_bfp8",
        "all_bfp16",
        "kv_bfp8",
        "kv_bfp4",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, single_iter, cur_pos_tensor",
    (
        [32, 8, 1, 32768, 128, (8, 6), True, True],  # Llama2-70B
        [16, 8, 1, 32768, 128, (8, 6), False, False],  # Llama2-70B
        [8, 8, 1, 32768, 128, (8, 6), True, False],  # Llama2-70B
        [4, 8, 1, 32768, 128, (8, 6), True, False],  # Llama2-70B
        [32, 8, 1, 32768, 128, (8, 8), True, True],  # Mixtral8x7b
    ),
)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, single_iter, cur_pos_tensor, use_program_cache):
    ttnn.device.DisablePersistentKernelCache()
    if single_iter:
        run_test_sdpa_decode_single_iter(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
        )
    else:
        run_test_sdpa_decode_multi_pos(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
        )


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.skip("Skipping due to potential nd pcc issue #9370")
@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.bfloat16, ttnn.bfloat16],
    ],
    ids=[
        "all_bfp8",
        "all_bfp16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    ([16, 8, 1, 32768, 128, (8, 6)],),  # Llama2-70B
)
def test_sdpa_decode_sharded(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    ttnn.device.DisablePersistentKernelCache()
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=True, sharded_out=False
    )
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=True, sharded_out=True
    )
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=False, sharded_out=True
    )


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.skip("Skipping Perf Test in CI")
def test_sdpa_decode_perf(device, use_program_cache):
    dtype = ttnn.bfloat8_b
    q_dtype = ttnn.bfloat16
    nh = 8
    nkv = 1
    d = 128
    grid_size = (8, 8)

    bs_combs = [
        (32, 2048),
        (16, 2048),
        (32, 8192),
        (16, 8192),
        (8, 8192),
        (16, 8192 * 2),
        (8, 8192 * 2),
        (16, 8192 * 4),
        (8, 8192 * 4),
        (4, 8192 * 4),
    ]

    for b, s in bs_combs:
        run_test_sdpa_decode_single_iter(  # different user pos
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            grid_size,
            q_dtype,
            sharded_in=True,
            sharded_out=True,
            start_indices=np.linspace(0, s - 1, b, dtype=np.int32).tolist(),
        )
        run_test_sdpa_decode_single_iter(  # all same user pos
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            grid_size,
            q_dtype,
            sharded_in=True,
            sharded_out=True,
            start_indices=[s - 1 for _ in range(b)],
        )


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.skip("Skipping due to potential nd pcc issue #9370")
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
    ids=["bfp8", "bf16"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([16, 8, 1, 8192, 128],),  # Llama2-70B
)
def test_sdpa_decode_program_cache(device, b, nh, nkv, s, d, dtype, use_program_cache):
    ttnn.device.DisablePersistentKernelCache()

    dummy_tensors = []
    for i in range(2):
        # generate random start indices from 0 to s-1
        start_indices = np.random.randint(0, s - 1, b).tolist()
        start_indices[0] = s - 1

        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(32, 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.types.MemoryConfig(
                    ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM
                ),
            )
        )
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(1, 1, 32, 32 * 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.types.MemoryConfig(
                    ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.types.BufferType.L1,
                    ttnn.experimental.tensor.ShardSpec(
                        ttnn.experimental.tensor.CoreRangeSet({num_to_corerange(32)}),
                        (32, 32),
                        ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                        False,
                    ),
                ),
            )
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=False,
            sharded_out=False,
            start_indices=start_indices,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=True,
            sharded_out=False,
            start_indices=start_indices,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=True,
            sharded_out=True,
            start_indices=start_indices,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=False,
            sharded_out=True,
            start_indices=start_indices,
        )

    assert device.num_program_cache_entries() == 4


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
    dram_memcfg = ttnn.types.MemoryConfig(ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM)

    K = fa_rand(nkv, b, s, d)
    V = fa_rand(nkv, b, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_idx = 0

    failed_start_pos = []

    while start_idx < 32000:
        Q = fa_rand(1, b, padded_num_heads, d)
        prev_pcc = None

        scale = d**-0.5

        k_chunk_size = get_chunk_size(start_idx + 1)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,  # device.compute_with_storage_grid_size(),
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
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
        K_slice = K[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
        V_slice = V[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
        attn_mask_slice = attn_mask[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S

        expect = torch.nn.functional.scaled_dot_product_attention(
            Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
        )  # b, nh, 1, d
        expect = expect.squeeze().unsqueeze(0)

        all_out_pass = True

        for i in range(200):
            tt_Q = ttnn.as_tensor(
                Q,
                device=device,
                dtype=q_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=dram_memcfg,  # height_sharded_memcfg
            )

            tt_back = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                [start_idx for _ in range(b)],
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

        start_idx += 20  # if start_idx < 4096 else 3001

    logger.info(f"ND Start Pos: {failed_start_pos}")


@pytest.mark.timeout(600)
@pytest.mark.skip("Skipping due to causing 45 minutes timeout on tt eager unit tests")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        # [ttnn.bfloat16, ttnn.bfloat16],
        # [ttnn.bfloat8_b, ttnn.bfloat8_b],
        [ttnn.bfloat4_b, ttnn.bfloat4_b],
    ],
    ids=[
        # "bfp16_bfp16",
        # "bfp8_bfp8",
        "bfp4_bfp4",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    (
        # [16, 8, 1, 32768, 128, (8, 6)],  # Llama2-70B
        # [32, 8, 1, 32768, 128, (8, 8)],  # Llama2-70B
        [16, 8, 1, 32768, 128, (8, 6)],
    ),
)
def test_sdpa_decode_ndpcc(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, use_program_cache):
    ttnn.device.DisablePersistentKernelCache()
    run_test_sdpa_decode_ndpcc(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype)
