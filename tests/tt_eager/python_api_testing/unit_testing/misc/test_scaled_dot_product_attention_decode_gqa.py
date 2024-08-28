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
    start_indices=None,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    num_parallel_cores = grid_size[0] * grid_size[1] // b * nkv
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

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_indices = [s // 2 for _ in range(b * nkv)] if start_indices is None else start_indices
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

    attn_mask = torch.zeros((1, b * nkv, padded_num_heads, padded_layer_len))
    for i in range(b * nkv):
        start_idx = start_indices[i]
        attn_mask[:, i, :, start_idx + 1 :] = torch.finfo(torch.float32).min

    Q = fa_rand(1, nh, b, d)

    tt_Q = ttnn.as_tensor(
        Q,
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
    )

    tt_back = ttnn.transformer.scaled_dot_product_attention_decode_gqa(
        tt_Q,
        tt_K,
        tt_V,
        start_indices,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram_memcfg,
    )

    tt_back = ttnn.to_torch(tt_back)

    Q_slice = Q.permute(2, 1, 0, 3)  # b, nh, 1, d
    K_slice = K[:, :, :padded_layer_len, :]  # b, nh, S, d
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    V_slice = V[:, :, :padded_layer_len, :]  # b, nh, S, d
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S

    attn_mask_slice = attn_mask[:, :b, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S
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
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, single_iter",
    (
        [1, 32, 8, 32768, 128, (8, 8), True],  # Llama3.1-8B
        [2, 32, 8, 32768, 128, (8, 8), True],  # Llama3.1-8B
        [4, 32, 8, 32768, 128, (8, 8), True],  # Llama3.1-8B
        [8, 16, 4, 32768, 128, (8, 8), True],  # Llama3.1-8B on N300
    ),
)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, single_iter, use_program_cache):
    ttnn.device.DisablePersistentKernelCache()
    run_test_sdpa_decode_single_iter(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype)


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.skip("Skipping due to potential nd pcc issue #9370")
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "bf16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([4, 32, 8, 8192, 128],),  # Llama3.1-8B
)
def test_sdpa_decode_program_cache(device, b, nh, nkv, s, d, dtype, use_program_cache):
    ttnn.device.DisablePersistentKernelCache()

    for i in range(2):
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 8),
            dtype,
            start_indices=None,
        )

    assert device.num_program_cache_entries() == 5
