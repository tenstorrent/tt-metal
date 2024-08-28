# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 7),  # current CI issue
        # compute_with_storage_grid_size=(8, 2), # old ND issue
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    Q = torch.randn(b, nh, s, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)
    attn_mask = torch.full((s, s), torch.finfo(torch.float32).min)
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, 1, -1, -1)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")
    logger.debug(f"attn_mask: {attn_mask.shape}")

    tt_Q = ttnn.Tensor(Q, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_K = ttnn.Tensor(K, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_V = ttnn.Tensor(V, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_attn_mask = ttnn.Tensor(attn_mask, dtype).to(ttnn.TILE_LAYOUT).to(device)
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)

    # for i in range(1000):
    for i in range(1):
        tt_back = ttnn.transformer.scaled_dot_product_attention(
            tt_Q, tt_K, tt_V, tt_attn_mask, is_causal=True, program_config=program_config
        )
        tt_back = tt_back.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        # out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
        # out_pass, out_pcc = comp_pcc(gt, tt_back, 0.99)
        out_pass, out_pcc = comp_pcc(gt, tt_back, 0.99)
        logger.debug(f"python vs pytorch: {out_pcc}")
        assert out_pass


# @pytest.mark.skip(reason="ND PCC issues")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
# @pytest.mark.parametrize("q_chunk_size", [128], ids=["q128"]) # CI failing after llk redundant cb init commit
# @pytest.mark.parametrize("q_chunk_size", [32], ids=["q32"]) # old ND issue
@pytest.mark.parametrize("q_chunk_size", [32, 128], ids=["q32", "q128"])
# @pytest.mark.parametrize("q_chunk_size", [512], ids=["q32"])
@pytest.mark.parametrize("k_chunk_size", [32], ids=["k64"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        # [1, 2, 1, 2048, 128],  # old ND issue
        [1, 8, 1, 128, 128],  # Llama2-70B - CI failing after llk redundant cb init commit
        # [1, 8, 1, 128, 128],  # Llama2-70B small sequence
        # [1, 8, 1, 2048, 128],  # Llama2-70B
        # [1, 16, 1, 2048, 64],  # Falcon-40B
        # [1, 71, 1, 2048, 64],  # Falcon-7B
        # [8, 8, 1, 2048, 128],  # Llama2-70B large batch
        # [1, 8, 1, 8192, 128],  # Llama2-70B large sequence
    ),
)
def test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    ttnn.experimental.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)


# @pytest.mark.skip(reason="ND PCC issues")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 2048, 128],  # Llama2-70B
        [1, 16, 1, 2048, 64],  # Falcon-40B
        [1, 71, 1, 2048, 64],  # Falcon-7B
    ),
)
def test_sdpa_tt_with_program_cache(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, use_program_cache):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")

    for _ in range(2):
        run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)

    assert device.num_program_cache_entries() == 1


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
    return ttnn.experimental.tensor.CoreRange(
        ttnn.experimental.tensor.CoreCoord(0, 0),
        ttnn.experimental.tensor.CoreCoord(num_x - 1, num_y - 1),
    )


def get_chunk_size(s):
    # Not sure if optimal
    if s <= 32:
        return 32
    if s <= 64:
        return 64
    if s <= 128:
        return 128
    if s <= 256:
        return 256
    if s <= 2048:
        return 512
    return 1024


def run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype):
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
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

    K = torch.randn(nkv, b, s, d)
    V = torch.randn(nkv, b, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_idx = 31

    while start_idx < s:
        scale = d**-0.5

        k_chunk_size = get_chunk_size(start_idx)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
        )

        padded_layer_len = nearest_n(start_idx, n=k_chunk_size)

        # Test various sequence lengths
        logger.debug(f"Testing with sequence length: {start_idx}")
        logger.debug(f"Using chunk size: {k_chunk_size}")
        logger.debug(f"Using padded layer length: {padded_layer_len}")
        logger.debug(f"Using padded num heads: {padded_num_heads}")

        attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
        # Assume all users are at same position
        attn_mask[:, :, :, start_idx:] = torch.finfo(torch.float32).min

        Q = torch.randn(1, b, padded_num_heads, d)

        tt_Q = ttnn.as_tensor(
            Q, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=height_sharded_memcfg
        )

        tt_attn_mask = ttnn.as_tensor(
            attn_mask, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
        )

        tt_back = ttnn.transformer.scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            tt_attn_mask,
            is_causal=False,
            scale=scale,
            program_config=program_config,
            valid_seq_len=padded_layer_len,
            compute_kernel_config=compute_kernel_config,
            memory_config=height_sharded_memcfg,
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

        out_pass, out_pcc = comp_pcc(expect, tt_back, 0.99)

        logger.debug(f"python vs pytorch: {out_pcc}")
        assert out_pass

        start_idx += 601


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
    ids=["bfp8", "bf16"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [16, 8, 1, 8192, 128],  # Llama2-70B
        [32, 16, 1, 2048, 64],  # Falcon-40B
        [32, 4, 1, 8192, 128],  # Mixtral
    ),
)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype):
    ttnn.experimental.device.DisablePersistentKernelCache()
    run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype)


def run_test_sdpa_decode_single_iter(device, b, nh, nkv, s, d, dtype):
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
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

    K = torch.randn(nkv, b, s, d)
    V = torch.randn(nkv, b, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    start_idx = s // 2
    scale = d**-0.5

    k_chunk_size = get_chunk_size(start_idx)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
    )

    padded_layer_len = nearest_n(start_idx, n=k_chunk_size)

    # Test various sequence lengths
    logger.debug(f"Testing with sequence length: {start_idx}")
    logger.debug(f"Using chunk size: {k_chunk_size}")
    logger.debug(f"Using padded layer length: {padded_layer_len}")
    logger.debug(f"Using padded num heads: {padded_num_heads}")

    attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
    # Assume all users are at same position
    attn_mask[:, :, :, start_idx:] = torch.finfo(torch.float32).min

    Q = torch.randn(1, b, padded_num_heads, d)

    tt_Q = ttnn.as_tensor(Q, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=height_sharded_memcfg)

    tt_attn_mask = ttnn.as_tensor(
        attn_mask, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
    )

    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        tt_attn_mask,
        is_causal=False,
        scale=scale,
        program_config=program_config,
        valid_seq_len=padded_layer_len,
        compute_kernel_config=compute_kernel_config,
        memory_config=height_sharded_memcfg,
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

    out_pass, out_pcc = comp_pcc(expect, tt_back, 0.99)

    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
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
    ttnn.experimental.device.DisablePersistentKernelCache()

    dummy_tensors = []
    for _ in range(2):
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
        run_test_sdpa_decode_single_iter(device, b, nh, nkv, s, d, dtype)

    assert device.num_program_cache_entries() == 1
