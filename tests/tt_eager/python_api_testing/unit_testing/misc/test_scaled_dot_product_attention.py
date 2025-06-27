# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

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
from models.utility_functions import skip_for_wormhole_b0, skip_for_blackhole


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def run_test_sdpa_tt(
    device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, use_high_precision_compute=False, rmse_threshold=None
):
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    if use_high_precision_compute:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
    )
    tt_back = ttnn.to_torch(tt_back)
    # Slice out any tile-padding
    tt_back = tt_back[:, :, :s, :]

    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_repeated, V_repeated, is_causal=True)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")
    if rmse_threshold is not None:
        assert rmse < rmse_threshold
    else:
        assert out_pass


def run_sdpa_noncausal(
    device, b, nh, nkv, sq, d, q_chunk_size, k_chunk_size, dtype, sk=None, use_mask=True, rmse_threshold=None
):
    torch.manual_seed(1234)
    if sk is None:
        sk = sq

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, sq, d)
    K = fa_rand(b, nkv, sk, d)
    V = fa_rand(b, nkv, sk, d)
    # Generate random non-causal attention mask
    tt_mask = None
    mask = None
    if use_mask:
        mask = torch.bernoulli(
            torch.full(
                (
                    b,
                    sq,
                    sk,
                ),
                0.25,
            )
        )
        mask = mask.unsqueeze(1)
        mask = mask * -1e9
        tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=False,
        attn_mask=tt_mask,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_back = ttnn.to_torch(tt_back)
    # Slice out any tile-padding
    tt_back = tt_back[:, :, :sq, :]

    if nkv > 1 and nkv != nh:
        assert nh % nkv == 0
        K = K.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)
        V = V.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False, attn_mask=mask)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")
    if rmse_threshold is not None:
        assert rmse < rmse_threshold

    assert out_pass


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [128, 512], ids=["q128", "q512"])
@pytest.mark.parametrize("k_chunk_size", [128, 512], ids=["k128", "k512"])
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize("b", [1, 2], ids=["b1", "b2"])
@pytest.mark.parametrize("nh", [1, 8], ids=["nh1", "nh8"])
@pytest.mark.parametrize("nkv", [1], ids=["nkv1"])
@pytest.mark.parametrize("s", [1, 160, 2011], ids=["s1", "s160", "s2011"])
@pytest.mark.parametrize(
    "d",
    [128],
    ids=[
        "d128",
    ],
)
def test_sdpa_tt_padded(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, is_causal, is_ci_env):
    if q_chunk_size == 512 and k_chunk_size == 512:
        pytest.skip("OOM config.")
    if nh % nkv != 0:
        pytest.skip("nkv must divide nh")
    if is_ci_env and (b == 1 or nh == 1 or s == 1 or q_chunk_size == 512 or k_chunk_size == 512):
        pytest.skip("Skipping to avoid CI timeout")
    ttnn.device.DisablePersistentKernelCache()
    if is_causal:
        run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, rmse_threshold=0.033)
    else:
        run_sdpa_noncausal(
            device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, use_mask=False, rmse_threshold=0.033
        )


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp4", "bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 2048, 128],  # Llama2-70B
        [1, 16, 1, 2048, 64],  # Falcon-40B
        [1, 71, 1, 2048, 64],  # Falcon-7B
        [8, 8, 1, 2048, 128],  # Llama2-70B large batch
        [1, 8, 1, 8192, 128],  # Llama2-70B large sequence
    ),
)
def test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if dtype == ttnn.bfloat4_b and (
        q_chunk_size > 128 or k_chunk_size > 128 or [b, nh, nkv, s, d] != [1, 8, 1, 2048, 128]
    ):
        pytest.skip("just need to single test case for sanity-check bfp4")
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    rmse_threshold = 0.0092 if (dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b) else 0.0093
    ttnn.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, rmse_threshold=rmse_threshold)


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [32], ids=["q32"])
@pytest.mark.parametrize("k_chunk_size", [32], ids=["k32"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([1, 16, 1, 32, 64],),  # Falcon-40B
    ids=["f40b"],
)
def test_sdpa_tt_small_chunks(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    ttnn.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [256], ids=["q256"])
@pytest.mark.parametrize("k_chunk_size", [512], ids=["k512"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 3, 3, 44 * 1024, 128],  # Llama2-70B
        [1, 1, 1, 2048, 128],
    ),
    ids=["full-grid", "single-core"],
)
def test_sdpa_perf(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    ttnn.device.DisablePersistentKernelCache()
    run_sdpa_noncausal(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, use_mask=False)


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [256], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([1, 8, 1, 128 * 1024, 128],),  # Llama2-70B 128K sequence
)
def test_sdpa_tt_large_seq(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    ttnn.device.DisablePersistentKernelCache()
    rmse_threshold = 0.0094
    run_test_sdpa_tt(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        use_high_precision_compute=True,
        rmse_threshold=rmse_threshold,
    )


@pytest.mark.skip(reason="Skip perf test in CI")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [256], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [256], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 128, 128],  # Llama2-70B 128K sequence
        [1, 8, 1, 8 * 1024, 128],  # Llama2-70B 128K sequence
        [1, 8, 1, 16 * 1024, 128],  # Llama2-70B 128K sequence
        [1, 8, 1, 32 * 1024, 128],  # Llama2-70B 128K sequence
        [1, 8, 1, 64 * 1024, 128],  # Llama2-70B 128K sequence
    ),
)
def test_sdpa_tt_perf(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    ttnn.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
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
def test_sdpa_tt_with_program_cache(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")

    for _ in range(2):
        run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)

    assert device.num_program_cache_entries() == 1


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 2048, 128],  # Llama2-70B
        [1, 16, 2, 128, 128],  # GQA
        [1, 16, 16, 4096, 96],  # Llama-3.2-11B-Vision
        [1, 71, 1, 2048, 64],  # Falcon-7B
        [8, 8, 1, 2048, 128],  # Llama2-70B large batch
    ),
)
def test_sdpa_noncausal(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if s > 2048 and (q_chunk_size == 128 or k_chunk_size == 128):
        pytest.skip("Bad PCC for small chunks")
    ttnn.device.DisablePersistentKernelCache()
    rmse_threshold = 0.0062
    run_sdpa_noncausal(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, rmse_threshold=rmse_threshold)


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, sq, sk, d",
    (
        [1, 8, 1, 4096, 2048, 128],
        # [1, 4, 4, 128*1024, 6528, 128],  # Llama-Vision long seq
        [1, 4, 1, 2048, 6528, 128],  # Llama-Vision
    ),
)
def test_sdpa_noncausal_unequal_seqlen(device, b, nh, nkv, sq, sk, d, q_chunk_size, k_chunk_size, dtype):
    if (sq % q_chunk_size != 0) or (sk % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    ttnn.device.DisablePersistentKernelCache()
    run_sdpa_noncausal(device, b, nh, nkv, sq, d, q_chunk_size, k_chunk_size, dtype, sk=sk)


def run_test_chunked_sdpa(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    q_chunk_size,
    k_chunk_size,
    prefill_chunk_size,
    page_block_size,
    q_dtype,
    k_dtype,
    use_high_precision_compute,
    grid_size=None,
):
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size or device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    if use_high_precision_compute:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)
    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_repeated, V_repeated, is_causal=True)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")

    assert s % prefill_chunk_size == 0, "s must be divisible by prefill_chunk_size"
    assert prefill_chunk_size % page_block_size == 0, "prefill_chunk_size must be divisible by page_block_size"
    num_prefill_chunks = s // prefill_chunk_size
    # Prepare K, V paged for TT
    max_num_blocks_per_seq = s // page_block_size
    assert max_num_blocks_per_seq * page_block_size == s
    max_num_blocks = b * max_num_blocks_per_seq
    assert max_num_blocks * page_block_size == b * s

    # Shuffle paged KV cache according to some random page_table
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    # page_table is the reverse permutation from shuffled -> unshuffled, and is used to map
    # a virtual block to the physical block id.
    page_table = reverse_permutation.reshape(b, max_num_blocks_per_seq)

    def page_cache(cache):
        paged_cache = (
            cache.reshape(b, nkv, max_num_blocks_per_seq, page_block_size, d)
            .transpose(1, 2)
            .reshape(max_num_blocks, nkv, page_block_size, d)
        )

        shuffled_page_cache = paged_cache[permutation]
        return shuffled_page_cache

    def unpage_cache(cache):
        unshuffled_page_cache = cache[reverse_permutation]
        paged_cache_back = (
            unshuffled_page_cache.reshape(b, nkv, max_num_blocks_per_seq, page_block_size, d)
            .transpose(1, 2)
            .reshape(b, nkv, s, d)
        )
        return paged_cache_back

    # Check that we can convert from normal to paged to normal
    assert torch.allclose(unpage_cache(page_cache(K)), K), "K is not equal to unpage_cache(page_cache(K))"
    assert torch.allclose(unpage_cache(page_cache(V)), V), "V is not equal to unpage_cache(page_cache(V))"

    tt_paged_K = ttnn.Tensor(page_cache(K), k_dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_paged_V = ttnn.Tensor(page_cache(V), k_dtype).to(ttnn.TILE_LAYOUT).to(device)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    for chunk_idx in range(num_prefill_chunks):
        # Chunk Q
        Q_chunk = Q[:, :, chunk_idx * prefill_chunk_size : (chunk_idx + 1) * prefill_chunk_size]
        tt_Q_chunk = ttnn.Tensor(Q_chunk, q_dtype).to(ttnn.TILE_LAYOUT).to(device)
        chunk_start_idx = chunk_idx * prefill_chunk_size

        tt_back = ttnn.transformer.chunked_scaled_dot_product_attention(
            tt_Q_chunk,
            tt_paged_K,
            tt_paged_V,
            page_table_tt,
            chunk_start_idx,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        tt_back = tt_back.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        gt_chunk = gt[:, :, chunk_idx * prefill_chunk_size : (chunk_idx + 1) * prefill_chunk_size]
        out_pass, out_pcc = comp_pcc(gt_chunk, tt_back, 0.998)
        logger.debug(f"python vs pytorch: {out_pcc}")
        assert out_pass


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("k_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize("prefill_chunk_size", [1024, 2048])
@pytest.mark.parametrize("page_block_size", [64, 128])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    [
        [1, 8, 1, 16 * 1024, 128],
    ],  # Llama2-70B
)
def test_sdpa_chunked(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    q_chunk_size,
    k_chunk_size,
    prefill_chunk_size,
    page_block_size,
    q_dtype,
    k_dtype,
    use_high_precision_compute=False,
):
    for _ in range(2):
        run_test_chunked_sdpa(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            q_chunk_size,
            k_chunk_size,
            prefill_chunk_size,
            page_block_size,
            q_dtype,
            k_dtype,
            use_high_precision_compute,
        )

    # Print number of program cache entries
    assert device.num_program_cache_entries() == 1, "Program cache should only have 1 entry but has {}".format(
        device.num_program_cache_entries()
    )


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("k_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("q_chunk_size", [128])
@pytest.mark.parametrize("k_chunk_size", [128])
@pytest.mark.parametrize("prefill_chunk_size", [1024])
@pytest.mark.parametrize("page_block_size", [64])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    [
        [2, 1, 1, 4096, 128],
    ],  # Llama2-70B
)
def test_sdpa_chunked_iterate_batch(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    q_chunk_size,
    k_chunk_size,
    prefill_chunk_size,
    page_block_size,
    q_dtype,
    k_dtype,
    use_high_precision_compute=False,
):
    """
    This tests chunked prefill where a single core has more than one user to process.
    """
    for _ in range(2):
        run_test_chunked_sdpa(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            q_chunk_size,
            k_chunk_size,
            prefill_chunk_size,
            page_block_size,
            q_dtype,
            k_dtype,
            use_high_precision_compute,
            grid_size=(1, 1),
        )

    # Print number of program cache entries
    assert device.num_program_cache_entries() == 1, "Program cache should only have 1 entry but has {}".format(
        device.num_program_cache_entries()
    )


def run_test_joint_sdpa(
    device,
    b,
    nh,
    seq_len,
    joint_seq_len,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    use_high_precision_compute=False,
    grid_size=None,
    rmse_threshold=None,
):
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size or device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    if use_high_precision_compute:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    Q = fa_rand(b, nh, seq_len, d)
    K = fa_rand(b, nh, seq_len, d)
    V = fa_rand(b, nh, seq_len, d)

    joint_Q = fa_rand(b, nh, joint_seq_len, d)
    joint_K = fa_rand(b, nh, joint_seq_len, d)
    joint_V = fa_rand(b, nh, joint_seq_len, d)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_joint_Q = ttnn.from_torch(joint_Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_joint_K = ttnn.from_torch(joint_K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_joint_V = ttnn.from_torch(joint_V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_out, tt_joint_out = ttnn.transformer.joint_scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        tt_joint_Q,
        tt_joint_K,
        tt_joint_V,
        joint_strategy="rear",
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_out = ttnn.to_torch(tt_out)
    tt_joint_out = ttnn.to_torch(tt_joint_out)
    # Slice out any tile-padding
    tt_out = tt_out[:, :, :seq_len, :]
    tt_joint_out = tt_joint_out[:, :, :joint_seq_len, :]
    logger.debug(f"tt_out: {tt_out.shape}")
    logger.debug(f"tt_joint_out: {tt_joint_out.shape}")

    pt_Q = torch.cat([Q, joint_Q], dim=2)
    pt_K = torch.cat([K, joint_K], dim=2)
    pt_V = torch.cat([V, joint_V], dim=2)
    gt = torch.nn.functional.scaled_dot_product_attention(pt_Q, pt_K, pt_V, is_causal=False)
    gt_out = gt[:, :, :seq_len, :]
    gt_joint_out = gt[:, :, seq_len:, :]

    for out, gt in [(tt_out, gt_out), (tt_joint_out, gt_joint_out)]:
        out_pass, out_pcc = comp_pcc(gt, out, 0.994)
        logger.debug(f"python vs pytorch: {out_pcc}")
        rmse = torch.sqrt(((gt - out) ** 2).mean()).item()
        logger.debug(f"rmse: {rmse}")
        assert out_pass
        if rmse_threshold is not None:
            assert rmse < rmse_threshold


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [128, 512], ids=["q128", "q512"])
@pytest.mark.parametrize("k_chunk_size", [128, 512], ids=["k128", "k512"])
@pytest.mark.parametrize("b", [1, 2], ids=["b1", "b2"])
@pytest.mark.parametrize("nh", [1, 3], ids=["nh1", "nh3"])
@pytest.mark.parametrize(
    "seq_len, joint_seq_len",
    [
        (15, 19),
        (2048, 256),
        (3000, 100),
        (20 * 1024 + 1, 118),
    ],
)
@pytest.mark.parametrize(
    "d",
    [128],
    ids=[
        "d128",
    ],
)
def test_joint_sdpa(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype):
    if q_chunk_size == 512 and k_chunk_size == 512:
        pytest.skip("OOM config.")
    ttnn.device.DisablePersistentKernelCache()
    rmse_threshold = 0.013
    run_test_joint_sdpa(
        device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype, rmse_threshold=rmse_threshold
    )


# @pytest.mark.skip(reason="ND PCC issues")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize("b", [1], ids=["b1"])
@pytest.mark.parametrize("nh", [1], ids=["nh1"])
@pytest.mark.parametrize(
    "seq_len, joint_seq_len",
    [
        (3000, 100),
    ],
)
@pytest.mark.parametrize(
    "d",
    [128],
    ids=[
        "d128",
    ],
)
def test_joint_sdpa_program_cache(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype):
    dummy_tensors = []
    for _ in range(3):
        dummy_tensors.append(
            ttnn.from_torch(fa_rand(b, nh, seq_len, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        )
        run_test_joint_sdpa(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype, dummy_tensors)


from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed
from tt_metal.tools.profiler.process_model_log import run_device_profiler, get_latest_ops_log_filename


@pytest.mark.skip()
def test_combine():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    old_df = pd.read_csv("old_sdpa.csv")
    new_df = pd.read_csv("new_sdpa.csv")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey="row")

    # For side-by-side bars
    bar_width = 0.35
    seq_order = [512, 1024, 2048, 4096, 8192, 16384]  # for tick labels
    head_order = [64, 128, 256]

    # Set up bar positions - shifted left/right depending on df
    bar_shifts = {"old": -bar_width / 2, "new": bar_width / 2}

    for df_idx, (df_name, df) in enumerate([("old", old_df), ("new", new_df)]):
        # Rename columns
        df = df.rename(
            columns={
                "INPUT_0_W": "batch",
                "INPUT_0_Z": "num_heads",
                "INPUT_0_Y": "seq_len",
                "INPUT_0_X": "head_dim",
            }
        )

        # Calculate TFLOP/s
        def flops(row):
            B = row.batch
            H = row.num_heads
            L = row.seq_len
            D = row.head_dim
            causal_multiplier = 1 + row.is_causal.astype(int)
            return 4 * L**2 * D * H * B / causal_multiplier

        df["tflops"] = flops(df) / df["min_kernel_duration_ns"] / 1e3  # ns → s, flop → Tflop
        print(df)

        # ------------------------------------------------------------------------------
        # Color map for different algorithms
        palette = {
            "new": "#1f77b4",
            "old": "#ff7f0e",
        }

        for j_head, head_dim in enumerate(head_order):
            for i_causal, causal in enumerate([False, True]):
                ax = axes[i_causal, j_head]
                data = df[df["head_dim"] == head_dim][df["is_causal"] == causal]

                # Base positions for sequence lengths
                x_positions = np.arange(len(seq_order))

                # Shift positions based on whether this is old or new data
                shifted_positions = x_positions + bar_shifts[df_name]

                # Only one implementation (TTNN) in our case
                ys = []
                for seq_len in seq_order:
                    seq_data = data[data["seq_len"] == seq_len]
                    if len(seq_data) > 0:
                        ys.append(seq_data["tflops"].values[0])
                    else:
                        ys.append(np.nan)

                ax.bar(
                    shifted_positions,
                    ys,
                    width=bar_width,
                    label=df_name if i_causal == 0 and j_head == 0 else "_nolegend_",
                    color=palette[df_name],
                )

                # Add value labels on top of each bar
                for i, y in enumerate(ys):
                    if not np.isnan(y):
                        value_text = f"{y:.1f}"

                        ax.text(shifted_positions[i], y, value_text, ha="center", va="bottom", fontsize=8, rotation=45)

                # axes cosmetics ------------------------------------------------------
                # Only set these once (for the first df)
                if df_idx == 0:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([f"{l//1024}k" if l >= 1024 else str(l) for l in seq_order], rotation=0)
                    ax.set_xlabel("Sequence length")
                    if j_head == 0:
                        ax.set_ylabel("Speed (TFLOP/s)")
                    title = (
                        "Forward, "
                        + ("with causal mask" if causal else "without causal mask")
                        + f", head dim {head_dim}"
                    )
                    ax.set_title(title, fontsize=9)

    # unified legend --------------------------------------------------------------
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=len(labels), frameon=False)

    fig.suptitle("Attention forward speed (FP16/BF16) on Wormhole", y=0.96, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig("attention_speed_grid.png", dpi=300)


@pytest.mark.skip()
def test_sdpa_benchmark_detailed():
    command = "pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention.py::test_sdpa_benchmark"
    cols = ["ATTRIBUTES", "INPUT_0_W", "INPUT_0_Z", "INPUT_0_Y", "INPUT_0_X", "DEVICE KERNEL DURATION [ns]"]
    op_name = "ScaledDotProductAttention"
    warmup_iters = 0  # 5 iterations per device
    step_name = "SDPA"
    subdir = "sdpa"

    run_device_profiler(command, subdir)
    # r = post_process_ops_log(subdir, cols, sum_vals=False)
    filename = get_latest_ops_log_filename(subdir)
    import pandas as pd

    df = pd.read_csv(filename)
    df["is_causal"] = (
        df["ATTRIBUTES"]
        .str.extract(r"'is_causal':\s*('true'|'false')", expand=False)  # → "true"/"false"/NaN
        .map({"'true'": True, "'false'": False})  # → bool/NaN
    )

    # TODO: Drop rows with NaN in device kernel duration?
    keep = ["INPUT_0_W", "INPUT_0_Z", "INPUT_0_Y", "INPUT_0_X", "is_causal", "DEVICE KERNEL DURATION [ns]"]
    df = df[keep]
    group_cols = ["INPUT_0_W", "INPUT_0_Z", "INPUT_0_Y", "INPUT_0_X", "is_causal"]
    result = (
        df.groupby(group_cols, as_index=False)["DEVICE KERNEL DURATION [ns]"]
        .min()
        .rename(columns={"DEVICE KERNEL DURATION [ns]": "min_kernel_duration_ns"})
    )

    result.to_csv("new_sdpa.csv", index=False)
    import matplotlib.pyplot as plt
    import numpy as np

    df = result
    # Rename columns
    df = df.rename(
        columns={
            "INPUT_0_W": "batch",
            "INPUT_0_Z": "num_heads",
            "INPUT_0_Y": "seq_len",
            "INPUT_0_X": "head_dim",
        }
    )

    # Calculate TFLOP/s
    def flops(row):
        B = row.batch
        H = row.num_heads
        L = row.seq_len
        D = row.head_dim
        causal_multiplier = 1 + row.is_causal.astype(int)
        return 4 * L**2 * D * H * B / causal_multiplier

    df["tflops"] = flops(df) / df["min_kernel_duration_ns"] / 1e3  # ns → s, flop → Tflop
    print(df)

    # ------------------------------------------------------------------------------
    # 2.  Plot (6 sub‑plots, 2 × 3 grid)
    # ------------------------------------------------------------------------------
    # Color map for different algorithms
    palette = {
        "TTNN": "#1f77b4",
        "FlashAttention‑2": "#ff7f0e",
        "FlashAttention‑3": "#9467bd",
        "Triton": "#2ca02c",
        "cuDNN": "#d62728",
    }

    seq_order = [512, 1024, 2048, 4096, 8192, 16384]  # for tick labels
    head_order = [64, 128, 256]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey="row")

    for j_head, head_dim in enumerate(head_order):
        for i_causal, causal in enumerate([False, True]):
            ax = axes[i_causal, j_head]
            data = df[df["head_dim"] == head_dim][df["is_causal"] == causal]
            # breakpoint()

            # Group data by sequence length for the bar plot
            bar_width = 0.15
            x_positions = np.arange(len(seq_order))

            # Only one implementation (TTNN) in our case
            ys = []
            for seq_len in seq_order:
                seq_data = data[data["seq_len"] == seq_len]
                if len(seq_data) > 0:
                    ys.append(seq_data["tflops"].values[0])
                else:
                    ys.append(np.nan)

            ax.bar(
                x_positions,
                ys,
                width=bar_width,
                label="TTNN" if i_causal == 0 and j_head == 0 else "_nolegend_",
                color=palette["TTNN"],
            )

            # axes cosmetics ------------------------------------------------------
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"{l//1024}k" if l >= 1024 else str(l) for l in seq_order], rotation=0)
            ax.set_xlabel("Sequence length")
            if j_head == 0:
                ax.set_ylabel("Speed (TFLOP/s)")
            title = "Forward, " + ("with causal mask" if causal else "without causal mask") + f", head dim {head_dim}"
            ax.set_title(title, fontsize=9)

    # unified legend --------------------------------------------------------------
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=len(labels), frameon=False)

    fig.suptitle("Attention forward speed (FP16/BF16) on Wormhole", y=0.96, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig("attention_speed_grid.png", dpi=300)


@pytest.mark.skip()
def test_sdpa_benchmark(device):
    dtype = ttnn.bfloat16

    def create_device_tensors(b, nh, s, d):
        Q = fa_rand(b, nh, s, d)
        K = fa_rand(b, nh, s, d)
        V = fa_rand(b, nh, s, d)
        tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
        tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
        tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
        return tt_Q, tt_K, tt_V

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    num_tokens = 16384
    hidden_dim = 2048
    # for causal in [True]:
    for causal in [True, False]:
        # for head_dim in [64]:
        for head_dim in [64, 128, 256]:
            nh = hidden_dim // head_dim
            # for s in [512]:
            for s in [512, 1024, 2048, 4096, 8192, 16384]:
                b = num_tokens // s
                tt_Q, tt_K, tt_V = create_device_tensors(b, nh, s, head_dim)
                # Start sweep for this config
                # for q_chunk_size in [64]:
                for q_chunk_size in [64, 128, 256, 512]:
                    # for k_chunk_size in [64]:
                    for k_chunk_size in [64, 128, 256, 512]:
                        print(
                            f"Running test for config: b={b}, nh={nh}, s={s}, head_dim={head_dim}, q_chunk_size={q_chunk_size}, k_chunk_size={k_chunk_size}, dtype={dtype}, causal={causal}"
                        )
                        program_config = ttnn.SDPAProgramConfig(
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                            q_chunk_size=q_chunk_size,
                            k_chunk_size=k_chunk_size,
                            exp_approx_mode=True,
                        )
                        try:
                            tt_back = ttnn.transformer.scaled_dot_product_attention(
                                tt_Q,
                                tt_K,
                                tt_V,
                                is_causal=causal,
                                program_config=program_config,
                                compute_kernel_config=compute_kernel_config,
                            )
                        except Exception as e:
                            logger.error(
                                f"Error running test for config: b={b}, nh={nh}, s={s}, head_dim={head_dim}, q_chunk_size={q_chunk_size}, k_chunk_size={k_chunk_size}, dtype={dtype}, causal={causal}"
                            )
                            logger.error(f"Error: {e}")

                ttnn.DumpDeviceProfiler(device)
