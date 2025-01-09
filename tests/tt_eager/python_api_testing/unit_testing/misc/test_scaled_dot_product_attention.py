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
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0, skip_for_blackhole


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


def run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, use_high_precision_compute=False):
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
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

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
    )
    tt_back = ttnn.to_torch(tt_back)

    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_repeated, V_repeated, is_causal=True)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


# @pytest.mark.skip(reason="ND PCC issues")
@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
# @pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [128], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 1, 1, 2011, 128],  # Llama2-70B
        [1, 1, 1, 2049, 128],  # Llama2-70B
        [1, 1, 1, 1, 128],  # Llama2-70B
        [1, 1, 1, 160, 128],  # Llama2-70B
    ),
)
def test_sdpa_tt_padded(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    """
    This case tests padding in multiple ways.
    SDPA should correctly handle tile-padded tensors. In addition,
    it should allow arbitrary chunk sizes even if they don't divide the sequence length.
    """
    ttnn.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)


# @pytest.mark.skip(reason="ND PCC issues")
@skip_for_blackhole("Mismatching on BH, see #12349")
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
        [8, 8, 1, 2048, 128],  # Llama2-70B large batch
        [1, 8, 1, 8192, 128],  # Llama2-70B large sequence
    ),
)
def test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    ttnn.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [256], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([1, 8, 1, 8192 * 16, 128],),  # Llama2-70B 128K sequence
)
def test_sdpa_tt_large_seq(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    ttnn.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, use_high_precision_compute=True)


@pytest.mark.skip(reason="Skip perf test in CI")
@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [256], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [256], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 128, 128],  # Llama2-70B 128K sequence
        [1, 8, 1, 2048 * 16, 128],  # Llama2-70B 128K sequence
        [1, 8, 1, 8192, 128],  # Llama2-70B 128K sequence
        [1, 8, 1, 8192 * 2, 128],  # Llama2-70B 128K sequence
        [1, 8, 1, 8192 * 4, 128],  # Llama2-70B 128K sequence
        [1, 8, 1, 8192 * 16, 128],  # Llama2-70B 128K sequence
    ),
)
def test_sdpa_tt_perf(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    ttnn.device.DisablePersistentKernelCache()
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)


# @pytest.mark.skip(reason="ND PCC issues")
@skip_for_blackhole("Mismatching on BH, see #12349")
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


def run_sdpa_noncausal(device, b, nh, nkv, sq, d, q_chunk_size, k_chunk_size, dtype, sk=None):
    torch.manual_seed(1234)
    if sk is None:
        sk = sq

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, sq, d)
    K = fa_rand(b, nkv, sk, d)
    V = fa_rand(b, nkv, sk, d)
    # Generate random non-causal attention mask
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

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")
    logger.debug(f"mask: {mask.shape}")

    tt_Q = ttnn.Tensor(Q, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_K = ttnn.Tensor(K, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_V = ttnn.Tensor(V, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_mask = ttnn.Tensor(mask, ttnn.bfloat4_b).to(ttnn.TILE_LAYOUT).to(device)
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

    if nkv > 1 and nkv != nh:
        assert nh % nkv == 0
        K = K.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)
        V = V.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False, attn_mask=mask)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


# @pytest.mark.skip(reason="ND PCC issues")
@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 2048, 128],  # Llama2-70B
        [1, 16, 2, 128, 128],  # GQA
        [1, 16, 16, 4096, 96],  # Llama3.2-11B-Vision
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
    run_sdpa_noncausal(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
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
        exp_approx_mode=False,
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


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
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
    use_program_cache,
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


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
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
    use_program_cache,
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
