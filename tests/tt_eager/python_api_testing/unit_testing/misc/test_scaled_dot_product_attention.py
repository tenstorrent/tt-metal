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


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, fp32_accum=False):
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4 if fp32_accum else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")

    tt_Q = ttnn.Tensor(Q, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_K = ttnn.Tensor(K, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_V = ttnn.Tensor(V, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
    )
    tt_back = tt_back.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_repeated, V_repeated, is_causal=True)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    logger.debug(f"maximum difference: {torch.max(torch.abs(gt - tt_back))}")
    logger.debug(f"RMSE: {torch.sqrt(torch.mean((gt - tt_back) ** 2))}")
    breakpoint()
    assert out_pass


# @pytest.mark.skip(reason="ND PCC issues")
@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "bf16",
    ],
)
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize("fp32_accum", [True, False])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        # [1, 8, 1, 8192, 128],  # Llama2-70B
        # [1, 16, 1, 2048, 64],  # Falcon-40B
        # [1, 71, 1, 2048, 64],  # Falcon-7B
        # [8, 8, 1, 2048, 128],  # Llama2-70B batch
        # [1, 32, 8, 8192, 128],  # Llama3.1-8B large sequence
        # [1, 8, 1, 8192*4, 128],  # Llama2-70B large sequence
        [1, 8, 1, 8192 * 16, 128],  # Llama2-70B 128K sequence
    ),
)
def test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, fp32_accum):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    if fp32_accum and q_chunk_size == 256 and k_chunk_size == 256:
        pytest.skip("Can cause OOM if fp32 acc.")
    ttnn.device.DisablePersistentKernelCache()
    with torch.no_grad():
        run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, fp32_accum)


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


def simulated_binary_block_inplace(Q, K, V, q_chunk_size):
    B, NH, S, D = Q.shape
    out_tensor = torch.zeros_like(Q)
    n_steps = S // q_chunk_size
    for b in range(B):
        for nh in range(NH):
            for i in range(n_steps):
                Q_slice = Q[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :].clone()
                for j in range(i + 1):
                    V_slice = V[b, nh, j * q_chunk_size : (j + 1) * q_chunk_size, :]
                    Q_slice = torch.max(Q_slice, V_slice)  # max block. change to *= for mul block, += for add block
                out_tensor[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :] = Q_slice
    return out_tensor


def simulated_reduce_c(Q, K, V, q_chunk_size, op="max"):
    B, NH, S, D = Q.shape
    out_tensor = torch.zeros_like(Q)
    n_steps = S // q_chunk_size
    DHt = D // 32
    for b in range(B):
        for nh in range(NH):
            for i in range(n_steps):
                Q_slice = Q[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :].clone()
                for j in range(i + 1):
                    V_slice = V[b, nh, j * q_chunk_size : (j + 1) * q_chunk_size, :]
                    V_slice_result = torch.zeros_like(Q_slice)
                    if op == "sum":
                        V_slice = torch.sum(V_slice, dim=-1)  # reduce_c block
                    elif op == "max":
                        V_slice = torch.max(V_slice, dim=-1)[0]
                    for dht in range(DHt):
                        V_slice_result[..., dht * 32] = V_slice
                    Q_slice += V_slice_result

                out_tensor[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :] = Q_slice
    return out_tensor


def simulated_matmul(Q, K, V, q_chunk_size):
    B, NH, S, D = Q.shape
    out_tensor = torch.zeros_like(Q)
    n_steps = S // q_chunk_size
    DHt = D // 32
    for b in range(B):
        for nh in range(NH):
            for i in range(n_steps):
                Q_slice = Q[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :].clone()
                out_sum = torch.zeros_like(Q_slice)
                for j in range(i + 1):
                    k_slice = K[b, nh, j * q_chunk_size : (j + 1) * q_chunk_size, :]
                    V_slice = V[b, nh, j * q_chunk_size : (j + 1) * q_chunk_size, :]
                    QK = torch.matmul(Q_slice, k_slice.transpose(-1, -2))
                    out_sum += torch.matmul(QK, V_slice)
                out_tensor[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :] = out_sum
    return out_tensor


def simulated_sub_exp(Q, K, V, q_chunk_size):
    B, NH, S, D = Q.shape
    out_tensor = torch.zeros_like(Q)
    n_steps = S // q_chunk_size
    DHt = D // 32
    for b in range(B):
        for nh in range(NH):
            for i in range(n_steps):
                Q_slice = Q[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :].clone()
                out_sum = torch.zeros_like(Q_slice)
                for j in range(i + 1):
                    V_slice = V[b, nh, j * q_chunk_size : (j + 1) * q_chunk_size, :]
                    out_sum += torch.exp(Q_slice - V_slice)  # sub_exp block
                out_tensor[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :] = out_sum
    return out_tensor


def simulated_recip_block(Q, K, V, q_chunk_size):
    B, NH, S, D = Q.shape
    out_tensor = torch.zeros_like(Q)
    n_steps = S // q_chunk_size
    DHt = D // 32
    for b in range(B):
        for nh in range(NH):
            for i in range(n_steps):
                Q_slice = Q[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :].clone()
                for j in range(i + 1):
                    Q_slice = 1 / Q_slice  # recip block
                out_tensor[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :] = Q_slice
    return out_tensor


def simulated_bcast_col(Q, K, V, q_chunk_size, op="b_cast_scalar"):
    B, NH, S, D = Q.shape
    out_tensor = torch.zeros_like(Q)
    n_steps = S // q_chunk_size
    DHt = D // 32
    scale = 1 / (D**0.5)
    logger.debug(f"scale: {scale}")
    for b in range(B):
        for nh in range(NH):
            for i in range(n_steps):
                Q_slice = Q[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :].clone()
                out_sum = torch.zeros_like(Q_slice)
                for j in range(i + 1):
                    V_slice = V[b, nh, j * q_chunk_size : (j + 1) * q_chunk_size, :]
                    V_slice_col = V_slice[..., 0].unsqueeze(-1).expand_as(Q_slice)
                    if op == "sub_exp":
                        out_sum += torch.exp(Q_slice - V_slice_col)  # sub_exp block
                    elif op == "mul":
                        out_sum += Q_slice * V_slice_col  # mul blocks
                    elif op == "b_cast_scalar":
                        out_sum += Q_slice * scale
                    else:
                        raise ValueError(f"Unknown op: {op}")
                out_tensor[b, nh, i * q_chunk_size : (i + 1) * q_chunk_size, :] = out_sum
    return out_tensor


def run_sdpa_op_test(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, fp32_accum=False):
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4 if fp32_accum else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")

    tt_Q = ttnn.Tensor(Q, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_K = ttnn.Tensor(K, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_V = ttnn.Tensor(V, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_back = tt_back.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    # gt = torch.nn.functional.scaled_dot_product_attention(Q, K_repeated, V_repeated, is_causal=True)
    gt = simulated_bcast_col(Q, K_repeated, V_repeated, q_chunk_size)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    logger.debug(f"maximum difference: {torch.max(torch.abs(gt - tt_back))}")
    logger.debug(f"RMSE: {torch.sqrt(torch.mean((gt - tt_back) ** 2))}")
    breakpoint()
    assert out_pass


# @pytest.mark.skip(reason="ND PCC issues")
@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "bf16",
    ],
)
@pytest.mark.parametrize("q_chunk_size", [128], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize("fp32_accum", [True, False])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        # [1, 8, 1, 8192, 128],  # Llama2-70B
        # [1, 16, 1, 2048, 64],  # Falcon-40B
        # [1, 71, 1, 2048, 64],  # Falcon-7B
        # [8, 8, 1, 2048, 128],  # Llama2-70B batch
        # [1, 32, 8, 8192, 128],  # Llama3.1-8B large sequence
        # [1, 8, 1, 8192*4, 128],  # Llama2-70B large sequence
        [1, 1, 1, 8192, 128],  # Llama2-70B 128K sequence
    ),
)
def test_op_sdpa(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, fp32_accum):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    ttnn.device.DisablePersistentKernelCache()
    with torch.no_grad():
        run_sdpa_op_test(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, fp32_accum)
