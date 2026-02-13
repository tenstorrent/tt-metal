# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


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
@pytest.mark.parametrize("q_chunk_size", [32, 128, 512], ids=["q32", "q128", "q512"])
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
