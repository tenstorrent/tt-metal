# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import torch.nn.functional as F
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0, skip_for_blackhole
from .test_scaled_dot_product_attention import fa_rand


def torch_sdpa(q, k, v, num_devices):
    scale = k.size(-1) ** -0.5
    seq_len = k.size(2)
    slice_seq_len = seq_len // num_devices
    out = None
    lse = None
    lse_list = []
    for ring_id in range(num_devices):
        k_slice = k[:, :, ring_id * slice_seq_len : (ring_id + 1) * slice_seq_len, :]
        v_slice = v[:, :, ring_id * slice_seq_len : (ring_id + 1) * slice_seq_len, :]
        attn_weights = torch.matmul(q, k_slice.transpose(-2, -1)) * scale
        cur_max, _ = torch.max(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - cur_max)
        cur_sum = torch.sum(attn_weights, dim=-1, keepdim=True)
        cur_out = torch.matmul(attn_weights, v_slice)
        cur_out = cur_out / cur_sum
        cur_lse = cur_max + torch.log(cur_sum)
        if ring_id == 0:
            out = cur_out
            lse = cur_lse
        else:
            sig = F.sigmoid(cur_lse - lse)
            out = out - sig * (out - cur_out)
            lse = lse - F.logsigmoid(lse - cur_lse)
        lse_list.append(lse)

    return out, lse_list


def run_ring_joint_sdpa(
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
    # topology=ttnn.MeshTopology.RING,
):
    num_devices = 8
    local_seq_len = seq_len // num_devices
    torch.manual_seed(1234)

    compute_grid = grid_size or device.compute_with_storage_grid_size()

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=compute_grid,
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
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    Q = fa_rand(b, nh, local_seq_len, d)
    K = fa_rand(b, nh, seq_len, d)
    V = fa_rand(b, nh, seq_len, d)

    joint_Q = fa_rand(b, nh, joint_seq_len, d)
    joint_K = fa_rand(b, nh, joint_seq_len, d)
    joint_V = fa_rand(b, nh, joint_seq_len, d)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_joint_Q = ttnn.from_torch(joint_Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_joint_K = ttnn.from_torch(joint_K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_joint_V = ttnn.from_torch(joint_V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out, tt_joint_out, tt_lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
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
    tt_lse = ttnn.to_torch(tt_lse)
    tt_lse_list = torch.chunk(tt_lse, num_devices, dim=0)
    # Slice out any tile-padding
    tt_out = tt_out[:, :, :local_seq_len, :]
    tt_joint_out = tt_joint_out[:, :, :joint_seq_len, :]
    logger.debug(f"tt_out: {tt_out.shape}")
    logger.debug(f"tt_joint_out: {tt_joint_out.shape}")

    pt_Q = torch.cat([Q, joint_Q], dim=2)
    pt_K = torch.cat([K, joint_K], dim=2)
    pt_V = torch.cat([V, joint_V], dim=2)

    gt, gt_lse_list = torch_sdpa(pt_Q, pt_K, pt_V, num_devices)
    gt_out = gt[:, :, :local_seq_len, :]
    gt_joint_out = gt[:, :, local_seq_len:, :]

    passing = True
    for out, gt in [(tt_out, gt_out), (tt_joint_out, gt_joint_out)]:
        out_pass, out_pcc = comp_pcc(gt, out, 0.994)
        logger.debug(f"python vs pytorch: {out_pcc}")
        logger.debug(f"mse: {((gt - out) ** 2).mean()}")
        passing = passing and out_pass

    for i, (lse, gt) in enumerate(zip(tt_lse_list, gt_lse_list)):
        lse_pass, lse_pcc = comp_pcc(gt, lse, 0.98)
        logger.debug(f"python vs pytorch LSE {i}: {lse_pcc}")
        logger.debug(f"mse: {((gt - lse) ** 2).mean()}")
        passing = passing and lse_pass
        print(f"tt: max: {lse.max()}, min: {lse.min()}, mean: {lse.mean()}")
        print(f"gt: max: {gt.max()}, min: {gt.min()}, mean: {gt.mean()}")

    assert passing


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [32], ids=["q256"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k256"])
@pytest.mark.parametrize("b", [1], ids=["b1"])
@pytest.mark.parametrize("nh", [3], ids=["nh3"])
@pytest.mark.parametrize("d", [128], ids=["d128"])
@pytest.mark.parametrize(
    "seq_len, joint_seq_len",
    [
        (4096, 128),
    ],
)
def test_joint_sdpa(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype):
    if q_chunk_size == 512 and k_chunk_size == 512:
        pytest.skip("OOM config.")
    ttnn.device.DisablePersistentKernelCache()
    run_ring_joint_sdpa(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype)


# @skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
# @pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
# @pytest.mark.parametrize("q_chunk_size", [128], ids=["q128"])
# @pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
# @pytest.mark.parametrize("b", [1], ids=["b1"])
# @pytest.mark.parametrize("nh", [1], ids=["nh1"])
# @pytest.mark.parametrize(
#     "seq_len, joint_seq_len",
#     [
#         (3000, 100),
#     ],
# )
# @pytest.mark.parametrize(
#     "d",
#     [128],
#     ids=[
#         "d128",
#     ],
# )
# def test_joint_sdpa_program_cache(
#     device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype, use_program_cache
# ):
#     dummy_tensors = []
#     for _ in range(3):
#         dummy_tensors.append(
#             ttnn.from_torch(fa_rand(b, nh, seq_len, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
#         )
#         run_ring_joint_sdpa(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype, dummy_tensors)
