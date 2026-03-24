# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Repro for https://github.com/tenstorrent/tt-metal/issues/40583

SDPA with high precision compute config (fp32_dest_acc_en=True) and bfp8 data
fails with UndefinedBehavior in ttsim:
  ERROR: UndefinedBehavior: tensix_execute_movd2b: incompatible use_dst32b=0 dest_32b_lo=1

Root cause: enforce_fp32_accumulation flag is not set consistently across the
scaled_dot_product_attention operation. The reduce_c step doesn't set
enforce_fp32_accumulation even though fp32_dest_acc_en is enabled.

Run with tt-sim:
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1 \
    pytest tests/ttnn/unit_tests/operations/sdpa/test_repro_issue_40583.py -svv
"""

import torch
import ttnn
import pytest
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from loguru import logger


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def test_sdpa_high_precision_fp32_dest_acc(device):
    """
    SDPA with fp32_dest_acc_en=True and bfp8 data triggers UndefinedBehavior
    in ttsim due to inconsistent enforce_fp32_accumulation in reduce_c.
    """
    torch.manual_seed(1234)

    b, nh, nkv, s, d = 1, 8, 1, 2048, 128
    q_chunk_size = 128
    k_chunk_size = 128
    dtype = ttnn.bfloat8_b

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    # High precision compute config — fp32_dest_acc_en=True triggers the bug
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_back = ttnn.to_torch(tt_back)
    tt_back = tt_back[:, :, :s, :]

    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_repeated, V_repeated, is_causal=True)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.info(f"PCC: {out_pcc}")
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.info(f"RMSE: {rmse}")

    assert rmse < 0.0092, f"RMSE {rmse} exceeds threshold 0.0092"
