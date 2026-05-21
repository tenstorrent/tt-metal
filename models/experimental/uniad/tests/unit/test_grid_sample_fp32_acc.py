# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for ttnn.grid_sample bilinear with fp32_dest_acc_en=True.

Previously, the bilinear program factory dropped the user-supplied
`dst_full_sync_en` value from the compute kernel config. With
`fp32_dest_acc_en=True` and `in_ntiles_c > 4`, `pack_untilize_dest` exceeded
the half-sync DEST capacity (4 tiles in fp32 mode) and silently corrupted
output whenever a core processed more than one output stick (>~130 sticks
total on BH p150b). All four cases (default, explicit False, fp32 acc True,
fp32 acc True + dst_full_sync) must now hit PCC ≥ 0.99 vs torch reference.
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize(
    "B,C,H_in,W_in,H_out,W_out",
    [
        # Bisecting fp32_dest_acc_en regression by output stick count.
        # 1 stick = 1 (B,h_o,w_o) coordinate. fp32_acc=true works for
        # small stick counts and breaks somewhere between 60 and 216.
        (1, 256, 40, 23, 10, 6),  # 60 sticks: works
        (1, 256, 40, 23, 10, 8),  # 80 sticks
        (1, 256, 40, 23, 10, 10),  # 100 sticks
        (1, 256, 40, 23, 12, 10),  # 120 sticks
        (1, 256, 40, 23, 14, 10),  # 140 sticks
        (1, 256, 40, 23, 16, 10),  # 160 sticks
        (1, 256, 40, 23, 18, 10),  # 180 sticks
        (1, 256, 40, 23, 20, 10),  # 200 sticks
        (1, 256, 40, 23, 20, 12),  # 240 sticks: broken
    ],
)
def test_grid_sample_fp32_acc(device, B, C, H_in, W_in, H_out, W_out):
    torch.manual_seed(0)
    x_nhwc = torch.randn(B, H_in, W_in, C, dtype=torch.bfloat16)
    grid = (torch.rand(B, H_out, W_out, 2, dtype=torch.float32) * 2.0 - 1.0).to(torch.bfloat16)

    # PyTorch reference (NCHW)
    ref = F.grid_sample(
        x_nhwc.permute(0, 3, 1, 2).to(torch.float32),
        grid.to(torch.float32),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    ref_nhwc = ref.permute(0, 2, 3, 1)

    x_tt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    grid_tt = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Case A: no compute_kernel_config (uses init defaults: fp32_acc=false)
    out_a = ttnn.grid_sample(x_tt, grid_tt, mode="bilinear", padding_mode="zeros", align_corners=False)
    a_torch = ttnn.to_torch(out_a).to(torch.float32)
    _, pcc_a = comp_pcc(ref_nhwc.to(torch.float32), a_torch, pcc=0.0)
    logger.info(f"fp32_acc=default(false) PCC: {pcc_a}")

    # Case B: fp32_dest_acc_en=False explicit
    cfg_false = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        math_approx_mode=False,
    )
    out_b = ttnn.grid_sample(
        x_tt, grid_tt, mode="bilinear", padding_mode="zeros", align_corners=False, compute_kernel_config=cfg_false
    )
    b_torch = ttnn.to_torch(out_b).to(torch.float32)
    _, pcc_b = comp_pcc(ref_nhwc.to(torch.float32), b_torch, pcc=0.0)
    logger.info(f"fp32_acc=False explicit PCC: {pcc_b}")

    # Case C: fp32_dest_acc_en=True
    cfg_true = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        math_approx_mode=False,
    )
    out_c = ttnn.grid_sample(
        x_tt, grid_tt, mode="bilinear", padding_mode="zeros", align_corners=False, compute_kernel_config=cfg_true
    )
    c_torch = ttnn.to_torch(out_c).to(torch.float32)
    _, pcc_c = comp_pcc(ref_nhwc.to(torch.float32), c_torch, pcc=0.0)
    logger.info(f"fp32_acc=True PCC: {pcc_c}")

    # Case D: fp32_dest_acc_en=True + dst_full_sync_en=True
    cfg_sync = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        math_approx_mode=False,
        dst_full_sync_en=True,
    )
    out_d = ttnn.grid_sample(
        x_tt, grid_tt, mode="bilinear", padding_mode="zeros", align_corners=False, compute_kernel_config=cfg_sync
    )
    d_torch = ttnn.to_torch(out_d).to(torch.float32)
    _, pcc_d = comp_pcc(ref_nhwc.to(torch.float32), d_torch, pcc=0.0)
    logger.info(f"fp32_acc=True + dst_full_sync=True PCC: {pcc_d}")

    logger.info(
        f"SUMMARY  B={B} C={C} H_in={H_in} W_in={W_in} H_out={H_out} W_out={W_out} sticks={B*H_out*W_out}: "
        f"default={pcc_a:.5f}  false={pcc_b:.5f}  true={pcc_c:.5f}  true+sync={pcc_d:.5f}"
    )

    assert pcc_a >= 0.99, f"default (fp32_acc=False) PCC {pcc_a} below 0.99"
    assert pcc_b >= 0.99, f"explicit fp32_acc=False PCC {pcc_b} below 0.99"
    assert pcc_c >= 0.99, f"fp32_acc=True PCC {pcc_c} below 0.99"
    assert pcc_d >= 0.99, f"fp32_acc=True + dst_full_sync=True PCC {pcc_d} below 0.99"
