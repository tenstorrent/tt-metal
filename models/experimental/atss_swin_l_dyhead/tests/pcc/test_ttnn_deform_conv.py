# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for TtDeformConv2dV2 vs torchvision.ops.deform_conv2d."""

import pytest
import torch
import torchvision.ops
import ttnn

from loguru import logger
from models.common.utility_functions import comp_pcc
from models.experimental.atss_swin_l_dyhead.tt.tt_deform_conv import TtDeformConv2dV2


@pytest.mark.parametrize(
    "H, W, C_in, C_out, stride",
    [
        (5, 5, 256, 256, 1),  # P7
        (10, 10, 256, 256, 1),  # P6
        (20, 20, 256, 256, 1),  # P5
        (40, 40, 256, 256, 1),  # P4
        (80, 80, 256, 256, 1),  # P3
        (10, 10, 256, 256, 2),  # spatial_conv_low stride-2
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_deform_conv2d_v2_pcc(device, H, W, C_in, C_out, stride):
    torch.manual_seed(42)
    kH, kW = 3, 3
    K = kH * kW
    pad = (1, 1)
    dil = (1, 1)
    H_in, W_in = H, W
    if stride == 1:
        H_out, W_out = H, W
    else:
        H_out, W_out = H // stride, W // stride

    # Random inputs
    x_torch = torch.randn(1, C_in, H_in, W_in, dtype=torch.float32) * 0.5
    offset_torch = torch.randn(1, 2 * K, H_out, W_out, dtype=torch.float32) * 0.3  # small offsets
    mask_torch_logit = torch.randn(1, K, H_out, W_out, dtype=torch.float32) * 0.5
    mask_torch = mask_torch_logit.sigmoid()
    weight_torch = torch.randn(C_out, C_in, kH, kW, dtype=torch.float32) * 0.1
    bias_torch = torch.randn(C_out, dtype=torch.float32) * 0.1

    # --- Reference: torchvision deform_conv2d ---
    with torch.no_grad():
        ref = torchvision.ops.deform_conv2d(
            input=x_torch,
            offset=offset_torch,
            weight=weight_torch,
            bias=bias_torch,
            stride=(stride, stride),
            padding=pad,
            dilation=dil,
            mask=mask_torch,
        )
    logger.info(f"Reference output shape: {ref.shape}, mean={ref.mean().item():.4f}, std={ref.std().item():.4f}")

    # --- TTNN: tt_deform_conv2d_v2 ---
    # NCHW -> NHWC for input, offset, mask
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()
    offset_nhwc = offset_torch.permute(0, 2, 3, 1).contiguous()
    mask_nhwc = mask_torch.permute(0, 2, 3, 1).contiguous()

    x_tt = ttnn.from_torch(
        x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    offset_tt = ttnn.from_torch(
        offset_nhwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    mask_tt = ttnn.from_torch(
        mask_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    dcn = TtDeformConv2dV2(
        device=device,
        weight=weight_torch,
        bias=bias_torch,
        C_in=C_in,
        C_out=C_out,
        kH=kH,
        kW=kW,
        H_in=H_in,
        W_in=W_in,
        H_out=H_out,
        W_out=W_out,
        stride=(stride, stride),
        padding=pad,
        dilation=dil,
        align_corners=True,
    )

    out_tt = dcn(x_tt, offset_tt, mask_tt)
    out_torch_nhwc = ttnn.to_torch(ttnn.from_device(out_tt)).float()
    # out_torch_nhwc: (1, H_out, W_out, C_out) → permute to NCHW for comparison
    out_torch = out_torch_nhwc.permute(0, 3, 1, 2)

    logger.info(
        f"TTNN output shape: {out_torch.shape}, mean={out_torch.mean().item():.4f}, std={out_torch.std().item():.4f}"
    )
    assert out_torch.shape == ref.shape, f"shape mismatch: ttnn={out_torch.shape} vs ref={ref.shape}"

    passing, pcc = comp_pcc(ref, out_torch, 0.96)
    logger.info(f"  DCNv2 H={H} W={W} stride={stride}: PCC={pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < 0.96"
