# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the device-side modulated_deform_conv prototype.

Verifies that `TtModulatedDeformConv2dDevice` reproduces the host DCNv2
reference (`torchvision.ops.deform_conv2d`, which is mathematically
equivalent to the mmcv kernel the weights were trained with) to within
bfloat16 tolerance on the small shapes UniAD's ResNet stages 3 and 4
actually use.
"""

import pytest
import torch
import ttnn
from loguru import logger

from torch.nn.modules.utils import _pair

from models.experimental.uniad.tt.ttnn_modulated_deform_conv import TtModulatedDeformConv2dDevice
from models.experimental.uniad.tt.ttnn_resnet import modulated_deform_conv2d as ref_modulated_deform_conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
@pytest.mark.parametrize(
    "B,C_in,C_out,H_in,W_in",
    [
        (6, 256, 512, 40, 23),  # single-chunk path (C_in <= grid_sample cap)
        (6, 512, 512, 20, 12),  # 2 c-chunks (e.g. stage 3 narrow case)
        (6, 1024, 512, 20, 12),  # 4 c-chunks (ResNet101 stage 3 DCN shape)
    ],
)
def test_modulated_deform_conv_device_matches_reference(device, B, C_in, C_out, H_in, W_in):
    K = 3
    stride = _pair(1)
    padding = _pair(1)
    dilation = _pair(1)
    groups = 1
    deform_groups = 1

    H_out = (H_in + 2 * padding[0] - dilation[0] * (K - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (K - 1) - 1) // stride[1] + 1

    torch.manual_seed(0)
    x_torch_nchw = torch.randn(B, C_in, H_in, W_in, dtype=torch.float32)
    # offset & mask coming from a conv_offset, simulated here with random
    offset_y_torch = torch.randn(B, K * K, H_out, W_out, dtype=torch.float32) * 0.3
    offset_x_torch = torch.randn(B, K * K, H_out, W_out, dtype=torch.float32) * 0.3
    mask_torch = torch.sigmoid(torch.randn(B, K * K, H_out, W_out, dtype=torch.float32))
    weight_torch = torch.randn(C_out, C_in, K, K, dtype=torch.float32) * (1.0 / (C_in * K * K) ** 0.5)
    bias_torch = None

    # The DCNv2 kernel (torchvision and mmcv alike) reads offset channel
    # `2 * (kh*K + kw) + 0` as y-offset and `+1` as x-offset for kernel
    # position (kh, kw) — i.e. interleaved (y0, x0, y1, x1, ..., y8, x8).
    # The standard wrapper produces conv_offset output where this layout
    # is whatever the model learns, so the convention is invisible in
    # normal training; with explicitly constructed offsets we have to
    # honour the actual layout the kernel reads. The scaffold consumes
    # the same interleaved layout, so we build it once and use it for
    # both the host reference and the device call.
    offset_yx_nchw = torch.stack([offset_y_torch, offset_x_torch], dim=2).reshape(B, 2 * K * K, H_out, W_out)

    ref_out_nchw = ref_modulated_deform_conv2d(
        x_torch_nchw,
        offset_yx_nchw,
        mask_torch,
        weight_torch,
        bias_torch,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
    )
    # ref shape: (B, C_out, H_out, W_out)

    # Device side: feed NHWC tensors
    x_nhwc = ttnn.from_torch(
        x_torch_nchw.permute(0, 2, 3, 1).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    offset_yx_nhwc = ttnn.from_torch(
        offset_yx_nchw.permute(0, 2, 3, 1).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    mask_nhwc = ttnn.from_torch(
        mask_torch.permute(0, 2, 3, 1).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    op = TtModulatedDeformConv2dDevice(
        weight=weight_torch,
        bias=bias_torch,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        deform_groups=deform_groups,
        device=device,
    )

    out_nhwc = op(x_nhwc, offset_yx_nhwc, mask_nhwc)
    out_torch_nhwc = ttnn.to_torch(out_nhwc)
    # Convert to NCHW for comparison
    dev_out_nchw = out_torch_nhwc.permute(0, 3, 1, 2)

    pcc_required = 0.99
    logger.info(f"shapes: x={x_torch_nchw.shape}, ref_out={ref_out_nchw.shape}, dev_out={dev_out_nchw.shape}")
    assert_with_pcc(ref_out_nchw, dev_out_nchw.to(torch.float32), pcc_required)
