# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


@pytest.mark.parametrize("width", [4096, 32768])
@pytest.mark.parametrize("mask_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_scale_mask_softmax_wide_rows_fp32_buffers(device, input_dtype, mask_dtype, width):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, width) * 3
    mask = torch.zeros(1, 1, 32, width)
    mask[..., torch.randperm(width)[: width // 4]] = -1e4
    x = x.float() if input_dtype == ttnn.float32 else x.bfloat16()
    mask = mask.float() if mask_dtype == ttnn.float32 else mask.bfloat16()
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    out = ttnn.scale_mask_softmax(
        ttnn.from_torch(x, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device),
        0.125,
        ttnn.from_torch(mask, dtype=mask_dtype, layout=ttnn.TILE_LAYOUT, device=device),
        compute_kernel_config=config,
    )
    ref = torch.softmax(x.double() * 0.125 + mask.double().repeat(1, 1, 2, 1), dim=-1)
    assert_numeric_metrics(
        ref, ttnn.to_torch(out).double(), pcc_threshold=0.9999, rtol=1e-2, atol=1e-6, frobenius_threshold=1e-2
    )
