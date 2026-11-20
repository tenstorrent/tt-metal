# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""full_inner_dim on height-sharded conv2d: the whole kh x kw window of a pixel forms one inner-dim block (no partial
sum spill/reload between kernel rows). Compares against the sliced (default) path and torch."""
import pytest
import torch
import torch.nn.functional as F

import ttnn


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "cin, cout, h, w, k, stride, pad, dil, groups, abh, out_dtype, fp32, force_split",
    [
        (64, 96, 64, 96, 3, 1, 1, 1, 1, 64, ttnn.bfloat16, True, None),  # stem-like, fp32 acc
        (64, 64, 64, 96, 3, 1, 1, 1, 2, 64, ttnn.bfloat16, True, None),  # grouped
        (96, 96, 32, 48, 3, 2, 1, 1, 1, 32, ttnn.bfloat8_b, False, None),  # stride 2, bfp8 out
        (160, 192, 40, 72, 3, 1, 1, 1, 1, 96, ttnn.bfloat8_b, False, True),  # split reader on, C*kw not a tile multiple
        (40, 64, 32, 32, 3, 1, 1, 1, 1, 32, ttnn.bfloat16, False, None),  # C*kw = 120 -> padded inner dim
        (32, 32, 24, 24, 5, 1, 2, 1, 1, 32, ttnn.bfloat16, False, None),  # 5x5
        (32, 32, 20, 20, 3, 1, 2, 2, 1, 32, ttnn.bfloat16, False, None),  # dilation 2
        (64, 64, 32, 32, 3, 1, 1, 1, 1, 32, ttnn.bfloat16, False, False),  # split reader off
    ],
)
def test_conv2d_full_inner_dim_height_sharded(
    device, cin, cout, h, w, k, stride, pad, dil, groups, abh, out_dtype, fp32, force_split
):
    torch.manual_seed(0)
    x = torch.randn(1, cin, h, w)
    wt = torch.randn(cout, cin // groups, k, k) * (2.0 / (cin // groups * k * k)) ** 0.5
    b = torch.randn(cout) * 0.1
    ref = F.conv2d(x, wt, b, stride=stride, padding=pad, dilation=dil, groups=groups).permute(0, 2, 3, 1)
    xd = ttnn.from_torch(
        x.permute(0, 2, 3, 1).reshape(1, 1, h * w, cin).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32
    )
    outs = {}
    for fid in (False, True):
        cfg = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            act_block_h_override=abh,
            full_inner_dim=fid,
            force_split_reader=force_split,
            deallocate_activation=False,
        )
        o = ttnn.conv2d(
            input_tensor=xd,
            weight_tensor=ttnn.from_torch(wt.to(torch.bfloat16), dtype=ttnn.bfloat16),
            bias_tensor=ttnn.from_torch(b.reshape(1, 1, 1, cout).to(torch.bfloat16), dtype=ttnn.bfloat16),
            device=device,
            in_channels=cin,
            out_channels=cout,
            batch_size=1,
            input_height=h,
            input_width=w,
            kernel_size=[k, k],
            stride=[stride, stride],
            padding=[pad, pad, pad, pad],
            dilation=[dil, dil],
            groups=groups,
            conv_config=cfg,
            compute_config=cc,
            dtype=out_dtype,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )
        outs[fid] = ttnn.to_torch(o).float().reshape(ref.shape)
        ttnn.deallocate(o)
    p_ref = pcc(outs[True], ref)
    p_sliced = pcc(outs[True], outs[False])
    print(f"PCC full_inner_dim vs torch {p_ref:.6f}, vs sliced {p_sliced:.6f}")
    assert p_ref > 0.998 and p_sliced > 0.9999
