"""TT-Metal (ttnn) implementation of DUSt3R layers.

Each function takes a device + torch weights/inputs and returns a torch tensor
on host so the test harness can compute PCC against the reference.
"""
from __future__ import annotations

import torch
import ttnn


def _to_dev(t: torch.Tensor, device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def patch_embed(img: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device):
    """Patch embedding via ttnn.conv2d with 16x16 stride.

    img: (B, 3, H, W)      weight: (1024, 3, 16, 16)      bias: (1024,)
    returns (B, N, 1024) on host.
    """
    B, C, H, W = img.shape
    out_H = H // 16
    out_W = W // 16

    # ttnn.conv2d wants input in NHWC, flattened to (1, 1, B*H*W, C) row major
    img_nhwc = img.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
    tt_img = ttnn.from_torch(
        img_nhwc.reshape(1, 1, B * H * W, C),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_w = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
    tt_b = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)

    out = ttnn.conv2d(
        input_tensor=tt_img,
        weight_tensor=tt_w,
        bias_tensor=tt_b,
        in_channels=C,
        out_channels=weight.shape[0],
        device=device,
        kernel_size=(16, 16),
        stride=(16, 16),
        padding=(0, 0),
        batch_size=B,
        input_height=H,
        input_width=W,
    )

    out_torch = ttnn.to_torch(out)  # shape (1, 1, B*out_H*out_W, E)
    out_torch = out_torch.reshape(B, out_H * out_W, -1)
    return out_torch
