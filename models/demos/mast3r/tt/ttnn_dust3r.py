"""TT-Metal (ttnn) implementation of DUSt3R layers.

Each function takes a device + torch weights/inputs and returns a torch tensor
on host so the test harness can compute PCC against the reference.
"""
from __future__ import annotations

import torch
import ttnn


def patch_embed(img: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device):
    """Patch embedding via im2col + matmul (equivalent to stride=kernel Conv2d).

    img: (B, 3, H, W)      weight: (1024, 3, 16, 16)      bias: (1024,)
    returns (B, N, 1024) on host, where N = (H/16) * (W/16).
    """
    B, C, H, W = img.shape
    p = 16
    hp, wp = H // p, W // p
    N = hp * wp

    # im2col on host: (B, C, H, W) -> (B, N, C*p*p)
    patches = img.unfold(2, p, p).unfold(3, p, p)  # (B, C, hp, wp, p, p)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, hp, wp, C, p, p)
    patches = patches.reshape(B, N, C * p * p)

    # Flatten weight to match: Conv2d weight (E, C, p, p) -> (C*p*p, E)
    w_flat = weight.reshape(weight.shape[0], -1).t().contiguous()  # (C*p*p, E)

    tt_patches = ttnn.from_torch(patches, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_w = ttnn.from_torch(w_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(bias.reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.matmul(tt_patches, tt_w)
    out = ttnn.add(out, tt_b)

    out_torch = ttnn.to_torch(out)  # (B, N, E)
    return out_torch
