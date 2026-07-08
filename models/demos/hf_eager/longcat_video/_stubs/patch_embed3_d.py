# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn port of `patch_embed3_d` (meituan-longcat/LongCat-Video's
`dit.x_embedder`, class `PatchEmbed3D` in the vendored
`longcat_video/modules/blocks.py`):

    proj = Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    forward(x):                         # x: [B, C, T, H, W]
        x = proj(x)                     # [B, embed_dim, N_t, N_h, N_w]
        return x.flatten(2).transpose(1, 2)   # [B, N, embed_dim]  (flatten=True, norm=None here)

`proj` has `kernel_size == stride == patch_size` -- a non-overlapping
"patchify" convolution, mathematically IDENTICAL to reshaping the input into
per-patch vectors and applying a Linear layer with the conv weight flattened
to `[embed_dim, in_chans*pt*ph*pw]` (not an approximation: for a stride-only
conv, `out[b,co,nt,nh,nw] = sum_{ci,dt,dh,dw} weight[co,ci,dt,dh,dw] *
input[b,ci,nt*pt+dt,nh*ph+dh,nw*pw+dw] + bias[co]` is exactly a dot product
over the flattened patch). The patch EXTRACTION (a pure reshape/permute, no
arithmetic) runs natively on-device via `ttnn.reshape`/`ttnn.permute` (no
host round-trip), and the learned projection runs as a native `ttnn.linear`.
This bring-up's synthetic PCC input is always patch-size-divisible, so the
reference's padding branch is never exercised.
"""

from __future__ import annotations

import ttnn


class TtPatchEmbed3D:
    def __init__(self, device: ttnn.Device, torch_module) -> None:
        self.device = device
        self.dtype = ttnn.bfloat16
        self.patch_size = tuple(torch_module.patch_size)

        conv_w = torch_module.proj.weight  # [embed_dim, in_chans, pt, ph, pw]
        self.w = ttnn.from_torch(
            conv_w.reshape(conv_w.shape[0], -1).transpose(0, 1).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.b = ttnn.from_torch(
            torch_module.proj.bias.reshape(1, -1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        B, C, T, H, W = tuple(x.shape)
        pt, ph, pw = self.patch_size
        assert T % pt == 0 and H % ph == 0 and W % pw == 0
        N_t, N_h, N_w = T // pt, H // ph, W // pw

        patches = ttnn.reshape(x, (B, C, N_t, pt, N_h, ph, N_w, pw))
        patches = ttnn.permute(patches, (0, 2, 4, 6, 1, 3, 5, 7))
        patches = ttnn.reshape(patches, (B, N_t * N_h * N_w, C * pt * ph * pw))

        return ttnn.linear(patches, self.w, bias=self.b)


def build(device: ttnn.Device, torch_module) -> TtPatchEmbed3D:
    return TtPatchEmbed3D(device, torch_module)
