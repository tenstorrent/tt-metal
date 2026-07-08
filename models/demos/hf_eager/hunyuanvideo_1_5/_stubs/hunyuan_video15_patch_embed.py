# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hunyuan_video15_patch_embed` of tencent/HunyuanVideo-1.5.

Reference submodule: `x_embedder`, a `HunyuanVideo15PatchEmbed`:

    hidden = self.proj(hidden_states)              # Conv3d(in, embed, k=patch, stride=patch)
    hidden = hidden.flatten(2).transpose(1, 2)     # (B, embed, F', H', W') -> (B, N, embed)
    return hidden

Because kernel == stride == patch_size, the Conv3d is a *non-overlapping*
patchifier: it is exactly `reshape into patches -> linear`, where the linear is
the conv weight flattened to (in_chans * kt * kh * kw, embed).

Input/output: (B, C, F, H, W) -> (B, F'·H'·W', embed).

Native ttnn strategy
--------------------
patch_size == (1, 1, 1) (this model): no patch gather is needed — the conv is a
per-voxel channel projection. In ttnn: row-major reshape (B,C,F,H,W)->(B,C,N),
permute to (B,N,C), then `matmul + bias` with the flattened conv weight. For a
larger patch the non-overlapping patch gather is a pure data rearrange (done on
host) followed by the SAME native `matmul + bias`. Float32 with a HiFi4 config.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def build(device, torch_module):
    """Bind the Conv3d patch weight as a linear projection; native ttnn forward."""

    proj = torch_module.proj  # nn.Conv3d
    embed = int(proj.out_channels)
    cin = int(proj.in_channels)
    kt, kh, kw = (int(k) for k in proj.kernel_size)
    patch_vol = kt * kh * kw

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Conv weight (embed, cin, kt, kh, kw) flattened per output to (cin*kt*kh*kw)
    # in (c, i, j, k) order -> matmul weight (cin*kt*kh*kw, embed).
    w = proj.weight.detach().reshape(embed, cin * patch_vol)
    Wc = f32(w.t())
    bias = f32(proj.bias.detach().reshape(1, embed)) if proj.bias is not None else None

    is_unit_patch = kt == 1 and kh == 1 and kw == 1

    def forward(hidden_states, *args, **kwargs):
        x = hidden_states
        if isinstance(x, ttnn.Tensor):
            shape = [int(d) for d in x.shape]
        else:
            shape = list(x.shape)
        B, C, F, H, W = shape
        OF, OH, OW = F // kt, H // kh, W // kw
        N = OF * OH * OW

        if is_unit_patch and isinstance(x, ttnn.Tensor):
            # Native path: (B,C,F,H,W) -> (B,C,N) -> (B,N,C), all on device.
            xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            xr = ttnn.reshape(xr, (B, C, N))
            xr = ttnn.permute(xr, (0, 2, 1))  # (B, N, C)
            xr = ttnn.to_layout(xr, ttnn.TILE_LAYOUT)
            if xr.get_dtype() != ttnn.float32:
                xr = ttnn.typecast(xr, ttnn.float32)
        else:
            # General non-overlapping patch gather is a pure data rearrange.
            xt = ttnn.to_torch(x).float() if isinstance(x, ttnn.Tensor) else x.float()
            xt = xt.reshape(B, C, OF, kt, OH, kh, OW, kw)
            xt = xt.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # (B, OF, OH, OW, C, kt, kh, kw)
            xt = xt.reshape(B, N, C * patch_vol)  # (c, i, j, k) order
            xr = ttnn.from_torch(xt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        out = ttnn.matmul(xr, Wc, compute_kernel_config=compute_config)
        if bias is not None:
            out = ttnn.add(out, bias)
        return out

    return forward


def hunyuan_video15_patch_embed(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_patch_embed requires build(device, torch_module) to bind the "
        "Conv3d patch weight; the bare callable has no parameters."
    )
