# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `wan_r_m_s`
(meituan-longcat/LongCat-Video's `vae.encoder.down_blocks.0.norm1`, a real
`diffusers.models.autoencoders.autoencoder_kl_wan.WanRMS_norm`).

`WanRMS_norm.forward` is `F.normalize(x, dim=1 if channel_first else -1) *
scale * gamma + bias`, where `scale = dim**0.5` (a constant fixed at
construction from the channel count) and `gamma`/`bias` are per-channel
parameters of shape `(dim, 1, 1)`/`(dim, 1, 1, 1)`. Per the bring-up gate's
TP principles, norms are elementwise/parameter-only ops that shard in no
scheme -- they stay REPLICATED: every mesh chip runs the identical forward
on its (replicated) input using the identical (host-resident) gamma/bias
values, no weight split and no collective. The sharded PCC harness gathers
via `ConcatMeshToTensor(dim=0)` and, since every chip produced the same
answer, trims back to one copy -- exactly recovering this replicated
result. This is why the bring-up gate routes this component straight to
`shard` with no single-device rung: a plain `ttnn.Device` and a
`ttnn.MeshDevice` run the exact same code path here.

`gamma`/`bias` are tiny per-channel scale factors; they are read to host
once at `build()` time (not used to drive any math on host -- the norm and
scale-multiply below run entirely as ttnn ops on the input's own device)
and applied as ttnn scalar operands, which sidesteps the odd `(dim, 1, 1,
1)` parameter shape (last two dims of size 1) that would otherwise need
ROW_MAJOR bookkeeping to avoid the TILE_LAYOUT 32x32 minimum.
"""

from __future__ import annotations

import torch

import ttnn


class TtWanRMSNorm:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.channel_first = bool(torch_module.channel_first)
        self.scale = float(torch_module.scale)

        gamma = torch_module.gamma.detach().to(torch.float32).reshape(-1)
        self.gamma_vals = gamma.tolist()
        self.dim = len(self.gamma_vals)

        bias = torch_module.bias
        self.bias_vals = (
            bias.detach().to(torch.float32).reshape(-1).tolist() if isinstance(bias, torch.Tensor) else None
        )

    def __call__(self, x: ttnn.Tensor, *args, **kwargs) -> ttnn.Tensor:
        shape = tuple(x.shape)
        assert len(shape) == 3, f"wan_r_m_s (native) expects a rank-3 (B, N, C) activation, got shape {shape}"
        b, n, c = shape

        # Canonicalize to an explicit 4D (N, C, H, W)-style tensor so the
        # reduction/broadcast below lands on ttnn's native bcast_h / bcast_w
        # elementwise-binary support (no data movement -- only a leading
        # size-1 axis is inserted; the tiled (n, c) trailing dims are
        # untouched).
        x4 = ttnn.reshape(x, (b, 1, n, c))
        reduce_dim = 2 if self.channel_first else 3  # H-slot (n) or W-slot (c)

        sumsq = ttnn.sum(ttnn.multiply(x4, x4), dim=reduce_dim, keepdim=True)
        sumsq = ttnn.add(sumsq, 1e-12)  # matches F.normalize's division-by-zero guard
        inv_norm = ttnn.rsqrt(sumsq)
        scaled = ttnn.multiply(ttnn.multiply(x4, inv_norm), self.scale)

        channels = []
        for i in range(self.dim):
            ch = ttnn.multiply(scaled, self.gamma_vals[i])
            if self.bias_vals is not None:
                ch = ttnn.add(ch, self.bias_vals[i])
            channels.append(ch)
        return channels[0] if self.dim == 1 else ttnn.concat(channels, dim=1)


def build(device, torch_module) -> TtWanRMSNorm:
    return TtWanRMSNorm(device, torch_module)
