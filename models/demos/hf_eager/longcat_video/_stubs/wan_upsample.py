# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `wan_upsample`
(meituan-longcat/LongCat-Video's
`vae.decoder.up_blocks.0.upsamplers.0.resample.0`, a real
`diffusers.models.autoencoders.autoencoder_kl_wan.WanUpsample` --
`nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact")`, computed in
fp32 then cast back to the input dtype).

Stateless (no weight/bias): per the bring-up gate's TP principles this is
a pure elementwise/spatial op that needs no weight split and no
collective -- every mesh chip runs the identical `ttnn.upsample` on its
own (replicated) input, matching the graduated `wan_resample`'s use of
the SAME `ttnn.upsample(..., scale_factor=2)` call for its spatial
upsample stage (see `models/tt_dit/models/vae/vae_wan2_1.py`'s
`WanResample.forward`). For an exact integer 2x scale, PyTorch's
"nearest" and "nearest-exact" modes are mathematically identical (both
just replicate each source pixel into a 2x2 output block), so
`ttnn.upsample`'s default `mode="nearest"` reproduces this module's
`mode="nearest-exact"` exactly.
"""

from __future__ import annotations

import ttnn


class TtWanUpsample:
    def __init__(self, device, torch_module) -> None:
        self.device = device

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: NCHW (torch.nn.Upsample's native convention); ttnn.upsample
        # expects NHWC, ROW_MAJOR.
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_nhwc = ttnn.permute(x, (0, 2, 3, 1))
        out_nhwc = ttnn.upsample(x_nhwc, 2)
        return ttnn.permute(out_nhwc, (0, 3, 1, 2))


def build(device, torch_module) -> TtWanUpsample:
    return TtWanUpsample(device, torch_module)
