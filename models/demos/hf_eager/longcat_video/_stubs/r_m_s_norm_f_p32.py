# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn port of `r_m_s_norm_f_p32` (meituan-longcat/LongCat-Video's
`dit.blocks.*.attn.q_norm`/`k_norm`, class `RMSNorm_FP32` in the vendored
`longcat_video/modules/blocks.py`):

    class RMSNorm_FP32(nn.Module):
        def __init__(self, dim, eps):
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            output = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).type_as(x)
            return output * self.weight

Standard learnable RMSNorm over the last dim -- exactly `ttnn.rms_norm`.
"""

from __future__ import annotations

import ttnn


class TtRMSNormFP32:
    def __init__(self, device: ttnn.Device, torch_module) -> None:
        self.device = device
        self.eps = torch_module.eps
        self.weight = ttnn.from_torch(
            torch_module.weight.reshape(1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)


def build(device: ttnn.Device, torch_module) -> TtRMSNormFP32:
    return TtRMSNormFP32(device, torch_module)
