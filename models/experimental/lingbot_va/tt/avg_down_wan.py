# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_dit.layers.module import Module


class TtAvgDown3D(Module):
    """
    Spatial-temporal downsampling via averaging.

    Takes [B, T, H, W, C_in] and produces
    [B, T/factor_t, H/factor_s, W/factor_s, C_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s

        self.factor = factor_t * factor_s * factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Match prior PyTorch path: accumulate mean in float32, return bfloat16.
        x_work = ttnn.typecast(x, ttnn.float32)
        B, T, H, W, C = (int(x_work.shape[i]) for i in range(5))
        x_bcthw = ttnn.permute(x_work, (0, 4, 1, 2, 3))

        pad_t = (self.factor_t - T % self.factor_t) % self.factor_t
        if pad_t > 0:
            pad_zeros = ttnn.zeros(
                (B, C, pad_t, H, W),
                device=x.device(),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            x_bcthw = ttnn.concat([pad_zeros, x_bcthw], dim=2)

        _b, c, tn, h, w = (int(x_bcthw.shape[i]) for i in range(5))
        ft, fs = self.factor_t, self.factor_s
        t1, h1, w1 = tn // ft, h // fs, w // fs

        x_bcthw = ttnn.reshape(x_bcthw, (_b, c, t1, ft, h1, fs, w1, fs))
        x_bcthw = ttnn.permute(x_bcthw, (0, 1, 3, 5, 7, 2, 4, 6))
        x_bcthw = ttnn.reshape(x_bcthw, (_b, c * self.factor, t1, h1, w1))
        x_bcthw = ttnn.reshape(x_bcthw, (_b, self.out_channels, self.group_size, t1, h1, w1))
        x_bcthw = ttnn.mean(x_bcthw, dim=2, keepdim=False)
        x_bthwc = ttnn.permute(x_bcthw, (0, 2, 3, 4, 1))
        return ttnn.typecast(x_bthwc, ttnn.bfloat16)
