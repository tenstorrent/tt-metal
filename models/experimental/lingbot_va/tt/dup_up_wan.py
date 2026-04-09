# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.tt_dit.layers.module import Module


class TtDupUp3D(Module):
    """Channel-to-spatial upsampling via duplication (no learned parameters).

    Inverse of TtAvgDown3D. Takes [B, T, H, W, C_in] and produces
    [B, T*factor_t, H*factor_s, W*factor_s, C_out] by repeating
    channel values and distributing them across spatial positions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: ttnn.Tensor, first_chunk: bool = False) -> ttnn.Tensor:
        """
        Args:
            x: [B, T, H, W, C] in BTHWC layout.
            first_chunk: If True, trim the first temporal slot from the output
                         (for causal streaming alignment).
        Returns:
            [B, T*factor_t, H*factor_s, W*factor_s, out_channels]
            (with temporal trim if first_chunk).
        """
        B, T, H, W = (int(x.shape[i]) for i in range(4))
        ft, fs = self.factor_t, self.factor_s
        oc = self.out_channels

        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_bcthw = ttnn.permute(x_rm, (0, 4, 1, 2, 3))
        x_bcthw = ttnn.repeat_interleave(x_bcthw, self.repeats, dim=1)

        x_bcthw = ttnn.reshape(x_bcthw, (B, oc, ft, fs, fs, T, H, W))
        x_bcthw = ttnn.permute(x_bcthw, (0, 1, 5, 2, 6, 3, 7, 4))
        x_bcthw = ttnn.reshape(x_bcthw, (B, oc, T * ft, H * fs, W * fs))

        if first_chunk:
            _, _, t_long, h_long, w_long = (int(x_bcthw.shape[i]) for i in range(5))
            x_bcthw = ttnn.slice(
                x_bcthw,
                [0, 0, ft - 1, 0, 0],
                [B, oc, t_long, h_long, w_long],
            )

        return ttnn.permute(x_bcthw, (0, 2, 3, 4, 1))
