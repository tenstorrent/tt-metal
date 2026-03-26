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
        B, T, H, W, C = x.shape

        x_torch = ttnn.to_torch(x)
        x_bcthw = x_torch.permute(0, 4, 1, 2, 3).contiguous()

        x_bcthw = x_bcthw.repeat_interleave(self.repeats, dim=1)

        x_bcthw = x_bcthw.view(
            B,
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            T,
            H,
            W,
        )
        x_bcthw = x_bcthw.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        x_bcthw = x_bcthw.view(
            B,
            self.out_channels,
            T * self.factor_t,
            H * self.factor_s,
            W * self.factor_s,
        )

        if first_chunk:
            x_bcthw = x_bcthw[:, :, self.factor_t - 1 :, :, :]

        x_bthwc = x_bcthw.permute(0, 2, 3, 4, 1).contiguous()

        return ttnn.from_torch(
            x_bthwc,
            dtype=x.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=x.device(),
        )
