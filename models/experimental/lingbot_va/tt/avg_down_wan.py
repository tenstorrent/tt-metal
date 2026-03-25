import ttnn
from models.tt_dit.layers.module import Module
import torch


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
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s

        self.factor = factor_t * factor_s * factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            x: [B, T, H, W, C]

        Returns:
            [B, T/factor_t, H/factor_s, W/factor_s, out_channels]
        """

        B, T, H, W, C = x.shape

        # Match PyTorch AvgDown3D in float32; bf16 pooling drifts vs diffusers reference.
        x_torch = ttnn.to_torch(x).float()  # [B, T, H, W, C]

        # ---- convert to BCTHW ----
        x_bcthw = x_torch.permute(0, 4, 1, 2, 3).contiguous()

        # ---- temporal padding ----
        pad_t = (self.factor_t - T % self.factor_t) % self.factor_t
        if pad_t > 0:
            x_bcthw = torch.nn.functional.pad(x_bcthw, (0, 0, 0, 0, pad_t, 0))

        B, C, T, H, W = x_bcthw.shape

        # ---- expose pooling factors ----
        x_bcthw = x_bcthw.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )

        # ---- move pooling dims near channel ----
        x_bcthw = x_bcthw.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()

        # [B, C, ft, fs, fs, T', H', W']

        # ---- merge pooling dims into channels ----
        x_bcthw = x_bcthw.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )

        # ---- split channels for averaging ----
        x_bcthw = x_bcthw.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )

        # ---- average ----
        x_bcthw = x_bcthw.mean(dim=2)

        # ---- convert back to BTHWC ----
        x_bthwc = x_bcthw.permute(0, 2, 3, 4, 1).to(torch.bfloat16).contiguous()

        return ttnn.from_torch(
            x_bthwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=x.device(),
        )
