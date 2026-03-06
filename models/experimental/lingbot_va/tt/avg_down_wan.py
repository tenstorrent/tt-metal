import ttnn
from models.tt_dit.layers.module import Module


class TtAvgDown3D(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
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
        x shape: [B, T, H, W, C]
        """
        B, T, H, W, C = x.shape
        pad_t = (self.factor_t - T % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = ttnn.pad(x, [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)], 0.0)
        # Reshape to expose pooling factors
        B, T, H, W, C = x.shape
        x = ttnn.reshape(
            x,
            (
                B,
                T // self.factor_t,
                self.factor_t,
                H // self.factor_s,
                self.factor_s,
                W // self.factor_s,
                self.factor_s,
                C,
            ),
        )
        x = ttnn.permute(x, (0, 1, 3, 5, 7, 2, 4, 6))
        # Merge channel factors
        x = ttnn.reshape(x, (B, T // self.factor_t, H // self.factor_s, W // self.factor_s, C * self.factor))
        # Group channels for averaging
        x = ttnn.reshape(
            x, (B, T // self.factor_t, H // self.factor_s, W // self.factor_s, self.out_channels, self.group_size)
        )
        # Average over grouped channels
        x = ttnn.mean(x, dim=5)
        return x
