from collections.abc import Sequence

import ttnn

from ...layers.audio_ops import _AlignedOutConv1d
from ...layers.module import Module, ModuleList
from .snake_ttnn import Snake


class HiFTConv1d(_AlignedOutConv1d):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        causal: bool,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding_mode="causal" if causal else "zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )


class HiFTResBlock(Module):
    """HiFT / HiFiGAN residual block.

    Input/output layout is BTC to match TTNN Conv1d.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 3, 5),
        causal: bool = False,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        self.num_layers = len(dilations)

        self.convs1 = ModuleList(
            [
                HiFTConv1d(
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    causal=causal,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
                for dilation in dilations
            ]
        )

        self.convs2 = ModuleList(
            [
                HiFTConv1d(
                    channels,
                    kernel_size=kernel_size,
                    dilation=1,
                    causal=causal,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
                for _ in dilations
            ]
        )

        self.activations1 = ModuleList(
            [
                Snake(
                    channels=channels,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
                for _ in dilations
            ]
        )

        self.activations2 = ModuleList(
            [
                Snake(
                    channels=channels,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
                for _ in dilations
            ]
        )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        for i in range(self.num_layers):
            xt = self.activations1[i](x_BTC)

            nxt = self.convs1[i](xt)
            ttnn.deallocate(xt)

            xt = self.activations2[i](nxt)
            ttnn.deallocate(nxt)

            nxt = self.convs2[i](xt)
            ttnn.deallocate(xt)

            x_new = ttnn.add(x_BTC, nxt)
            ttnn.deallocate(nxt)

            # Don't deallocate the caller-owned original input.
            if i > 0:
                ttnn.deallocate(x_BTC)

            x_BTC = x_new

        return x_BTC
