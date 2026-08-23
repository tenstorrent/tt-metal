# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from ...layers.linear import Linear
from ...layers.module import Module, ModuleList
from .conv1d_simple_ttnn import Conv1dSimpleTTNN


class _ELU(Module):
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.elu(x)


class ConvRNNF0Predictor(Module):
    """TTNN port of CosyVoice2 ConvRNNF0Predictor.

    Device convention:
        input:  [B, T, 80]  ROW_MAJOR
        output: [B, T, 1]

    The reference PyTorch model uses [B, 80, T] internally and squeezes
    the final channel. Keeping the singleton channel here is convenient
    for the following TTNN audio pipeline.
    """

    def __init__(
        self,
        num_class: int = 1,
        in_channels: int = 80,
        cond_channels: int = 512,
        *,
        mesh_device,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        self.num_class = num_class
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.mesh_device = mesh_device
        self.dtype = dtype

        layers = []

        channels = [
            (in_channels, cond_channels),
            (cond_channels, cond_channels),
            (cond_channels, cond_channels),
            (cond_channels, cond_channels),
            (cond_channels, cond_channels),
        ]

        for cin, cout in channels:
            layers.append(
                Conv1dSimpleTTNN(
                    cin,
                    cout,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
            )
            layers.append(_ELU())

        # Indices intentionally match PyTorch Sequential:
        # convs = 0,2,4,6,8; ELUs = 1,3,5,7,9.
        self.condnet = ModuleList(layers)

        self.classifier = Linear(
            cond_channels,
            num_class,
            bias=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )

    @staticmethod
    def _materialize_weight_norm(
        g: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize weight_norm(g, v) with dim=0."""

        vf = v.float()
        gf = g.float()

        dims = tuple(range(1, vf.ndim))
        norm = torch.linalg.vector_norm(
            vf,
            dim=dims,
            keepdim=True,
        )

        weight = vf * (gf / norm.clamp_min(torch.finfo(torch.float32).tiny))

        return weight.to(v.dtype)

    def _prepare_torch_state(
        self,
        state: dict[str, torch.Tensor],
    ) -> None:
        # CosyVoice uses torch weight_norm around each Conv1d.
        #
        # Support both:
        #   new parametrizations.weight.original{0,1}
        # and:
        #   old weight_g / weight_v
        for index in (0, 2, 4, 6, 8):
            prefix = f"condnet.{index}."

            weight_key = prefix + "weight"

            if weight_key in state:
                continue

            new_g = prefix + "parametrizations.weight.original0"
            new_v = prefix + "parametrizations.weight.original1"

            old_g = prefix + "weight_g"
            old_v = prefix + "weight_v"

            if new_g in state and new_v in state:
                g = state.pop(new_g)
                v = state.pop(new_v)

                state[weight_key] = self._materialize_weight_norm(
                    g,
                    v,
                )

            elif old_g in state and old_v in state:
                g = state.pop(old_g)
                v = state.pop(old_v)

                state[weight_key] = self._materialize_weight_norm(
                    g,
                    v,
                )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        if x_BTC.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(
                x_BTC,
                ttnn.ROW_MAJOR_LAYOUT,
            )
        else:
            x = x_BTC

        for i in range(len(self.condnet)):
            x = self.condnet[i](x)

        # Conv path is ROW_MAJOR; Linear uses TILE.
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(
                x,
                ttnn.TILE_LAYOUT,
            )

        x = self.classifier(x)
        x = ttnn.abs(x)

        return x
