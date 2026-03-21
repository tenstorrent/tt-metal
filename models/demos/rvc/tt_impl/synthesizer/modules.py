# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import Conv1d
from models.demos.rvc.tt_impl.linear import Linear

LRELU_SLOPE = 0.1


class WN:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        gin_channels: int = 0,
    ) -> None:
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.device = device
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.gin_channels = gin_channels
        self.in_layers: list[Conv1d] = []
        self.res_skip_layers: list[Linear] = []

        self.cond_layer: Linear | None = None
        if gin_channels != 0:
            self.cond_layer = Linear(
                device=device,
                in_features=gin_channels,
                out_features=2 * hidden_channels * num_layers,
            )

        for i in range(num_layers):
            dilation = dilation_rate**i
            self.in_layers.append(
                Conv1d(
                    device=device,
                    in_channels=hidden_channels,
                    out_channels=2 * hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding="same",
                )
            )

            res_skip_channels = 2 * hidden_channels if i < num_layers - 1 else hidden_channels
            self.res_skip_layers.append(
                Linear(
                    device=device,
                    in_features=hidden_channels,
                    out_features=res_skip_channels,
                )
            )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if self.cond_layer is not None:
            self.cond_layer.load_state_dict(state_dict, key="cond_layer", module_prefix=module_prefix)
        for i, layer in enumerate(self.in_layers):
            layer.load_state_dict(state_dict, key=f"in_layers.{i}", module_prefix=module_prefix)
        for i, layer in enumerate(self.res_skip_layers):
            layer.load_state_dict(state_dict, key=f"res_skip_layers.{i}", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        out = ttnn.zeros_like(x)
        g_proj = None
        if g is not None:
            if self.cond_layer is None:
                raise ValueError("g is provided but gin_channels is 0.")
            g_proj = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers, strict=True)):
            conv_out = in_layer(x)
            if g_proj is not None:
                cond_offset = i * 2 * self.hidden_channels
                layer_conditioning = ttnn.slice(
                    g_proj,
                    (0, 0, cond_offset),
                    (g_proj.shape[0], g_proj.shape[1], cond_offset + 2 * self.hidden_channels),
                )
            else:
                layer_conditioning = ttnn.zeros_like(conv_out)

            input_activation = ttnn.add(conv_out, layer_conditioning, output_tensor=conv_out)
            t_activation, s_activation = ttnn.chunk(input_activation, 2, dim=-1)
            gates_activations = ttnn.multiply(
                ttnn.sigmoid(s_activation, output_tensor=s_activation),
                ttnn.tanh(t_activation, output_tensor=t_activation),
                output_tensor=s_activation,
            )

            res_skip_out = res_skip_layer(gates_activations)
            if i < self.num_layers - 1:
                residual_out, skip_out = ttnn.chunk(res_skip_out, 2, dim=-1)
                x = ttnn.add(x, residual_out, output_tensor=x)
                out = ttnn.add(out, skip_out, output_tensor=out)
            else:
                out = ttnn.add(out, res_skip_out, output_tensor=out)

        return out


class ResBlock1:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        self.convs1: list[Conv1d] = []
        self.convs2: list[Conv1d] = []
        self.lrelu_slope = LRELU_SLOPE
        for d_value in dilation:
            self.convs1.append(
                Conv1d(
                    device=device,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d_value,
                    padding="same",
                    activation=("leaky_relu", {"negative_slope": self.lrelu_slope}),
                )
            )
            self.convs2.append(
                Conv1d(
                    device=device,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                )
            )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        for i, conv in enumerate(self.convs1):
            conv.load_state_dict(state_dict, key=f"convs1.{i}", module_prefix=module_prefix)
        for i, conv in enumerate(self.convs2):
            conv.load_state_dict(state_dict, key=f"convs2.{i}", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # needed since x is modified in-place in the loop, and we want to keep the original x for the residual connection
        x = ttnn.clone(x)
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            hidden = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope)
            hidden = c1(hidden)
            hidden = c2(hidden)
            x = ttnn.add(hidden, x, output_tensor=x)
        return x


class ResBlock2:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int] = (1, 3),
    ) -> None:
        self.convs: list[Conv1d] = []
        self.lrelu_slope = LRELU_SLOPE
        for d_value in dilation:
            self.convs.append(
                Conv1d(
                    device=device,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d_value,
                    padding="same",
                )
            )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        for i, conv in enumerate(self.convs):
            conv.load_state_dict(state_dict, key=f"convs.{i}", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # needed since x is modified in-place in the loop, and we want to keep the original x for the residual connection
        x = ttnn.clone(x)
        for conv in self.convs:
            xt = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = conv(xt)
            x = ttnn.add(xt, x, output_tensor=x)
        return x


class ResidualCouplingLayer:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        gin_channels: int = 0,
    ) -> None:
        if channels % 2 != 0:
            raise ValueError("channels should be divisible by 2")
        self.half_channels = channels // 2
        self.pre_linear = Linear(
            device=device,
            in_features=self.half_channels,
            out_features=hidden_channels,
        )
        self.enc = WN(
            device=device,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            num_layers=num_layers,
            gin_channels=gin_channels,
        )
        self.post_linear = Linear(
            device=device,
            in_features=hidden_channels,
            out_features=self.half_channels,
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        enc_module_prefix = f"{module_prefix}enc."
        self.pre_linear.load_state_dict(state_dict, key="pre_linear", module_prefix=module_prefix)
        self.enc.load_state_dict(state_dict, module_prefix=enc_module_prefix)
        self.post_linear.load_state_dict(state_dict, key="post_linear", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x0, x1 = ttnn.chunk(x, 2, dim=-1)
        h = self.pre_linear(x0)
        h = self.enc(h, g=g)
        stats = self.post_linear(h)
        x1 = ttnn.subtract(x1, stats, output_tensor=x1)
        out = ttnn.concat([x0, x1], dim=-1)
        return out
