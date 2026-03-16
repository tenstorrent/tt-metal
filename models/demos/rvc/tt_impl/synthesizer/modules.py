# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import Conv1d
from models.demos.rvc.tt_impl.linear import Linear

LRELU_SLOPE = 0.1


class LayerNorm:
    def __init__(self, device: ttnn.MeshDevice, channels: int, eps: float = 1e-5) -> None:
        self.device = device
        self.channels = channels
        self.eps = eps
        self.gamma: ttnn.Tensor | None = None
        self.beta: ttnn.Tensor | None = None

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        gamma_key = f"{prefix}gamma" if prefix else "gamma"
        beta_key = f"{prefix}beta" if prefix else "beta"
        if gamma_key not in parameters:
            raise KeyError(f"Missing required parameter: {gamma_key}")
        if beta_key not in parameters:
            raise KeyError(f"Missing required parameter: {beta_key}")
        self.gamma = ttnn.from_torch(
            parameters[gamma_key].reshape(1, 1, self.channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.beta = ttnn.from_torch(
            parameters[beta_key].reshape(1, 1, self.channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(x, weight=self.gamma, bias=self.beta, epsilon=self.eps)


class WN:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        conv_config: ttnn.Conv1dConfig | None = None,
        compute_config: ttnn.DeviceComputeKernelConfig | None = None,
    ) -> None:
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.device = device
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.in_layers: list[Conv1d] = []
        self.res_skip_layers: list[Linear] = []

        self.cond_layer: Linear | None = None
        if gin_channels != 0:
            self.cond_layer = Linear(
                device=device,
                in_features=gin_channels,
                out_features=2 * hidden_channels * n_layers,
            )

        for i in range(n_layers):
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

            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(
                Linear(
                    device=device,
                    in_features=hidden_channels,
                    out_features=res_skip_channels,
                )
            )

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        if self.cond_layer is not None:
            self.cond_layer.load_parameters(parameters, key="cond_layer", prefix=prefix)
        for i, layer in enumerate(self.in_layers):
            layer.load_parameters(parameters, key=f"in_layers.{i}", prefix=prefix)
        for i, layer in enumerate(self.res_skip_layers):
            layer.load_parameters(parameters, key=f"res_skip_layers.{i}", prefix=prefix)

    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        output = ttnn.zeros_like(x)
        g_proj = None
        if g is not None:
            if self.cond_layer is None:
                raise ValueError("g is provided but gin_channels is 0.")
            g_proj = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers, strict=True)):
            x_in = in_layer(x)
            if g_proj is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = ttnn.slice(
                    g_proj,
                    (0, 0, cond_offset),
                    (g_proj.shape[0], g_proj.shape[1], cond_offset + 2 * self.hidden_channels),
                )
            else:
                g_l = ttnn.zeros_like(x_in)

            in_act = ttnn.add(x_in, g_l, output_tensor=x_in)
            t_act = ttnn.slice(in_act, (0, 0, 0), (in_act.shape[0], in_act.shape[1], self.hidden_channels))
            s_act = ttnn.slice(
                in_act,
                (0, 0, self.hidden_channels),
                (in_act.shape[0], in_act.shape[1], 2 * self.hidden_channels),
            )
            acts = ttnn.multiply(
                ttnn.sigmoid(s_act, output_tensor=s_act), ttnn.tanh(t_act, output_tensor=t_act), output_tensor=s_act
            )

            res_skip_acts = res_skip_layer(acts)
            if i < self.n_layers - 1:
                res_acts = ttnn.slice(
                    res_skip_acts,
                    (0, 0, 0),
                    (res_skip_acts.shape[0], res_skip_acts.shape[1], self.hidden_channels),
                )
                skip_acts = ttnn.slice(
                    res_skip_acts,
                    (0, 0, self.hidden_channels),
                    (res_skip_acts.shape[0], res_skip_acts.shape[1], 2 * self.hidden_channels),
                )
                x = ttnn.add(x, res_acts, output_tensor=x)
                output = ttnn.add(output, skip_acts, output_tensor=output)
            else:
                output = ttnn.add(output, res_skip_acts, output_tensor=output)

        return output


class ResBlock1:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
        conv_config: ttnn.Conv1dConfig | None = None,
        compute_config: ttnn.DeviceComputeKernelConfig | None = None,
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

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        for i, conv in enumerate(self.convs1):
            conv.load_parameters(parameters, key=f"convs1.{i}", prefix=prefix)
        for i, conv in enumerate(self.convs2):
            conv.load_parameters(parameters, key=f"convs2.{i}", prefix=prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # needed since x is modified in-place in the loop, and we want to keep the original x for the residual connection
        x = ttnn.clone(x)
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt0 = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt1 = c1(xt0)
            xt2 = c2(xt1)
            x = ttnn.add(xt2, x, output_tensor=x)
        return x


class ResBlock2:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int] = (1, 3),
        conv_config: ttnn.Conv1dConfig | None = None,
        compute_config: ttnn.DeviceComputeKernelConfig | None = None,
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

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        for i, conv in enumerate(self.convs):
            conv.load_parameters(parameters, key=f"convs.{i}", prefix=prefix)

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
        n_layers: int,
        gin_channels: int = 0,
        conv_config: ttnn.Conv1dConfig | None = None,
        compute_config: ttnn.DeviceComputeKernelConfig | None = None,
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
            n_layers=n_layers,
            gin_channels=gin_channels,
            conv_config=conv_config,
            compute_config=compute_config,
        )
        self.post_linear = Linear(
            device=device,
            in_features=hidden_channels,
            out_features=self.half_channels,
        )

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        enc_prefix = f"{prefix}enc." if prefix else "enc."
        pre_key = (
            "pre_linear" if (f"{prefix}pre_linear.weight" if prefix else "pre_linear.weight") in parameters else "pre"
        )
        post_key = (
            "post_linear"
            if (f"{prefix}post_linear.weight" if prefix else "post_linear.weight") in parameters
            else "post"
        )
        self.pre_linear.load_parameters(parameters, key=pre_key, prefix=prefix)
        self.enc.load_parameters(parameters, prefix=enc_prefix)
        self.post_linear.load_parameters(parameters, key=post_key, prefix=prefix)

    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x0 = ttnn.slice(x, (0, 0, 0), (x.shape[0], x.shape[1], self.half_channels))
        x1 = ttnn.slice(x, (0, 0, self.half_channels), (x.shape[0], x.shape[1], 2 * self.half_channels))
        h = self.pre_linear(x0)
        h = self.enc(h, g=g)
        stats = self.post_linear(h)
        x1 = ttnn.subtract(x1, stats, output_tensor=x1)
        out = ttnn.concat([x0, x1], dim=-1)
        return out
