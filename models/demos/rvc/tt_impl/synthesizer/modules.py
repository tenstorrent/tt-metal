# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import TTConv1d

LRELU_SLOPE = 0.1


def _conv_output_to_nlc(x: ttnn.Tensor) -> ttnn.Tensor:
    if len(x.shape) == 4:
        batch, _, length, channels = x.shape
        return ttnn.reshape(x, (batch, length, channels))
    return x


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
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        self.beta = ttnn.from_torch(
            parameters[beta_key].reshape(1, 1, self.channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if x.shape[-1] != self.channels and x.shape[1] == self.channels:
            x = ttnn.permute(x, (0, 2, 1))
        if self.gamma is None or self.beta is None:
            raise ValueError("LayerNorm parameters are not loaded.")
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        gamma = ttnn.to_layout(self.gamma, ttnn.TILE_LAYOUT)
        beta = ttnn.to_layout(self.beta, ttnn.TILE_LAYOUT)
        return ttnn.layer_norm(x, weight=gamma, bias=beta, epsilon=self.eps)


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
        self.in_layers: list[TTConv1d] = []
        self.res_skip_layers: list[TTConv1d] = []

        self.cond_layer: TTConv1d | None = None
        if gin_channels != 0:
            self.cond_layer = TTConv1d(
                device=device,
                in_channels=gin_channels,
                out_channels=2 * hidden_channels * n_layers,
                kernel_size=1,
            )

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(
                TTConv1d(
                    device=device,
                    in_channels=hidden_channels,
                    out_channels=2 * hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )

            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(
                TTConv1d(
                    device=device,
                    in_channels=hidden_channels,
                    out_channels=res_skip_channels,
                    kernel_size=1,
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
            g_proj = _conv_output_to_nlc(self.cond_layer(g))

        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers, strict=True)):
            x_in = _conv_output_to_nlc(in_layer(x))
            # if i == 1:
            #     return x_in
            if g_proj is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = ttnn.slice(
                    g_proj,
                    (0, 0, cond_offset),
                    (g_proj.shape[0], g_proj.shape[1], cond_offset + 2 * self.hidden_channels),
                )
            else:
                g_l = ttnn.zeros_like(x_in)

            in_act = x_in + g_l
            t_act = ttnn.slice(in_act, (0, 0, 0), (in_act.shape[0], in_act.shape[1], self.hidden_channels))
            s_act = ttnn.slice(
                in_act,
                (0, 0, self.hidden_channels),
                (in_act.shape[0], in_act.shape[1], 2 * self.hidden_channels),
            )
            acts = ttnn.tanh(t_act) * ttnn.sigmoid(s_act)

            res_skip_acts = _conv_output_to_nlc(res_skip_layer(acts))
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
                x = x + res_acts
                # output = output + skip_acts
                output = ttnn.add(output, skip_acts)  # , output_tensor=output)
            else:
                # output = output + res_skip_acts
                output = ttnn.add(output, res_skip_acts)  # , output_tensor=output)

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
        self.convs1: list[TTConv1d] = []
        self.convs2: list[TTConv1d] = []
        self.lrelu_slope = LRELU_SLOPE
        for d_value in dilation:
            self.convs1.append(
                TTConv1d(
                    device=device,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d_value,
                    padding=int((kernel_size * d_value - d_value) / 2),
                )
            )
            self.convs2.append(
                TTConv1d(
                    device=device,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding=int((kernel_size - 1) / 2),
                )
            )

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        for i, conv in enumerate(self.convs1):
            conv.load_parameters(parameters, key=f"convs1.{i}", prefix=prefix)
        for i, conv in enumerate(self.convs2):
            conv.load_parameters(parameters, key=f"convs2.{i}", prefix=prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt0 = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt1 = _conv_output_to_nlc(c1(xt0))
            xt2 = ttnn.leaky_relu(xt1, negative_slope=self.lrelu_slope)
            xt3 = _conv_output_to_nlc(c2(xt2))
            x = xt3 + x
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
        self.convs: list[TTConv1d] = []
        self.lrelu_slope = LRELU_SLOPE
        for d_value in dilation:
            self.convs.append(
                TTConv1d(
                    device=device,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d_value,
                    padding=int((kernel_size * d_value - d_value) / 2),
                )
            )

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        for i, conv in enumerate(self.convs):
            conv.load_parameters(parameters, key=f"convs.{i}", prefix=prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for conv in self.convs:
            xt = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = _conv_output_to_nlc(conv(xt))
            x = xt + x
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
        self.pre = TTConv1d(
            device=device,
            in_channels=self.half_channels,
            out_channels=hidden_channels,
            kernel_size=1,
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
        self.post = TTConv1d(
            device=device,
            in_channels=hidden_channels,
            out_channels=self.half_channels,
            kernel_size=1,
        )

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        enc_prefix = f"{prefix}enc." if prefix else "enc."
        self.pre.load_parameters(parameters, key="pre", prefix=prefix)
        self.enc.load_parameters(parameters, prefix=enc_prefix)
        self.post.load_parameters(parameters, key="post", prefix=prefix)

    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x0 = ttnn.slice(x, (0, 0, 0), (x.shape[0], x.shape[1], self.half_channels))
        x1 = ttnn.slice(x, (0, 0, self.half_channels), (x.shape[0], x.shape[1], 2 * self.half_channels))
        h = _conv_output_to_nlc(self.pre(x0))
        h = self.enc(h, g=g)
        stats = _conv_output_to_nlc(self.post(h))
        x1 = x1 - stats
        x0 = ttnn.to_layout(x0, ttnn.TILE_LAYOUT)
        x1 = ttnn.to_layout(x1, ttnn.TILE_LAYOUT)
        x_out = ttnn.concat([x0, x1], dim=-1)
        return ttnn.to_layout(x_out, ttnn.ROW_MAJOR_LAYOUT)
