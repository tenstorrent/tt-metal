# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import functional as F

from models.demos.rvc.torch_impl.utils import linear_channel_first


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts


LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()

        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        if gin_channels != 0:
            self.cond_layer = nn.Linear(gin_channels, 2 * hidden_channels * n_layers)

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_linear = nn.Linear(hidden_channels, res_skip_channels)
            self.res_skip_layers.append(res_skip_linear)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None):
        output = torch.zeros_like(x)
        # n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g is not None:
            g = linear_channel_first(g, self.cond_layer)

        for i, (in_layer, res_skip_linear) in enumerate(zip(self.in_layers, self.res_skip_layers, strict=True)):
            x_in = in_layer(x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            res_skip_acts = linear_channel_first(acts, res_skip_linear)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = x + res_acts
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d_value,
                    padding=int((kernel_size * d_value - d_value) / 2),
                )
                for d_value in dilation
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=int((kernel_size * 1 - 1) / 2))
                for _ in dilation
            ]
        )
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x: torch.Tensor):
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d_value,
                    padding=int((kernel_size * d_value - d_value) / 2),
                )
                for d_value in dilation
            ]
        )
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c(xt)
            x = xt + x
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.pre_linear = nn.Linear(self.half_channels, hidden_channels)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.post_linear = nn.Linear(hidden_channels, self.half_channels)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = linear_channel_first(x0, self.pre_linear)
        h = self.enc(h, g=g)
        stats = linear_channel_first(h, self.post_linear)
        x1 = x1 - stats
        x = torch.cat([x0, x1], 1)
        return x
