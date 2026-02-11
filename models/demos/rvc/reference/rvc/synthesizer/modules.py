import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from rvc.synthesizer.commons import fused_add_tanh_sigmoid_multiply, get_padding

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
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.parametrizations.weight_norm(cond_layer, name="weight")

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
            in_layer = torch.nn.utils.parametrizations.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.parametrizations.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g is not None:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers, strict=True)):
            x_in = in_layer(x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)

            res_skip_acts = res_skip_layer(acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d_value,
                        padding=get_padding(kernel_size, d_value),
                    )
                )
                for d_value in dilation
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
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
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d_value,
                        padding=get_padding(kernel_size, d_value),
                    )
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
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        x1 = (x1 - stats) * x_mask
        x = torch.cat([x0, x1], 1)
        return x
