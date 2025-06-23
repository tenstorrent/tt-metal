# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn
import torch.nn.functional as f


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DFL(nn.Module):
    def __init__(self):
        super(DFL, self).__init__()
        self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Conv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        enable_act=True,
        bias=False,
        enable_autopad=True,
    ):
        super().__init__()
        self.enable_act = enable_act
        if enable_act:
            self.conv = nn.Conv2d(
                in_channel,
                out_channel,
                kernel,
                stride=stride,
                padding=autopad(k=kernel, p=None, d=dilation) if enable_autopad else padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            self.bn = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.03)
            self.act = nn.SiLU(inplace=True)
        else:
            self.conv = nn.Conv2d(
                in_channel,
                out_channel,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            self.bn = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.03)

    def forward(self, x):
        if self.enable_act:
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
        return x


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=[1, 1]
    ):
        super().__init__()
        self.cv1 = Conv(
            in_channel[0],
            out_channel[0],
            kernel[0],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
            groups=groups[0],
        )
        self.cv2 = Conv(
            in_channel[1],
            out_channel[1],
            kernel[1],
            stride=stride[1],
            padding=padding[1],
            dilation=dilation[1],
            groups=groups[1],
        )

    def forward(self, x):
        input = x
        x = self.cv1(x)
        x = self.cv2(x)
        return input + x


class C3k(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation, groups):
        super().__init__()
        self.cv1 = Conv(
            in_channel[0],
            out_channel[0],
            kernel[0],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
            groups=groups[0],
        )
        self.cv2 = Conv(
            in_channel[1],
            out_channel[1],
            kernel[1],
            stride=stride[1],
            padding=padding[1],
            dilation=dilation[1],
            groups=groups[1],
        )
        self.cv3 = Conv(
            in_channel[2],
            out_channel[2],
            kernel[2],
            stride=stride[2],
            padding=padding[2],
            dilation=dilation[2],
            groups=groups[2],
        )
        self.m = nn.Sequential(
            Bottleneck(
                in_channel[3:5],
                out_channel[3:5],
                kernel[3:5],
                stride=stride[3:5],
                padding=padding[3:5],
                dilation=dilation[3:5],
                groups=groups[3:5],
            ),
            Bottleneck(
                in_channel[5:7],
                out_channel[5:7],
                kernel[5:7],
                stride=stride[5:7],
                padding=padding[5:7],
                dilation=dilation[5:7],
                groups=groups[5:7],
            ),
        )

    def forward(self, x, i=0, j=0):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x = self.m(x1)
        x = torch.cat((x, x2), 1)
        x = self.cv3(x)
        return x


class C3k2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation, groups, is_bk_enabled=False):
        super().__init__()
        self.is_bk_enabled = is_bk_enabled
        if is_bk_enabled:
            self.cv1 = Conv(
                in_channel[0],
                out_channel[0],
                kernel[0],
                stride=stride[0],
                padding=padding[0],
                dilation=dilation[0],
                groups=groups[0],
            )
            self.cv2 = Conv(
                in_channel[1],
                out_channel[1],
                kernel[1],
                stride=stride[1],
                padding=padding[1],
                dilation=dilation[1],
                groups=groups[1],
            )
            self.m = nn.ModuleList(
                [
                    Bottleneck(
                        in_channel[2:4],
                        out_channel[2:4],
                        kernel[2:4],
                        stride=stride[2:4],
                        padding=padding[2:4],
                        dilation=dilation[2:4],
                        groups=groups[2:4],
                    ),
                ]
            )

        else:
            self.cv1 = Conv(
                in_channel[0],
                out_channel[0],
                kernel[0],
                stride=stride[0],
                padding=padding[0],
                dilation=dilation[0],
                groups=groups[0],
            )
            self.cv2 = Conv(
                in_channel[1],
                out_channel[1],
                kernel[1],
                stride=stride[1],
                padding=padding[1],
                dilation=dilation[1],
                groups=groups[1],
            )
            self.m = nn.ModuleList(
                [
                    C3k(
                        in_channel[2:9],
                        out_channel[2:9],
                        kernel[2:9],
                        stride[2:9],
                        padding[2:9],
                        dilation[2:9],
                        groups[2:9],
                    ),
                    C3k(
                        in_channel[2:9],
                        out_channel[2:9],
                        kernel[2:9],
                        stride[2:9],
                        padding[2:9],
                        dilation[2:9],
                        groups[2:9],
                    ),
                ]
            )

    def forward(self, x, i=0):
        if self.is_bk_enabled:
            x = self.cv1(x)
            y = list(x.chunk(2, 1))
            y.extend(self.m[0](y[-1]))
            y[-1] = y[-1].unsqueeze(0)
            x = torch.cat(y, 1)
            x = self.cv2(x)
        else:
            x = self.cv1(x)

            y = list(x.chunk(2, 1))
            for j, m in enumerate(self.m):
                y.extend(m(y[-1], i=i, j=j))
                y[-1] = y[-1].unsqueeze(0)

            x = torch.cat(y, 1)
            x = self.cv2(x)
        return x


class AAttn(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel, stride, padding, dilation, groups, dim=384, num_heads=8, area=1
    ):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * self.num_heads

        self.qkv = Conv(
            in_channel[0],
            out_channel[0],
            kernel[0],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
            groups=groups[0],
            enable_act=False,
        )
        self.proj = Conv(
            in_channel[1],
            out_channel[1],
            kernel[1],
            stride=stride[1],
            padding=padding[1],
            dilation=dilation[1],
            groups=groups[1],
            enable_act=False,
        )
        self.pe = Conv(
            in_channel[2],
            out_channel[2],
            kernel[2],
            stride=stride[2],
            padding=padding[2],
            dilation=dilation[2],
            groups=groups[2],
            enable_act=False,
            bias=True,
        )

    def forward(self, x, i=0, j=0):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape

        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        y = self.pe(v)
        x = x + y
        x = self.proj(x)
        return x


class ABlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel,
        stride,
        padding,
        dilation,
        groups,
        dim=384,
        num_heads=12,
        mlp_ratio=1.2,
        area=1,
    ):
        super().__init__()
        dim, num_heads = 384, 12
        mlp_ratio = 1.2
        self.attn = AAttn(
            in_channel[0:3],
            out_channel[0:3],
            kernel[0:3],
            stride[0:3],
            padding[0:3],
            dilation[0:3],
            groups[0:3],
            dim=dim,
            num_heads=num_heads,
            area=area,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(
                in_channel[3],
                out_channel[3],
                kernel[3],
                stride=stride[3],
                padding=padding[3],
                dilation=dilation[3],
                groups=groups[3],
                enable_act=True,
            ),
            Conv(
                in_channel[4],
                out_channel[4],
                kernel[4],
                stride=stride[4],
                padding=padding[4],
                dilation=dilation[4],
                groups=groups[4],
                enable_act=False,
            ),
        )

    def forward(self, x, i=0, j=0):
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel,
        stride,
        padding,
        dilation,
        groups,
        c1,
        c2,
        n=1,
        a2=True,
        area=1,
        residual=False,
        mlp_ratio=2.0,
        e=0.5,
        g=1,
        shortcut=True,
    ):
        super().__init__()
        self.cv1 = Conv(
            in_channel[0],
            out_channel[0],
            kernel[0],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
            groups=groups[0],
        )
        self.cv2 = Conv(
            in_channel[1],
            out_channel[1],
            kernel[1],
            stride=stride[1],
            padding=padding[1],
            dilation=dilation[1],
            groups=groups[1],
        )

        residual = True
        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            (
                nn.Sequential(
                    *(
                        ABlock(
                            in_channel[2:7],
                            out_channel[2:7],
                            kernel[2:7],
                            stride[2:7],
                            padding[2:7],
                            dilation[2:7],
                            groups[2:7],
                            dim=384,
                            num_heads=12,
                            mlp_ratio=1.2,
                            area=area,
                        )
                        for _ in range(2)
                    )
                )
                if a2
                else C3k(
                    in_channel[2:10],
                    out_channel[2:9],
                    kernel[2:9],
                    stride[2:9],
                    padding[2:9],
                    dilation[2:9],
                    groups[2:9],
                )
            )
            for _ in range(n)
        )

    def forward(self, x, i=0):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


class Detect(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation, groups):
        super().__init__()
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.cv2 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(
                        in_channel[0],
                        out_channel[0],
                        kernel[0],
                        stride=stride[0],
                        padding=padding[0],
                        dilation=dilation[0],
                        groups=groups[0],
                    ),
                    Conv(
                        in_channel[1],
                        out_channel[1],
                        kernel[1],
                        stride=stride[1],
                        padding=padding[1],
                        dilation=dilation[1],
                        groups=groups[1],
                    ),
                    nn.Conv2d(
                        in_channel[2],
                        out_channel[2],
                        kernel[2],
                        stride=stride[2],
                        padding=padding[2],
                        dilation=dilation[2],
                        groups=groups[2],
                    ),
                ),
                nn.Sequential(
                    Conv(
                        in_channel[3],
                        out_channel[3],
                        kernel[3],
                        stride=stride[3],
                        padding=padding[3],
                        dilation=dilation[3],
                        groups=groups[3],
                    ),
                    Conv(
                        in_channel[4],
                        out_channel[4],
                        kernel[4],
                        stride=stride[4],
                        padding=padding[4],
                        dilation=dilation[4],
                        groups=groups[4],
                    ),
                    nn.Conv2d(
                        in_channel[5],
                        out_channel[5],
                        kernel[5],
                        stride=stride[5],
                        padding=padding[5],
                        dilation=dilation[5],
                        groups=groups[5],
                    ),
                ),
                nn.Sequential(
                    Conv(
                        in_channel[6],
                        out_channel[6],
                        kernel[6],
                        stride=stride[6],
                        padding=padding[6],
                        dilation=dilation[6],
                        groups=groups[6],
                    ),
                    Conv(
                        in_channel[7],
                        out_channel[7],
                        kernel[7],
                        stride=stride[7],
                        padding=padding[7],
                        dilation=dilation[7],
                        groups=groups[7],
                    ),
                    nn.Conv2d(
                        in_channel[8],
                        out_channel[8],
                        kernel[8],
                        stride=stride[8],
                        padding=padding[8],
                        dilation=dilation[8],
                        groups=groups[8],
                    ),
                ),
            ]
        )

        self.cv3 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Sequential(
                        Conv(
                            in_channel[9],
                            out_channel[9],
                            kernel[9],
                            stride=stride[9],
                            padding=padding[9],
                            dilation=dilation[9],
                            groups=groups[9],
                        ),
                        Conv(
                            in_channel[10],
                            out_channel[10],
                            kernel[10],
                            stride=stride[10],
                            padding=padding[10],
                            dilation=dilation[10],
                            groups=groups[10],
                        ),
                    ),
                    nn.Sequential(
                        Conv(
                            in_channel[11],
                            out_channel[11],
                            kernel[11],
                            stride=stride[11],
                            padding=padding[11],
                            dilation=dilation[11],
                            groups=groups[11],
                        ),
                        Conv(
                            in_channel[12],
                            out_channel[12],
                            kernel[12],
                            stride=stride[12],
                            padding=padding[12],
                            dilation=dilation[12],
                            groups=groups[12],
                        ),
                    ),
                    nn.Conv2d(
                        in_channel[13],
                        out_channel[13],
                        kernel[13],
                        stride=stride[13],
                        padding=padding[13],
                        dilation=dilation[13],
                        groups=groups[13],
                    ),
                ),
                nn.Sequential(
                    nn.Sequential(
                        Conv(
                            in_channel[14],
                            out_channel[14],
                            kernel[14],
                            stride=stride[14],
                            padding=padding[14],
                            dilation=dilation[14],
                            groups=groups[14],
                        ),
                        Conv(
                            in_channel[15],
                            out_channel[15],
                            kernel[15],
                            stride=stride[15],
                            padding=padding[15],
                            dilation=dilation[15],
                            groups=groups[15],
                        ),
                    ),
                    nn.Sequential(
                        Conv(
                            in_channel[16],
                            out_channel[16],
                            kernel[16],
                            stride=stride[16],
                            padding=padding[16],
                            dilation=dilation[16],
                            groups=groups[16],
                        ),
                        Conv(
                            in_channel[17],
                            out_channel[17],
                            kernel[17],
                            stride=stride[17],
                            padding=padding[17],
                            dilation=dilation[17],
                            groups=groups[17],
                        ),
                    ),
                    nn.Conv2d(
                        in_channel[18],
                        out_channel[18],
                        kernel[18],
                        stride=stride[18],
                        padding=padding[18],
                        dilation=dilation[18],
                        groups=groups[18],
                    ),
                ),
                nn.Sequential(
                    nn.Sequential(
                        Conv(
                            in_channel[19],
                            out_channel[19],
                            kernel[19],
                            stride=stride[19],
                            padding=padding[19],
                            dilation=dilation[19],
                            groups=groups[19],
                        ),
                        Conv(
                            in_channel[20],
                            out_channel[20],
                            kernel[20],
                            stride=stride[20],
                            padding=padding[20],
                            dilation=dilation[20],
                            groups=groups[20],
                        ),
                    ),
                    nn.Sequential(
                        Conv(
                            in_channel[21],
                            out_channel[21],
                            kernel[21],
                            stride=stride[21],
                            padding=padding[21],
                            dilation=dilation[21],
                            groups=groups[21],
                        ),
                        Conv(
                            in_channel[22],
                            out_channel[22],
                            kernel[22],
                            stride=stride[22],
                            padding=padding[22],
                            dilation=dilation[22],
                            groups=groups[22],
                        ),
                    ),
                    nn.Conv2d(
                        in_channel[23],
                        out_channel[23],
                        kernel[23],
                        stride=stride[23],
                        padding=padding[23],
                        dilation=dilation[23],
                        groups=groups[23],
                    ),
                ),
            ]
        )
        self.dfl = DFL()

    def forward(self, y1, y2, y3):
        x1 = self.cv2[0](y1)
        x2 = self.cv2[1](y2)
        x3 = self.cv2[2](y3)
        x4 = self.cv3[0](y1)
        x5 = self.cv3[1](y2)
        x6 = self.cv3[2](y3)

        y1 = torch.cat((x1, x4), 1)
        y2 = torch.cat((x2, x5), 1)
        y3 = torch.cat((x3, x6), 1)
        y_all = [y1, y2, y3]

        y1 = torch.reshape(y1, (y1.shape[0], y1.shape[1], y1.shape[2] * y1.shape[3]))
        y2 = torch.reshape(y2, (y2.shape[0], y2.shape[1], y2.shape[2] * y2.shape[3]))
        y3 = torch.reshape(y3, (y3.shape[0], y3.shape[1], y3.shape[2] * y3.shape[3]))

        y = torch.cat((y1, y2, y3), 2)
        ya, yb = y.split((self.out_channel[2], self.out_channel[13]), 1)
        ya = torch.reshape(ya, (ya.shape[0], int(ya.shape[1] / self.in_channel[24]), self.in_channel[24], ya.shape[2]))

        ya = torch.permute(ya, (0, 2, 1, 3))
        ya = f.softmax(ya, dim=1)
        c = self.dfl(ya)
        c1 = torch.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))
        c2 = c1
        c1 = c1[:, 0:2, :]
        c2 = c2[:, 2:4, :]

        anchor, strides = (y_all.transpose(0, 1) for y_all in make_anchors(y_all, [8, 16, 32], 0.5))
        anchor.unsqueeze(0)

        c1 = anchor - c1
        c2 = anchor + c2

        z1 = c2 - c1
        z2 = c1 + c2

        z2 = z2 / 2

        z = torch.concat((z2, z1), 1)
        z = z * strides
        yb = torch.sigmoid(yb)
        out = torch.concat((z, yb), 1)
        return out


class YoloV12x(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv(3, 96, kernel=3, stride=2, padding=1),  # 0
            Conv(96, 192, kernel=3, stride=2, padding=1),  # 1
            C3k2(  # 2
                [192, 384, 96, 96, 96, 48, 48, 48, 48],
                [192, 384, 48, 48, 96, 48, 48, 48, 48],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            Conv(384, 384, kernel=3, stride=2, padding=1),  # 3
            C3k2(  # 4
                [384, 768, 192, 192, 192, 96, 96, 96, 96],
                [384, 768, 96, 96, 192, 96, 96, 96, 96],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            Conv(768, 768, kernel=3, stride=2, padding=1),  # 5
            A2C2f(  # 6
                in_channel=[768, 1920, 384, 384, 384, 384, 460, 384, 384, 384, 384, 460],
                out_channel=[384, 768, 1152, 384, 384, 460, 384, 1152, 384, 384, 460, 384],
                kernel=[1, 1, 1, 1, 7, 1, 1, 1, 1, 7, 1, 1],
                stride=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                padding=[
                    0,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                ],
                dilation=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                groups=[1, 1, 1, 1, 384, 1, 1, 1, 1, 384, 1, 1],
                c1=768,
                c2=768,
                n=4,
                a2=True,
                area=4,
                residual=True,
                mlp_ratio=1.2,
                e=0.5,
                g=1,
                shortcut=True,
            ),
            Conv(768, 768, kernel=3, stride=2, padding=1),  # 7
            A2C2f(  # 8
                [768, 1920, 384, 384, 384, 384, 460, 384, 384, 384, 384, 460],
                [384, 768, 1152, 384, 384, 460, 384, 1152, 384, 384, 460, 384],
                [1, 1, 1, 1, 7, 1, 1, 1, 1, 7, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [
                    0,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                ],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 384, 1, 1, 1, 1, 384, 1, 1],
                c1=768,
                c2=768,
                n=4,
                a2=True,
                area=1,
                residual=True,
                mlp_ratio=1.2,
                e=0.5,
                g=1,
                shortcut=True,
            ),
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # 9
            Concat(),  # 10
            A2C2f(  # 11
                [1536, 1152, 384, 384, 384, 192, 192, 192, 192],
                [384, 768, 192, 192, 384, 192, 192, 192, 192],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                c1=1536,
                c2=768,
                n=2,
                a2=False,
                area=-1,
                residual=True,
                mlp_ratio=1.2,
                e=0.5,
                g=1,
                shortcut=True,
            ),
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # 12
            Concat(),  # 13
            A2C2f(  # 14
                [1536, 576, 192, 192, 192, 96, 96, 96, 96],
                [192, 384, 96, 96, 192, 96, 96, 96, 96],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                c1=1536,
                c2=384,
                n=2,
                a2=False,
                area=-1,
                residual=True,
                mlp_ratio=1.2,
                e=0.5,
                g=1,
                shortcut=True,
            ),
            Conv(384, 384, kernel=3, stride=2, padding=1),  # 15
            Concat(),  # 16
            A2C2f(  # 17
                [1152, 1152, 384, 384, 384, 192, 192, 192, 192],
                [384, 768, 192, 192, 384, 192, 192, 192, 192],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                c1=1152,
                c2=768,
                n=2,
                a2=False,
                area=-1,
                residual=True,
                mlp_ratio=1.2,
                e=0.5,
                g=1,
                shortcut=True,
            ),
            Conv(768, 768, kernel=3, stride=2, padding=1),  # 18
            Concat(),  # 19
            C3k2(  # 20
                [1536, 1536, 384, 384, 384, 192, 192, 192, 192],
                [768, 768, 192, 192, 384, 192, 192, 192, 192],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            Detect(  # 21
                [
                    384,
                    96,
                    96,
                    768,
                    96,
                    96,
                    768,
                    96,
                    96,
                    384,
                    384,
                    384,
                    384,
                    384,
                    768,
                    768,
                    384,
                    384,
                    384,
                    768,
                    768,
                    384,
                    384,
                    384,
                    16,
                ],
                [
                    96,
                    96,
                    64,
                    96,
                    96,
                    64,
                    96,
                    96,
                    64,
                    384,
                    384,
                    384,
                    384,
                    80,
                    768,
                    384,
                    384,
                    384,
                    80,
                    768,
                    384,
                    384,
                    384,
                    80,
                    1,
                ],
                [3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 384, 1, 384, 1, 1, 768, 1, 384, 1, 1, 768, 1, 384, 1, 1, 1],
            ),
        )

    def forward(self, x):
        input = x
        x = self.model[0](x)  # 0
        x = self.model[1](x)  # 1
        x = self.model[2](x, i=4)  # 2
        x = self.model[3](x)  # 3
        x = self.model[4](x, i=6)  # 4
        x4 = x
        x = self.model[5](x)  # 5
        x = self.model[6](x, i=8)  # 6
        x6 = x
        x = self.model[7](x)  # 7
        x = self.model[8](x, i=10)  # 8
        x8 = x
        x = f.upsample(x, scale_factor=2.0)  # 9
        x = torch.cat((x, x6), 1)  # 10
        x = self.model[11](x, i=13)  # 11
        x11 = x
        x = f.upsample(x, scale_factor=2.0)  # 12
        x = torch.cat((x, x4), 1)  # 13
        x = self.model[14](x, i=16)  # 14
        x14 = x
        x = self.model[15](x)  # 15
        x = torch.cat((x, x11), 1)  # 16
        x = self.model[17](x, i=19)  # 17
        x17 = x
        x = self.model[18](x)  # 18
        x = torch.cat((x, x8), 1)  # 19
        x = self.model[20](x, i=22)  # 20
        x20 = x
        x = self.model[21](y1=x14, y2=x17, y3=x20)  # 21
        return (x, x14, x17, x20)


class BaseModel(nn.Module):
    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


class DetectionModel(BaseModel):
    def __init__(self, cfg="yolo12x.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
