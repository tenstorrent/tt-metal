# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as f


def make_anchors(feats, strides, grid_cell_offset=0.5):
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


class DFL(nn.Module):
    def __init__(self):
        super(DFL, self).__init__()
        self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, stride=1, padding=0, dilation=1, groups=1, enable_act=True):
        super().__init__()
        self.enable_act = enable_act
        if enable_act:
            self.conv = nn.Conv2d(
                in_channel,
                out_channel,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
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
                bias=False,
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


class SPPF(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=[1, 1],
        m_kernel=5,
        m_padding=2,
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
        self.m = nn.MaxPool2d(kernel_size=m_kernel, stride=1, padding=m_padding)

    def forward(self, x):
        x = self.cv1(x)
        x1 = x
        m1 = self.m(x)
        m2 = self.m(m1)
        m3 = self.m(m2)
        y = torch.cat((x1, m1, m2, m3), 1)
        x = self.cv2(y)
        return x


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

    def forward(self, x):
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
                ]
            )

    def forward(self, x):
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
            y.extend(self.m[0](y[-1]))
            y[-1] = y[-1].unsqueeze(0)
            x = torch.cat(y, 1)
            x = self.cv2(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation, groups):
        super().__init__()
        self.num_heads = 2
        self.key_dim = 32
        self.head_dim = 64
        self.scale = self.key_dim**-0.5

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
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation, groups):
        super().__init__()
        self.attn = Attention(
            in_channel[0:3], out_channel[0:3], kernel[0:3], stride[0:3], padding[0:3], dilation[0:3], groups[0:3]
        )
        self.ffn = nn.Sequential(
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
                enable_act=False,
            ),
        )

    def forward(self, x):
        x1 = x
        x = self.attn(x)
        x = x1 + x
        x1 = x
        x = self.ffn(x)
        return x + x1


class C2PSA(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation, groups):
        super().__init__()
        self.out_channel = out_channel
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
        self.m = nn.Sequential(
            PSABlock(
                in_channel[2:7],
                out_channel[2:7],
                kernel[2:7],
                stride=stride[2:7],
                padding=padding[2:7],
                dilation=dilation[2:7],
                groups=groups[2:7],
            )
        )

    def forward(self, x):
        x = self.cv1(x)
        a, b = x.split((int(self.out_channel[0] / 2), int(self.out_channel[0] / 2)), 1)
        x = self.m(b)
        x = self.cv2(torch.cat((a, x), 1))
        return x


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

        ya, yb = y.split((self.out_channel[0], self.out_channel[10]), 1)

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


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class YoloV11(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv(3, 16, kernel=3, stride=2, padding=1),  # 0
            Conv(16, 32, kernel=3, stride=2, padding=1),  # 1
            C3k2(  # 2
                [32, 48, 16, 8],
                [32, 64, 8, 16],
                [1, 1, 3, 3],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                is_bk_enabled=True,
            ),
            Conv(64, 64, kernel=3, stride=2, padding=1),  # 3
            C3k2(  # 4
                [64, 96, 32, 16],
                [64, 128, 16, 32],
                [1, 1, 3, 3],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                is_bk_enabled=True,
            ),
            Conv(128, 128, kernel=3, stride=2, padding=1),  # 5
            C3k2(
                [128, 192, 64, 64, 64, 32, 32, 32, 32],  # 6
                [128, 128, 32, 32, 64, 32, 32, 32, 32],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            Conv(128, 256, kernel=3, stride=2, padding=1),  # 7
            C3k2(
                [256, 384, 128, 128, 128, 64, 64, 64, 64],  # 8
                [256, 256, 64, 64, 128, 64, 64, 64, 64],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            SPPF([256, 512], [128, 256], [1, 1], [1, 1]),  # 9
            C2PSA(
                [256, 256, 128, 128, 128, 128, 256],  # 10
                [256, 256, 256, 128, 128, 256, 128],
                [1, 1, 1, 1, 3, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 128, 1, 1],
            ),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(),
            C3k2(  # 13
                [384, 192, 64, 32],
                [128, 128, 32, 64],
                [1, 1, 3, 3],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                is_bk_enabled=True,
            ),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(),
            C3k2(  # 16
                [256, 96, 32, 16],
                [64, 64, 16, 32],
                [1, 1, 3, 3],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                is_bk_enabled=True,
            ),
            Conv(64, 64, kernel=3, stride=2, padding=1),  # 17
            Concat(),
            C3k2(  # 19
                [192, 192, 64, 32],
                [128, 128, 32, 64],
                [1, 1, 3, 3],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                is_bk_enabled=True,
            ),
            Conv(128, 128, kernel=3, stride=2, padding=1),  # 20
            Concat(),
            C3k2(
                [384, 384, 128, 128, 128, 64, 64, 64, 64],  # 22
                [256, 256, 64, 64, 128, 64, 64, 64, 64],
                [1, 1, 1, 1, 1, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            Detect(  # 23
                [
                    64,
                    64,
                    64,
                    128,
                    64,
                    64,
                    256,
                    64,
                    64,
                    64,
                    64,
                    80,
                    80,
                    80,
                    128,
                    128,
                    80,
                    80,
                    80,
                    256,
                    256,
                    80,
                    80,
                    80,
                    16,
                ],
                [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 80, 80, 80, 80, 128, 80, 80, 80, 80, 256, 80, 80, 80, 80, 1],
                [3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 1, 80, 1, 1, 128, 1, 80, 1, 1, 256, 1, 80, 1, 1, 1],
            ),
        )

    def forward(self, x):
        x = self.model[0](x)  # 0
        x = self.model[1](x)  # 1
        x = self.model[2](x)  # 2
        x = self.model[3](x)  # 3
        x = self.model[4](x)  # 4
        x4 = x
        x = self.model[5](x)  # 5
        x = self.model[6](x)  # 6
        x6 = x
        x = self.model[7](x)  # 7
        x = self.model[8](x)  # 8
        x = self.model[9](x)  # 9
        x = self.model[10](x)  # 10
        x10 = x
        x = f.upsample(x, scale_factor=2.0)  # 11
        x = torch.cat((x, x6), 1)  # 12
        x = self.model[13](x)  # 13
        x13 = x
        x = f.upsample(x, scale_factor=2.0)  # 14
        x = torch.cat((x, x4), 1)  # 15
        x = self.model[16](x)  # 16
        x16 = x
        x = self.model[17](x)  # 17
        x = torch.cat((x, x13), 1)  # 18
        x = self.model[19](x)  # 19
        x19 = x
        x = self.model[20](x)  # 20
        x = torch.cat((x, x10), 1)  # 21
        x = self.model[22](x)  # 22
        x22 = x
        x = self.model[23](x16, x19, x22)  # 23
        return x
