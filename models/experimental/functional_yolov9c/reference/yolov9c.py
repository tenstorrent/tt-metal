# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as f
import math


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


class Conv(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel=1, stride=1, padding=0, dilation=1, groups=1, enable_identity=False
    ):
        super().__init__()
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
        if enable_identity:
            self.act = nn.Identity()
        else:
            self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RepConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1):
        super().__init__()
        self.act = nn.SiLU(inplace=True)
        self.conv1 = Conv(in_channel, out_channel, 3, 1, padding=1, groups=1, enable_identity=True)
        self.conv2 = Conv(in_channel, out_channel, 1, 1, padding=0, groups=1, enable_identity=True)

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cv1 = Conv(in_channel, out_channel)
        self.cv2 = Conv(in_channel, out_channel, 3, padding=1)

    def forward(self, x):
        torch_input = x
        return torch_input + self.cv2(self.cv1(x))


class RepBottleneck(Bottleneck):
    def __init__(self, in_channel, out_channel):
        super().__init__(in_channel // 2, out_channel // 2)
        self.cv1 = RepConv(in_channel // 2, out_channel // 2)


class C3(nn.Module):
    def __init__(self, in_channel, out_channel, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.cv1 = Conv(in_channel // 2, out_channel // 4, 1, 1)
        self.cv2 = Conv(in_channel // 2, out_channel // 4, 1, 1)
        self.cv3 = Conv(in_channel // 2, out_channel // 2)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepCSP(C3):
    def __init__(self, in_channel, out_channel):
        super().__init__(in_channel, out_channel)
        self.m = nn.Sequential(*(RepBottleneck(in_channel // 2, out_channel // 2) for _ in range(1)))


class RepNCSPELAN4(nn.Module):
    def __init__(self, input_channel, output_channel, cv2_inc, cv2_outc, cv3_inc, cv3_outc, cv4_inc, cv4_out_c, n=1):
        super().__init__()
        self.cv1 = Conv(input_channel, output_channel)
        self.cv2 = nn.Sequential(RepCSP(output_channel, output_channel), Conv(cv2_inc, cv2_outc, 3, 1, padding=1))
        self.cv3 = nn.Sequential(RepCSP(output_channel, output_channel), Conv(cv3_inc, cv3_outc, 3, 1, padding=1))
        self.cv4 = Conv(cv4_inc, cv4_out_c, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cv1 = Conv(in_channel, out_channel, 3, 2, 1)
        self.cv2 = Conv(in_channel, out_channel, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    def __init__(self, input_channel, output_channel, k=5):
        super().__init__()
        self.cv1 = Conv(input_channel, output_channel, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(input_channel * 2, input_channel)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class DFL(nn.Module):
    def __init__(self):
        super(DFL, self).__init__()
        self.conv = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Detect(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.cv2 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(
                        in_channel[0],
                        64,
                        3,
                        padding=1,
                    ),
                    Conv(
                        in_channel[1],
                        64,
                        3,
                        padding=1,
                    ),
                    nn.Conv2d(
                        in_channel[2],
                        64,
                        1,
                    ),
                ),
                nn.Sequential(
                    Conv(
                        in_channel[3],
                        64,
                        3,
                        padding=1,
                    ),
                    Conv(
                        in_channel[4],
                        64,
                        3,
                        padding=1,
                    ),
                    nn.Conv2d(
                        in_channel[5],
                        64,
                        1,
                    ),
                ),
                nn.Sequential(
                    Conv(
                        in_channel[6],
                        64,
                        3,
                        padding=1,
                    ),
                    Conv(
                        in_channel[7],
                        64,
                        3,
                        padding=1,
                    ),
                    nn.Conv2d(
                        in_channel[8],
                        64,
                        1,
                    ),
                ),
            ]
        )
        self.cv3 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(
                        in_channel[9],
                        256,
                        3,
                        padding=1,
                    ),
                    Conv(
                        in_channel[10],
                        256,
                        3,
                        padding=1,
                    ),
                    nn.Conv2d(
                        in_channel[11],
                        80,
                        1,
                    ),
                ),
                nn.Sequential(
                    Conv(
                        in_channel[12],
                        256,
                        3,
                        padding=1,
                    ),
                    Conv(
                        in_channel[13],
                        256,
                        3,
                        padding=1,
                    ),
                    nn.Conv2d(
                        in_channel[14],
                        80,
                        1,
                    ),
                ),
                nn.Sequential(
                    Conv(
                        in_channel[15],
                        256,
                        3,
                        padding=1,
                    ),
                    Conv(
                        in_channel[16],
                        256,
                        3,
                        padding=1,
                    ),
                    nn.Conv2d(
                        in_channel[17],
                        80,
                        1,
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
        ya, yb = y.split((64, 80), 1)

        ya = torch.reshape(ya, (ya.shape[0], int(ya.shape[1] / 16), 16, ya.shape[2]))
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


class YoloV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv(3, 64, kernel=3, stride=2, padding=1),  # 0
            Conv(64, 128, kernel=3, stride=2, padding=1),  # 1
            RepNCSPELAN4(128, 128, 64, 64, 64, 64, 256, 256),  # 2
            ADown(128, 128),  # 3
            RepNCSPELAN4(256, 256, 128, 128, 128, 128, 512, 512),  # 4
            ADown(256, 256),  # 5
            RepNCSPELAN4(512, 512, 256, 256, 256, 256, 1024, 512),  # 6
            ADown(256, 256),  # 7
            RepNCSPELAN4(512, 512, 256, 256, 256, 256, 1024, 512),  # 8
            SPPELAN(512, 256),  # 9
            RepNCSPELAN4(1024, 512, 256, 256, 256, 256, 1024, 512),  # 10
            RepNCSPELAN4(1024, 256, 128, 128, 128, 128, 512, 256),  # 11
            ADown(128, 128),  # 12
            RepNCSPELAN4(768, 512, 256, 256, 256, 256, 1024, 512),  # 13
            ADown(256, 256),  # 14
            RepNCSPELAN4(1024, 512, 256, 256, 256, 256, 1024, 512),  # 15
            Detect([256, 64, 64, 512, 64, 64, 512, 64, 64, 256, 256, 256, 512, 256, 256, 512, 256, 256]),  # 16
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
        x10 = x
        x = f.upsample(x, scale_factor=2.0)  # 11
        x = torch.cat((x, x6), 1)  # 12
        x = self.model[10](x)  # 12
        x13 = x
        x = f.upsample(x, scale_factor=2.0)  # 14
        x = torch.cat((x, x4), 1)  # 15
        x = self.model[11](x)  # 16
        x16 = x
        x = self.model[12](x)  # 17
        x = torch.cat((x, x13), 1)  # 18
        x = self.model[13](x)  # 19
        x19 = x
        x = self.model[14](x)  # 20
        x = torch.cat((x, x10), 1)  # 21
        x = self.model[15](x)  # 22
        x22 = x
        x = self.model[16](x16, x19, x22)  # 23
        return x


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


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
    def __init__(self, cfg="yolov9c.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
