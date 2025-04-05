# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU(inplace=True))
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=nn.ReLU(inplace=True))
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


class Detect(nn.Module):
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.ch = ch
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.tensor([8.0, 16.0, 32.0])  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3, act=nn.ReLU(inplace=True)),
                Conv(c2, c2, 3, act=nn.ReLU(inplace=True)),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3, act=nn.ReLU(inplace=True)),
                Conv(c3, c3, 3, act=nn.ReLU(inplace=True)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x):
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)


class Yolov6x_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.nc = 80
        self.act = nn.ReLU(inplace=True)
        self.scales = [1.00, 1.25, 512]
        self.model = nn.Sequential(  # backbone
            Conv(3, 80, 3, 2, act=self.act),  # 0
            Conv(80, 160, 3, 2, act=self.act),  # 1
            nn.Sequential(  # 2
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
            ),
            Conv(160, 320, 3, 2, act=self.act),  # 3
            nn.Sequential(  # 4
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
            ),
            Conv(320, 640, 3, 2, act=self.act),  # 5
            nn.Sequential(  # 6
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
            ),
            Conv(640, 640, 3, 2, act=self.act),  # 7
            nn.Sequential(  # 8
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
            ),
            SPPF(640, 640),  # 9
            # Head
            Conv(640, 320, 1, 1, act=self.act),  # 10
            nn.ConvTranspose2d(320, 320, 2, 2),
            Concat(),  # 12
            Conv(960, 320, 3, 1, act=self.act),  # 13
            nn.Sequential(  # 14
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
            ),
            Conv(320, 160, 1, 1, act=self.act),  # 15
            nn.ConvTranspose2d(160, 160, 2, 2),
            Concat(),  # 17
            Conv(480, 160, 3, 1, act=self.act),  # 18
            nn.Sequential(  # 19
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
                Conv(160, 160, 3, 1, act=self.act),
            ),
            Conv(160, 160, 3, 2, act=self.act),  # 20
            Concat(),  # 21
            Conv(320, 320, 3, 1, act=self.act),  # 22
            nn.Sequential(  # 23
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
                Conv(320, 320, 3, 1, act=self.act),
            ),
            Conv(320, 320, 3, 2, act=self.act),  # 24
            Concat(),  # 25
            Conv(640, 640, 3, 1, act=self.act),  # 26
            nn.Sequential(  # 27
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
                Conv(640, 640, 3, 1, act=self.act),
            ),
            Detect(nc=self.nc, ch=(160, 320, 640)),  # 28
        )

    def forward(self, x):
        x0 = self.model[0](x)
        x1 = self.model[1](x0)
        x2 = self.model[2](x1)
        x3 = self.model[3](x2)
        x4 = self.model[4](x3)
        x5 = self.model[5](x4)
        x6 = self.model[6](x5)
        x7 = self.model[7](x6)
        x8 = self.model[8](x7)
        x9 = self.model[9](x8)

        x10 = self.model[10](x9)
        x11 = self.model[11](x10)
        x12 = self.model[12]([x11, x6])
        x13 = self.model[13](x12)
        x14 = self.model[14](x13)

        x15 = self.model[15](x14)
        x16 = self.model[16](x15)
        x17 = self.model[17]([x16, x4])
        x18 = self.model[18](x17)
        x19 = self.model[19](x18)

        x20 = self.model[20](x19)
        x21 = self.model[21]([x20, x15])
        x22 = self.model[22](x21)
        x23 = self.model[23](x22)

        x24 = self.model[24](x23)
        x25 = self.model[25]([x24, x10])
        x26 = self.model[26](x25)
        x27 = self.model[27](x26)

        x28 = self.model[28]([x19, x23, x27])

        return x28
