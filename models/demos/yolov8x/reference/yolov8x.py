# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.default_act = nn.SiLU(inplace=True)
        self.conv = nn.Conv2d(c1, c2, k, s, self._autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)  # Modified BatchNorm parameters
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def _autopad(self, k, p=None, d=1):
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.shortcut = shortcut

    def forward(self, x):
        cv1_out = self.cv1(x)
        cv2_out = self.cv2(cv1_out)
        add = self.shortcut and x.shape[1] == cv2_out.shape[1]  # Check channels
        return x + cv2_out if add else cv2_out


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)])
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        cv1_out = self.cv1(x)
        y = list(cv1_out.chunk(2, 1))
        for i, bottleneck in enumerate(self.m):
            z = bottleneck(y[-1])
            y.append(z)
        x = torch.cat(y, 1)
        x = self.cv2(x)
        return x


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        self.c_ = c1 // 2
        self.cv1 = Conv(c1, self.c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(self.c_ * 4, c2, 1, 1)

    def forward(self, x):
        cv1_out = self.cv1(x)
        y = [cv1_out]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class DetectCv2(nn.Module):
    def __init__(self, c1, c2, k, reg_max):
        super().__init__()
        self.conv1 = Conv(c1, c2, k)
        self.conv2 = Conv(c2, c2, k)
        self.conv3 = nn.Conv2d(c2, reg_max, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        x = x.view(b, 4, self.c1, a)
        x = x.transpose(2, 1)
        x = x.softmax(dim=1)
        x = self.conv(x)
        x = x.view(b, 4, a)
        return x


class Detect(nn.Module):
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.c2, self.c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(nc, 100))

        self.cv2 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(ch[i], self.c2, 3), Conv(self.c2, self.c2, 3), nn.Conv2d(self.c2, 4 * self.reg_max, 1)
                )
                for i in range(self.nl)
            ]
        )

        self.cv3 = nn.ModuleList(
            [
                nn.Sequential(Conv(ch[i], self.c3, 3), Conv(self.c3, self.c3, 3), nn.Conv2d(self.c3, nc, 1))
                for i in range(self.nl)
            ]
        )

        self.dfl = DFL(self.reg_max)
        self.anchors = None
        self.strides = None
        self.self_shape = None

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        shape = x[0].shape
        if self.anchors is None or self.self_shape != shape:
            temp = self._make_anchors(x)
            self.anchors, self.strides = (i.transpose(0, 1) for i in temp)
            self.anchors = self.anchors.unsqueeze(0)
            self.strides = self.strides.unsqueeze(0)
            self.self_shape = shape

        x_cat = torch.cat([xi.reshape(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dfl_out = self.dfl(box)
        dbox = self._decode_bboxes(dfl_out, self.anchors)
        dbox = dbox * self.strides

        return (torch.cat((dbox, cls.sigmoid()), 1), x)

    def _make_anchors(self, feats, grid_cell_offset=0.5):
        anchor_points, stride_tensor = [], []
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(self.stride):
            h, w = feats[i].shape[2:]
            sx = torch.arange(end=w) + grid_cell_offset
            sy = torch.arange(end=h) + grid_cell_offset

            sy, sx = torch.meshgrid(sy, sx)

            temp = torch.stack((sx, sy), -1)
            anchor_points.append(temp.view(-1, 2))
            temp = torch.full((h * w, 1), stride, dtype=dtype, device=device)
            stride_tensor.append(temp)

        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def _decode_bboxes(self, distance, anchor_points, xywh=True, dim=1):
        lt, rb = distance.chunk(2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv(3, 80, 3, 2, 1),
            Conv(80, 160, 3, 2, 1),
            C2f(160, 160, n=3, shortcut=True),
            Conv(160, 320, 3, 2, 1),
            C2f(320, 320, n=6, shortcut=True),
            Conv(320, 640, 3, 2, 1),
            C2f(640, 640, n=6, shortcut=True),
            Conv(640, 640, 3, 2, 1),
            C2f(640, 640, n=3, shortcut=True),
            SPPF(640, 640),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(),
            C2f(1280, 640, n=3, shortcut=False),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(),
            C2f(960, 320, n=3, shortcut=False),
            Conv(320, 320, 3, 2, 1),
            Concat(),
            C2f(960, 640, n=3, shortcut=False),
            Conv(640, 640, 3, 2, 1),
            Concat(),
            C2f(1280, 640, n=3, shortcut=False),
            Detect(
                nc=80,
                ch=(320, 640, 640),
            ),
        )

    def forward(self, x):
        x = self.model[:5](x)
        four = x.detach().clone()
        x = self.model[5:7](x)
        six = x.detach().clone()
        x = self.model[7:10](x)
        nine = x.detach().clone()
        x = self.model[10](x)
        x = self.model[11](x, six)
        x = self.model[12](x)
        twelve = x.detach().clone()
        x = self.model[13](x)
        x = self.model[14](x, four)
        x = self.model[15](x)
        fifteen = x.detach().clone()
        x = self.model[16](x)
        x = self.model[17](x, twelve)
        x = self.model[18](x)
        eighteen = x.detach().clone()
        x = self.model[19](x)
        x = self.model[20](x, nine)
        x = self.model[21](x)
        twentyone = x.detach().clone()
        return self.model[22]([fifteen, eighteen, twentyone])


class YOLOv8(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        self.model = DetectionModel()
        if weights_path:
            self.load_state_dict(torch.load(weights_path), strict=False)

    def forward(self, x):
        return self.model(x)
