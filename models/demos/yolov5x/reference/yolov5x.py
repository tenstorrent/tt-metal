# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.default_act = nn.SiLU(inplace=True)
        self.conv = nn.Conv2d(c1, c2, k, s, self._autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
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
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, *x):
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


class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv(3, 80, 6, 2, 2),  # 0
            Conv(80, 160, 3, 2, 1),  # 1
            C3(160, 160, n=4, shortcut=True),  # 2
            Conv(160, 320, 3, 2, 1),  # 3
            C3(320, 320, n=8, shortcut=True),  # 4
            Conv(320, 640, 3, 2, 1),  # 5
            C3(640, 640, n=12, shortcut=True),  # 6
            Conv(640, 1280, 3, 2, 1),  # 7
            C3(1280, 1280, n=4, shortcut=True),  # 8
            SPPF(1280, 1280),  # 9
            Conv(1280, 640, 1, 1),  # 10
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # 11
            Concat(),  # 12
            C3(1280, 640, n=4, shortcut=False),  # 13
            Conv(640, 320, 1, 1),  # 14
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # 15
            Concat(),  # 16
            C3(640, 320, n=4, shortcut=False),  # 17
            Conv(320, 320, 3, 2, 1),  # 18
            Concat(),  # 19
            C3(640, 640, n=4, shortcut=False),  # 20
            Conv(640, 640, 3, 2, 1),  # 21
            Concat(),  # 22
            C3(1280, 1280, n=4, shortcut=False),  # 23
            Detect(
                nc=80,
                ch=(320, 640, 1280),
            ),  # 24
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
        x = self.model[11](x)  # 11
        x = self.model[12](x, x6)  # 12
        x = self.model[13](x)  # 13
        x = self.model[14](x)  # 14
        x14 = x
        x = self.model[15](x)  # 15
        x = self.model[16](x, x4)  # 16
        x = self.model[17](x)  # 17
        x17 = x
        x = self.model[18](x)  # 18
        x = self.model[19](x, x14)  # 19
        x = self.model[20](x)  # 20
        x20 = x
        x = self.model[21](x)  # 21
        x = self.model[22](x, x10)  # 22
        x = self.model[23](x)  # 23
        x23 = x
        x = self.model[24]([x17, x20, x23])  # 24
        return x


class YOLOv5(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        self.model = DetectionModel()
        if weights_path:
            self.load_state_dict(torch.load(weights_path), strict=False)

    def forward(self, x):
        return self.model(x)
