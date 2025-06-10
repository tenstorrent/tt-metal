# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x1 = self.cv3(x1)
        x1 = self.cv4(x1)
        y1 = torch.cat([x1] + [m(x1) for m in self.m], 1)
        y1 = self.cv5(y1)
        y1 = self.cv6(y1)
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        assert k == 3
        assert autopad(k, p) == 1
        padding_11 = autopad(k, p) - k // 2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        out = self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        return out


class Detect(nn.Module):
    stride = None
    export = False
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super(Detect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        self.training = False
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)
                    xy = xy * (2.0 * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))
                    wh = wh**2 * (4 * self.anchor_grid[i].data)
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)
        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,
            device=z.device,
        )
        box @= convert_matrix
        return (box, score)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Yolov7_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.nc = 80
        self.anchors = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]
        self.ch = [256, 512, 1024]
        self.model = nn.Sequential(
            Conv(3, 32, 3, 1),  # 0
            Conv(32, 64, 3, 2),  # 1
            Conv(64, 64, 3, 1),  # 2
            Conv(64, 128, 3, 2),  # 3
            Conv(128, 64, 1, 1),  # 4
            Conv(128, 64, 1, 1),  # 5
            Conv(64, 64, 3, 1),  # 6
            Conv(64, 64, 3, 1),  # 7
            Conv(64, 64, 3, 1),  # 8
            Conv(64, 64, 3, 1),  # 9
            Concat(),  # 10
            Conv(256, 256, 1, 1),  # 11
            MP(),  # 12
            Conv(256, 128, 1, 1),  # 13
            Conv(256, 128, 1, 1),  # 14
            Conv(128, 128, 3, 2),  # 15
            Concat(),  # 16
            Conv(256, 128, 1, 1),  # 17
            Conv(256, 128, 1, 1),  # 18
            Conv(128, 128, 3, 1),  # 19
            Conv(128, 128, 3, 1),  # 20
            Conv(128, 128, 3, 1),  # 21
            Conv(128, 128, 3, 1),  # 22
            Concat(),  # 23              # Done
            Conv(512, 512, 1, 1),  # 24
            MP(),  # 25
            Conv(512, 256, 1, 1),  # 26
            Conv(512, 256, 1, 1),  # 27
            Conv(256, 256, 3, 2),  # 28
            Concat(),  # 29
            Conv(512, 256, 1, 1),  # 30
            Conv(512, 256, 1, 1),  # 31
            Conv(256, 256, 3, 1),  # 32
            Conv(256, 256, 3, 1),  # 33
            Conv(256, 256, 3, 1),  # 34
            Conv(256, 256, 3, 1),  # 35
            Concat(),  # 36
            Conv(1024, 1024, 1, 1),  # 37
            MP(),  # 38
            Conv(1024, 512, 1, 1),  # 39
            Conv(1024, 512, 1, 1),  # 40
            Conv(512, 512, 3, 2),  # 41
            Concat(),  # 42
            Conv(1024, 256, 1, 1),  # 43
            Conv(1024, 256, 1, 1),  # 44
            Conv(256, 256, 3, 1),  # 45
            Conv(256, 256, 3, 1),  # 46
            Conv(256, 256, 3, 1),  # 47
            Conv(256, 256, 3, 1),  # 48
            Concat(),  # 49
            Conv(1024, 1024, 1, 1),  # 50
            # HEAD Module
            SPPCSPC(1024, 512),  # 51
            Conv(512, 256, 1, 1),  # 52
            # upsample 53
            nn.Upsample(None, 2, "nearest"),  # 53
            Conv(1024, 256, 1, 1),  # 54
            Concat(),  # 5
            Conv(512, 256, 1, 1),  # 56
            Conv(512, 256, 1, 1),  # 57
            Conv(256, 128, 3, 1),  # 58
            Conv(128, 128, 3, 1),  # 59
            Conv(128, 128, 3, 1),  # 60
            Conv(128, 128, 3, 1),  # 61
            Concat(),  # 62
            Conv(1024, 256, 1, 1),  # 63
            Conv(256, 128, 1, 1),  # 64
            # upsample 65
            nn.Upsample(None, 2, "nearest"),  # 65
            Conv(512, 128, 1, 1),  # 66
            Concat(),  # 67
            Conv(256, 128, 1, 1),  # 68
            Conv(256, 128, 1, 1),  # 69
            Conv(128, 64, 3, 1),  # 70
            Conv(64, 64, 3, 1),  # 71
            Conv(64, 64, 3, 1),  # 72
            Conv(64, 64, 3, 1),  # 73
            Concat(),  # 74
            Conv(512, 128, 1, 1),  # 75
            MP(),  # 76
            Conv(128, 128, 1, 1),  # 77
            Conv(128, 128, 1, 1),  # 78
            Conv(128, 128, 3, 2),  # 79
            Concat(),  # 80
            Conv(512, 256, 1, 1),  # 81
            Conv(512, 256, 1, 1),  # 82
            Conv(256, 128, 3, 1),  # 83
            Conv(128, 128, 3, 1),  # 84
            Conv(128, 128, 3, 1),  # 85
            Conv(128, 128, 3, 1),  # 86
            Concat(),  # 87
            Conv(1024, 256, 1, 1),  # 88
            MP(),  # 89
            Conv(256, 256, 1, 1),  # 90
            Conv(256, 256, 1, 1),  # 91
            Conv(256, 256, 3, 2),  # 92
            Concat(),  # 93
            Conv(1024, 512, 1, 1),  # 94
            Conv(1024, 512, 1, 1),  # 95
            Conv(512, 256, 3, 1),  # 96
            Conv(256, 256, 3, 1),  # 97
            Conv(256, 256, 3, 1),  # 98
            Conv(256, 256, 3, 1),  # 99
            Concat(),  # 100
            Conv(2048, 512, 1, 1),  # 101
            RepConv(128, 256),  # 102
            RepConv(256, 512),  # 103
            RepConv(512, 1024),  # 104
            Detect(self.nc, self.anchors, self.ch),  # 105
        )

    def forward(self, x):
        x0 = self.model[0](x)
        x1 = self.model[1](x0)
        x2 = self.model[2](x1)
        x3 = self.model[3](x2)
        x4 = self.model[4](x3)
        x5 = self.model[5](x3)
        x6 = self.model[6](x5)
        x7 = self.model[7](x6)
        x8 = self.model[8](x7)
        x9 = self.model[9](x8)
        x10 = self.model[10]([x9, x7, x5, x4])
        x11 = self.model[11](x10)
        x12 = self.model[12](x11)
        x13 = self.model[13](x12)
        x14 = self.model[14](x11)
        x15 = self.model[15](x14)
        x16 = self.model[16]([x15, x13])
        x17 = self.model[17](x16)
        x18 = self.model[18](x16)
        x19 = self.model[19](x18)
        x20 = self.model[20](x19)
        x21 = self.model[21](x20)
        x22 = self.model[22](x21)
        x23 = self.model[23]([x22, x20, x18, x17])
        x24 = self.model[24](x23)
        x25 = self.model[25](x24)
        x26 = self.model[26](x25)
        x27 = self.model[27](x24)
        x28 = self.model[28](x27)
        x29 = self.model[29]([x28, x26])  # Done
        x30 = self.model[30](x29)
        x31 = self.model[31](x29)
        x32 = self.model[32](x31)
        x33 = self.model[33](x32)
        x34 = self.model[34](x33)
        x35 = self.model[35](x34)
        x36 = self.model[36]([x35, x33, x31, x30])
        x37 = self.model[37](x36)
        x38 = self.model[38](x37)
        x39 = self.model[39](x38)
        x40 = self.model[40](x37)
        x41 = self.model[41](x40)
        x42 = self.model[42]([x41, x39])
        x43 = self.model[43](x42)
        x44 = self.model[44](x42)
        x45 = self.model[45](x44)
        x46 = self.model[46](x45)
        x47 = self.model[47](x46)
        x48 = self.model[48](x47)
        x49 = self.model[49]([x48, x46, x44, x43])
        x50 = self.model[50](x49)

        x51 = self.model[51](x50)
        x52 = self.model[52](x51)
        x53 = self.model[53](x52)
        x54 = self.model[54](x37)
        x55 = self.model[55]([x54, x53])
        x56 = self.model[56](x55)
        x57 = self.model[57](x55)
        x58 = self.model[58](x57)
        x59 = self.model[59](x58)
        x60 = self.model[60](x59)
        x61 = self.model[61](x60)
        x62 = self.model[62]([x61, x60, x59, x58, x57, x56])
        x63 = self.model[63](x62)
        x64 = self.model[64](x63)
        x65 = self.model[65](x64)
        x66 = self.model[66](x24)
        x67 = self.model[67]([x66, x65])
        x68 = self.model[68](x67)
        x69 = self.model[69](x67)
        x70 = self.model[70](x69)
        x71 = self.model[71](x70)
        x72 = self.model[72](x71)
        x73 = self.model[73](x72)
        x74 = self.model[74]([x73, x72, x71, x70, x69, x68])
        x75 = self.model[75](x74)
        x76 = self.model[76](x75)
        x77 = self.model[77](x76)
        x78 = self.model[78](x75)
        x79 = self.model[79](x78)
        x80 = self.model[80]([x79, x77, x63])
        x81 = self.model[81](x80)
        x82 = self.model[82](x80)
        x83 = self.model[83](x82)
        x84 = self.model[84](x83)
        x85 = self.model[85](x84)
        x86 = self.model[86](x85)
        x87 = self.model[87]([x86, x85, x84, x83, x82, x81])
        x88 = self.model[88](x87)
        x89 = self.model[89](x88)
        x90 = self.model[90](x89)
        x91 = self.model[91](x88)
        x92 = self.model[92](x91)
        x93 = self.model[93]([x92, x90, x51])
        x94 = self.model[94](x93)
        x95 = self.model[95](x93)
        x96 = self.model[96](x95)
        x97 = self.model[97](x96)
        x98 = self.model[98](x97)
        x99 = self.model[99](x98)
        x100 = self.model[100]([x99, x98, x97, x96, x95, x94])
        x101 = self.model[101](x100)
        x102 = self.model[102](x75)
        x103 = self.model[103](x88)
        x104 = self.model[104](x101)
        x105 = self.model[105]([x102, x103, x104])

        return x105
