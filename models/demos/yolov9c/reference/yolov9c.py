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


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


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
        enable_identity=False,
        enable_bn=True,
        enable_autopad=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel,
            stride=stride,
            padding=autopad(k=kernel, p=None, d=dilation) if enable_autopad else padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        if enable_bn:
            self.bn = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.03)
        if enable_identity:
            self.act = nn.Identity()
        else:
            self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
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
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = [8, 16, 32]
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)

        self.dfl = DFL(16)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        y = self._inference(x)
        return (y, x)

    def _inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dfl = self.dfl(box)
        cls = cls.sigmoid()

        dbox = self.decode_bboxes(dfl, self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls), 1)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        return dist2bbox(bboxes, anchors, dim=1)


class Proto(nn.Module):
    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = Conv(c1, c_, kernel=3, enable_autopad=True)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, kernel=3, enable_autopad=True)
        self.cv3 = Conv(c_, c2, enable_autopad=True)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Segment(Detect):
    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = Detect.forward(self, x)
        return (torch.cat([x[0], mc], 1), (x[1], mc, p))


class YoloV9(nn.Module):
    def __init__(self, enable_segment=True):
        super().__init__()
        if enable_segment:
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
                nn.Upsample(scale_factor=2.0, mode="nearest"),  # 10
                Concat(),  # 11
                RepNCSPELAN4(1024, 512, 256, 256, 256, 256, 1024, 512),  # 12
                nn.Upsample(scale_factor=2.0, mode="nearest"),  # 13
                Concat(),  # 14
                RepNCSPELAN4(1024, 256, 128, 128, 128, 128, 512, 256),  # 15
                ADown(128, 128),  # 16
                Concat(),  # 17
                RepNCSPELAN4(768, 512, 256, 256, 256, 256, 1024, 512),  # 18
                ADown(256, 256),  # 19
                Concat(),  # 20
                RepNCSPELAN4(1024, 512, 256, 256, 256, 256, 1024, 512),  # 21
                Segment(80, 32, 256, [256, 512, 512]),  # 22 Enable this to use the model only for instance segmentation
            )
        else:
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
                nn.Upsample(scale_factor=2.0, mode="nearest"),  # 10
                Concat(),  # 11
                RepNCSPELAN4(1024, 512, 256, 256, 256, 256, 1024, 512),  # 12
                nn.Upsample(scale_factor=2.0, mode="nearest"),  # 13
                Concat(),  # 14
                RepNCSPELAN4(1024, 256, 128, 128, 128, 128, 512, 256),  # 15
                ADown(128, 128),  # 16
                Concat(),  # 17
                RepNCSPELAN4(768, 512, 256, 256, 256, 256, 1024, 512),  # 18
                ADown(256, 256),  # 19
                Concat(),  # 20
                RepNCSPELAN4(1024, 512, 256, 256, 256, 256, 1024, 512),  # 21
                Detect(
                    nc=80,
                    ch=(256, 512, 512),
                ),  # 22 Enable this to use the model only for object detection
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
        x = f.upsample(x, scale_factor=2.0)  # 10
        x = torch.cat((x, x6), 1)  # 11
        x = self.model[12](x)  # 12
        x13 = x
        x = f.upsample(x, scale_factor=2.0)  # 13
        x = torch.cat((x, x4), 1)  # 14
        x = self.model[15](x)  # 15
        x16 = x
        x = self.model[16](x)  # 16
        x = torch.cat((x, x13), 1)  # 17
        x = self.model[18](x)  # 18
        x19 = x
        x = self.model[19](x)  # 19
        x = torch.cat((x, x10), 1)  # 20
        x = self.model[21](x)  # 21
        x22 = x
        x = self.model[22]([x16, x19, x22])  # 22

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


class SegmentModel(BaseModel):
    def __init__(self, cfg="yolov9c-seg.pt", ch=3, nc=None, verbose=True):
        super().__init__()
