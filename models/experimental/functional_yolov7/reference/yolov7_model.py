import argparse
import logging
import sys
from pathlib import Path
from copy import deepcopy
import math

sys.path.append("./")
logger = logging.getLogger(__name__)
import torch
from torch import nn
from models.experimental.functional_yolov7.reference.yolov7_utils import *
from models.experimental.functional_yolov7.reference.yolov7_config import config
import yaml

try:
    import thop
except ImportError:
    thop = None


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
        print(
            "input shape to detect: ",
            len(x),
            "first input:",
            x[0].shape,
            "second input:",
            x[1].shape,
            "third input:",
            x[2].shape,
        )
        z = []
        self.training |= self.export
        print("self.training: ", self.training)
        print("self.nl: ", self.nl)
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            print("before x[i].shape: ", x[i].shape)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            print("x[i]: ", x[i].shape)
            if not self.training:
                print("yes training")
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
            print("Training")
            out = x
        elif self.end2end:
            print("end2end")
            out = torch.cat(z, 1)
        elif self.include_nms:
            print("include_nms")
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            print("concat")
            out = torch.cat(z, 1)
        else:
            print("nothing")
            out = (torch.cat(z, 1), x)
        print("out shape:", out[0].shape)
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


class Model(nn.Module):
    def __init__(self, cfg=config, ch=3, nc=None, anchors=None):
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        print("ch : ", ch)
        if nc and nc != self.yaml["nc"]:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        if anchors:
            logger.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        # print("self.model: ", self.model)
        self.names = [str(i) for i in range(self.yaml["nc"])]
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()
        initialize_weights(self)

    def forward(self, x, augment=False, profile=False):
        if augment:
            print("Augmenting")
            img_size = x.shape[-2:]
            s = [1, 0.83, 0.67]
            f = [None, 3, None]
            y = []
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]
                yi[..., :4] /= si
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]
                y.append(yi)
            return torch.cat(y, 1), None
        else:
            print("No augmenting")
            return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if not hasattr(self, "traced"):
                self.traced = False
            if self.traced:
                if isinstance(m, Detect):
                    break
            if profile:
                c = isinstance(m, (Detect))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print("%10.1f%10.0f%10.1fms %-40s" % (o, m.np, dt[-1], m.type))
            x = m(x)
            y.append(x if m.i in self.save else None)
        if profile:
            print("%.1fms total" % sum(dt))
        return x

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def parse_model(d, ch):
    logger.info("\n%3s%18s%3s%10s  %-40s%-30s" % ("", "from", "n", "params", "module", "arguments"))
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, RepConv, SPPCSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)
                n = 1
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Detect]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        logger.info("%3s%18s%3s%10.0f  %-40s%-30s" % (i, f, n, np, t, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolor-csp-c.yaml", help="model.yaml")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)
    device = select_device(opt.device)
    model = Model(opt.cfg).to(device)
    model.train()

    if opt.profile:
        img = torch.rand(1, 3, 640, 640).to(device)
        y = model(img, profile=True)
