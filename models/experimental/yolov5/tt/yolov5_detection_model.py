# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from copy import deepcopy
from loguru import logger

from models.utility_functions import torch2tt_tensor

from models.experimental.yolov5.reference.models.common import DetectMultiBackend
from models.experimental.yolov5.tt.yolov5_conv import TtYolov5Conv
from models.experimental.yolov5.tt.yolov5_c3 import TtYolov5C3
from models.experimental.yolov5.tt.yolov5_bottleneck import TtYolov5Bottleneck
from models.experimental.yolov5.tt.yolov5_sppf import TtYolov5SPPF
from models.experimental.yolov5.tt.yolov5_upsample import TtYolov5Upsample
from models.experimental.yolov5.tt.yolov5_concat import TtYolov5Concat
from models.experimental.yolov5.tt.yolov5_detect import TtYolov5Detect
from pathlib import Path

import contextlib
import math

from models.experimental.yolov5.reference.utils.general import make_divisible
from models.experimental.yolov5.reference.utils.autoanchor import (
    check_anchor_order,
)

from models.experimental.yolov5.reference.utils.torch_utils import (
    fuse_conv_and_bn,
    model_info,
    profile,
    scale_img,
    time_sync,
)


def parse_model(state_dict, base_address, yaml_dict, ch, device):  # model_dict, input_channels(3)
    # with open(yaml_file_name, encoding="ascii", errors="ignore") as f:
    #     d = yaml.safe_load(f)  # model dict

    d = yaml_dict

    # Parse a YOLOv5 model.yaml dictionary
    logger.info(f"{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
    )

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        logger.info(f"{colorstr('activation:')} {act}")  # print

    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain

        if m in {
            "Conv",
            "Bottleneck",
            "SPPF",
            "C3",
        }:
            c1, c2 = ch[f], args[0]

            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]

            if m in {"BottleneckCSP", "C3", "C3TR", "C3Ghost", "C3x"}:
                args.insert(2, n)  # number of repeats
                n = 1

            m = eval(f"TtYolov5{m}")

        elif m == "Concat":
            c2 = sum(ch[x] for x in f)
            m = TtYolov5Concat

        # TODO: channel, gw, gd
        elif m in {"Detect", "Segment"}:
            args.append([ch[x] for x in f])

            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

            if m == "Detect":
                m = TtYolov5Detect
            elif m == "Segment":
                args[3] = make_divisible(args[3] * gw, 8)
            else:
                assert False  # Something went wrong

        elif m == "nn.Upsample":
            c2 = ch[f]
            m = TtYolov5Upsample

        else:
            c2 = ch[f]

        args.insert(0, device)
        args.insert(0, f"{base_address}.{i}")
        args.insert(0, state_dict)

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)

        if i == 0:
            ch = []

        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for i, m in enumerate(self.model):
            logger.debug(f"Running layer {i}")

            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        logger.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            logger.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info("Fusing layers... ")
        for m in self.model.modules():
            if hasattr(m, "bn") and hasattr(m, "conv") and hasattr(m, "forward_fuse"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()

        if isinstance(m, TtYolov5Detect):  # (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class TtYolov5DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(
        self,
        state_dict,
        base_address,
        device,
        cfg="yolov5s.yaml",
        ch=3,
        nc=None,
        anchors=None,
    ):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            logger.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value

        self.model, self.save = parse_model(
            state_dict, base_address, deepcopy(self.yaml), ch=[ch], device=device
        )  # model, savelist

        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()

        if isinstance(m, TtYolov5Detect):  # (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            forward = lambda x: self.forward(x) if isinstance(m, TtYolov5Detect) else self.forward(x)[0]

            zeros_tensor = torch2tt_tensor(torch.zeros(1, ch, s, s), device)
            forwaded_zeros = forward(zeros_tensor)

            m.stride = torch.tensor([s / x.shape[-2] for x in forwaded_zeros])  # forward

            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride

        self.info()

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None

        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = (
                p[..., 0:1] / scale,
                p[..., 1:2] / scale,
                p[..., 2:4] / scale,
            )  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module

        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def _yolov5_detection_model(cfg_path, state_dict, base_address, device) -> TtYolov5DetectionModel:
    tt_model = TtYolov5DetectionModel(
        cfg=cfg_path,
        state_dict=state_dict,
        base_address=base_address,
        device=device,
    )
    return tt_model


def yolov5s_detection_model(device) -> TtYolov5DetectionModel:
    cfg_path = "models/experimental/yolov5/reference/yolov5s.yaml"
    weights = "models/experimental/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False

    refence_model = DetectMultiBackend(weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half)

    tt_model = TtYolov5DetectionModel(
        cfg=cfg_path,
        state_dict=refence_model.state_dict(),
        base_address="model.model",
        device=device,
    )

    return tt_model
