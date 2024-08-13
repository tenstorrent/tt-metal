# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import warnings
from copy import deepcopy
from loguru import logger
from pathlib import Path
import contextlib
import math
from models.experimental.yolov3.reference.models.common import DetectMultiBackend
from models.experimental.yolov3.tt.yolov3_conv import TtConv
from models.experimental.yolov3.tt.yolov3_bottleneck import TtBottleneck
from models.experimental.yolov3.tt.yolov3_upsample import TtUpsample
from models.experimental.yolov3.tt.yolov3_concat import TtConcat
from models.experimental.yolov3.tt.yolov3_detect import TtDetect
from models.experimental.yolov3.reference.models.yolo import Segment
from models.experimental.yolov3.reference.utils.general import make_divisible
from models.experimental.yolov3.reference.utils.autoanchor import (
    check_anchor_order,
)
from models.experimental.yolov3.reference.utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


def parse_model(state_dict, base_address, yaml_dict, ch, device):  # model_dict, input_channels(3)
    d = yaml_dict

    # Parse a YOLOv3 model.yaml dictionary
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
        # m = eval(m) if isinstance(m, str) else m  # eval strings

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

            m = eval(f"Tt{m}")

        elif m == "Concat":
            c2 = sum(ch[x] for x in f)
            m = TtConcat

        # TODO: channel, gw, gd
        elif m in {"Detect", "Segment"}:
            args.append([ch[x] for x in f])
            m = TtDetect

            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

            if m == "Segment":
                raise NotImplementedError
                args[3] = make_divisible(args[3] * gw, 8)

        elif m == "nn.Upsample":
            c2 = ch[f]
            m = TtUpsample

        else:
            c2 = ch[f]

        args.insert(0, f"{base_address}.{i}")
        args.insert(0, state_dict)
        args.insert(0, device)

        if n > 1:
            list_modules = []
            for iter_layer in range(n):
                args[2] = f"{base_address}.{i}.{iter_layer}"
                list_modules.append((m(*args)))

            m_ = nn.Sequential(*list_modules)
        else:
            m_ = nn.Sequential(m(*args))

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
    # YOLOv3 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
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
            # logger.info(f'Module type  {type(m)}')
            # if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
            if hasattr(m, "bn") and hasattr(m, "conv") and hasattr(m, "forward_fuse"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # logger.info('fuse_conv_and_bn... ')
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
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class TtDetectionModel(BaseModel):
    # YOLOv3 detection model
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        cfg="yolov3.yaml",
        ch=3,
        nc=None,
        anchors=None,
        stride=None,
    ):  # model, input channels, number of classes
        super().__init__()
        self.device = device

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
        if anchors is not None:
            logger.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value

        self.model, self.save = parse_model(
            state_dict, base_address, deepcopy(self.yaml), ch=[ch], device=self.device
        )  # model, savelist

        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Get llast module and Build strides, anchors
        m = self.model[-1][0]  # Detect()
        if isinstance(m, (TtDetect, Segment)):
            logger.info("Initialize strides and anchors")
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in forward(torch2tt_tensor(torch.zeros(1, ch, s, s), self.device))]
            )  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride

        self.info()
        logger.info("Initialization compelted")

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        raise NotImplementedError

        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def _yolov3_fused_model(cfg_path, state_dict, base_address, device) -> TtDetectionModel:
    tt_model = TtDetectionModel(
        cfg=cfg_path,
        state_dict=state_dict,
        base_address=base_address,
        device=device,
    )
    return tt_model


def yolov3_fused_model(device, model_location_generator) -> TtDetectionModel:
    # Load yolo
    model_path = model_location_generator("models", model_subdir="Yolo")
    data_path = model_location_generator("data", model_subdir="Yolo")
    cfg_path = str(data_path / "yolov3.yaml")
    data_coco = str(data_path / "coco128.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False)

    tt_model = _yolov3_fused_model(
        cfg_path=cfg_path,
        state_dict=reference_model.state_dict(),
        base_address="model.model",
        device=device,
    )

    return tt_model
