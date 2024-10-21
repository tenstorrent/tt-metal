import numpy as np
import torch
import torch.nn as nn
import glob
from pathlib import Path
import torchvision
import requests
import math
import datetime
import logging
import os
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F
import re
import sys

logger = logging.getLogger(__name__)


def check_anchor_order(m):
    a = m.anchor_grid.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        print("Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
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
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


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
                nn.BatchNorm2d(num_features=c2),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        out = self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        print("repconv out shape: ", out.shape)
        return out


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        w = Path(__file__).parent.parent / "tests" / "yolov7.pt"
        sys.modules["models.yolo"] = sys.modules["models.yolov7.reference.yolov7_model"]
        sys.modules["models.common"] = sys.modules["models.yolov7.reference.yolov7_utils"]
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()
    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


def check_file(file):
    if Path(file).is_file() or file == "":
        return file
    else:
        files = glob.glob("./**/" + file, recursive=True)
        assert len(files), f"File Not Found: {file}"
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"
        return files[0]


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])


def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    min_wh, max_wh = 2, 4096
    max_det = 300
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        if nc == 1:
            x[:, 5:] = x[:, 4:5]
        else:
            x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3e3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break
    return output


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def attempt_download(file, repo="WongKinYiu/yolov7"):
    tests = Path(__file__).parent.parent / "tests"
    file = tests / Path(str(file).strip().replace("'", "").lower())
    file = Path(str(file).strip().replace("'", "").lower())
    if not file.exists():
        assets = [
            "yolov7.pt",
            "yolov7-tiny.pt",
            "yolov7x.pt",
            "yolov7-d6.pt",
            "yolov7-e6.pt",
            "yolov7-e6e.pt",
            "yolov7-w6.pt",
        ]
        name = file.name
        if name in assets:
            msg = f"{file} missing, try downloading from https://github.com/{repo}/releases/"
            redundant = False
            try:
                url = f"https://github.com/{repo}/releases/download/v0.1/{name}"
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1e6
            except Exception as e:
                assert redundant, "No secondary mirror"
                url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
                print(f"Downloading {url} to {file}...")
                os.system(f"curl -L {url} -o {file}")
            finally:
                if not file.exists() or file.stat().st_size < 1e6:
                    file.unlink(missing_ok=True)
                    print(f"ERROR: Download failure: {msg}")
                return


def date_modified(path=__file__):
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def git_describe(path=Path(__file__).parent):
    s = f"git -C {path} describe --tags --long --always"
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ""


def select_device(device="", batch_size=None):
    s = f"YOLOR 🚀 {git_describe() or date_modified()} torch {torch.__version__} "
    cpu = device.lower() == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * len(s)
        for i, d in enumerate(device.split(",") if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"
    else:
        s += "CPU\n"

    logger.info(s.encode().decode("ascii", "ignore") if platform.system() == "Windows" else s)
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))
        img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)
        if not same_shape:
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        return


def revert_sync_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


class TracedModel(nn.Module):
    def __init__(self, model=None, device=None, img_size=(640, 640)):
        super(TracedModel, self).__init__()
        self.stride = model.stride
        self.names = model.names
        self.model = model
        self.model = revert_sync_batchnorm(self.model)
        self.model.to("cpu")
        self.model.eval()
        self.detect_layer = self.model.model[-1]
        self.model.traced = True
        rand_example = torch.rand(1, 3, img_size, img_size)
        traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
        self.model = traced_script_module
        self.model.to(device)
        self.detect_layer.to(device)

    def forward(self, x, augment=False, profile=False):
        out = self.model(x)
        out = self.detect_layer(out)
        return out


def increment_path(path, exist_ok=True, sep=""):
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{sep}{n}"
