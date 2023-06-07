# YOLOv3 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv3 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import argparse
import os
import platform
import sys
from pathlib import Path

import torchvision
import cv2
import torch
import torch.nn as nn
import numpy as np
import math
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from typing import Optional
import inspect
import yaml
from urllib.parse import urlparse
from PIL import Image, ImageDraw
import glob

from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.yolov3.modeling_yolo import Detect, Model

from loguru import logger

LOGGER = logger

from python_api_testing.models.yolov3.yolov3_common import (
    model_location_generator,
    LoadImages,
    check_img_size,
    IMG_FORMATS,
    VID_FORMATS,
)


################### General ###########################
def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv3 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4]
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[
            x[:, 4].argsort(descending=True)[:max_nms]
        ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_file(file, suffix=""):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file
    elif file.startswith(("http:/", "https:/")):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(
            urllib.parse.unquote(file).split("?")[0]
        ).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            LOGGER.info(f"Found {url} locally at {file}")  # file already exists
        else:
            LOGGER.info(f"Downloading {url} to {file}...")
            torch.hub.download_url_to_file(url, file)
            assert (
                Path(file).exists() and Path(file).stat().st_size > 0
            ), f"File download failed: {url}"  # check
        return file
    elif file.startswith("clearml://"):  # ClearML Dataset ID
        assert (
            "clearml" in sys.modules
        ), "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:  # search
        files = []
        for d in "data", "models", "utils":  # search directories
            files.extend(
                glob.glob(str(ROOT / d / "**" / file), recursive=True)
            )  # find file
        assert len(files), f"File not found: {file}"  # assert file was found
        assert (
            len(files) == 1
        ), f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


#################### PLOTS ############################
def save_one_box(
    xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True
):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[
        int(xyxy[0, 1]) : int(xyxy[0, 3]),
        int(xyxy[0, 0]) : int(xyxy[0, 2]),
        :: (1 if BGR else -1),
    ]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop


def is_ascii(s=""):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode("ascii", "ignore")) == len(s)


class Annotator:
    # YOLOv3 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, pil=False, example="abc"):
        assert (
            im.data.contiguous
        ), "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images."
        non_ascii = not is_ascii(
            example
        )  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = (cv2.FONT_HERSHEY_DUPLEX,)
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(
        self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)
    ):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(
                    label
                )  # text width, height (WARNING: deprecated) in 9.2.0
                # _, _, w, h = self.font.getbbox(label)  # text width, height (New)
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (
                        box[0],
                        box[1] - h if outside else box[1],
                        box[0] + w + 1,
                        box[1] + 1 if outside else box[1] + h + 1,
                    ),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text(
                    (box[0], box[1] - h if outside else box[1]),
                    label,
                    fill=txt_color,
                    font=self.font,
                )
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(
                self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA
            )
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[
                    0
                ]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
        """
        if self.pil:
            # convert to numpy first
            self.im = np.asarray(self.im).copy()
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        colors = torch.tensor(colors, device=im_gpu.device, dtype=torch.float32) / 255.0
        colors = colors[:, None, None]  # shape(n,1,1,3)
        masks = masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

        inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = (masks_color * inv_alph_masks).sum(
            0
        ) * 2  # mask color summand shape(n,h,w,3)

        im_gpu = im_gpu.flip(dims=[0])  # flip channel
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs
        im_mask = (im_gpu * 255).byte().cpu().numpy()
        self.im[:] = (
            im_mask
            if retina_masks
            else scale_image(im_gpu.shape, im_mask, self.im.shape)
        )
        if self.pil:
            # convert im back to PIL and update draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top"):
        # Add text to image (PIL-only)
        if anchor == "bottom":  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        # Update self.im from a numpy array
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


##################### COLORS ################################
def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


#################### MODEL AND WEIGHT LOADING #############################
class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_download(file, repo="ultralytics/yolov5", release="v7.0"):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v7.0', etc.

    def github_assets(repository, version="latest"):
        # Return GitHub repo tag (i.e. 'v7.0') and assets (i.e. ['yolov5s.pt', 'yolov5m.pt', ...])
        if version != "latest":
            version = f"tags/{version}"  # i.e. tags/v7.0
        response = requests.get(
            f"https://api.github.com/repos/{repository}/releases/{version}"
        ).json()  # github api
        return response["tag_name"], [
            x["name"] for x in response["assets"]
        ]  # tag, assets

    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = name.split("?")[
                0
            ]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"Found {url} locally at {file}")  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file

        # GitHub assets
        assets = [
            f"yolov5{size}{suffix}.pt"
            for size in "nsmlx"
            for suffix in ("", "6", "-cls", "-seg")
        ]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = (
                        subprocess.check_output(
                            "git tag", shell=True, stderr=subprocess.STDOUT
                        )
                        .decode()
                        .split()[-1]
                    )
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        if name in assets:
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}",
            )

    return str(file)


# def attempt_load(weights, device=None, inplace=True, fuse=True):
#     # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a

#     model = Ensemble()
#     for w in weights if isinstance(weights, list) else [weights]:
#         ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
#         print(ckpt)
#         # ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

#         # Model compatibility updates
#         if not hasattr(ckpt, 'stride'):
#             ckpt.stride = torch.tensor([32.])
#         if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
#             ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

#         # model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

#     # Module compatibility updates
#     for m in model.modules():
#         t = type(m)
#         if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
#             m.inplace = inplace  # torch 1.7.0 compatibility
#             if t is Detect and not isinstance(m.anchor_grid, list):
#                 delattr(m, 'anchor_grid')
#                 setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
#         elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
#             m.recompute_scale_factor = None  # torch 1.11.0 compatibility

#     # Return model
#     if len(model) == 1:
#         return model[-1]

#     # Return detection ensemble
#     print(f'Ensemble created with {weights}\n')
#     for k in 'names', 'nc', 'yaml':
#         setattr(model, k, getattr(model[0], k))
#     model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
#     assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
#     return model


def yaml_load(file="data.yaml"):
    # Single-line safe yaml loading
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)


def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


class DetectMultiBackend(nn.Module):
    # YOLOv3 MultiBackend class for python inference on various backends
    def __init__(
        self,
        weights_loc,
        model_cfg,
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        fuse=True,
    ):
        super().__init__()

        # w = str(weights[0] if isinstance(weights, list) else weights)
        self.stride = 32  # default stride
        self.pt = True

        model = Model(model_cfg).fuse().eval()

        logger.info("Loading weights....")
        weights = torch.load(weights_loc)["model"]

        model.load_state_dict(state_dict=weights, strict=False)
        self.device = "cpu"
        self.stride = max(int(model.stride.max()), 32)  # model stride
        names = (
            model.module.names if hasattr(model, "module") else model.names
        )  # get class names
        model.half() if fp16 else model.float()
        self.model = model
        self.names = (
            yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        )
        self.fp16 = fp16

        # class names
        if "names" not in locals():
            self.names = (
                yaml_load(data)["names"]
                if data
                else {i: f"class{i}" for i in range(999)}
            )

        if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
            self.names = yaml_load(ROOT / "data/ImageNet.yaml")[
                "names"
            ]  # human-readable names

        # self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv3 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        # if self.nhwc:
        #     im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = (
                self.model(im, augment=augment, visualize=visualize)
                if augment or visualize
                else self.model(im)
            )

        if isinstance(y, (list, tuple)):
            return (
                self.from_numpy(y[0])
                if len(y) == 1
                else [self.from_numpy(x) for x in y]
            )
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        pass

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None


def run(
    weights=ROOT / "yolov3NEW.tar",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Load model
    model_path = model_location_generator("tt_dnn-models/Yolo/models/")
    weights_loc = str(model_path / "yolov3_weights.pt")
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")
    model_config_path = str(data_path / "yolov3.yaml")

    device = torch.device("cpu")
    model = DetectMultiBackend(
        weights_loc=weights_loc,
        model_cfg=model_config_path,
        device=device,
        dnn=dnn,
        data=data,
        fp16=half,
    )

    logger.info(model)

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size

    data_image_path = str(data_path / "images")

    dataset = LoadImages(
        data_image_path, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride
    )
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows = 0, []
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference

        visualize = (
            increment_path(save_dir / Path(path).stem, mkdir=True)
            if visualize
            else False
        )
        pred = model(im, augment=augment, visualize=visualize)

        # NMS

        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (
                            (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        )  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = (
                            None
                            if hide_labels
                            else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        )
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(
                            xyxy,
                            imc,
                            file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
                            BGR=True,
                        )

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(
                        str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                    )  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(
                            Path(save_path).with_suffix(".mp4")
                        )  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


def parse_opt():
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov3.pt",
        help="model path or triton URL",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(data_path / "images"),
        help="file/dir/URL/glob/screen/0(webcam)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(data_path / "coco128.yaml"),
        help="(optional) dataset.yaml path",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--vid-stride", type=int, default=1, help="video frame-rate stride"
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    logger.info(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
