# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import glob
import os
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torchvision
from loguru import logger


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scaleFill=True,
    scaleup=True,
    stride=32,
):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


class LoadImages:
    def __init__(self, path, img_size=640, stride=32):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())
            if os.path.isfile(a):
                files.append(a)
            else:
                raise FileNotFoundError(f"{p} does not exist")
        img_formats = ["jpg", "jpeg", "png", "bmp", "tiff"]
        images = [x for x in files if x.split(".")[-1].lower() in img_formats]
        ni = len(images)
        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni
        self.mode = "image"
        assert self.nf > 0, f"No images found in {p}. Supported formats are: {img_formats}"

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        img0 = cv2.imread(path)
        assert img0 is not None, "Image Not Found " + path
        self.count += 1
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return path, img, img0

    def __len__(self):
        return self.nf


def load_coco_class_names():
    url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    path = f"models/demos/yolov4/demo/coco.names"
    response = requests.get(url)
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip().split("\n")
    except requests.RequestException:
        pass
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    raise Exception("Failed to fetch COCO class names from both online and local sources.")


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])


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
            logger.info(f"WARNING: NMS time limit {time_limit}s exceeded")
            break
    return output


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


def preprocess(im):
    device = "cpu"
    img = torch.from_numpy(im)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def postprocess(
    preds, img, orig_imgs, batch, names, path, im0s, dataset, save_dir="models/demos/yolov7/demo/runs/detect"
):
    args = {"conf": 0.5, "iou": 0.7, "agnostic_nms": False, "max_det": 300, "classes": None}
    save_txt = False
    save_dir = Path(increment_path(Path(save_dir) / "exp", exist_ok=False))
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    pred = non_max_suppression(
        preds,
        args["conf"],
        args["iou"],
        classes=args["classes"],
        agnostic=args["agnostic_nms"],
    )
    results = []
    from models.experimental.yolo_eval.utils import Results

    for _, det in enumerate(pred):
        if det.numel() == 0:
            # If not prediction for a image
            results.append(Results(im0s, path=path, names=names, boxes=torch.full((1, 6), -1)))
            continue
        p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)
        p = Path(p)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        save_path = str(save_dir / p.name)
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            results.append(Results(im0s, path=path, names=names, boxes=det))
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            for *xyxy, conf, cls in reversed(det):
                # if save_img or view_img:
                conf = torch.Tensor(conf)
                conf = conf.to(torch.float32)
                label = f"{names[int(cls)]} {conf:.2f}"
                plot_one_box(
                    xyxy,
                    im0,
                    label=label,
                    color=colors[int(cls)],
                    line_thickness=1,
                )
        if dataset and dataset.mode == "image":
            cv2.imwrite(save_path, im0)
        logger.info(f"Predictions saved to {save_path}")

    return results
