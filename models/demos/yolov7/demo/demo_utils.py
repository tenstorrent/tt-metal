# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import glob
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from loguru import logger


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


def postprocess(preds, img, orig_imgs, batch, names, path, dataset, save_dir="models/demos/yolov7/demo/runs/detect"):
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
    from models.demos.utils.common_demo_utils import Results

    for i, det in enumerate(pred):
        im0 = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        path_i = path[i] if isinstance(path, (list, tuple)) else path
        save_path = str(save_dir / Path(path_i).name)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        s = ""

        if det is None or det.numel() == 0:
            results.append(Results(im0, path=path_i, names=names, boxes=torch.full((1, 6), -1)))
            continue

        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        results.append(Results(im0, path=path_i, names=names, boxes=det))

        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

        for *xyxy, conf, cls in reversed(det):
            label = f"{names[int(cls)]} {conf:.2f}"
            plot_one_box(
                xyxy,
                im0,
                label=label,
                color=colors[int(cls)],
                line_thickness=1,
            )

        if dataset and getattr(dataset, "mode", "") == "image":
            cv2.imwrite(save_path, im0)
        logger.info(f"Predictions saved to {save_path}")

    return results
