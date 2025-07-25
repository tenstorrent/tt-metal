# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision
from loguru import logger


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}


class LoadImages:
    def __init__(self, path, batch=1, vid_stride=1):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())
            if os.path.isfile(a):
                files.append(a)
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = []
        for f in files:
            suffix = f.split(".")[-1].lower()
            if suffix in IMG_FORMATS:
                images.append(f)
        ni = len(images)

        self.files = images
        self.nf = ni
        self.ni = ni
        self.mode = "image"
        self.vid_stride = vid_stride
        self.bs = batch
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}")

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:
                if imgs:
                    return paths, imgs, info
                else:
                    raise StopIteration

            path = self.files[self.count]

            self.mode = "image"
            im0 = imread(path)
            if im0 is None:
                logger.warning(f"WARNING ⚠️ Image Read Error {path}")
            else:
                paths.append(path)
                imgs.append(im0)
                info.append(f"image {self.count + 1}/{self.nf} {path}: ")
            self.count += 1
            if self.count >= self.ni:
                break

        return paths, imgs, info

    def _new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        return math.ceil(self.nf / self.bs)


def LetterBox(img, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    if center:
        dw /= 2
        dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img


def pre_transform(im):
    return [LetterBox(img=x) for x in im]


def preprocess(im):
    device = "cpu"
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im)

    im = im.half() if device != "cpu" else im.float()
    if not_tensor:
        im /= 255
    return im


def empty_like(x):
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )


def postprocess(preds, img, orig_imgs, batch, names):
    nc = 80
    max_det = 300
    args = {"conf": 0.5, "iou": 0.7, "agnostic_nms": False, "max_det": 300, "classes": None}
    preds = preds.permute(0, 2, 1)
    batch_size, anchors, _ = preds.shape
    boxes, scores = preds.split([4, nc], dim=-1)
    index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
    boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
    scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
    scores, index = scores.flatten(1).topk(min(max_det, anchors))
    i = torch.arange(batch_size)[..., None]
    preds = torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
    preds = non_max_suppression(
        preds,
        args["conf"],
        args["iou"],
        agnostic=args["agnostic_nms"],
        max_det=args["max_det"],
        classes=args["classes"],
    )

    results = []
    for pred, orig_img, img_path in zip(preds, orig_imgs, batch[0]):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(Results(orig_img, path=img_path, names=names, boxes=pred))

    return results


def empty_like(x):
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )


def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
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
    nc=0,
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    prediction = prediction.transpose(-1, -2)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]

        boxes = x[:, :4] + c
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            logger.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break

    return output


def Boxes(data):
    return {"xyxy": data[:, :4], "conf": data[:, -2], "cls": data[:, -1]}


def Results(orig_img, path, names, boxes):
    return {"orig_img": orig_img, "path": path, "names": names, "boxes": Boxes(boxes)}


def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]
        boxes[..., 1] -= pad[1]
        if not xywh:
            boxes[..., 2] -= pad[0]
            boxes[..., 3] -= pad[1]
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_download(file, repo="ultralytics/assets"):
    tests = Path(__file__).parent.parent / "reference"
    file_path = tests / Path(str(file).strip().replace("'", "").lower())

    if not file_path.exists():
        name = "yolov10x.pt"  # file_path.name
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"

        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"

            print(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            # Validate the file
            assert file_path.exists() and file_path.stat().st_size > 1e6, f"Download failed for {name}"

        except Exception as e:
            print(f"Error downloading from GitHub: {e}. Trying secondary source...")

            url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
            print(f"Downloading {url} to {file_path}...")
            os.system(f"curl -L {url} -o {file_path}")

            if not file_path.exists() or file_path.stat().st_size < 1e6:
                file_path.unlink(missing_ok=True)
                print(f"ERROR: Download failure for {msg}")
            else:
                print(f"Download succeeded from secondary source!")
    return file_path


# Function to load weights into the model
def attempt_load(weights, map_location=None):
    model = Ensemble()

    # Iterate through the weights and load them
    for w in weights if isinstance(weights, list) else [weights]:
        weight_path = attempt_download(w)
        ckpt = torch.load(weight_path, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True
        elif isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


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


def save_yolo_predictions_by_model(result, save_dir, image_path, model_name):
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model_name == "torch_model":
        bounding_box_color, label_color = (0, 255, 0), (0, 255, 0)
    else:
        bounding_box_color, label_color = (255, 0, 0), (255, 0, 0)

    boxes = result["boxes"]["xyxy"]
    scores = result["boxes"]["conf"]
    classes = result["boxes"]["cls"]
    names = result["names"]

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {score.item():.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), bounding_box_color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"prediction_{timestamp}.jpg"
    output_path = os.path.join(model_save_dir, output_name)
    cv2.imwrite(output_path, image)

    logger.info(f"Predictions saved to {output_path}")
