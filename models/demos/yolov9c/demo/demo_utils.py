# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from loguru import logger
from ultralytics import YOLO

from models.demos.yolov9c.reference.yolov9c import YoloV9
from models.experimental.yolo_eval.utils import non_max_suppression, scale_boxes


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


def load_torch_model(use_weights_from_ultralytics=True, module=None, model_task="segment"):
    state_dict = None

    weights = "yolov9c-seg.pt" if model_task == "segment" else "yolov9c.pt"
    enable_segment = model_task == "segment"

    if use_weights_from_ultralytics:
        torch_model = YOLO(weights)  # Use "yolov9c.pt" weight for detection
        torch_model.eval()
        state_dict = torch_model.state_dict()

    model = YoloV9(enable_segment=enable_segment)
    state_dict = model.state_dict() if state_dict is None else state_dict

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2

    model.load_state_dict(new_state_dict)
    model.eval()

    return model


def get_consistent_color(index):
    cmap = plt.get_cmap("tab20")
    color = cmap(index % 20)[:3]
    return tuple(int(c * 255) for c in color)


def save_seg_predictions_by_model(result, save_dir, image_path, model_name):
    os.makedirs(os.path.join(save_dir, model_name), exist_ok=True)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = result.masks.data.cpu().detach().numpy()
    mask_h, mask_w = masks.shape[1], masks.shape[2]

    image = cv2.resize(image, (mask_w, mask_h))
    overlay = image.copy()

    for i in range(len(masks)):
        mask = masks[i]
        color = get_consistent_color(i)
        mask_rgb = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            mask_rgb[:, :, c] = (mask * color[c]).astype(np.uint8)

        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * mask_rgb[mask_bool]).astype(np.uint8)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(save_dir, model_name, f"segmentation_{timestamp}.jpg")
    cv2.imwrite(out_path, overlay_bgr)
    logger.info(f"Saved to {out_path}")


# For Segmentation task


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}


class LetterBox:
    def __init__(self, new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = False
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        h, w, c = img.shape
        if c == 3:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img


# Postprocess
class SimpleClass:
    def __str__(self):
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class BaseTensor(SimpleClass):
    def __init__(self, data, orig_shape) -> None:
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        return self.data.shape

    def cpu(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def to(self, *args, **kwargs):
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        return len(self.data)

    def __getitem__(self, idx):
        return self.__class__(self.data[idx], self.orig_shape)


class Boxes(BaseTensor):
    def __init__(self, boxes, orig_shape) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def conf(self):
        return self.data[:, -2]

    @property
    def cls(self):
        return self.data[:, -1]

    @property
    def id(self):
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    def __init__(self, masks, orig_shape) -> None:
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]


class Results:
    def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, speed=None
    ) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        return self._apply("__getitem__", idx)

    def __len__(self):
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None, keypoints=None):
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)
        if keypoints is not None:
            self.keypoints = Keypoints(keypoints, self.orig_shape)

    def _apply(self, fn, *args, **kwargs):
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        return self._apply("cpu")

    def numpy(self):
        return self._apply("numpy")

    def to(self, *args, **kwargs):
        return self._apply("to", *args, **kwargs)

    def new(self):
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
        color_mode="class",
        txt_color=(255, 255, 255),
    ):
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()
        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.id is not None and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)
        if show:
            annotator.show(self.path)
        if save:
            annotator.save(filename or f"results_{Path(self.path).name}")

        return annotator.im if pil else annotator.result()

    def show(self, *args, **kwargs):
        self.plot(show=True, *args, **kwargs)

    def save(self, filename=None, *args, **kwargs):
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self):
        log_string = ""
        probs = self.probs
        if len(self) == 0:
            return log_string if probs is not None else f"{log_string}(no detections), "
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes := self.boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(text + "\n" for text in texts)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        if self.probs is not None:
            LOGGER.warning("Classify task do not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("OBB task do not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / Path(file_name).with_suffix(".jpg"),
                BGR=True,
            )


def crop_mask(masks, boxes):
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def scale_masks(masks, shape, padding=True):
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)
    return masks


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    c, mh, mw = protos.shape
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]
    return masks.gt_(0.0)


def process_mask_native(protos, masks_in, bboxes, shape):
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)


def construct_result(pred, img, orig_img, img_path, proto, retina_masks=True):
    if not len(pred):
        masks = None
    elif retina_masks:
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        masks = process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])
    else:
        masks = process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    if masks is not None:
        keep = masks.sum((-2, -1)) > 0
        pred, masks = pred[keep], masks[keep]

    return Results(orig_img, path=img_path, names={0: "object"}, boxes=pred[:, :6], masks=masks)


def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    h, w = image_shape
    boxes[boxes[:, 0] < threshold, 0] = 0
    boxes[boxes[:, 1] < threshold, 1] = 0
    boxes[boxes[:, 2] > w - threshold, 2] = w
    boxes[boxes[:, 3] > h - threshold, 3] = h
    return boxes


def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def postprocess(preds, img, orig_imgs, batch):
    args = {
        "conf": 0.25,
        "iou": 0.7,
        "agnostic_nms": False,
        "max_det": 300,
        "classes": None,
        "nc": 80,
        "end2end": False,
        "rotated": False,
        "return_idxs": False,
    }

    protos = preds[1][-1]
    preds = preds[0]

    preds = non_max_suppression(
        preds,
        args["conf"],
        args["iou"],
        args["classes"],
        args["agnostic_nms"],
        max_det=args["max_det"],
        nc=args["nc"],
        rotated=args["rotated"],
    )

    results = []
    for pred, orig_img, img_path, proto in zip(preds, orig_imgs, batch[0], protos):
        results.append(construct_result(pred, img, orig_img, img_path, proto))

    for result in results:
        full_box = torch.tensor(
            [0, 0, result.orig_shape[1], result.orig_shape[0]], device=preds[0].device, dtype=torch.float32
        )
        boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
        idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
        if idx.numel() != 0:
            result.boxes.xyxy[idx] = full_box

    return results
