import argparse
import time
from pathlib import Path
import os
import cv2
import numpy as np
import torch
from numpy import random
from models.experimental.functional_yolov7.reference.yolov7_utils import *


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
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
    elif scaleFill:  # stretch
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
        p = str(Path(path).absolute())
        if os.path.isfile(p):
            files = [p]
        else:
            raise Exception(f"ERROR: {p} does not exist")
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
        return path, img, img0, None

    def __len__(self):
        return self.nf


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


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


def detect(opt, model, save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
        not opt.no_trace,
    )
    save_img = not opt.nosave and not source.endswith(".txt")
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    device = select_device(opt.device)
    half = device.type != "cpu"
    if trace:
        model = TracedModel(model, device, opt.img_size)
    dataset = LoadImages(source, img_size=imgsz, stride=32)
    names = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    t0 = time.time()
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        print("initial img shape: ", img.shape)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            print("img shape: ", img.shape)
            torch.save(img, "yolov7_input.pt")
            pred = model(img, augment=opt.augment)[0]
            print("pred shape: ", pred.shape)
    #     pred = non_max_suppression(
    #         pred,
    #         opt.conf_thres,
    #         opt.iou_thres,
    #         classes=opt.classes,
    #         agnostic=opt.agnostic_nms,
    #     )
    #     for _, det in enumerate(pred):
    #         p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)
    #         p = Path(p)
    #         save_path = str(save_dir / p.name)
    #         if len(det):
    #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    #             for c in det[:, -1].unique():
    #                 n = (det[:, -1] == c).sum()
    #                 s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
    #             for *xyxy, conf, cls in reversed(det):
    #                 if save_img or view_img:
    #                     label = f"{names[int(cls)]} {conf:.2f}"
    #                     plot_one_box(
    #                         xyxy,
    #                         im0,
    #                         label=label,
    #                         color=colors[int(cls)],
    #                         line_thickness=1,
    #                     )
    #         if save_img:
    #             if dataset.mode == "image":
    #                 cv2.imwrite(save_path, im0)

    # print(f"Done. ({time.time() - t0:.3f}s)")
    return pred


def parse_opt(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov7.pt", help="model.pt path(s)")
    parser.add_argument("--source", type=str, default="inference/images", help="source")
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")

    opt = parser.parse_args(args)
    return opt
