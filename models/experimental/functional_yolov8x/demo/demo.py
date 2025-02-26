# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import cv2
import sys
import ttnn
import torch
import pytest
from pathlib import Path
import torch.nn as nn
from loguru import logger
from datetime import datetime
from models.experimental.functional_yolov8x.reference import yolov8x
from models.experimental.functional_yolov8x.reference import yolov8x_utils
from models.experimental.functional_yolov8x.tt.ttnn_yolov8x import YOLOv8xModel
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.functional_yolov8x.tt.ttnn_yolov8x_utils import custom_preprocessor
from models.experimental.functional_yolov8x.demo.demo_utils import LoadImages, preprocess, postprocess

try:
    sys.modules["ultralytics"] = yolov8x_utils
    sys.modules["ultralytics.nn.tasks"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov8x_utils

except KeyError:
    logger.info("models.experimental.functional_yolov8x.reference.yolov8x_utils not found.")


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
        name = "yolov8x.pt"
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"
        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"
            logger.info(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            assert file_path.exists() and file_path.stat().st_size > 1e6, f"Download failed for {name}"
        except Exception as e:
            logger.info(f"Error downloading from GitHub: {e}. Trying secondary source...")

            url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
            logger.info(f"Downloading {url} to {file_path}...")
            os.system(f"curl -L {url} -o {file_path}")

            if not file_path.exists() or file_path.stat().st_size < 1e6:
                file_path.unlink(missing_ok=True)
                logger.info(f"ERROR: Download failure for {msg}")
            else:
                logger.info(f"Download succeeded from secondary source!")
    return file_path


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        weight_path = attempt_download(w)
        logger.info("Loading weights from:", weight_path)
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


def save_yolo_predictions_by_model(result, save_dir, image_path, model_name):
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model_name == "torch_model":
        bounding_box_color, label_color = (0, 255, 0), (0, 255, 0)
    else:
        bounding_box_color, label_color = (255, 0, 0), (255, 255, 0)

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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "source, model_type",
    [
        ("models/experimental/functional_yolov8x/demo/images/bus.jpg", "torch_model"),
        ("models/experimental/functional_yolov8x/demo/images/bus.jpg", "tt_model"),
        ("models/experimental/functional_yolov8x/demo/images/test1.jpg", "torch_model"),
        ("models/experimental/functional_yolov8x/demo/images/test1.jpg", "tt_model"),
        ("models/experimental/functional_yolov8x/demo/images/test2.jpg", "torch_model"),
        ("models/experimental/functional_yolov8x/demo/images/test2.jpg", "tt_model"),
        ("models/experimental/functional_yolov8x/demo/images/test3.jpg", "torch_model"),
        ("models/experimental/functional_yolov8x/demo/images/test3.jpg", "tt_model"),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo(device, source, model_type, res, use_pretrained_weight):
    disable_persistent_kernel_cache()

    if model_type == "torch_model":
        if use_pretrained_weight:
            model = attempt_load("yolov8x.pt", map_location="cpu")
        else:
            model = yolov8x.DetectionModel()

        logger.info("Inferencing using Torch Model")
    else:
        if use_pretrained_weight:
            state_dict = attempt_load("yolov8x.pt", map_location="cpu").state_dict()

        else:
            torch_model = yolov8x.DetectionModel()
            state_dict = torch_model.state_dict()
        parameters = custom_preprocessor(device, state_dict, inp_h=res[0], inp_w=res[1])
        model = YOLOv8xModel(device=device, parameters=parameters)
        logger.info("Inferencing using ttnn Model")

    save_dir = "models/experimental/functional_yolov8x/demo/runs"

    dataset = LoadImages(path=source)

    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    names = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "TV",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }

    for batch in dataset:
        paths, im0s, s = batch

        im = preprocess(im0s, res=res)

        ttnn_im = im.permute((0, 2, 3, 1))
        ttnn_im = ttnn.from_torch(ttnn_im, dtype=ttnn.bfloat16)

        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = model(x=ttnn_im)
            preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)

        results = postprocess(preds, im, im0s, batch, names)[0]

        save_yolo_predictions_by_model(results, save_dir, source, model_type)

    logger.info("Inference done")
