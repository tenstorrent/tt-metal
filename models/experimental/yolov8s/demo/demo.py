# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime

import cv2
import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.experimental.yolov8s.demo.demo_utils import LoadImages, postprocess, preprocess
from models.experimental.yolov8s.tt.ttnn_yolov8s import TtYolov8sModel
from models.experimental.yolov8s.tt.tt_yolov8s_utils import custom_preprocessor
from models.utility_functions import disable_persistent_kernel_cache


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
        ("models/experimental/yolov8s/demo/images/bus.jpg", "torch_model"),
        ("models/experimental/yolov8s/demo/images/bus.jpg", "tt_model"),
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo(device, source, model_type, res, use_weights_from_ultralytics):
    disable_persistent_kernel_cache()

    if use_weights_from_ultralytics:
        torch_model = YOLO("yolov8s.pt")
        torch_model = torch_model.model
        model = torch_model.eval()

    if model_type == "tt_model":
        state_dict = torch_model.state_dict()
        parameters = custom_preprocessor(device, state_dict, inp_h=res[0], inp_w=res[1])
        model = TtYolov8sModel(device=device, parameters=parameters, res=(res[0], res[1]))
        logger.info("Inferencing using ttnn Model")

    save_dir = "models/experimental/yolov8s/demo/runs"

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

        """
        # pad input channels to 16 to avoid slow interleaved2sharded codepath for 3/8 channels
        ttnn_input = torch.nn.functional.pad(im, (0, 0, 0, 0, 0, 13, 0, 0), value=0)
        ttnn_input = ttnn_input.permute((0, 2, 3, 1))
        ttnn_input = ttnn_input.reshape(
            1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]
        )
        ttnn_input = ttnn.from_torch(im, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        """

        n, c, h, w = im.shape
        if c == 3:
            c = 16
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        ttnn_input = ttnn.from_torch(im, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn_input = ttnn_input.to(device, input_mem_config)

        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = model(x=ttnn_input)
            preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)

        results = postprocess(preds, im, im0s, batch, names)[0]

        save_yolo_predictions_by_model(results, save_dir, source, model_type)

    logger.info("Inference done")
