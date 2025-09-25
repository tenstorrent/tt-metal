# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from loguru import logger
from datasets import load_dataset

from models.experimental.yolov5.reference.models.common import DetectMultiBackend
from models.experimental.yolov5.reference.utils.dataloaders import LoadImages
from models.experimental.yolov5.reference.utils.general import check_img_size
from models.experimental.yolov5.tt.yolov5_detection_model import (
    yolov5s_detection_model,
)

from models.utility_functions import (
    torch2tt_tensor,
    comp_pcc,
)


def download_images(path, imgsz):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image = image.resize(imgsz)
    image.save(path / "input_image.jpg")


def test_Yolov5_detection_model(device):
    weights = "models/experimental/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False

    refence_model = DetectMultiBackend(weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half)
    refence_module = refence_model.model

    torch.manual_seed(0)
    # test_input = torch.rand(1, 3, 640, 640)

    stride = max(int(max(refence_module.stride)), 32)  # model stride
    imgsz = check_img_size((640, 640), s=stride)  # check image size

    download_images(Path(ROOT), imgsz)
    dataset = LoadImages(ROOT, img_size=imgsz, stride=stride, auto=True)

    for path, test_input, im0s, _, s in dataset:
        test_input = torch.from_numpy(test_input)
        test_input = test_input.float()
        test_input /= 255  # 0 - 255 to 0.0 - 1.0

        if len(test_input.shape) == 3:
            test_input = test_input[None]  # expand for batch dim

    logger.debug(f"Running inference on {path}")

    with torch.no_grad():
        refence_module.eval()
        pt_out = refence_module(test_input)

    test_input = torch2tt_tensor(test_input, device)
    tt_module = yolov5s_detection_model(device)

    with torch.no_grad():
        tt_module.eval()
        tt_out = tt_module(test_input)

    does_all_pass, pcc_message = comp_pcc(pt_out[0], tt_out[0], 0.99)
    logger.info(f"out[0] PCC: {pcc_message}")

    for i in range(len(pt_out[1])):
        does_pass, pcc_message = comp_pcc(pt_out[1][i], tt_out[1][i], 0.99)
        logger.info(f"out[1][{i}] PCC: {pcc_message}")

        if not does_pass:
            does_all_pass = False

    if does_all_pass:
        logger.info("test_Yolov5_detection_model Passed!")
    else:
        logger.warning("test_Yolov5_detection_model Failed!")

    assert does_pass
