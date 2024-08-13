# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger

import ttnn

from models.experimental.yolov3.reference.models.common import (
    autopad,
    DetectMultiBackend,
)
from models.experimental.yolov3.tt.yolov3_detection_model import (
    yolov3_fused_model,
)
from models.experimental.yolov3.reference.utils.dataloaders import LoadImages
from models.experimental.yolov3.reference.utils.general import check_img_size

from models.utility_functions import (
    comp_pcc,
    torch2tt_tensor,
)


def test_detection_model(device, model_location_generator):
    torch.manual_seed(1234)

    # Load yolo
    model_path = model_location_generator("models", model_subdir="Yolo")
    data_path = model_location_generator("data", model_subdir="Yolo")
    data_image_path = str(data_path / "images")
    data_coco = str(data_path / "coco128.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False)
    reference_model = reference_model.model

    tt_module = yolov3_fused_model(device, model_location_generator)

    with torch.no_grad():
        tt_module.eval()
        reference_model.eval()

        # Load data
        stride = max(int(max(reference_model.stride)), 32)  # model stride
        imgsz = check_img_size((640, 640), s=stride)  # check image size
        dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride, auto=True)

        path, im, _, _, _ = next(iter(dataset))
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference- fused torch
        pt_out = reference_model(im)
        # Inference- fused tt
        tt_im = torch2tt_tensor(im, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_out = tt_module(tt_im)

    # Check all outputs PCC
    does_all_pass = True

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out[0], 0.99)
    does_all_pass &= does_pass
    logger.info(f"Output prediction from the highest scale: {pcc_message}")

    for i in range(len(pt_out[1])):
        does_pass, pcc_message = comp_pcc(pt_out[1][i], tt_out[1][i], 0.99)
        logger.info(f"Object detection {i}: {pcc_message}")
        does_all_pass &= does_pass

    if does_all_pass:
        logger.info(f"Yolov3 Full Detection Model Passed!")
    else:
        logger.warning(f"Yolov3 Full Detection Model Failed!")

    assert does_all_pass
