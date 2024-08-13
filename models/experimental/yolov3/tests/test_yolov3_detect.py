# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger

import ttnn

from models.experimental.yolov3.reference.models.common import (
    autopad,
    DetectMultiBackend,
)
from models.experimental.yolov3.reference.utils.dataloaders import LoadImages
from models.experimental.yolov3.reference.utils.general import check_img_size
from models.experimental.yolov3.tt.yolov3_detect import TtDetect
from models.utility_functions import (
    comp_pcc,
    torch2tt_tensor,
)


def test_detect_module(device, model_location_generator):
    torch.manual_seed(1234)

    # Load yolo
    model_path = model_location_generator("models", model_subdir="Yolo")
    data_path = model_location_generator("data", model_subdir="Yolo")

    data_image_path = str(data_path / "images")
    data_coco = str(data_path / "coco128.yaml")
    model_config_path = str(data_path / "yolov3.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False)

    state_dict = reference_model.state_dict()

    INDEX = 28
    base_address = f"model.model.{INDEX}"

    nc = 80
    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ]
    ch = [128, 256, 512]

    torch_model = reference_model.model.model[INDEX]
    logger.info("Set detection-module strides")

    tt_model = TtDetect(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        nc=nc,
        anchors=anchors,
        ch=ch,
    )
    tt_model.anchors = torch_model.anchors
    tt_model.stride = torch.tensor([8.0, 16.0, 32.0])

    a = torch.rand(1, 256, 64, 80)
    b = torch.rand(1, 512, 32, 40)
    c = torch.rand(1, 1024, 16, 20)
    test_input = [a, b, c]

    with torch.no_grad():
        torch_model.eval()
        pt_out = torch_model(test_input)

    tt_a = torch2tt_tensor(a, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_b = torch2tt_tensor(b, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_c = torch2tt_tensor(c, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_test_input = [tt_a, tt_b, tt_c]

    with torch.no_grad():
        tt_model.eval()
        tt_out = tt_model(tt_test_input)

    does_all_pass = True

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out[0], 0.99)
    does_all_pass &= does_pass
    logger.info(f"Output prediction from the highest scale: {pcc_message}")

    for i in range(len(pt_out[1])):
        does_pass, pcc_message = comp_pcc(pt_out[1][i], tt_out[1][i], 0.99)
        logger.info(f"Object detection {i}: {pcc_message}")
        does_all_pass &= does_pass

    if does_all_pass:
        logger.info(f"Yolov3 Detection Head Passed!")
    else:
        logger.warning(f"Yolov3 Detection Head Failed!")

    assert does_all_pass
