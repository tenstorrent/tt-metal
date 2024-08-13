# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger

from models.experimental.yolov3.reference.utils.dataloaders import LoadImages
from models.experimental.yolov3.reference.utils.general import check_img_size
from models.experimental.yolov3.reference.models.yolo import Bottleneck
from models.experimental.yolov3.tt.yolov3_bottleneck import TtBottleneck
from models.experimental.yolov3.reference.models.common import (
    autopad,
    DetectMultiBackend,
)
from models.utility_functions import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_bottleneck_module(model_location_generator, device):
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

    INDEX = 2
    base_address = f"model.model.{INDEX}"

    torch_model = reference_model.model.model[INDEX]

    in_channels = reference_model.model.model[INDEX].cv1.conv.in_channels
    out_channels = reference_model.model.model[INDEX].cv2.conv.out_channels

    tt_model = TtBottleneck(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        c1=in_channels,
        c2=out_channels,
    )

    # Create random Input image with channels > 3
    im = torch.rand(1, 64, 512, 640)

    # Inference
    pred = torch_model(im)

    tt_im = torch2tt_tensor(im, device)
    tt_pred = tt_model(tt_im)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_pred)

    does_pass, pcc_message = comp_pcc(pred, tt_output_torch, 0.98)
    logger.info(pcc_message)

    if does_pass:
        logger.info("Yolo TtBottleneck Passed!")
    else:
        logger.warning("Yolo TtBottleneck Failed!")

    assert does_pass
