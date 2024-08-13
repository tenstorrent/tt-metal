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
from models.experimental.yolov3.tt.yolov3_upsample import TtUpsample
from models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_upsample_module(device, model_location_generator):
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

    INDEX = 17
    base_address = f"model.model.{INDEX}"

    torch_model = reference_model.model.model[INDEX]

    tt_model = TtUpsample(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        scale_factor=2.0,
        mode="nearest",
    )

    # Create random Input image with channels > 3
    im = torch.rand(1, 64, 512, 640)
    # Inference
    pred = torch_model(im)

    tt_im = torch2tt_tensor(im, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_pred = tt_model(tt_im)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_pred)

    does_pass, pcc_message = comp_pcc(pred, tt_output_torch)
    logger.info(pcc_message)
    _, comp_out = comp_allclose_and_pcc(pred, tt_output_torch)
    logger.info(comp_out)

    if does_pass:
        logger.info("Yolo TtUpsample Passed!")
    else:
        logger.warning("Yolo TtUpsample Failed!")

    assert does_pass
