# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger

from models.experimental.yolov5.reference.models.common import DetectMultiBackend
from models.experimental.yolov5.tt.yolov5_sppf import TtYolov5SPPF
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)


def test_Yolov5_sppf(device):
    weights = "models/experimental/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False

    refence_model = DetectMultiBackend(weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half)
    refence_module = refence_model.model.model[9]

    in_channels = refence_module.cv1.conv.in_channels
    out_channels = refence_module.cv2.conv.out_channels

    logger.info(f"in_channels {in_channels}")
    logger.info(f"out_channels {out_channels}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 512, 64, 64)

    pt_out = refence_module(test_input)
    logger.info(f"pt_out shape {pt_out.shape}")

    tt_module = TtYolov5SPPF(
        state_dict=refence_model.state_dict(),
        base_address="model.model.9",
        device=device,
        c1=in_channels,
        c2=out_channels,
    )

    test_input = torch2tt_tensor(test_input, device)

    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_sppf Passed!")
    else:
        logger.warning("test_Yolov5_sppf Failed!")

    assert does_pass
