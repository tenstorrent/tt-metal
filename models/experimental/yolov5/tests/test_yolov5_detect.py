# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from loguru import logger

from models.experimental.yolov5.reference.models.common import DetectMultiBackend
from models.experimental.yolov5.tt.yolov5_detect import TtYolov5Detect
from models.utility_functions import (
    torch2tt_tensor,
    comp_pcc,
)


def test_Yolov5_detect(device):
    weights = "models/experimental/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False

    refence_model = DetectMultiBackend(weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half)
    refence_module = refence_model.model.model[24]

    torch.manual_seed(0)

    a = torch.rand(1, 128, 32, 32)
    b = torch.rand(1, 256, 16, 16)
    c = torch.rand(1, 512, 8, 8)
    test_input = [a, b, c]

    nc = 80
    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ]
    ch = [128, 256, 512]

    with torch.no_grad():
        refence_module.eval()
        pt_out = refence_module(test_input)

    tt_module = TtYolov5Detect(
        state_dict=refence_model.state_dict(),
        base_address="model.model.24",
        device=device,
        nc=nc,
        anchors=anchors,
        ch=ch,
    )

    tt_module.anchors = refence_module.anchors
    tt_module.stride = torch.tensor([8.0, 16.0, 32.0])

    tt_a = torch2tt_tensor(a, tt_device=device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_b = torch2tt_tensor(b, tt_device=device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_c = torch2tt_tensor(c, tt_device=device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    test_input = [tt_a, tt_b, tt_c]

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
        logger.info("test_Yolov5_detect Passed!")
    else:
        logger.warning("test_Yolov5_detect Failed!")

    assert does_pass
