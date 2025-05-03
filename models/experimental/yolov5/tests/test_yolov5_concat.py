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

from models.experimental.yolov5.reference.models.common import DetectMultiBackend
from models.experimental.yolov5.tt.yolov5_concat import TtYolov5Concat
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)


def test_Yolov5_concat(device):
    weights = "models/experimental/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False

    refence_model = DetectMultiBackend(weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half)
    refence_module = refence_model.model.model[16]

    torch.manual_seed(0)
    im_1 = torch.rand(1, 64, 512, 640)
    im_2 = torch.rand(1, 64, 512, 640)

    pt_out = refence_module([im_1, im_2])

    tt_module = TtYolov5Concat(
        state_dict=refence_model.state_dict(),
        base_address="model.model.16",
        device=device,
    )

    im_1 = torch2tt_tensor(im_1, device)
    im_2 = torch2tt_tensor(im_2, device)

    tt_out = tt_module([im_1, im_2])
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_c3 Passed!")
    else:
        logger.warning("test_Yolov5_c3 Failed!")

    assert does_pass
