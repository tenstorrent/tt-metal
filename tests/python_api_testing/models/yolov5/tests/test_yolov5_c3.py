import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import tt_lib
import torch
from loguru import logger

from python_api_testing.models.yolov5.reference.models.common import DetectMultiBackend
from python_api_testing.models.yolov5.tt.yolov5_c3 import TtYolov5C3
from tt_models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)


def test_Yolov5_c3():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    weights = "tests/python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False

    refence_model = DetectMultiBackend(
        weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half
    )
    refence_module = refence_model.model.model[2]

    in_channels = refence_module.cv1.conv.in_channels
    out_channels = refence_module.cv3.conv.out_channels

    logger.debug(f"in_channels {in_channels}")
    logger.debug(f"out_channels {out_channels}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 64, 128, 128)

    pt_out = refence_module(test_input)

    tt_module = TtYolov5C3(
        state_dict=refence_model.state_dict(),
        base_address="model.model.2",
        c1=in_channels,
        c2=out_channels,
        device=device,
    )

    test_input = torch2tt_tensor(test_input, device)

    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_c3 Passed!")
    else:
        logger.warning("test_Yolov5_c3 Failed!")

    assert does_pass
