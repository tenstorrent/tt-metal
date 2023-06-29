import os
import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import tt_lib
import torch
from torch import nn
from loguru import logger
from python_api_testing.models.yolov5.reference.models.common import DetectMultiBackend
from python_api_testing.models.yolov5.tt.yolov5_detection_model import (
    TtYolov5DetectionModel,
)
from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def test_Yolov5_detection_model():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    cfg_path = "tests/python_api_testing/models/yolov5/reference/yolov5s.yaml"
    weights = "tests/python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False

    refence_model = DetectMultiBackend(
        weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half
    )
    refence_module = refence_model.model

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        refence_module.eval()
        pt_out = refence_module(test_input)

    test_input = torch2tt_tensor(test_input, device)

    tt_module = TtYolov5DetectionModel(
        cfg=cfg_path,
        state_dict=refence_model.state_dict(),
        base_address="model.model",
        device=device,
    )

    with torch.no_grad():
        tt_module.eval()
        tt_out = tt_module(test_input)

    tt_lib.device.CloseDevice(device)

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
