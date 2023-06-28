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
from loguru import logger
from python_api_testing.models.yolov5.reference.models.common import DetectMultiBackend
from python_api_testing.models.yolov5.tt.yolov5_detect import TtYolov5Detect
from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def test_Yolov5_detect():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    weights = "python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = "python_api_testing/models/yolov5/reference/data/coco128.yaml"
    half = False

    refence_model = DetectMultiBackend(
        weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half
    )
    refence_module = refence_model.model.model[24]

    torch.manual_seed(0)
    test_input = torch.rand(3, 128, 255, 255)

    nc = 80
    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ]
    ch = [128, 256, 512]

    pt_out = refence_module(test_input)

    logger.info(f"pt_out shape {pt_out.shape}")

    tt_module = TtYolov5Detect(
        state_dict=refence_model.state_dict(),
        base_address="model.model.24",
        device=device,
        nc=nc,
        anchors=anchors,
        ch=ch,
    )

    test_input = torch2tt_tensor(test_input, device)

    tt_out = tt_module(test_input)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_c3 Passed!")
    else:
        logger.warning("test_Yolov5_c3 Failed!")

    assert does_pass
