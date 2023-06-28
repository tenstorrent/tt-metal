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


def test_Yolov5_model():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    weights = "python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = "python_api_testing/models/yolov5/reference/data/coco128.yaml"
    half = False

    refence_model = DetectMultiBackend(
        weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half
    )
    refence_module = refence_model.model

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 256, 256)

    pt_out = refence_module(test_input)
    logger.info(f"pt_out type {type(pt_out)}")
    logger.info(f"pt_out len {len(pt_out)}")
    logger.info(f"pt_out 0-element {pt_out[0].shape}")
    logger.info(f"pt_out 1-element {pt_out[1][0].shape}")
    pt_out = pt_out[1][0]

    cfg_path = "tests/python_api_testing/models/yolov5/reference/yolov5s.yaml"

    tt_module = TtYolov5DetectionModel(
        cfg=cfg_path,
        state_dict=refence_model.state_dict(),
        base_address="model.model",
        device=device,
    )

    tt_out = tt_module(test_input)
    logger.info(f"2 pt_out type {type(tt_out)}")
    logger.info(f"2 pt_out len {len(tt_out)}")
    logger.info(f"2 pt_out 0-element {tt_out[0].shape}")
    logger.info(f"2 pt_out 1-element {tt_out[1].shape}")
    logger.info(f"2 pt_out 2-element {tt_out[2].shape}")
    tt_out = tt_out[0]

    # tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.5)

    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_sppf Passed!")
    else:
        logger.warning("test_Yolov5_sppf Failed!")

    assert does_pass
