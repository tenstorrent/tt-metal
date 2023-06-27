import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.yolov3.reference.models.common import autopad
from python_api_testing.models.yolov3.reference.utils.dataloaders import LoadImages
from python_api_testing.models.yolov3.reference.utils.general import check_img_size
from python_api_testing.models.yolov3.reference.models.yolo import Conv
from python_api_testing.models.yolov3.reference.models.common import DetectMultiBackend
from python_api_testing.models.yolov3.tt.yolov3_conv import TtConv
import tt_lib

from utility_functions_new import (
    comp_allclose_and_pcc,
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_conv_module(model_location_generator):
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    # Load yolo
    model_path = model_location_generator("tt_dnn-models/Yolo/models/")
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")

    data_image_path = str(data_path / "images")
    data_coco = str(data_path / "coco128.yaml")
    model_config_path = str(data_path / "yolov3.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(
        weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False
    )
    state_dict = reference_model.state_dict()

    INDEX = 0
    base_address = f"model.model.{INDEX}"

    torch_model = reference_model.model.model[INDEX]
    tt_model = TtConv(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        c1=3,
        c2=32,
        k=(3, 3),
        s=(1, 1),
        p=(1, 1),
    )

    # Load data
    stride = max(int(max(torch_model.conv.stride)), 32)  # model stride
    imgsz = check_img_size((640, 640), s=stride)  # check image size
    dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride, auto=True)

    # Run inference
    with torch.no_grad():
        for path, im, _, _, _ in dataset:
            im = torch.from_numpy(im)
            im = im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference- fused
            pred = torch_model(im)
            tt_im = torch2tt_tensor(
                im, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
            )
            tt_pred = tt_model(tt_im)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_pred)

    does_pass, pcc_message = comp_pcc(pred, tt_output_torch)

    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("Yolo TtConv Passed!")
    else:
        logger.warning("Yolo TtConv Failed!")

    assert does_pass
