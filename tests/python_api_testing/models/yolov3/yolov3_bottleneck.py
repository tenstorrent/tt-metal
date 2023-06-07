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

from python_api_testing.models.yolov3.yolov3_common import (
    autopad,
    model_location_generator,
    LoadImages,
    check_img_size,
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.yolov3.yolov3_conv import TtConv

from python_api_testing.models.yolov3.modeling_yolo import Bottleneck, Model
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


class TtBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        c1,
        c2,
        shortcut=True,
        g=1,
        e=0.5,
        fuse_model=True,
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        self.device = device
        self.base_address = base_address

        self.fuse_model = fuse_model

        c_ = int(c2 * e)  # hidden channels

        self.cv1 = TtConv(
            base_address=base_address + ".cv1",
            state_dict=state_dict,
            device=device,
            c1=c1,
            c2=c_,
            k=1,
            s=1,
        )
        self.cv2 = TtConv(
            base_address=base_address + ".cv2",
            state_dict=state_dict,
            device=device,
            c1=c_,
            c2=c2,
            k=3,
            s=1,
            g=g,
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        if self.fuse_model:
            output = self.cv1.forward_fuse(x)
            output = self.cv2.forward_fuse(output)
        else:
            output = self.cv2(self.cv1(x))

        if self.add:
            output = tt_lib.tensor.add(x, output)

        return output


def test_bottleneck_module():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    # Load yolo
    model_path = model_location_generator("tt_dnn-models/Yolo/models/")
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")

    data_image_path = str(data_path / "images")
    model_config_path = str(data_path / "yolov3.yaml")
    weights_loc = str(model_path / "yolov3_weights.pt")

    model = Model(model_config_path).fuse().eval()
    logger.info("Loading weights....")
    weights = torch.load(weights_loc)["model"]

    model.load_state_dict(state_dict=weights, strict=False)
    state_dict = model.state_dict()

    INDEX = 2
    base_address = f"model.{INDEX}"

    torch_model = model.model[INDEX]

    tt_model = TtBottleneck(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        fuse_model=True,
        c1=64,
        c2=64,
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

    logger.info(comp_allclose(pred, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("Yolo TtBottleneck Passed!")
    else:
        logger.warning("Yolo TtBottleneck Failed!")

    assert does_pass
