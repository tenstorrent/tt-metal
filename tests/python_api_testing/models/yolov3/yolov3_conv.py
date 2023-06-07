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
from python_api_testing.models.yolov3.modeling_yolo import Conv, Model
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


class TtConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    def __init__(
        self,
        device,
        state_dict,
        base_address,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
    ):
        super().__init__()

        self.device = device
        self.base_address = base_address

        self.conv_weight = state_dict[f"{base_address}.conv.weight"]
        self.conv_bias = state_dict[f"{base_address}.conv.bias"]
        self.conv = fallback_ops.Conv2d(
            self.conv_weight,
            self.conv_bias,
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(c2)

        self.act = act
        if self.act != True:
            logger.warning(
                f"Configuration for activation function {self.act} not supported. Using fallback.SiLU act function"
            )
            raise NotImplementedError

    def forward(self, x):
        x = self.conv(x)
        torch_x = tt2torch_tensor(x)
        torch_x = self.bn(torch_x)
        x = torch2tt_tensor(torch_x, self.device)
        return fallback_ops.silu(x)

    def forward_fuse(self, x):
        return fallback_ops.silu(self.conv(x))


def test_conv_module():
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

    INDEX = 0
    base_address = f"model.{INDEX}"

    torch_model = model.model[INDEX]
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
    stride = max(int(model.stride.max()), 32)  # model stride
    imgsz = check_img_size((640, 640), s=stride)  # check image size
    dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride, auto=True)

    # Run inference
    for path, im, _, _, _ in dataset:
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference- fused
        pred = torch_model(im)
        tt_im = torch2tt_tensor(im, device)
        tt_pred = tt_model.forward_fuse(tt_im)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_pred)

    does_pass, pcc_message = comp_pcc(pred, tt_output_torch, 0.98)

    logger.info(comp_allclose(pred, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("Yolo TtConv Passed!")
    else:
        logger.warning("Yolo TtConv Failed!")

    assert does_pass
