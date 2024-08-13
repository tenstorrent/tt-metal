# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger

from models.experimental.yolov3.reference.models.common import (
    autopad,
    DetectMultiBackend,
)
from models.experimental.yolov3.reference.utils.dataloaders import LoadImages
from models.experimental.yolov3.reference.utils.general import check_img_size
from models.experimental.yolov3.reference.models.yolo import Conv
from models.experimental.yolov3.tt.yolov3_conv import TtConv
from models.utility_functions import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_conv_module(device, model_location_generator):
    torch.manual_seed(1234)

    # Load yolo
    model_path = model_location_generator("models", model_subdir="Yolo")
    data_path = model_location_generator("data", model_subdir="Yolo")

    data_image_path = str(data_path / "images")
    data_coco = str(data_path / "coco128.yaml")
    model_config_path = str(data_path / "yolov3.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False)
    state_dict = reference_model.state_dict()

    INDEX = 0
    base_address = f"model.model.{INDEX}"
    torch_model = reference_model.model.model[INDEX]

    in_channels = torch_model.conv.in_channels
    out_channels = torch_model.conv.out_channels
    kernel_size = torch_model.conv.kernel_size[0]
    stride = torch_model.conv.stride[0]
    padding = torch_model.conv.padding[0]
    groups = torch_model.conv.groups
    dilation = torch_model.conv.dilation[0]

    tt_model = TtConv(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        c1=in_channels,
        c2=out_channels,
        k=kernel_size,
        s=stride,
        p=padding,
        g=groups,
        d=dilation,
    )

    # Load data
    stride = max(int(max(torch_model.conv.stride)), 32)  # model stride
    imgsz = check_img_size((640, 640), s=stride)  # check image size
    dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride, auto=True)

    real_input = True

    # Run inference
    with torch.no_grad():
        if real_input:
            path, im, _, _, _ = next(iter(dataset))
            im = torch.from_numpy(im)
            im = im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
        else:
            im = torch.rand(1, 3, 640, 640)

        # Inference- fused
        pt_out = torch_model(im)

        tt_im = torch2tt_tensor(im, device)
        tt_out = tt_model(tt_im)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_output_torch)

    logger.info(pcc_message)

    if does_pass:
        logger.info("Yolo TtConv Passed!")
    else:
        logger.warning("Yolo TtConv Failed!")

    assert does_pass
