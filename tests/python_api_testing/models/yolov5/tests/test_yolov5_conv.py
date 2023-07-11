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
from datasets import load_dataset

from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.yolov5.reference.models.common import DetectMultiBackend
from python_api_testing.models.yolov5.reference.utils.general import check_img_size
from python_api_testing.models.yolov5.reference.utils.dataloaders import LoadImages
from python_api_testing.models.yolov5.tt.yolov5_conv import TtYolov5Conv, TtYolov5Conv2D
from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)


def download_images(path, imgsz):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    if imgsz is not None:
        image = image.resize(imgsz)

    image.save(path / "input_image.jpg")


def get_test_input(refence_module):
    stride = max(int(max(refence_module.stride)), 32)  # model stride
    imgsz = check_img_size((640, 640), s=stride)  # check image size

    download_images(Path(ROOT), None)  # imgsz)
    dataset = LoadImages(ROOT, img_size=imgsz, stride=stride, auto=True)

    for path, test_input, im0s, _, s in dataset:
        test_input = torch.from_numpy(test_input)
        test_input = test_input.float()
        test_input /= 255  # 0 - 255 to 0.0 - 1.0

        if len(test_input.shape) == 3:
            test_input = test_input[None]  # expand for batch dim

    logger.debug(f"Running inference on {path}")
    return test_input


def test_Yolov5_Conv2D():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    weights = "tests/python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False
    block = 1

    refence_model = DetectMultiBackend(
        weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half
    )
    refence_module = refence_model.model.model[block].conv

    in_channels = refence_module.in_channels
    out_channels = refence_module.out_channels
    kernel_size = refence_module.kernel_size[0]
    stride = refence_module.stride[0]
    padding = refence_module.padding[0]
    groups = refence_module.groups
    dilation = refence_module.dilation

    logger.debug(f"in_channels {in_channels}")
    logger.debug(f"out_channels {out_channels}")
    logger.debug(f"kernel_size {kernel_size}")
    logger.debug(f"stride {stride}")
    logger.debug(f"padding {padding}")
    logger.debug(f"groups {groups}")
    logger.debug(f"dilation {dilation}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 32, 64, 64)
    pt_out = refence_module(test_input)

    tt_module = TtYolov5Conv2D(
        state_dict=refence_model.state_dict(),
        base_address=f"model.model.{block}.conv",
        device=device,
        c1=in_channels,
        c2=out_channels,
        k=kernel_size,
        s=stride,
        p=padding,
        g=groups,
        d=dilation[0],
    )

    # CHANNELS_LAST
    test_input = torch2tt_tensor(test_input, device)
    tt_out = tt_module(test_input)

    tt_out = tt_out.to(tt_lib.device.GetHost())
    tt_out = tt_out.to(tt_lib.tensor.Layout.ROW_MAJOR)

    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    logger.debug(f"pt_out shape {pt_out.shape}")
    logger.debug(f"tt_out shape {tt_out.shape}")

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_Conv2D Passed!")
    else:
        logger.warning("test_Yolov5_Conv2D Failed!")

    assert does_pass


def test_Yolov5_Silu():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    weights = "tests/python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False
    block = 1

    refence_model = DetectMultiBackend(
        weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half
    )
    refence_module = refence_model.model.model[block].act

    torch.manual_seed(0)
    test_input = torch.rand(1, 32, 64, 64)
    pt_out = refence_module(test_input)

    test_input = torch2tt_tensor(test_input, device)
    tt_out = fallback_ops.silu(test_input)

    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_Silu Passed!")
    else:
        logger.warning("test_Yolov5_Silu Failed!")

    assert does_pass


def test_Yolov5_conv():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    weights = "tests/python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False
    block = 1

    refence_model = DetectMultiBackend(
        weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half
    )

    refence_module = refence_model.model.model[block]

    in_channels = refence_module.conv.in_channels
    out_channels = refence_module.conv.out_channels
    kernel_size = refence_module.conv.kernel_size[0]
    stride = refence_module.conv.stride[0]
    padding = refence_module.conv.padding[0]
    groups = refence_module.conv.groups
    dilation = refence_module.conv.dilation
    act = refence_module.act

    logger.debug(f"in_channels {in_channels}")
    logger.debug(f"out_channels {out_channels}")
    logger.debug(f"kernel_size {kernel_size}")
    logger.debug(f"stride {stride}")
    logger.debug(f"padding {padding}")
    logger.debug(f"groups {groups}")
    logger.debug(f"dilation {dilation}")
    logger.debug(f"act {act}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 32, 64, 64)
    pt_out = refence_module(test_input)

    tt_module = TtYolov5Conv(
        state_dict=refence_model.state_dict(),
        base_address=f"model.model.{block}",
        device=device,
        c1=in_channels,
        c2=out_channels,
        k=kernel_size,
        s=stride,
        p=padding,
        g=groups,
        d=dilation[0],
        act=True,
    )
    test_input = torch2tt_tensor(test_input, device)

    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_conv Passed!")
    else:
        logger.warning("test_Yolov5_conv Failed!")

    assert does_pass


def test_Yolov5_Conv2D_real():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    weights = "tests/python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = None
    half = False
    block = 0

    refence_model = DetectMultiBackend(
        weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half
    )
    refence_module = refence_model.model.model[block].conv

    in_channels = refence_module.in_channels
    out_channels = refence_module.out_channels
    kernel_size = refence_module.kernel_size[0]
    stride = refence_module.stride[0]
    padding = refence_module.padding[0]
    groups = refence_module.groups
    dilation = refence_module.dilation

    logger.debug(f"in_channels {in_channels}")
    logger.debug(f"out_channels {out_channels}")
    logger.debug(f"kernel_size {kernel_size}")
    logger.debug(f"stride {stride}")
    logger.debug(f"padding {padding}")
    logger.debug(f"groups {groups}")
    logger.debug(f"dilation {dilation}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 480, 640)
    pt_out = refence_module(test_input)

    tt_module = TtYolov5Conv2D(
        state_dict=refence_model.state_dict(),
        base_address=f"model.model.{block}.conv",
        device=device,
        c1=in_channels,
        c2=out_channels,
        k=kernel_size,
        s=stride,
        p=padding,
        g=groups,
        d=dilation[0],
        conv_on_device=True,
    )

    # CHANNELS_LAST
    test_input = torch2tt_tensor(test_input, device)
    tt_out = tt_module(test_input)

    tt_out = tt_out.to(tt_lib.device.GetHost())
    tt_out = tt_out.to(tt_lib.tensor.Layout.ROW_MAJOR)

    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    logger.debug(f"pt_out shape {pt_out.shape}")
    logger.debug(f"tt_out shape {tt_out.shape}")

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_Conv2D_real Passed!")
    else:
        logger.warning("test_Yolov5_Conv2D_real Failed!")

    assert does_pass
