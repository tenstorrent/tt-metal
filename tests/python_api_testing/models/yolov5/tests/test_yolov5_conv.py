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
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.yolov5.reference.models.common import DetectMultiBackend
from python_api_testing.models.yolov5.tt.yolov5_conv import TtYolov5Conv, TtYolov5Conv2D
from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def test_Yolov5_Conv2D():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    weights = "python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = "python_api_testing/models/yolov5/reference/data/coco128.yaml"
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

    logger.info(f"in_channels {in_channels}")
    logger.info(f"out_channels {out_channels}")
    logger.info(f"kernel_size {kernel_size}")
    logger.info(f"stride {stride}")
    logger.info(f"padding {padding}")
    logger.info(f"groups {groups}")
    logger.info(f"dilation {dilation}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 32, 64, 64)

    pt_out = refence_module(test_input)
    logger.info(f"pt_out shape {pt_out.shape}")

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

    test_input = torch2tt_tensor(
        test_input, device
    )  # , tt_layout=tt_lib.tensor.Layout.ROW_MAJOR) # CHANNELS_LAST
    tt_out = tt_module(test_input)

    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    logger.debug(f"pt_out shape {pt_out.shape}")
    logger.debug(f"tt_out shape {tt_out.shape}")

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_Conv2D Passed!")
    else:
        logger.warning("test_Yolov5_Conv2D Failed!")

    assert does_pass


def test_Yolov5_Silu():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    weights = "python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = "python_api_testing/models/yolov5/reference/data/coco128.yaml"
    half = False
    block = 1

    refence_model = DetectMultiBackend(
        weights, device=torch.device("cpu"), dnn=dnn, data=data, fp16=half
    )
    refence_module = refence_model.model.model[block].act

    torch.manual_seed(0)
    test_input = torch.rand(1, 32, 64, 64)

    pt_out = refence_module(test_input)
    logger.info(f"pt_out shape {pt_out.shape}")

    test_input = torch2tt_tensor(test_input, device)
    tt_out = fallback_ops.silu(test_input)

    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_Silu Passed!")
    else:
        logger.warning("test_Yolov5_Silu Failed!")

    assert does_pass


def test_Yolov5_conv():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    weights = "python_api_testing/models/yolov5/reference/yolov5s.pt"
    dnn = False
    data = "python_api_testing/models/yolov5/reference/data/coco128.yaml"
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

    logger.info(f"in_channels {in_channels}")
    logger.info(f"out_channels {out_channels}")
    logger.info(f"kernel_size {kernel_size}")
    logger.info(f"stride {stride}")
    logger.info(f"padding {padding}")
    logger.info(f"groups {groups}")
    logger.info(f"dilation {dilation}")
    logger.info(f"act {act}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 32, 64, 64)

    pt_out = refence_module(test_input)
    logger.info(f"pt_out shape {pt_out.shape}")

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

    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Yolov5_conv Passed!")
    else:
        logger.warning("test_Yolov5_conv Failed!")

    assert does_pass
