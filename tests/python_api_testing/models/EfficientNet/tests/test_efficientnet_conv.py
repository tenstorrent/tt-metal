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
import torchvision

from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.utility_functions_new import torch2tt_tensor, tt2torch_tensor
from python_api_testing.models.EfficientNet.tt.efficientnet_conv import TtEfficientnetConv2dNormActivation
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc



def download_images(path, imgsz):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    if imgsz is not None:
        image = image.resize(imgsz)

    image.save(path / "input_image.jpg")


def test_efficientnet_conv():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    refence_model = torchvision.models.efficientnet_b0(pretrained=True)
    refence_model.eval()

    block = 0
    refence_module = refence_model.features[block]

    in_channels = refence_module[0].in_channels
    out_channels = refence_module[0].out_channels
    kernel_size = refence_module[0].kernel_size[0]
    stride = refence_module[0].stride[0]
    padding = refence_module[0].padding[0]
    groups = refence_module[0].groups
    dilation = refence_module[0].dilation
    #activation_layer = refence_module.activation_layer

    logger.debug(f"in_channels {in_channels}")
    logger.debug(f"out_channels {out_channels}")
    logger.debug(f"kernel_size {kernel_size}")
    logger.debug(f"stride {stride}")
    logger.debug(f"padding {padding}")
    logger.debug(f"groups {groups}")
    logger.debug(f"dilation {dilation}")
    #logger.debug(f"act {activation_layer}")

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 224, 224)
    pt_out = refence_module(test_input)

    tt_module = TtEfficientnetConv2dNormActivation(
        state_dict=refence_model.state_dict(),
        base_address=f"features.{block}",
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        activation_layer=True,
        dilation=dilation,
        conv_on_device=False,
    )

    test_input = torch2tt_tensor(test_input, tt_device=device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
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
