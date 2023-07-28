from pathlib import Path
import sys
import torch
import pytest
from loguru import logger
from torchvision.utils import save_image

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

from utility_functions_new import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
import tt_lib
from hrnet.tt.hrnet_model import hrnet_w18_small


@pytest.mark.parametrize(
    "model_name",
    (("hrnet_w18_small"),),
)
def test_gs_demo(imagenet_sample_input, imagenet_label_dict, model_name, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    tt_model = hrnet_w18_small(device, host, multi_scale_output=True)

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device, put_on_device=False)

    with torch.no_grad():
        tt_output = tt_model(tt_input)

    tt_output_torch = tt_to_torch_tensor(tt_output).view(1, -1)

    logger.info("GS's Predicted Output")
    logger.info(imagenet_label_dict[tt_output_torch[0].argmax(-1).item()])

    save_image(imagenet_sample_input, "hrnet_input.jpg")
    logger.info("Input image is saved for reference as hrnet_input.jpg")
    tt_lib.device.CloseDevice(device)
