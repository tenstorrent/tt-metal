from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torchvision import models
from loguru import logger

from libs import tt_lib as ttl
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc, comp_pcc
from vgg import *


_batch_size = 16


def test_vgg16_inference(imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size
    with torch.no_grad():
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        torch_vgg.eval()

        state_dict = torch_vgg.state_dict()

        tt_vgg = vgg16(device, host, state_dict)


        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_vgg(image)

        passing = comp_pcc(torch_output, tt_output)

        assert passing[0], passing[1:]

    logger.info(f"vgg16 PASSED {passing[1]}")
