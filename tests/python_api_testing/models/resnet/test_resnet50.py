from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from loguru import logger
import torch
import torchvision
from torchvision import models
from torchvision import transforms
import pytest

from resnetBlock import ResNet, Bottleneck
from libs import tt_lib as ttl
from utility_functions import comp_allclose_and_pcc, comp_pcc

batch_size=1

@pytest.mark.parametrize("fold_batchnorm", [False, True], ids=['Batchnorm not folded', "Batchnorm folded"])
def test_run_resnet50_inference(fold_batchnorm, imagenet_sample_input):
    image = imagenet_sample_input
    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        torch_resnet50.eval()

        state_dict = torch_resnet50.state_dict()

        tt_resnet50 = ResNet(Bottleneck, [3, 4, 6, 3],
                        device=device,
                        host=host,
                        state_dict=state_dict,
                        base_address="",
                        fold_batchnorm=fold_batchnorm)


        torch_output = torch_resnet50(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_resnet50(image)


        passing, info = comp_pcc(torch_output, tt_output)
        logger.info(info)
        assert passing
