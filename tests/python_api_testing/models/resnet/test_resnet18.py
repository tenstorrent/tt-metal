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
from torchvision import models
import pytest
from resnetBlock import ResNet, BasicBlock
import tt_lib

from sweep_tests.comparison_funcs import comp_allclose_and_pcc, comp_pcc


@pytest.mark.parametrize("fold_batchnorm", [True], ids=["Batchnorm folded"])
def test_run_resnet18_inference(fold_batchnorm, imagenet_sample_input):
    image = imagenet_sample_input

    with torch.no_grad():
        torch.manual_seed(1234)

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)

        torch_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        torch_resnet.eval()

        state_dict = torch_resnet.state_dict()

        tt_resnet18 = ResNet(BasicBlock, [2, 2, 2, 2],
                        device=device,
                        state_dict=state_dict,
                        base_address="",
                        fold_batchnorm=fold_batchnorm)

        torch_output = torch_resnet(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_resnet18(image)

        logger.info(comp_allclose_and_pcc(torch_output, tt_output))
        passing, info = comp_pcc(torch_output, tt_output)
        logger.info(info)

        tt_lib.device.CloseDevice(device)
        assert passing
