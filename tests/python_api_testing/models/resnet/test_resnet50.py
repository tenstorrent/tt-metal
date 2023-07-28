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
import tt_lib

from resnetBlock import ResNet, Bottleneck
from sweep_tests.comparison_funcs import comp_allclose_and_pcc, comp_pcc


@pytest.mark.parametrize("fold_batchnorm", [True], ids=["Batchnorm folded"])
def test_run_resnet50_inference(fold_batchnorm, imagenet_sample_input):
    image = imagenet_sample_input

    with torch.no_grad():
        torch.manual_seed(1234)

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)

        torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet50.eval()

        state_dict = torch_resnet50.state_dict()

        tt_resnet50 = ResNet(Bottleneck, [3, 4, 6, 3],
                        device=device,
                        state_dict=state_dict,
                        base_address="",
                        fold_batchnorm=fold_batchnorm)


        torch_output = torch_resnet50(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_resnet50(image)

        passing, info = comp_pcc(torch_output, tt_output, pcc=0.985)
        logger.info(info)
        tt_lib.device.CloseDevice(device)
        assert passing
