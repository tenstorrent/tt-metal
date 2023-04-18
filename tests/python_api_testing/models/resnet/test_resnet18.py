from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from tqdm import tqdm
from loguru import logger
import torch
import torchvision
from torchvision import models
from torchvision import transforms
import pytest
from resnetBlock import ResNet, BasicBlock
from libs import tt_lib as ttl

from imagenet import prep_ImageNet
from utility_functions import comp_allclose_and_pcc, comp_pcc
batch_size=1

@pytest.mark.parametrize("fold_batchnorm", [False, True], ids=['Batchnorm not folded', "Batchnorm folded"])
def test_run_resnet18_inference(fold_batchnorm):
    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        torch_resnet.eval()

        state_dict = torch_resnet.state_dict()

        tt_resnet18 = ResNet(BasicBlock, [2, 2, 2, 2],
                        device=device,
                        host=host,
                        state_dict=state_dict,
                        base_address="",
                        fold_batchnorm=fold_batchnorm)
        dataloader = prep_ImageNet(batch_size=batch_size)
        for i, (images, targets, _, _, _) in enumerate(tqdm(dataloader)):
            torch_output = torch_resnet(images).unsqueeze(1).unsqueeze(1)
            tt_output = tt_resnet18(images)
            break

        print(comp_allclose_and_pcc(torch_output, tt_output))
        passing, info = comp_pcc(torch_output, tt_output)
        logger.info(info)
        assert passing

test_run_resnet18_inference(True)
