from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
from tqdm import tqdm

import torch
import torchvision
from torchvision import models
from torchvision import transforms

from BasicBlock import BasicBlock
from libs import tt_lib as ttl
from resnet import _resnet
from imagenet import prep_ImageNet
from utility_functions import comp_allclose_and_pcc, comp_pcc
batch_size=1


def test_run_resnet_inference():
    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        torch_resnet.eval()
        print(torch_resnet.layer1)

        state_dict = torch_resnet.state_dict()
        tt_resnet18 = _resnet(BasicBlock, [2, 2, 2, 2], state_dict, device=device, host=host)

        dataloader = prep_ImageNet(batch_size=batch_size)
        for i, (images, targets, _, _, _) in enumerate(tqdm(dataloader)):
            torch_output = torch_resnet(images).unsqueeze(1).unsqueeze(1)
            tt_output = tt_resnet18(images)

            print(comp_allclose_and_pcc(torch_output, tt_output))
            break

test_run_resnet_inference()
