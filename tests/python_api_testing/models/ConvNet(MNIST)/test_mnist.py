from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch

from loguru import logger

from libs import tt_lib as ttl
from utility_functions import comp_pcc
from mnist import *

def test_mnist_convnet_inference():

    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device

        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_ConvNet, state_dict = load_torch()
        test_dataset, test_loader = prep_data()

        tt_convnet = TtConvNet(device, host, state_dict)
        for image, labels in test_loader:
            img = image.to('cpu')
            # unsqueeze to go from [batch, 10] to [batch, 1, 1, 10]
            torch_output = torch_ConvNet(img).unsqueeze(1).unsqueeze(1)
            _, torch_predicted = torch.max(torch_output.data, -1)

            tt_output = tt_convnet(img)

            _, tt_predicted = torch.max(tt_output.data, -1)
            print(tt_output.shape, torch_output.shape)

            passing = comp_pcc(torch_output, tt_output)
            assert passing[0], passing[1:]
            break
    logger.info(f"ConvNet MNIST PASSED {passing[1]}")
    ttl.device.CloseDevice(device)
