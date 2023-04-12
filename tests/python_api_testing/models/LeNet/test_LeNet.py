from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from loguru import logger
from libs import tt_lib as ttl

from LeNet5 import *

from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc, comp_pcc



def test_LeNet_inference():
    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device

        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        torch_LeNet, state_dict = load_torch_LeNet()
        test_dataset, test_loader = prep_data()

        TTLeNet = TtLeNet5(num_classes, device, host, state_dict)


        for image, labels in test_loader:

            img = image.to('cpu')
            torch_output = torch_LeNet(img).unsqueeze(1).unsqueeze(1)
            _, torch_predicted = torch.max(torch_output.data, -1)

            tt_output = TTLeNet(img)
            _, tt_predicted = torch.max(tt_output.data, -1)

            passing = comp_pcc(torch_output, tt_output)
            assert passing[0], passing[1:]
            break

    logger.info(f"LeNet PASSED {passing[1]}")

    ttl.device.CloseDevice(device)

test_LeNet_inference()
