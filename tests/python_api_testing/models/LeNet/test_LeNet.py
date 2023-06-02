from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import tt_lib
import torch
from loguru import logger
from LeNet5 import *

from sweep_tests.comparison_funcs import comp_allclose_and_pcc, comp_pcc


def test_LeNet_inference(model_location_generator):
    with torch.no_grad():
        torch.manual_seed(1234)

        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        host = tt_lib.device.GetHost()

        pt_model_path = model_location_generator("tt_dnn-models/LeNet/model.pt")
        torch_LeNet, state_dict = load_torch_LeNet(pt_model_path)
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
    tt_lib.device.CloseDevice(device)
