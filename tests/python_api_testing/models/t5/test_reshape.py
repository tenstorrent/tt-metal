from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import numpy as np
#from pymetal import ttlib as ttm
from libs import tt_lib as ttm

from utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from loguru import logger
from python_api_testing.models.t5.t5_utils import torch2tt_tensor, tt2torch_tensor, print_corr_coef


def test_reshape(device):

    torch.manual_seed(0)
    test_input = (torch.rand(1, 1, 64, 32) * 2) - 1
    tt_input = torch2tt_tensor(test_input, device)
    tt_input_host = tt2torch_tensor(tt_input)

    logger.info("Pytorch input tensor:")
    print(test_input[0, 0, 1:10, 1:10])

    logger.info("Tt input tensor:")
    print(tt_input_host[0, 0, 1:10, 1:10])

    print_diff_argmax(test_input, tt_input_host)

    pt_out = torch.reshape(test_input, (1, 1, 32, 64))
    #pt_out = test_input.reshape((1, 1, 32, 64))
    ttm.tensor.reshape(tt_input, 1, 1, 32, 64)
    tt_out = tt2torch_tensor(tt_input)

    logger.info("Pytorch tensor reshaped:")
    print(pt_out[0, 0, 1:10, 1:10])

    logger.info("Tt tensor reshaped:")
    print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    if does_pass:
        logger.info("Test reshape Passed!")
    else:
        logger.warning("Test reshape Failed!")


if __name__ == "__main__":
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    test_reshape(device)
    ttm.device.CloseDevice(device)
