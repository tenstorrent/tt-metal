import torch
from loguru import logger
import tt_lib

from tt_models.utility_functions import (
    comp_pcc,
    tt2torch_tensor,
    torch2tt_tensor,
)
from tt_models.ConvNet_MNIST.tt.convnet_mnist import convnet_mnist
from tt_models.ConvNet_MNIST.convnet_mnist_utils import get_test_data


def test_mnist_inference():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    tt_convnet, pt_convnet = convnet_mnist(device)
    test_input, images = get_test_data(64)

    with torch.no_grad():
        pt_output = pt_convnet(test_input).unsqueeze(1).unsqueeze(1)

        tt_input = torch2tt_tensor(
            test_input, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        tt_output = tt_convnet(tt_input)
        tt_output = tt2torch_tensor(tt_output)

        pcc_passing, pcc_output = comp_pcc(pt_output, tt_output)
        logger.info(f"Output {pcc_output}")

        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

    tt_lib.device.CloseDevice(device)
