from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")

import torch

from loguru import logger

import tt_lib
from utility_functions_new import comp_pcc, tt2torch_tensor
from tests.python_api_testing.models.conftest import model_location_generator_
from mnist import *
import pytest


def run_mnist_inference(model, on_weka, pcc, model_location_generator):

    with torch.no_grad():

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)


        if on_weka:
            model_name = str(
                model_location_generator("tt_dnn-models/ConvNetMNIST/") / model
            )
        else:
            model_name = model

        torch_ConvNet, state_dict = load_torch(model_name)
        test_dataset, test_loader = prep_data()
        first_input, label = next(iter(test_loader))

        tt_convnet = TtConvNet(device, state_dict)
        with torch.no_grad():
            img = first_input.to("cpu")
            # unsqueeze to go from [batch, 10] to [batch, 1, 1, 10]

            torch_output = torch_ConvNet(img).unsqueeze(1).unsqueeze(1)
            _, torch_predicted = torch.max(torch_output.data, -1)

            tt_image = tt_lib.tensor.Tensor(
                img.reshape(-1).tolist(),
                img.shape,
                tt_lib.tensor.DataType.BFLOAT16,
                tt_lib.tensor.Layout.ROW_MAJOR,
            )
            tt_output = tt_convnet(tt_image)
            tt_output = tt2torch_tensor(tt_output)

            pcc_passing, pcc_output = comp_pcc(torch_output, tt_output)
            logger.info(f"Output {pcc_output}")
            assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "model, on_weka, pcc",
    (("convnet_mnist.pt", True, 0.99),),
)
def test_mnist_inference(model, on_weka, pcc, model_location_generator, reset_seeds):
    run_mnist_inference(model, on_weka, pcc, model_location_generator)
