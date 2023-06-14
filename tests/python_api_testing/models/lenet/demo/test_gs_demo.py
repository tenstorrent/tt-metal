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
import pytest
import tt_lib

from tt.lenet import lenet5
from lenet_utils import load_torch_lenet, prepare_image


@pytest.mark.parametrize(
    "",
    ((),),
)
def test_gs_demo(mnist_sample_input, model_location_generator):
    sample_image = mnist_sample_input
    image = prepare_image(sample_image)
    num_classes = 10
    batch_size = 1
    with torch.no_grad():
        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()

        tt_lenet = lenet5(num_classes, device, host, model_location_generator)

        tt_image = tt_lib.tensor.Tensor(
            image.reshape(-1).tolist(),
            image.shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        tt_output = tt_lenet(tt_image)
        tt_output = tt_output.to(host)
        tt_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())

        _, tt_predicted = torch.max(tt_output.data, -1)

        sample_image.save("input_image.jpg")
        logger.info(f"Input image savd as input_image.jpg.")
        logger.info(f"GS's predicted Output: {tt_predicted[0][0][0]}.")

        tt_lib.device.CloseDevice(device)
