from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import pytest
from loguru import logger

from lenet_utils import load_torch_lenet, prepare_image


@pytest.mark.parametrize(
    "",
    ((),),
)
def test_cpu_demo(mnist_sample_input, model_location_generator):
    sample_image = mnist_sample_input
    image = prepare_image(sample_image)
    num_classes = 10
    batch_size = 1
    with torch.no_grad():
        pt_model_path = model_location_generator("tt_dnn-models/LeNet/model.pt")
        torch_LeNet, state_dict = load_torch_lenet(pt_model_path, num_classes)

        torch_output = torch_LeNet(image).unsqueeze(1).unsqueeze(1)
        _, torch_predicted = torch.max(torch_output.data, -1)

        sample_image.save("input_image.jpg")
        logger.info(f"Input image savd as input_image.jpg.")
        logger.info(f"CPU's predicted Output: {torch_predicted[0][0][0]}.")
