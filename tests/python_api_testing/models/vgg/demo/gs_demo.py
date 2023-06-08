from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torchvision.transforms as transforms
import torch
import pytest
import ast
from torchvision import models
from loguru import logger
from PIL import Image

import tt_lib

from tt.vgg import *


_batch_size = 1



@pytest.mark.parametrize("image_path", [f"{f}/../sample_image.JPEG"])
def test_gs_demo(image_path):
    im = Image.open(image_path)
    im = im.resize((224, 224))

    # Apply the transformation to the random image and Add an extra dimension at the beginning
    # to match the desired shape of 3x224x224
    image = transforms.ToTensor()(im).unsqueeze(0)

    batch_size = _batch_size
    with torch.no_grad():
        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()


        # TODO: enable conv on tt device after adding fast dtx transform
        tt_vgg = vgg16(device, host, disable_conv_on_tt_device=True)

        tt_image = tt_lib.tensor.Tensor(
            image.reshape(-1).tolist(),
            get_shape(image.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        tt_output = tt_vgg(tt_image)

        with open(f"{f}/../imagenet_class_labels.txt", "r") as file:
            class_labels = ast.literal_eval(file.read())

        tt_output = tt_output.to(host)
        tt_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())

        logger.info(
            f"GS's predicted Output: {class_labels[torch.argmax(tt_output).item()]}\n"
        )

        tt_lib.device.CloseDevice(device)
