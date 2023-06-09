from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torchvision import models
from loguru import logger

import tt_lib

from tt.vgg import *


_batch_size = 1



def test_gs_demo(imagenet_sample_input, imagenet_label_dict):
    image = imagenet_sample_input
    class_labels = imagenet_label_dict

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

        tt_output = tt_output.to(host)
        tt_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())

        logger.info(
            f"GS's predicted Output: {class_labels[torch.argmax(tt_output).item()]}\n"
        )

        tt_lib.device.CloseDevice(device)
