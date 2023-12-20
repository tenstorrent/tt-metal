# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from torchvision import models
from loguru import logger

import tt_lib

from models.experimental.vgg.tt.vgg import *


_batch_size = 1


def test_gs_demo(device, imagenet_sample_input, imagenet_label_dict):
    image = imagenet_sample_input
    class_labels = imagenet_label_dict

    batch_size = _batch_size
    with torch.no_grad():
        # TODO: enable conv on tt device after adding fast dtx transform
        tt_vgg = vgg16(device, disable_conv_on_tt_device=True)

        tt_image = tt_lib.tensor.Tensor(
            image.reshape(-1).tolist(),
            get_shape(image.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        tt_output = tt_vgg(tt_image)

        tt_output = tt_output.cpu()
        tt_output = torch.Tensor(tt_output.to_torch())

        logger.info(f"GS's predicted Output: {class_labels[torch.argmax(tt_output).item()]}\n")
