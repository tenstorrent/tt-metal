# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from torchvision.utils import save_image

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from models.experimental.hrnet.tt.hrnet_model import hrnet_w18_small


@pytest.mark.parametrize(
    "model_name",
    (("hrnet_w18_small"),),
)
def test_gs_demo(device, imagenet_sample_input, imagenet_label_dict, model_name, reset_seeds):
    tt_model = hrnet_w18_small(device, multi_scale_output=True)

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device, put_on_device=False)

    with torch.no_grad():
        tt_output = tt_model(tt_input)

    tt_output_torch = tt_to_torch_tensor(tt_output).view(1, -1)

    logger.info("GS's Predicted Output")
    logger.info(imagenet_label_dict[tt_output_torch[0].argmax(-1).item()])

    save_image(imagenet_sample_input, "hrnet_input.jpg")
    logger.info("Input image is saved for reference as hrnet_input.jpg")
