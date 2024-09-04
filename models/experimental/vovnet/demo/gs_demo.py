# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import timm

from torchvision.utils import save_image
from loguru import logger

from models.experimental.vovnet.tt.vovnet import vovnet_for_image_classification
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)


@pytest.mark.parametrize(
    "model_name",
    (("hf_hub:timm/ese_vovnet19b_dw.ra_in1k"),),
)
def test_vovnet_model_inference(device, imagenet_sample_input, imagenet_label_dict, model_name, reset_seeds):
    model = timm.create_model(model_name, pretrained=True)

    torch_model = model

    tt_model = vovnet_for_image_classification(
        device=device,
    )

    input = imagenet_sample_input
    model_output = torch_model(input)

    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_output = tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)

    predicted_label = tt_output_torch.argmax(-1).item()

    logger.info("GS's Predicted Output")
    logger.info(imagenet_label_dict[predicted_label])

    save_image(imagenet_sample_input, "vonet_input.jpg")
    logger.info("Input image is saved for reference as vovnet_input.jpg")
