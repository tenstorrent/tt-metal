# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger


from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)

from models.experimental.swin.tt.swin_model import TtSwinModel
from models.experimental.swin.swin_utils import get_shape
from transformers import SwinModel
from transformers import AutoFeatureExtractor


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_swin_model_inference(device, imagenet_sample_input, pcc, reset_seeds):
    image = imagenet_sample_input

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    inputs = feature_extractor(images=image, return_tensors="pt")

    base_address = f""

    with torch.no_grad():
        torch_model = model

        tt_model = TtSwinModel(
            config=model.config,
            state_dict=model.state_dict(),
            base_address=base_address,
            device=device,
        )

        # Run torch model
        torch_output = torch_model(**inputs)

        # Run tt model
        tt_pixel_values = inputs.pixel_values
        tt_pixel_values = torch_to_tt_tensor_rm(tt_pixel_values, device)
        tt_output = tt_model(tt_pixel_values)

        tt_output_torch = tt_to_torch_tensor(tt_output.last_hidden_state)
        tt_output_torch = tt_output_torch.squeeze(0)

        does_pass, pcc_message = comp_pcc(torch_output.last_hidden_state, tt_output_torch, pcc)

        logger.info(comp_allclose(torch_output.last_hidden_state, tt_output_torch))
        logger.info(pcc_message)

        if does_pass:
            logger.info("SwinModel Passed!")
        else:
            logger.warning("SwinModel Failed!")

        assert does_pass
