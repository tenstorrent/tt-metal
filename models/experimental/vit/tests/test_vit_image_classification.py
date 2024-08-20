# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from loguru import logger
from transformers import ViTForImageClassification as HF_ViTForImageClassication
from transformers import AutoImageProcessor as HF_AutoImageProcessor

from models.experimental.vit.tt.modeling_vit import TtViTForImageClassification
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose_and_pcc,
)


@pytest.mark.skip(reason="#7527: Test needs review")
def test_vit_image_classification(device, hf_cat_image_sample_input, pcc=0.95):
    image = hf_cat_image_sample_input

    with torch.no_grad():
        HF_model = HF_ViTForImageClassication.from_pretrained("google/vit-base-patch16-224")
        image_processor = HF_AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        inputs = image_processor(image, return_tensors="pt")

        reference = HF_model
        state_dict = HF_model.state_dict()

        config = HF_model.config
        HF_output = reference(**inputs).logits

        tt_inputs = torch_to_tt_tensor_rm(inputs["pixel_values"], device, put_on_device=False)
        tt_model = TtViTForImageClassification(config, base_address="", state_dict=state_dict, device=device)
        tt_model.vit.get_head_mask = reference.vit.get_head_mask
        tt_output = tt_model(tt_inputs)[0]
        tt_output = tt_to_torch_tensor(tt_output).squeeze(0)[:, 0, :]
        pcc_passing, _ = comp_pcc(HF_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(HF_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")

        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
