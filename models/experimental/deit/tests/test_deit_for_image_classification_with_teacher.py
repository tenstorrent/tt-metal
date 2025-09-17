# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher
from loguru import logger


from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose_and_pcc,
)

from models.experimental.deit.tt.deit_for_image_classification_with_teacher import (
    TtDeiTForImageClassificationWithTeacher,
)


def test_deit_for_image_classification_with_teacher_inference(device, hf_cat_image_sample_input, pcc=0.95):
    with torch.no_grad():
        image = hf_cat_image_sample_input

        # real input
        image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
        inputs = image_processor(images=image, return_tensors="pt")

        torch_model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")
        torch_model.eval()
        state_dict = torch_model.state_dict()
        config = torch_model.config

        torch_output = torch_model(**inputs).logits

        tt_inputs = torch_to_tt_tensor_rm(inputs["pixel_values"], device, put_on_device=False)
        tt_model = TtDeiTForImageClassificationWithTeacher(
            config, device=device, state_dict=state_dict, base_address=""
        )

        tt_model.deit.get_head_mask = torch_model.deit.get_head_mask
        tt_output = tt_model(tt_inputs)[0]
        tt_output = tt_to_torch_tensor(tt_output).squeeze(0)[:, 0, :]

        pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Failed! Low pcc: {pcc}."
