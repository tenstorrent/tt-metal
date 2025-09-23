# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoImageProcessor, DeiTForImageClassification
import torch
from loguru import logger


from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.deit.tt.deit_for_image_classification import deit_for_image_classification


def test_gs_demo(hf_cat_image_sample_input, device):
    image = hf_cat_image_sample_input

    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    inputs = image_processor(images=image, return_tensors="pt")

    torch_model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
    torch_model.eval()

    tt_inputs = torch_to_tt_tensor_rm(inputs["pixel_values"], device, put_on_device=False)
    tt_model = deit_for_image_classification(device)

    with torch.no_grad():
        tt_output = tt_model(tt_inputs)[0]
        tt_output = tt_to_torch_tensor(tt_output).squeeze(0)[:, 0, :]

    # model prediction
    image.save("deit_without_teacher_gs_input_image.jpg")
    predicted_label = tt_output.argmax(-1).item()

    logger.info(f"Input image saved as deit_without_teacher_gs_input_image.jpg")
    logger.info(f"TT's prediction: {torch_model.config.id2label[predicted_label]}.")
