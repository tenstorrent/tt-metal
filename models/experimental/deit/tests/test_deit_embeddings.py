# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoImageProcessor, DeiTModel
from loguru import logger


from models.experimental.deit.tt.deit_config import DeiTConfig
from models.experimental.deit.tt.deit_embeddings import DeiTEmbeddings
from models.utility_functions import (
    comp_pcc,
    comp_allclose_and_pcc,
)


def test_deit_embeddings_inference(device, hf_cat_image_sample_input, pcc=0.99):
    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()
    base_address = "embeddings"
    torch_embeddings = model.embeddings
    use_mask_token = False
    bool_masked_pos = None
    head_mask = None

    # real input
    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    image = hf_cat_image_sample_input
    input_image = image_processor(images=image, return_tensors="pt")
    input_image = input_image["pixel_values"]

    torch_output = torch_embeddings(input_image, bool_masked_pos)

    # setup tt model
    tt_embeddings = DeiTEmbeddings(DeiTConfig(), base_address, state_dict, use_mask_token)

    tt_output = tt_embeddings(input_image, bool_masked_pos)

    pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
    _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
    logger.info(f"Output {pcc_output}")
    assert pcc_passing, f"Failed! Low pcc: {pcc}."
