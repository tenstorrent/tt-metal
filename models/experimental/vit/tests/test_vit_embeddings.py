# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger
from transformers import ViTForImageClassification as HF_ViTForImageClassication

from models.utility_functions import comp_allclose_and_pcc, comp_pcc
from models.experimental.vit.tt.modeling_vit import ViTEmbeddings


@pytest.mark.skip(reason="#7527: Test needs review")
def test_vit_embeddings(imagenet_sample_input, pcc=0.99):
    image = imagenet_sample_input

    with torch.no_grad():
        HF_model = HF_ViTForImageClassication.from_pretrained("google/vit-base-patch16-224")

        state_dict = HF_model.state_dict()
        reference = HF_model.vit.embeddings

        config = HF_model.config
        HF_output = reference(image, None, None)

        tt_image = image
        tt_layer = ViTEmbeddings(config, base_address="vit.embeddings", state_dict=state_dict)

        tt_output = tt_layer(tt_image, None, None)
        pcc_passing, _ = comp_pcc(HF_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(HF_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
