# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from transformers import DeiTModel

from models.utility_functions import (
    comp_pcc,
    comp_allclose_and_pcc,
)

from models.experimental.deit.tt.deit_config import DeiTConfig

from models.experimental.deit.tt.deit_patch_embeddings import DeiTPatchEmbeddings


def test_deit_patch_embeddings_inference(device, pcc=0.99):
    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address = "embeddings.patch_embeddings"
    torch_patch_embeddings = model.embeddings.patch_embeddings

    input_shape = torch.Size([1, 3, 224, 224])
    pixel_values = torch.randn(input_shape)

    torch_output = torch_patch_embeddings(pixel_values)

    # setup tt model
    tt_patch_embeddings = DeiTPatchEmbeddings(DeiTConfig(), state_dict, base_address)

    tt_output = tt_patch_embeddings(pixel_values)

    pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
    _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
    logger.info(f"Output {pcc_output}")
    assert pcc_passing, f"Failed! Low pcc: {pcc}."
