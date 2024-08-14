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

from models.experimental.swin.tt.swin_patch_embedding import TtSwinPatchEmbeddings
from transformers import SwinModel


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_swin_patch_embeddings_inference(device, pcc, reset_seeds):
    base_address = f"embeddings.patch_embeddings"

    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # Torch swinpatchembedding
    torch_model = model.embeddings.patch_embeddings

    # Tt swinpatchembedding
    tt_model = TtSwinPatchEmbeddings(
        config=model.config,
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
    )

    # Run torch model
    pixel_values = torch.rand(1, 3, 224, 224)

    torch_output = torch_model(pixel_values)

    # Run tt model
    tt_pixel_values = torch_to_tt_tensor_rm(pixel_values, device)

    tt_output = tt_model(tt_pixel_values)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)
    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("SwinPatchEmbedding Passed!")
    else:
        logger.warning("SwinPatchEmbedding Failed!")

    assert does_pass
