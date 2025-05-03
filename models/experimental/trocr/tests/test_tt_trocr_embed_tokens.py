# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from loguru import logger
from transformers import VisionEncoderDecoderModel


from models.experimental.trocr.tt.trocr_embed_tokens import TtTrOCREmbedTokens
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_embed_tokens_inference(device, pcc, reset_seeds):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        config = model.decoder.config

        base_address = f"decoder.model.decoder.embed_tokens"

        torch_model = model.decoder.model.decoder.embed_tokens

        tt_model = TtTrOCREmbedTokens(
            config=config,
            base_address=base_address,
            state_dict=model.state_dict(),
            device=device,
        )

        # run torch model
        input = torch.rand(1, 3, 577, 768)

        model_output = torch_model(input.long()).squeeze(0)

        # run tt model
        tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)
        tt_output = tt_model(tt_input)
        tt_output_torch = tt_to_torch_tensor(tt_output)

        # compare output
        passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

        logger.info(comp_allclose(model_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("TrOCREmbedTokens Passed!")
        else:
            logger.warning("TrOCEmbedTokens Failed!")

        assert passing, f"TrOCEmbedTokens output does not meet PCC requirement {pcc}."
