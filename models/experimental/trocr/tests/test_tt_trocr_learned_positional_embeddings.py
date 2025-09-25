# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from loguru import logger
from transformers import VisionEncoderDecoderModel


from models.experimental.trocr.tt.trocr_learned_positional_embeddings import TtTrOCRLearnedPositionalEmbedding
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_attention_inference(device, pcc, reset_seeds):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        config = model.decoder.config

        base_address = f"decoder.model.decoder.embed_positions"

        torch_model = model.decoder.model.decoder.embed_positions

        num_heads = 512
        embed_dim = 1024
        tt_model = TtTrOCRLearnedPositionalEmbedding(
            num_embeddings=num_heads,
            embedding_dim=embed_dim,
            config=config,
            base_address=base_address,
            state_dict=model.state_dict(),
            device=device,
        )

        # run torch model
        input = torch.rand(1, 4)

        model_output = torch_model(input)

        # run tt model
        tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)
        tt_output_torch = tt_model(tt_input)

        # compare output
        passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

        logger.info(comp_allclose(model_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("TrOCRLearnedPositionalEmbeddings Passed!")
        else:
            logger.warning("TrOCRLearnedPositionalEmbeddings Failed!")

        assert passing, f"TrOCRLearnedPositionalEmbeddings output does not meet PCC requirement {pcc}."
