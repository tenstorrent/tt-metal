# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from loguru import logger
from transformers import VisionEncoderDecoderModel

from models.experimental.trocr.tt.trocr_decoder import TtTrOCRDecoder
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.66),),
)
def test_trocr_decoder_inference(device, pcc, reset_seeds):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        config = model.decoder.config

        base_address = f"decoder"

        torch_model = model.decoder.model.decoder

        # run torch model
        input = torch.rand(1, 5).long()

        model_output = torch_model(input.long()).last_hidden_state

        tt_model = TtTrOCRDecoder(
            config=config,
            base_address=base_address,
            state_dict=model.state_dict(),
            device=device,
        )

        # run tt model
        tt_input = torch_to_tt_tensor_rm(input, device=device, put_on_device=False)
        tt_output = tt_model(tt_input)
        tt_output_torch = tt_to_torch_tensor(tt_output.last_hidden_state).squeeze(0)

        # compare output
        passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

        logger.info(comp_allclose(model_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("TrOCRDecoder Passed!")
        else:
            logger.warning("TrOCRDecoder Failed!")

        assert passing, f"TrOCRDecoder output does not meet PCC requirement {pcc}."
