import torch
import pytest
from loguru import logger

from transformers import VisionEncoderDecoderModel

import tt_lib

from tt_models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from tt_models.utility_functions import (
    comp_pcc,
    comp_allclose,
)

from tt_models.trocr.tt.trocr_decoder_layer import TtTrOCRDecoderLayer


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_decoder_layer_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        config = model.decoder.config

        base_address = f"decoder.model.decoder.layers.0"

        torch_model = model.decoder.model.decoder.layers[0]

        tt_model = TtTrOCRDecoderLayer(
            config=config,
            base_address=base_address,
            state_dict=model.state_dict(),
            device=device,
        )

        # run torch model
        input = torch.rand(1, 3, 1024)

        model_output = torch_model(input)[0]

        # run tt model
        tt_input = torch_to_tt_tensor_rm(input, device)
        tt_output = tt_model(tt_input)
        tt_output_torch = tt_to_torch_tensor(tt_output[0])
        tt_output_torch = tt_output_torch.squeeze(0)

        # compare output
        passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

        logger.info(comp_allclose(model_output, tt_output_torch))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)
        if passing:
            logger.info("TrOCRDecoderLayer Passed!")
        else:
            logger.warning("TrOCRDecoderLayer Failed!")

        assert passing, f"TrOCRDecoderLayer output does not meet PCC requirement {pcc}."
