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

from tt_models.trocr.tt.trocr_for_causal_llm import TtTrOCRForCausalLM


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_causal_llm_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        config = model.decoder.config

        base_address = f"decoder"

        torch_model = model.decoder
        # run torch model
        input = torch.rand(1, 20).long()

        model_output = torch_model(input.long()).logits

        tt_model = TtTrOCRForCausalLM(
            config=config,
            base_address=base_address,
            state_dict=model.state_dict(),
            device=device,
        )

        # run tt model
        tt_input = torch_to_tt_tensor_rm(input, device=device, put_on_device=False)
        tt_output = tt_model(tt_input)
        tt_output_torch = tt_to_torch_tensor(tt_output.logits).squeeze(0).float()

        # compare output
        passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

        logger.info(comp_allclose(model_output, tt_output_torch))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)
        if passing:
            logger.info("TrOCRCausalLlm Passed!")
        else:
            logger.warning("TrOCRCausalLlm Failed!")

        assert passing, f"TrOCRCausalLlm output does not meet PCC requirement {pcc}."
