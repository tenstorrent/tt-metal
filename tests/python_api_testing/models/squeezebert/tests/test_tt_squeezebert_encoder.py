import pytest
from loguru import logger
import torch
import tt_lib
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
)
from models.squeezebert.tt.squeezebert_encoder import TtSqueezeBert_Encoder
from transformers import (
    SqueezeBertForQuestionAnswering as HF_SqueezeBertForQuestionAnswering,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_squeezebert_encoder_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    HF_model = HF_SqueezeBertForQuestionAnswering.from_pretrained(
        "squeezebert/squeezebert-uncased"
    )

    torch_model = HF_model.transformer.encoder

    # Tt squeezebert_encoder
    config = HF_model.config
    tt_model = TtSqueezeBert_Encoder(
        config,
        state_dict=HF_model.state_dict(),
        base_address=f"transformer.encoder",
        device=device,
    )

    hidden_states = torch.rand(1, 19, 768)
    attention_mask = torch.rand(1, 1, 1, 19)
    head_mask = [None, None, None, None, None, None, None, None, None, None, None, None]

    tt_hidden_states = torch_to_tt_tensor_rm(hidden_states, device, put_on_device=True)
    tt_attention_mask = torch_to_tt_tensor_rm(
        attention_mask, device, put_on_device=False
    )

    with torch.no_grad():
        torch_output = torch_model(hidden_states, attention_mask, head_mask)
        tt_output = tt_model(tt_hidden_states, tt_attention_mask, head_mask)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output.last_hidden_state).squeeze(0)
    torch_output = torch_output.last_hidden_state
    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("SqueezeBertEncoder Passed!")

    assert does_pass, f"SqueezeBertEncoder does not meet PCC requirement {pcc}."
