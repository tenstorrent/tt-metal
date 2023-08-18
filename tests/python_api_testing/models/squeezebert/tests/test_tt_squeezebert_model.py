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
from models.squeezebert.tt.squeezebert_model import TtSqueezeBertModel
from transformers import (
    SqueezeBertForQuestionAnswering as HF_SqueezeBertForQuestionAnswering,
)
from transformers import AutoTokenizer


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_squeezebert_model_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
    HF_model = HF_SqueezeBertForQuestionAnswering.from_pretrained(
        "squeezebert/squeezebert-uncased"
    )

    torch_model = HF_model.transformer

    # Tt squeezebert_model
    config = HF_model.config
    tt_model = TtSqueezeBertModel(
        config,
        state_dict=HF_model.state_dict(),
        base_address=f"transformer",
        device=device,
    )

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )

    inputs = tokenizer(question, context, return_tensors="pt")

    tt_attention_mask = torch_to_tt_tensor_rm(
        inputs.attention_mask, device, put_on_device=False
    )

    with torch.no_grad():
        torch_output = torch_model(**inputs)
        """
        Torch tensor is passed as input for embedding to address low pcc
        """
        tt_output = tt_model(inputs.input_ids, tt_attention_mask, inputs.token_type_ids)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output.last_hidden_state).squeeze(0)
    torch_output = torch_output.last_hidden_state
    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("SqueezeBertModel Passed!")

    assert does_pass, f"SqueezeBertModel does not meet PCC requirement {pcc}."
