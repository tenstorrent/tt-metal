import pytest
from loguru import logger
import torch
import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from tests.python_api_testing.models.utility_functions_new import (
    comp_allclose,
    comp_pcc,
)
from models.distilbert.tt.distilbert_model import TtDistilBertModel
from transformers import (
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
)
from transformers import AutoTokenizer

@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_distilbert_model_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(
        "distilbert-base-uncased-distilled-squad"
    )

    torch_model = HF_model.distilbert
    base_address = f"distilbert"

    # Tt distilbert_model
    config = HF_model.config
    tt_model = TtDistilBertModel(
        config,
        state_dict=HF_model.state_dict(),
        base_address=base_address,
        device=device,
    )

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )

    inputs = tokenizer(question, context, return_tensors="pt")
    tt_attn_mask = torch_to_tt_tensor_rm(inputs.attention_mask, device, put_on_device=False)

    with torch.no_grad():
        torch_output = torch_model(**inputs)
        """
        Torch tensor is passed as input for embedding to address low pcc
        """
        tt_output = tt_model(inputs.input_ids, tt_attn_mask)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output[0]).squeeze(0)
    torch_output = torch_output[0]

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("DistilBertModel Passed!")

    assert does_pass, f"DistilBertModel does not meet PCC requirement {pcc}."
