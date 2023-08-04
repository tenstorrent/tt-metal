import pytest
from loguru import logger
import torch
import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from tests.python_api_testing.models.utility_functions_new import (
    comp_allclose,
    comp_pcc,
)
from models.distilbert.tt.distilbert_multihead_self_attention import (
    TtMultiHeadSelfAttention,
)
from transformers import (
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_distilbert_multihead_self_attention_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(
        "distilbert-base-uncased-distilled-squad"
    )
    LAYER_INDEX = 0
    torch_model = HF_model.distilbert.transformer.layer[LAYER_INDEX].attention
    base_address = f"distilbert.transformer.layer.{LAYER_INDEX}"

    # Tt distilbert_mhsa
    config = HF_model.config
    tt_model = TtMultiHeadSelfAttention(
        config,
        state_dict=HF_model.state_dict(),
        base_address=base_address,
        device=device,
    )

    query = torch.rand(1, 19, 768)
    key = torch.rand(1, 19, 768)
    value = torch.rand(1, 19, 768)
    mask = torch.rand(1, 19)

    tt_query = torch_to_tt_tensor_rm(query, device, put_on_device=True)
    tt_key = torch_to_tt_tensor_rm(key, device, put_on_device=True)
    tt_value = torch_to_tt_tensor_rm(value, device, put_on_device=True)
    tt_mask = torch_to_tt_tensor_rm(mask, device, put_on_device=False)

    with torch.no_grad():
        torch_output = torch_model(query, key, value, mask)
        tt_output = tt_model(tt_query, tt_key, tt_value, tt_mask)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output[0]).squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("DistilBertMultiHeadSelfAttention Passed!")

    assert (
        does_pass
    ), f"DistilBertMultiHeadSelfAttention does not meet PCC requirement {pcc}."
