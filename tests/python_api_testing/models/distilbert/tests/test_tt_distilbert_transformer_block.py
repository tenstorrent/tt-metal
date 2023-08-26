import pytest
from loguru import logger
import torch
import tt_lib
from tt_models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from tt_models.utility_functions import (
    comp_allclose,
    comp_pcc,
)
from tt_models.distilbert.tt.distilbert_transformer_block import TtTransformerBlock
from transformers import (
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_distilbert_transformer_block_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(
        "distilbert-base-uncased-distilled-squad"
    )
    LAYER_INDEX = 0
    torch_model = HF_model.distilbert.transformer.layer[LAYER_INDEX]
    base_address = f"distilbert.transformer.layer.{LAYER_INDEX}"

    # Tt distilbert_transformer_block
    config = HF_model.config
    tt_model = TtTransformerBlock(
        config,
        state_dict=HF_model.state_dict(),
        base_address=base_address,
        device=device,
    )

    input = torch.rand(1, 19, 768)
    attn_mask = torch.rand(1, 19)

    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=True)
    tt_attn_mask = torch_to_tt_tensor_rm(attn_mask, device, put_on_device=False)

    with torch.no_grad():
        torch_output = torch_model(input, attn_mask)
        tt_output = tt_model(tt_input, tt_attn_mask)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output[0]).squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("DistilBertTransformerBlock Passed!")

    assert does_pass, f"DistilBertTransformerBlock does not meet PCC requirement {pcc}."
