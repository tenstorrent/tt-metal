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
from models.squeezebert.tt.squeezebert_self_attention import TtSqueezeBertSelfAttention
from transformers import (
    SqueezeBertForQuestionAnswering as HF_SqueezeBertForQuestionAnswering,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_squeezebert_self_attention_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    HF_model = HF_SqueezeBertForQuestionAnswering.from_pretrained(
        "squeezebert/squeezebert-uncased"
    )

    LAYER_IDX = 0
    torch_model = HF_model.transformer.encoder.layers[LAYER_IDX].attention
    cin, q_groups, k_groups, v_groups = 768, 4, 4, 4
    # Tt squeezebert_self_attention
    config = HF_model.config
    tt_model = TtSqueezeBertSelfAttention(
        config,
        cin=cin,
        q_groups=q_groups,
        k_groups=k_groups,
        v_groups=v_groups,
        state_dict=HF_model.state_dict(),
        base_address=f"transformer.encoder.layers.{LAYER_IDX}.attention",
        device=device,
    )

    hidden_states = torch.rand(1, 768, 19)
    attention_mask = torch.rand(1, 1, 1, 19)
    output_attention = False

    tt_hidden_states = torch_to_tt_tensor_rm(hidden_states, device, put_on_device=False)
    tt_attention_mask = torch_to_tt_tensor_rm(
        attention_mask, device, put_on_device=False
    )

    with torch.no_grad():
        torch_output = torch_model(hidden_states, attention_mask, output_attention)
        tt_output = tt_model(tt_hidden_states, tt_attention_mask, output_attention)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output["context_layer"]).squeeze(0)
    torch_output = torch_output["context_layer"]
    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("SqueezeBertSelfAttention Passed!")

    assert does_pass, f"SqueezeBertSelfAttention does not meet PCC requirement {pcc}."
