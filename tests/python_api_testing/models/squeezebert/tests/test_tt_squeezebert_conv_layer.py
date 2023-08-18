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
from models.squeezebert.tt.squeezebert_conv_layernorm import TtConvLayerNorm
from transformers import (
    SqueezeBertForQuestionAnswering as HF_SqueezeBertForQuestionAnswering,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_squeezebert_conv_layernorm_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    HF_model = HF_SqueezeBertForQuestionAnswering.from_pretrained(
        "squeezebert/squeezebert-uncased"
    )

    LAYER_IDX = 0
    torch_model = HF_model.transformer.encoder.layers[LAYER_IDX].post_attention
    cin, cout, groups = 768, 768, 1
    # Tt squeezebert_conv_layernorm
    config = HF_model.config
    tt_model = TtConvLayerNorm(
        config,
        cin=cin,
        cout=cout,
        groups=groups,
        state_dict=HF_model.state_dict(),
        base_address=f"transformer.encoder.layers.{LAYER_IDX}.post_attention",
        device=device,
    )

    hidden_states = torch.rand(1, 768, 19)
    input_tensor = torch.rand(1, 768, 19)

    tt_hidden_states = torch_to_tt_tensor_rm(hidden_states, device, put_on_device=False)
    tt_input_tensor = torch_to_tt_tensor_rm(input_tensor, device, put_on_device=False)

    with torch.no_grad():
        torch_output = torch_model(hidden_states, input_tensor)
        tt_output = tt_model(tt_hidden_states, tt_input_tensor)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output).squeeze(0)
    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("SqueezeBertConvLayernorm Passed!")

    assert does_pass, f"SqueezeBertConvLayernorm does not meet PCC requirement {pcc}."
