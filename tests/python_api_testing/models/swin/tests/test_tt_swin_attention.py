from pathlib import Path
import sys
import torch
import pytest
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

from tt_models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
)
import tt_lib
from python_api_testing.models.swin.tt.swin_attention import (
    TtSwinAttention,
)
from transformers import SwinModel


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_swin_attention_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    ATTN_LAYER_INDEX = 0
    base_address = (
        f"encoder.layers.{ATTN_LAYER_INDEX}.blocks.{ATTN_LAYER_INDEX}.attention"
    )

    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # Torch swinattention
    torch_model = (
        model.encoder.layers[ATTN_LAYER_INDEX].blocks[ATTN_LAYER_INDEX].attention
    )

    # Tt swinattention
    num_heads, window_size, dim = 3, 7, 96
    tt_model = TtSwinAttention(
        config=model.config,
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
        host=host,
    )

    # Run torch model
    hidden_states = torch.rand(64, 49, 96)
    torch_output = torch_model(hidden_states)

    # Run tt model
    hidden_states = torch.unsqueeze(hidden_states, 0)
    tt_hidden_states = torch_to_tt_tensor_rm(hidden_states, device)

    tt_output = tt_model(tt_hidden_states)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)
    if does_pass:
        logger.info("SwinAttention Passed!")
    else:
        logger.warning("SwinAttention Failed!")

    assert does_pass
