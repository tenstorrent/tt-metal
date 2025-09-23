# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
from loguru import logger


from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
from models.experimental.swin.tt.swin_encoder import TtSwinEncoder
from transformers import SwinModel


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_swin_encoder_inference(device, pcc, reset_seeds):
    base_address = f"encoder"

    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # Torch swinencoder
    torch_model = model.encoder

    # Tt swinencoder
    grid_size = (56, 56)

    tt_model = TtSwinEncoder(
        config=model.config,
        grid_size=grid_size,
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
    )

    # Run torch model
    hidden_states = torch.rand(1, 3136, 96)
    input_dimensions = (56, 56)

    torch_output = torch_model(hidden_states, input_dimensions)

    # Run tt model
    hidden_states = torch.unsqueeze(hidden_states, 0)
    tt_hidden_states = torch_to_tt_tensor_rm(hidden_states, device)

    tt_output = tt_model(tt_hidden_states, input_dimensions)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output.last_hidden_state)
    tt_output_torch = tt_output_torch.squeeze(0)
    does_pass, pcc_message = comp_pcc(torch_output.last_hidden_state, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output.last_hidden_state, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("SwinEncoder Passed!")
    else:
        logger.warning("SwinEncoder Failed!")

    assert does_pass
