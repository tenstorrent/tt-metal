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

from models.experimental.swin.tt.swin_output import (
    TtSwinOutput,
)
from transformers import SwinModel


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_swin_output_inference(device, pcc, reset_seeds):
    OUTPUT_LAYER_INDEX = 0
    base_address = f"encoder.layers.{OUTPUT_LAYER_INDEX}.blocks.{OUTPUT_LAYER_INDEX}.output"

    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # Torch swinoutput
    torch_model = model.encoder.layers[OUTPUT_LAYER_INDEX].blocks[OUTPUT_LAYER_INDEX].output

    # Tt swinoutput
    dim = 96
    tt_model = TtSwinOutput(
        config=model.config,
        dim=dim,
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
    )

    # Run torch model
    hidden_states = torch.rand(1, 3136, 384)
    torch_output = torch_model(hidden_states)

    # Run tt model
    hidden_states = torch.unsqueeze(hidden_states, 0)
    tt_hidden_states = torch_to_tt_tensor_rm(hidden_states, device)

    tt_output = tt_model(tt_hidden_states)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("SwinOutput Passed!")
    else:
        logger.warning("SwinOutput Failed!")

    assert does_pass
