# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from transformers import RobertaModel


from models.experimental.roberta.tt.roberta_output import TtRobertaOutput
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


def test_roberta_output_inference(device):
    torch.manual_seed(1234)

    SELF_ATTN_LAYER_INDEX = 0
    base_address = f"encoder.layer.{SELF_ATTN_LAYER_INDEX}.output"

    model = RobertaModel.from_pretrained("roberta-base")

    # Torch roberta
    torch_model = model.encoder.layer[SELF_ATTN_LAYER_INDEX].output

    # Tt roberta
    tt_model = TtRobertaOutput(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
    )

    # Run torch model
    hidden_states = torch.rand(1, 8, 3072)
    input_tensor = torch.rand(1, 8, 768)

    torch_output = torch_model(hidden_states, input_tensor)

    # Run tt model
    hidden_states = torch.unsqueeze(hidden_states, 0)
    input_tensor = torch.unsqueeze(input_tensor, 0)
    tt_hidden_states = torch2tt_tensor(hidden_states, device)
    tt_input_tensor = torch2tt_tensor(input_tensor, device)

    tt_output = tt_model(tt_hidden_states, tt_input_tensor)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("RobertaOutput Passed!")
    else:
        logger.warning("RobertaOutput Failed!")

    assert does_pass
