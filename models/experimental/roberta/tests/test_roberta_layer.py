# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from transformers import RobertaModel

import pytest

from models.experimental.roberta.tt.roberta_layer import TtRobertaLayer
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
    skip_for_wormhole_b0,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


@skip_for_wormhole_b0()
def test_roberta_layer_inference(device):
    torch.manual_seed(1234)

    SELF_ATTN_LAYER_INDEX = 0
    base_address = f"encoder.layer.{SELF_ATTN_LAYER_INDEX}"

    model = RobertaModel.from_pretrained("roberta-base")

    # Torch roberta
    torch_model = model.encoder.layer[SELF_ATTN_LAYER_INDEX]

    # Tt roberta
    tt_model = TtRobertaLayer(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
    )
    # Run torch model
    hidden_states = torch.rand(1, 32, 768)
    attention_mask = torch.ones(1, 1, 32)
    torch_output = torch_model(hidden_states, attention_mask=attention_mask)
    torch_output = torch_output[0]

    # Run tt model
    hidden_states = torch.unsqueeze(hidden_states, 0)
    tt_hidden_states = torch2tt_tensor(hidden_states, device)
    attention_mask = torch.unsqueeze(attention_mask, 0)
    tt_attention_mask = torch2tt_tensor(attention_mask, device)

    tt_output = tt_model(tt_hidden_states, attention_mask=tt_attention_mask)
    tt_output = tt_output[0]

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_output)
    while len(tt_output_torch.size()) != len(torch_output.size()):
        tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("RobertaLayer Passed!")
    else:
        logger.warning("RobertaLayer Failed!")

    assert does_pass
