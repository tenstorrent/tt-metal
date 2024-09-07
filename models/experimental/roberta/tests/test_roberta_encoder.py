# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from transformers import RobertaModel

import pytest

from models.experimental.roberta.tt.roberta_encoder import TtRobertaEncoder
from models.utility_functions import tt2torch_tensor, comp_allclose, comp_pcc, is_wormhole_b0, is_blackhole
from models.experimental.roberta.roberta_common import torch2tt_tensor


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_roberta_encoder_inference(device):
    torch.manual_seed(1234)
    base_address = f"encoder"

    model = RobertaModel.from_pretrained("roberta-base")

    # Torch roberta self attn
    torch_model = model.encoder

    # Tt roberta self attn
    tt_model = TtRobertaEncoder(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
    )
    # Run torch model
    hidden_states = torch.rand(1, 32, 768)
    attention_mask = torch.ones(1, 1, 32)
    torch_output = torch_model(hidden_states, attention_mask=attention_mask)

    # Run tt model
    hidden_states = torch.unsqueeze(hidden_states, 0)
    tt_hidden_states = torch2tt_tensor(hidden_states, device)
    attention_mask = torch.unsqueeze(attention_mask, 0)
    tt_attention_mask = torch2tt_tensor(attention_mask, device)

    tt_output = tt_model(tt_hidden_states, attention_mask=tt_attention_mask)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_output.last_hidden_state)
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output.last_hidden_state, tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output.last_hidden_state, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("RobertaEncoder Passed!")
    else:
        logger.warning("RobertaEncoder Failed!")

    assert does_pass
