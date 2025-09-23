# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from transformers import RobertaModel


from models.experimental.roberta.tt.roberta_pooler import TtRobertaPooler
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


def test_roberta_pooler_inference(device):
    torch.manual_seed(1234)

    base_address = f"pooler"

    model = RobertaModel.from_pretrained("roberta-base")

    # Torch roberta
    torch_model = model.pooler

    # Tt roberta
    tt_model = TtRobertaPooler(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
    )
    # Run torch model
    hidden_states = torch.rand(1, 32, 768)
    torch_output = torch_model(hidden_states)

    # Run tt model
    hidden_states = torch.unsqueeze(hidden_states, 0)
    tt_hidden_states = torch2tt_tensor(hidden_states, device)

    tt_output = tt_model(tt_hidden_states)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)
    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("RobertaPooler Passed!")
    else:
        logger.warning("RobertaPooler Failed!")

    assert does_pass
