# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from transformers import RobertaForMaskedLM


from models.experimental.roberta.tt.roberta_lm_head import TtRobertaLMHead
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


def test_roberta_lm_head(device):
    torch.manual_seed(1234)

    base_address = f"lm_head"

    model = RobertaForMaskedLM.from_pretrained("roberta-base")
    torch_model = model.lm_head

    # Tt roberta
    tt_model = TtRobertaLMHead(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
    )

    input_x = torch.rand([1, 9, 768])

    # Run torch model
    torch_output = torch_model(input_x)

    # Run tt model
    input_x = torch.unsqueeze(input_x, 0)
    input_x = torch2tt_tensor(input_x, device)

    tt_output = tt_model(input_x)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("RobertaLMHead Passed!")
    else:
        logger.warning("RobertaLMHead Failed!")

    assert does_pass
