# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from transformers import AutoTokenizer, RobertaModel

import pytest

from models.experimental.roberta.tt.roberta_model import TtRobertaModel
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
    is_wormhole_b0,
    is_blackhole,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_roberta_model_inference(device):
    torch.manual_seed(1234)

    base_address = f""
    torch_model = RobertaModel.from_pretrained("roberta-base")

    torch_model.eval()
    with torch.no_grad():
        # Tt roberta
        tt_model = TtRobertaModel(
            config=torch_model.config,
            base_address=base_address,
            device=device,
            state_dict=torch_model.state_dict(),
            reference_model=torch_model,
            add_pooling_layer=False,
        )

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

        # Run torch model
        torch_output = torch_model(**inputs)
        torch_output = torch_output.last_hidden_state

        # Run tt model
        tt_attention_mask = torch.unsqueeze(inputs.attention_mask.float(), 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, device)

        tt_output = tt_model(inputs.input_ids, tt_attention_mask)
        tt_output = tt_output.last_hidden_state

        # Compare outputs
        tt_output = tt2torch_tensor(tt_output)
        while len(torch_output.size()) < len(tt_output.size()):
            tt_output = tt_output.squeeze(0)

        does_pass, pcc_message = comp_pcc(torch_output, tt_output, 0.98)

        logger.info(comp_allclose(torch_output, tt_output))
        logger.info(pcc_message)

        if does_pass:
            logger.info("RobertaModel Passed!")
        else:
            logger.warning("RobertaModel Failed!")

        assert does_pass
