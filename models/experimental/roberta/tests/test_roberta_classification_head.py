# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from transformers import RobertaForSequenceClassification


from models.experimental.roberta.tt.roberta_classification_head import TtRobertaClassificationHead
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


def test_roberta_classification_head(device):
    torch.manual_seed(1234)
    base_address = f"classifier"
    with torch.no_grad():
        model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        torch_model = model.classifier

        # Tt roberta
        tt_model = TtRobertaClassificationHead(
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
        tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)

        does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

        logger.info(comp_allclose(torch_output, tt_output_torch))
        logger.info(pcc_message)

        if does_pass:
            logger.info("RobertaClassificationHead Passed!")
        else:
            logger.warning("RobertaClassificationHead Failed!")

        assert does_pass
