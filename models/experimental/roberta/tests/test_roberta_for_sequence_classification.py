# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from transformers import AutoTokenizer, RobertaForSequenceClassification

import pytest

from models.experimental.roberta.tt.roberta_for_sequence_classification import TtRobertaForSequenceClassification
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
    skip_for_wormhole_b0,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


@skip_for_wormhole_b0()
def test_roberta_for_sequence_classification(device):
    torch.manual_seed(1234)
    base_address = ""

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        model.eval()

        # Tt roberta
        tt_model = TtRobertaForSequenceClassification(
            config=model.config,
            base_address=base_address,
            device=device,
            state_dict=model.state_dict(),
            reference_model=model,
        )
        tt_model.eval()

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

        # Run torch model
        logger.info("Running torch model ...")
        torch_output = model(**inputs).logits

        # Run tt model
        logger.info("Running tt model ...")
        tt_attention_mask = torch.unsqueeze(inputs.attention_mask, 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, device)
        tt_output = tt_model(inputs.input_ids, tt_attention_mask).logits

        # Compare outputs
        predicted_class_id = torch_output.argmax().item()
        torch_predicted_class = model.config.id2label[predicted_class_id]
        logger.info(f"Torch Predicted {torch_predicted_class}")

        tt_output_torch = tt2torch_tensor(tt_output)
        tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)

        tt_predicted_class_id = tt_output_torch.argmax().item()
        tt_predicted_class = tt_model.config.id2label[tt_predicted_class_id]
        logger.info(f"Tt Predicted {tt_predicted_class}")

        does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

        logger.info(comp_allclose(torch_output, tt_output_torch))
        logger.info(pcc_message)

        if does_pass:
            logger.info("RobertaForSequenceClassification Passed!")
        else:
            logger.warning("RobertaForSequenceClassification Failed!")

        assert does_pass
