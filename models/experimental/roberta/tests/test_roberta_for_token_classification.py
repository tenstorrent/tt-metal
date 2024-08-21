# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from transformers import AutoTokenizer, RobertaForTokenClassification

import pytest

from models.experimental.roberta.tt.roberta_for_token_classification import TtRobertaForTokenClassification
from models.utility_functions import (
    tt2torch_tensor,
    comp_allclose,
    comp_pcc,
    skip_for_wormhole_b0,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


@skip_for_wormhole_b0()
@pytest.mark.skip(reason="Test is failing. see issue #7533")
def test_roberta_for_token_classification(device):
    torch.manual_seed(1234)

    base_address = ""

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        model = RobertaForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        model.eval()

        # Tt roberta
        tt_model = TtRobertaForTokenClassification(
            config=model.config,
            base_address=base_address,
            device=device,
            state_dict=model.state_dict(),
            reference_model=model,
        )
        tt_model.eval()

        inputs = tokenizer(
            "HuggingFace is a company based in Paris and New York",
            add_special_tokens=False,
            return_tensors="pt",
        )

        # Run torch model
        logger.info("Running torch model ...")
        torch_output = model(**inputs).logits
        print(torch_output.size())

        # Run tt model
        logger.info("Running tt model ...")
        tt_attention_mask = torch.unsqueeze(inputs.attention_mask, 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, device)
        tt_output = tt_model(inputs.input_ids, tt_attention_mask).logits

        # Compare outputs

        # Torch output
        torch_predicted_token_class_ids = torch_output.argmax(-1)
        torch_predicted_tokens_classes = [model.config.id2label[t.item()] for t in torch_predicted_token_class_ids[0]]
        logger.info(f"Torch Predicted {torch_predicted_tokens_classes}")

        # Tt ouptut
        tt_output_torch = tt2torch_tensor(tt_output)
        tt_output_torch = tt_output_torch.squeeze(0)

        tt_predicted_token_class_ids = tt_output_torch.argmax(-1)
        tt_predicted_tokens_classes = [tt_model.config.id2label[t.item()] for t in tt_predicted_token_class_ids[0]]
        logger.info(f"Tt Predicted {tt_predicted_tokens_classes}")

        does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

        logger.info(comp_allclose(torch_output, tt_output_torch))
        logger.info(pcc_message)

        if does_pass:
            logger.info("RobertaForTokenClassification Passed!")
        else:
            logger.warning("RobertaForTokenClassification Failed!")

        assert does_pass
