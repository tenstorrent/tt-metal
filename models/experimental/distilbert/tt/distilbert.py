# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.distilbert.tt.distilbert_for_question_answering import (
    TtDistilBertForQuestionAnswering,
)
from transformers import (
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
)


def _distilbert(config, state_dict, base_address, device):
    return TtDistilBertForQuestionAnswering(
        config=config,
        state_dict=state_dict,
        base_address=base_address,
        device=device,
    )


def distilbert_for_question_answering(device) -> TtDistilBertForQuestionAnswering:
    model_name = "distilbert-base-uncased-distilled-squad"
    model = HF_DistilBertForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    state_dict = model.state_dict()
    config = model.config
    base_address = f""
    model = _distilbert(config, state_dict, base_address, device)
    return model
