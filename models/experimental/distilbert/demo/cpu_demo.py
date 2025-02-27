# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

from transformers import (
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
)
from transformers import AutoTokenizer


@pytest.mark.parametrize(
    "model_name",
    (("distilbert-base-uncased-distilled-squad"),),
)
def test_cpu_demo(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(model_name)

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )
    inputs = tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        torch_output = HF_model(**inputs)

        answer_start_index = torch_output.start_logits.argmax()
        answer_end_index = torch_output.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

        answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

    logger.info("Context: ")
    logger.info(context)
    logger.info("Question: ")
    logger.info(question)
    logger.info("HF Model Predicted answer: ")
    logger.info(answer)
