# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

from models.experimental.squeezebert.squeezebert_utils import get_answer
from transformers import SqueezeBertForQuestionAnswering, AutoTokenizer


def test_cpu_demo():
    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
    HF_model = SqueezeBertForQuestionAnswering.from_pretrained("squeezebert/squeezebert-uncased")

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )
    inputs = tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        HF_output = HF_model(**inputs)

    HF_answer = get_answer(inputs, HF_output, tokenizer)

    logger.info("Input Question: Where do I live?")
    logger.info("Input Context: My name is Merve and I live in İstanbul.")
    logger.info("HF Model answered")
    logger.info(HF_answer)
