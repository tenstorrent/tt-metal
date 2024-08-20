# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from datasets import load_dataset
import evaluate
from models.experimental.bloom.bloom_utils import pad_input_32

from transformers import BloomTokenizerFast, BloomForQuestionAnswering
from models.experimental.bloom.tt.bloom_qa import TtBloomForQuestionAnswering


@pytest.mark.parametrize(
    "model_name,iterations",
    (("bigscience/bloom-560m", 10),),
)
def test_bloom_qa(model_name, device, iterations):
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    hf_reference_model = BloomForQuestionAnswering.from_pretrained(model_name)
    config = hf_reference_model.config

    with torch.no_grad():
        tt_model = TtBloomForQuestionAnswering(config, hf_reference_model.state_dict(), device)

    squad_dataset = load_dataset("squad_v2")
    validation_split = squad_dataset["validation"]
    predicted_answers = []
    reference_answer = []
    index = 0
    iteration = 0
    while iteration < iterations:
        question = validation_split["question"][index]
        context = validation_split["context"][index]
        answers = validation_split["answers"][index]
        id = validation_split["id"][index]
        index += 1

        if len(answers["text"]) > 0:
            iteration += 1

            inputs = tokenizer(question, context, return_tensors="pt")

            input_ids = pad_input_32(inputs.input_ids, hf_reference_model.config.pad_token_id)

            attention_mask = pad_input_32(inputs.attention_mask, 0)

            tt_output = tt_model.forward(input_ids=input_ids, attention_mask=attention_mask, device=device)

            answer_start_index = tt_output[0].argmax()
            answer_end_index = tt_output[1].argmax()

            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

            tt_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

            predicted_answers.append(
                {
                    "prediction_text": tt_answer,
                    "id": id,
                    "no_answer_probability": 0.0,
                }
            )
            reference_answer.append(
                {
                    "answers": {
                        "answer_start": [answers["answer_start"][0]],
                        "text": [answers["text"][0]],
                    },
                    "id": id,
                }
            )

    squad_metric = evaluate.load("squad_v2")
    eval_score = squad_metric.compute(predictions=predicted_answers, references=reference_answer)
    logger.info("Exact Match :")
    logger.info(eval_score["exact"])
    logger.info("F1 Score :")
    logger.info(eval_score["f1"])
