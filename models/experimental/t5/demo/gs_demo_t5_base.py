# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
from loguru import logger
from datasets import load_dataset
import evaluate
import pytest

from models.experimental.t5.tt.t5_for_conditional_generation import (
    t5_base_for_conditional_generation,
)
from models.experimental.t5.t5_utils import run_generate
from transformers import T5ForConditionalGeneration, AutoTokenizer


@pytest.mark.parametrize(
    "iterations",
    (35,),
)
def test_demo_t5_base(device, iterations):
    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    tt_model, _ = t5_base_for_conditional_generation(device)

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
            input_text = f"question: {question} context: {context}"
            tt_output = run_generate(
                input_text,
                tokenizer,
                model,
                tt_model,
                device,
                run_tt_model=True,
                log=False,
            )
            predicted_answers.append(
                {
                    "prediction_text": tt_output,
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
