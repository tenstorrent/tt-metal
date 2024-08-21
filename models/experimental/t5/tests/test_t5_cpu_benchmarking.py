# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer
import evaluate
from loguru import logger
import pytest


def t5_cpu_demo(model_name, iterations):
    squad_dataset = load_dataset("squad_v2")

    validation_split = squad_dataset["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

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
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512)
            generated_output = model.generate(input_ids)
            generated_answer = tokenizer.decode(generated_output[0], skip_special_tokens=True)

            predicted_answers.append(
                {
                    "prediction_text": generated_answer,
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


@pytest.mark.parametrize(
    "model_name, iterations",
    (("t5-small", 50),),
)
def test_cpu_demo_t5_small(model_name, iterations):
    t5_cpu_demo(model_name, iterations)


@pytest.mark.parametrize(
    "model_name,iterations",
    (("t5-base", 50),),
)
def test_cpu_demo_t5_base(model_name, iterations):
    t5_cpu_demo(model_name, iterations)


@pytest.mark.parametrize(
    "model_name,iterations",
    (("google/flan-t5-small", 50),),
)
def test_cpu_demo_flan_t5_small(model_name, iterations):
    t5_cpu_demo(model_name, iterations)
