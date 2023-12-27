# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from datasets import load_dataset
import evaluate
import pytest

from models.experimental.t5.tt.t5_for_conditional_generation import (
    t5_small_for_conditional_generation,
)
from models.experimental.t5.t5_utils import run_generate
from transformers import T5ForConditionalGeneration, AutoTokenizer


@pytest.mark.parametrize(
    "iterations",
    (9,),
)
def test_demo_t5_small(device, iterations):
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    tt_model, _ = t5_small_for_conditional_generation(device)

    dataset = load_dataset("openai/summarize_from_feedback", "axis")
    bert_score = evaluate.load("bertscore")

    validation_split = dataset["validation"]["info"]
    reference_split = dataset["validation"]["summary"]

    prediction = []
    references = []

    for i in range(iterations):
        input_text = f"summarize: {validation_split[i]['post']}"
        reference_answer = reference_split[i]["text"][1:]
        tt_output = run_generate(
            input_text,
            tokenizer,
            model,
            tt_model,
            device,
            run_tt_model=True,
        )
        prediction.append(tt_output)
        references.append(reference_answer)

    results = bert_score.compute(predictions=prediction, references=references, lang="en")
    Accuracy = sum(results["f1"]) / len(results["f1"])
    logger.info("Accuracy:")
    logger.info(Accuracy)
