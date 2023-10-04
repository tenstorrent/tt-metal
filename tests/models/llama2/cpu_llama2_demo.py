# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tests.models.llama2.reference.generation import Llama
from tests.models.llama2.dataset_utils import get_data
import evaluate
import numpy as np

from loguru import logger
import pytest


@pytest.mark.parametrize(
    "iterations",
    ((20),),
)
def test_cpu_demo(iterations, model_location_generator):
    llama2_path = str(model_location_generator("llama-2-7b", model_subdir="llama-2"))
    bert_score = evaluate.load("bertscore")

    facebook_research_reference_model = Llama.build(llama2_path, llama2_path, 50, 1)

    facebook_research_reference_model.model.eval()

    input_loc = model_location_generator("nanogpt/inputs/hellaswag_validation.jsonl")
    val_examples = get_data(input_loc)
    calculated_label = []

    for i in range(iterations):
        prompt = val_examples[i].input_sentence
        output = Llama.text_completion(facebook_research_reference_model, [prompt])
        prediction = output[0]
        score = []
        for end in val_examples[i].endings:
            results = bert_score.compute(predictions=[prediction], references=[end], lang="en")
            score.append(results["f1"])
        calculated_label.append(score)

    calculated_label = np.array(calculated_label)
    golden_labels = np.array([x.label for x in val_examples])
    accuracy = np.mean(calculated_label.argmax(1) == golden_labels[:iterations])
    logger.info("Accuracy")
    logger.info(accuracy)
