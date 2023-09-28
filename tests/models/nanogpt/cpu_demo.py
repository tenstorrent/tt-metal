# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import evaluate
from tests.models.nanogpt.dataset_utils import get_data


@pytest.mark.parametrize(
    "model_name, iterations",
    (("gpt2", 50),),
)
def test_nanogpt_cpu_demo(model_name, iterations, model_location_generator):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    HF_model = GPT2LMHeadModel.from_pretrained(model_name)
    bert_score = evaluate.load("bertscore")

    calculated_label = []
    input_loc = model_location_generator("nanogpt/inputs/hellaswag_validation.jsonl")
    val_examples = get_data(input_loc)

    HF_model.eval()

    for i in range(iterations):
        # Prepare input
        prompt = val_examples[i].input_sentence
        inputs = tokenizer(prompt, return_tensors="pt", padding=False)

        with torch.no_grad():
            output = HF_model.generate(**inputs)

        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        prediction = answer[len(prompt) + 1 :]

        score = []
        for end in val_examples[i].endings:
            results = bert_score.compute(predictions=[prediction], references=[end], lang="en")
            score.append(results["f1"])

        calculated_label.append(score)

    calculated_label = np.array(calculated_label)
    golden_labels = np.array([x.label for x in val_examples])
    accuracy = np.mean(calculated_label.argmax(1) == golden_labels[:iterations])

    logger.info("Accuracy: ")
    logger.info(accuracy)
