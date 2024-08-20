# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import BloomForCausalLM, BloomTokenizerFast
import numpy as np
import pytest

from models.experimental.bloom.bloom_utils import run_generate
from models.experimental.bloom.dataset_utils import get_data
import evaluate

from loguru import logger
import models.experimental.bloom.tt.bloom_causal_lm as bloom_causal_lm


@pytest.mark.parametrize(
    "iterations",
    ((4),),
)
def test_bloom_causal_lm(device, model_location_generator, iterations):
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    hf_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hf_reference_model.eval()

    tt_model = bloom_causal_lm.TtBloomForCausalLM(hf_reference_model.config, hf_reference_model.state_dict(), device)

    accuracy_metric = evaluate.load("accuracy")
    input_loc = model_location_generator("nanogpt/inputs/hellaswag_validation.jsonl")
    val_examples = get_data(input_loc)
    calculated_label = []
    bert_score = evaluate.load("bertscore")
    for i in range(iterations):
        prompt = val_examples[i].input_sentence
        output = run_generate(
            hf_reference_model=hf_reference_model,
            tt_model=tt_model,
            tokenizer=tokenizer,
            input_sentance=prompt,
            max_tokens=35,
            device=device,
        )
        prediction = output[len(prompt) :]
        score = []
        for end in val_examples[i].endings:
            results = bert_score.compute(predictions=[prediction], references=[end], lang="en")
            score.append(results["f1"])
        calculated_label.append(score)
    calculated_label = np.array(calculated_label)
    calculated_label = list(calculated_label.argmax(1))
    golden_labels = [x.label for x in val_examples]
    accuracy = accuracy_metric.compute(references=golden_labels[:iterations], predictions=calculated_label)
    logger.info("Accuracy")
    logger.info(accuracy)
