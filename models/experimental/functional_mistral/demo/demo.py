# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import torch

from loguru import logger

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)
import numpy as np
from pathlib import Path
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.mistral.reference.model import Transformer
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.reference.tokenizer import Tokenizer
from models.experimental.functional_mistral.tt.ttnn_functional_transformer import mistral_transformer
from models.experimental.functional_mistral.tt.mistral_utils import generate
from models.experimental.functional_mistral.dataset_utils import get_data

import evaluate

from ttnn.model_preprocessing import *

# from models import generation_utils

"""
def generate_next_token(model, input_ids, parameters, num_heads, logits_processor, max_length, **kwargs):
    num_tokens = input_ids.shape[-1]
    padded_input_ids, alibi, causal_mask = model.preprocess_inputs(
        input_ids=input_ids,
        num_heads=num_heads,
        max_length=max_length,
        attention_mask=None,
        **kwargs,
    )

    logits = model.bloom_for_causal_lm(padded_input_ids, alibi, causal_mask, parameters, num_heads)
    next_token_logits = logits[:, num_tokens - 1, :]  # Get the logits for the last token
    processed_logits = logits_processor(input_ids, next_token_logits)
    next_token = torch.argmax(processed_logits, dim=-1).unsqueeze(-1)
    return next_token
"""


def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    input_prompt = []
    for i in range(batch):
        input_prompt.append(user_input[i]["input"])
    return input_prompt


def run_mistral_causal_LM_inference(
    model_version,
    functional_model,
    batch_size,
    input_path,
    model_location_generator,
    device,
):
    torch.manual_seed(1234)
    model_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), n_layers=1, max_batch_size=batch_size, is_whole_model=False)
    with open(model_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    model_args.max_batch_size = batch_size
    model_args.n_layers = 1
    # load input
    prompts = load_inputs(input_path, batch_size)

    profiler.start("preprocessing_parameter")
    reference_model = Transformer(args=model_args)
    parameters = preprocess_model_parameters(
        model_name=f"ttnn-functional-bloom-for-causal-lm",
        initialize_model=lambda: reference_model,
        device=device,
    )

    profiler.end("preprocessing_parameter")

    profiler.start("Ouput Generation")
    output = generate(
        model_args,
        prompts,
        functional_model,
        tokenizer,
        parameters,
        max_tokens=5,
        device=device,
    )

    profiler.end("Ouput Generation")

    logger.info("Input Prompt")
    logger.info(prompts)
    logger.info("Mistral Model output")
    logger.info(output)

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "Ouput Generation": profiler.get("Ouput Generation"),
    }

    return measurements


def run_mistral_causal_LM_inference_hellaswag(
    model_version,
    functional_model,
    batch_size,
    model_location_generator,
    device,
    n_iterations,
):
    torch.manual_seed(1234)
    model_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), n_layers=1, max_batch_size=batch_size, is_whole_model=False)
    with open(model_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    model_args.max_batch_size = batch_size
    model_args.n_layers = 1

    dataset_path = model_location_generator("nanogpt/inputs/hellaswag_validation.jsonl")

    val_inputs = get_data(dataset_path)

    # num_heads = config.n_head

    reference_model = Transformer(args=model_args)
    parameters = preprocess_model_parameters(
        model_name=f"ttnn-functional-bloom-for-causal-lm",
        initialize_model=lambda: reference_model,
        device=device,
    )
    bert_score = evaluate.load("bertscore")
    accuracy_metric = evaluate.load("accuracy")
    calculated_label = []
    for i in range(n_iterations):
        prompts = [val_inputs[i * batch_size + j].input_sentence for j in range(batch_size)]
        endings = [val_inputs[(i * batch_size) + j].endings for j in range(batch_size)]
        generated_text = generate(
            model_args,
            prompts,
            functional_model,
            tokenizer,
            parameters,
            max_tokens=5,
            device=device,
        )

        prediction = [generated_text[i][len(prompts[i]) :] for i in range(len(prompts))]
        score = []
        for i in range(len(endings)):
            results = bert_score.compute(predictions=[prediction[i]], references=[endings[i]], lang="en")
            score.append(results["f1"])
        calculated_label.append(score)
    calculated_label = np.array(calculated_label)
    calculated_label = list(calculated_label.argmax(1))
    golden_labels = [val_inputs[(i * batch_size) + j].label for j in range(batch_size)]
    accuracy = accuracy_metric.compute(references=golden_labels[:n_iterations], predictions=calculated_label)
    logger.info("Accuracy")
    logger.info(accuracy)


def test_demo(
    input_path,
    model_location_generator,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_mistral_causal_LM_inference(
        model_version="mistral-7B-v0.1",
        functional_model=mistral_transformer,
        batch_size=8,
        input_path=input_path,
        model_location_generator=model_location_generator,
        device=device,
    )


@pytest.mark.parametrize(
    "n_iterations",
    ((2),),
)
def test_demo_hellaswag(model_location_generator, device, use_program_cache, n_iterations):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_mistral_causal_LM_inference_hellaswag(
        model_version="mistral-7B-v0.1",
        functional_model=mistral_transformer,
        batch_size=8,
        model_location_generator=model_location_generator,
        device=device,
        n_iterations=n_iterations,
    )
