# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import torch
import evaluate
import numpy as np
from loguru import logger

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)
from models import generation_utils
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import BloomTokenizerFast, BloomForCausalLM, BloomConfig
from models.demos.grayskull.functional_bloom.tt.ttnn_optimized_functional_bloom import *
from models.demos.grayskull.functional_bloom.tt import ttnn_functional_bloom, ttnn_optimized_functional_bloom
from models.demos.grayskull.functional_bloom.dataset_utils import get_data


def generate_next_token(
    model, config, input_ids, parameters, num_heads, logits_processor, device, max_length, **kwargs
):
    num_tokens = input_ids.shape[-1]
    padded_input_ids, alibi, causal_mask = model.preprocess_inputs(
        input_ids=input_ids,
        num_heads=num_heads,
        max_length=max_length,
        attention_mask=None,
        device=device,
        **kwargs,
    )
    logits = model.bloom_for_causal_lm(
        config=config, input_ids=padded_input_ids, alibi=alibi, causal_mask=causal_mask, parameters=parameters
    )
    next_token_logits = logits[:, num_tokens - 1, :]  # Get the logits for the last token
    processed_logits = logits_processor(input_ids, next_token_logits)
    next_token = torch.argmax(processed_logits, dim=-1).unsqueeze(-1)
    return next_token


def generate(
    model,
    config,
    input_ids,
    parameters,
    tokenizer,
    logits_processor,
    num_heads,
    num_tokens_to_decode,
    device,
    max_length=384,
    **kwargs,
):
    # Tokenize the input text and get initial input_ids
    for _ in range(num_tokens_to_decode):
        next_token = generate_next_token(
            model,
            config,
            input_ids,
            parameters,
            num_heads,
            logits_processor,
            device,
            max_length,
            **kwargs,
        )

        # Check if the next token is the end-of-sequence token
        if torch.all(next_token == tokenizer.eos_token_id):
            break

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids


def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    input_prompt = []
    for i in range(batch):
        input_prompt.append(user_input[i]["input"])
    return input_prompt


def run_bloom_causal_LM_inference(
    model_version,
    functional_model,
    batch_size,
    input_path,
    model_location_generator,
    device,
    num_tokens_to_decode=10,
):
    torch.manual_seed(1234)
    config = BloomConfig.from_pretrained(model_version)
    tokenizer = BloomTokenizerFast.from_pretrained(model_version)

    # load input
    input_text = load_inputs(input_path, batch_size)

    num_heads = config.n_head

    profiler.start("preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=f"ttnn-functional-bloom-for-causal-lm",
        initialize_model=lambda: BloomForCausalLM.from_pretrained(model_version).eval(),
        device=device,
        custom_preprocessor=functional_model.custom_preprocessor,
        convert_to_ttnn=lambda model, name: name != "lm_head",
    )
    profiler.end("preprocessing_parameter")

    encoded_prompts = [tokenizer.encode(prompt) for prompt in input_text]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    input_ids = []
    for i in encoded_prompts:
        input_ids.append(i[:min_prompt_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)

    logits_processor = generation_utils.get_logits_processor(input_ids, config)

    generated_ids = generate(
        model=functional_model,
        config=config,
        input_ids=input_ids,
        parameters=parameters,
        tokenizer=tokenizer,
        logits_processor=logits_processor,
        num_heads=num_heads,
        num_tokens_to_decode=num_tokens_to_decode,
        device=device,
    )

    profiler.start("post_processing_output_to_string")
    generated_text = []
    for i in range(len(generated_ids)):
        generated_text.append(tokenizer.decode(generated_ids[i], skip_special_tokens=True))
    profiler.end("post_processing_output_to_string")

    for i in range(len(input_ids)):
        logger.info("Input Prompt ")
        logger.info(input_text[i])
        logger.info("Output Prompt")
        logger.info(generated_text[i])

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "post_processing": profiler.get("post_processing_output_to_string"),
    }

    return measurements, generated_text


def run_bloom_causal_LM_inference_hellaswag(
    model_version,
    functional_model,
    batch_size,
    model_location_generator,
    device,
    loop_count=5,
    num_tokens_to_decode=10,
):
    torch.manual_seed(1234)
    config = BloomConfig.from_pretrained(model_version)
    tokenizer = BloomTokenizerFast.from_pretrained(model_version)

    dataset_path = model_location_generator("nanogpt/inputs/hellaswag_validation.jsonl")

    val_inputs = get_data(dataset_path)

    num_heads = config.n_head

    parameters = preprocess_model_parameters(
        model_name=f"ttnn-functional-bloom-for-causal-lm",
        initialize_model=lambda: BloomForCausalLM.from_pretrained(model_version).eval(),
        device=device,
        custom_preprocessor=functional_model.custom_preprocessor,
        convert_to_ttnn=lambda model, name: name != "lm_head",
    )
    bert_score = evaluate.load("bertscore")
    accuracy_metric = evaluate.load("accuracy")
    calculated_label = []
    for i in range(loop_count):
        input_ids = tokenizer.encode(val_inputs[i].input_sentence, return_tensors="pt")
        input_ids = input_ids.expand((batch_size, input_ids.shape[-1]))
        logits_processor = generation_utils.get_logits_processor(input_ids, config)
        generated_ids = generate(
            model=functional_model,
            config=config,
            input_ids=input_ids,
            parameters=parameters,
            tokenizer=tokenizer,
            logits_processor=logits_processor,
            num_heads=num_heads,
            num_tokens_to_decode=num_tokens_to_decode,
            device=device,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        prediction = generated_text[len(val_inputs[i].input_sentence) :]
        score = []
        for end in val_inputs[i].endings:
            results = bert_score.compute(predictions=[prediction], references=[end], lang="en")
            score.append(results["f1"])
        calculated_label.append(score)
    calculated_label = np.array(calculated_label)
    calculated_label = list(calculated_label.argmax(1))
    golden_labels = [x.label for x in val_inputs]
    accuracy = accuracy_metric.compute(references=golden_labels[:loop_count], predictions=calculated_label)
    logger.info("Accuracy")
    logger.info(accuracy)

    return accuracy


@pytest.mark.parametrize(
    "functional_model",
    (ttnn_functional_bloom, ttnn_optimized_functional_bloom),
)
def test_demo(
    input_path,
    functional_model,
    model_location_generator,
    device,
    use_program_cache,
    batch_size=8,
    num_tokens_to_decode=10,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_bloom_causal_LM_inference(
        model_version="bigscience/bloom-560m",
        functional_model=functional_model,
        batch_size=batch_size,
        input_path=input_path,
        model_location_generator=model_location_generator,
        device=device,
        num_tokens_to_decode=num_tokens_to_decode,
    )


@pytest.mark.parametrize(
    "functional_model",
    (ttnn_functional_bloom, ttnn_optimized_functional_bloom),
)
@pytest.mark.parametrize(
    "loop_count",
    ((4),),
)
def test_demo_hellaswag(
    model_location_generator,
    functional_model,
    device,
    use_program_cache,
    loop_count,
    batch_size=8,
    num_tokens_to_decode=10,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_bloom_causal_LM_inference_hellaswag(
        model_version="bigscience/bloom-560m",
        functional_model=functional_model,
        batch_size=batch_size,
        model_location_generator=model_location_generator,
        device=device,
        loop_count=loop_count,
        num_tokens_to_decode=num_tokens_to_decode,
    )
