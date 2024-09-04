# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import torch
import evaluate
from loguru import logger
from datasets import load_dataset
from models.generation_utils import get_logits_processor
import ttnn

from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from models.demos.grayskull.t5.tt import ttnn_functional_t5
from models.demos.grayskull.t5.tt import ttnn_optimized_functional_t5
from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)


def load_inputs(input_path, batch):
    with open(input_path) as f:
        input_data = json.load(f)
        assert len(input_data) >= batch, f"Input data needs to have at least {batch} (batch size) entries."

        input_text = []
        for i in range(batch):
            input_text.append(input_data[i]["content"])

        return input_text


def run_generate(input_ids, model, config, parameters, device, max_tokens, batch_size):
    tt_model = ttnn_optimized_functional_t5

    logits_processor = get_logits_processor(input_ids, config)

    decoder_input_ids = model.generation_config.pad_token_id * torch.ones(batch_size, input_ids.shape[-1]).to(
        torch.long
    )

    input_ids = ttnn.from_torch(input_ids, device=device)

    profiler.start(f"inference_time")
    for iteration in range(max_tokens):
        decoder_input_ids = ttnn.from_torch(decoder_input_ids, device=device, dtype=ttnn.uint32)

        tt_output, encoder_hidden_states = tt_model.t5_for_conditional_generation(
            config,
            input_ids,
            decoder_input_ids,
            parameters=parameters,
        )
        tt_output = ttnn.from_device(tt_output)
        next_token_logits = ttnn.to_torch(tt_output)

        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        decoder_input_ids = ttnn.from_device(decoder_input_ids)
        decoder_input_ids = ttnn.to_torch(decoder_input_ids)
        decoder_input_ids[:, iteration + 1] = next_tokens[:, iteration]

    profiler.end(f"inference_time")

    return decoder_input_ids


def run_summarization_inference(input_path, device, batch_size, sequence_length, max_tokens, model_name):
    config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)

    input_sentance = load_inputs(input_path, batch_size)

    profiler.start(f"preprocessing_input")
    input_ids = tokenizer(
        input_sentance,
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    profiler.end(f"preprocessing_input")

    tt_model_name = "ttnn_optimized_" + model_name

    decoded_tt_output = []

    convert_to_ttnn = ttnn_optimized_functional_t5.convert_to_ttnn

    custom_preprocessor = ttnn_optimized_functional_t5.custom_preprocessor

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: model,
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    profiler.end(f"preprocessing_parameter")

    tt_output = run_generate(
        input_ids,
        model,
        config,
        parameters,
        device,
        max_tokens,
        batch_size,
    )

    profiler.start(f"post_processing_output_to_string")
    for batch in range(batch_size):
        output = tokenizer.decode(tt_output[batch], skip_special_tokens=True)
        decoded_tt_output.append(output)
    profiler.end(f"post_processing_output_to_string")

    for i in range(batch_size):
        logger.info(
            f"------------------------------------------------------------------------------------------------------------------------"
        )
        logger.info(f"Input text {i} >> {input_sentance[i]}")
        logger.info(f"Output text {i} >> {decoded_tt_output[i]}")
        logger.info("")

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "preprocessing_input": profiler.get("preprocessing_input"),
        "inference_time": profiler.get("inference_time"),
        "post_processing": profiler.get("post_processing_output_to_string"),
    }
    logger.info(f"preprocessing_parameter: {measurements['preprocessing_parameter']} s")
    logger.info(f"preprocessing_input: {measurements['preprocessing_input']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")
    logger.info(f"post_processing : {measurements['post_processing']} s")

    return measurements


def run_summarization_dataset_inference(device, batch_size, sequence_length, max_tokens, model_name):
    config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)

    dataset = load_dataset("openai/summarize_from_feedback", "axis")
    dataset = dataset.shuffle(seed=19)
    bert_score = evaluate.load("bertscore")

    validation_split = dataset["validation"]["info"]
    reference_split = dataset["validation"]["summary"]

    input_sentance = []
    references = []

    for i in range(batch_size):
        references.append(reference_split[i]["text"][1:])
        input_sentance.append(f"summarize: {validation_split[i]['post']}")

    profiler.start(f"preprocessing_input")
    input_ids = tokenizer(
        input_sentance,
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    profiler.end(f"preprocessing_input")

    tt_model_name = "ttnn_optimized_" + model_name

    decoded_tt_output = []

    convert_to_ttnn = ttnn_optimized_functional_t5.convert_to_ttnn

    custom_preprocessor = ttnn_optimized_functional_t5.custom_preprocessor

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: model,
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    profiler.end(f"preprocessing_parameter")

    tt_output = run_generate(
        input_ids,
        model,
        config,
        parameters,
        device,
        max_tokens,
        batch_size,
    )

    profiler.start(f"post_processing_output_to_string")
    for batch in range(batch_size):
        output = tokenizer.decode(tt_output[batch], skip_special_tokens=True)
        decoded_tt_output.append(output)
    profiler.end(f"post_processing_output_to_string")

    for i in range(batch_size):
        logger.info(
            f"------------------------------------------------------------------------------------------------------------------------"
        )
        logger.info(f"Input text {i} >> {input_sentance[i]}")
        logger.info(f"Output text {i} >> {decoded_tt_output[i]}")
        logger.info("")

    results = bert_score.compute(predictions=decoded_tt_output, references=references, lang="en")
    avg_f1 = sum(results["f1"]) / len(results["f1"])
    logger.info("")
    logger.info(f"Average F1 score: {avg_f1}")

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "preprocessing_input": profiler.get("preprocessing_input"),
        "inference_time": profiler.get("inference_time"),
        "post_processing": profiler.get("post_processing_output_to_string"),
    }
    logger.info(f"preprocessing_parameter: {measurements['preprocessing_parameter']} s")
    logger.info(f"preprocessing_input: {measurements['preprocessing_input']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")
    logger.info(f"post_processing : {measurements['post_processing']} s")

    return measurements


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "max_tokens", "model_name"),
    (
        (8, 128, 64, "t5-small"),
        (8, 128, 64, "google/flan-t5-small"),
    ),
)
def test_t5_demo_for_summarize(
    input_path, device, use_program_cache, batch_size, sequence_length, max_tokens, model_name
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_summarization_inference(input_path, device, batch_size, sequence_length, max_tokens, model_name)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "max_tokens", "model_name"),
    (
        (8, 128, 64, "t5-small"),
        (8, 128, 64, "google/flan-t5-small"),
    ),
)
def test_t5_demo_for_summarize_dataset(device, use_program_cache, batch_size, sequence_length, max_tokens, model_name):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_summarization_dataset_inference(device, batch_size, sequence_length, max_tokens, model_name)
