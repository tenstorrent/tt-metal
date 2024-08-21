# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import torch
from loguru import logger
from models.generation_utils import get_logits_processor
import ttnn
import time

from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from models.demos.grayskull.t5.tt import ttnn_optimized_functional_t5
from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)


def load_inputs(input_path, batch):
    with open(input_path) as f:
        input_data = json.load(f)
        assert len(input_data) >= batch, f"Input data needs to have at least {batch} (batch size) entries."

        input_text = []
        for i in range(batch):
            input_text.append(input_data[i]["content"])

        return input_text


durations = []


def run_generate(input_ids, model, config, parameters, device, max_tokens, batch_size):
    tt_model = ttnn_optimized_functional_t5

    logits_processor = get_logits_processor(input_ids, config)

    decoder_input_ids = model.generation_config.pad_token_id * torch.ones(batch_size, input_ids.shape[-1]).to(
        torch.long
    )

    input_ids = ttnn.from_torch(input_ids, device=device)

    for iteration in range(max_tokens):
        decoder_input_ids = ttnn.from_torch(decoder_input_ids, device=device, dtype=ttnn.uint32)

        start = time.time()
        tt_output, encoder_hidden_states = tt_model.t5_for_conditional_generation(
            config,
            input_ids,
            decoder_input_ids,
            parameters=parameters,
        )
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

        tt_output = ttnn.from_device(tt_output)
        next_token_logits = ttnn.to_torch(tt_output)

        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        decoder_input_ids = ttnn.from_device(decoder_input_ids)
        decoder_input_ids = ttnn.to_torch(decoder_input_ids)
        decoder_input_ids[:, iteration + 1] = next_tokens[:, iteration]

    inference_and_compile_time, *_, inference_time = durations

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Tokens per second: {1 / inference_time * batch_size}")

    return decoder_input_ids


def run_summarization_inference(device, batch_size, sequence_length, max_tokens, model_name):
    input_path = "models/demos/grayskull/t5/demo/input_data_cg.json"
    config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)

    input_sentance = load_inputs(input_path, batch_size)

    input_ids = tokenizer(
        input_sentance,
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    tt_model_name = "ttnn_optimized_" + model_name

    decoded_tt_output = []

    convert_to_ttnn = ttnn_optimized_functional_t5.convert_to_ttnn

    custom_preprocessor = ttnn_optimized_functional_t5.custom_preprocessor

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: model,
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    tt_output = run_generate(
        input_ids,
        model,
        config,
        parameters,
        device,
        max_tokens,
        batch_size,
    )

    for batch in range(batch_size):
        output = tokenizer.decode(tt_output[batch], skip_special_tokens=True)
        decoded_tt_output.append(output)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "max_tokens", "model_name"),
    ((8, 128, 2, "t5-small"),),
)
def test_t5_demo_for_summarize(device, use_program_cache, batch_size, sequence_length, max_tokens, model_name):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_summarization_inference(device, batch_size, sequence_length, max_tokens, model_name)
