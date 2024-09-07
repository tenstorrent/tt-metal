# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer, T5Model
import torch
import json
import pytest
import ttnn
from loguru import logger

from models.utility_functions import (
    Profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
from models.experimental.t5.tt.t5_model import TtT5Model

BATCH_SIZE = 1


def run_perf_t5(expected_inference_time, expected_compile_time, device):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    comments = "small"
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    use_attention_mask = True

    tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=32)
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    tt_model = TtT5Model(config, hf_reference_model.state_dict(), device)

    # Prepare input
    input_sentance = "Studies have been shown that owning a dog is good for you"
    tokenized = tokenizer(input_sentance, padding="max_length", max_length=32, return_tensors="pt")  # Batch size 1

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask if use_attention_mask else None

    decoder_input_sentence = "Studies show that"
    tokenized = tokenizer(
        decoder_input_sentence, padding="max_length", max_length=32, return_tensors="pt"
    )  # Batch size 1

    decoder_input_ids = tokenized.input_ids
    decoder_attention_mask = tokenized.attention_mask if use_attention_mask else None

    # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    decoder_input_ids = hf_reference_model._shift_right(decoder_input_ids)

    with torch.no_grad():
        # PyTorch forward pass
        profiler.start(cpu_key)
        pt_out = hf_reference_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_model_outputs = tt_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_model_outputs

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_model_outputs = tt_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_model_outputs

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    prep_perf_report(
        model_name="T5",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"t5 small inference time: {second_iter_time}")
    logger.info(f"t5 compile time: {compile_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            0.068,
            6.5,
        ),
    ),
)
def test_perf_bare_metal(device, use_program_cache, expected_inference_time, expected_compile_time):
    run_perf_t5(expected_inference_time, expected_compile_time, device)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            0.10,
            7,
        ),
    ),
)
def test_perf_virtual_machine(device, use_program_cache, expected_inference_time, expected_compile_time):
    run_perf_t5(expected_inference_time, expected_compile_time, device)
