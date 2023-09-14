# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from loguru import logger
import pytest

import tt_lib

from models.utility_functions import Profiler
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    prep_report,
)
from models.experimental.nanogpt.tt.nanogpt import *

BATCH_SIZE = 1


def run_perf_nanogpt(expected_inference_time, expected_compile_time, device):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    prompt = "Hello, my dog is a little"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    HF_model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)

    tt_model = nanogpt_model(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        hf_output = HF_model.generate(inputs.input_ids)
        tt_lib.device.Synchronize()
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model.generate(inputs.input_ids)
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model.generate(inputs.input_ids)
        tt_lib.device.Synchronize()
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report(
        model_name="nanogpt",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="text-generation",
        inference_time_cpu=cpu_time,
    )

    compile_time = first_iter_time - second_iter_time

    logger.info(f"nanogpt inference time: {second_iter_time}")
    logger.info(f"nanogpt compile time: {compile_time}")
    assert second_iter_time < expected_inference_time, "nanogpt is too slow"
    assert compile_time < expected_compile_time, "nanogpt compile time is too slow"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            38.75,
            11.35,
        ),
    ),
)
def test_perf_bare_metal(use_program_cache, expected_inference_time, expected_compile_time, device):
    run_perf_nanogpt(expected_inference_time, expected_compile_time, device)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            47.04,
            16.03,
        ),
    ),
)
def test_perf_virtual_machine(use_program_cache, expected_inference_time, expected_compile_time, device):
    run_perf_nanogpt(expected_inference_time, expected_compile_time, device)
