# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from transformers import AutoTokenizer, RobertaForSequenceClassification
from loguru import logger

import ttnn

from models.experimental.roberta.tt.roberta_for_sequence_classification import TtRobertaForSequenceClassification
from models.utility_functions import (
    Profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
from models.experimental.roberta.roberta_common import torch2tt_tensor

BATCH_SIZE = 1


def run_perf_roberta(expected_inference_time, expected_compile_time, device):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    comments = "Base Emotion"
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        model.eval()

        # Tt roberta
        tt_model = TtRobertaForSequenceClassification(
            config=model.config,
            base_address="",
            device=device,
            state_dict=model.state_dict(),
            reference_model=model,
        )
        tt_model.eval()

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

        profiler.start(cpu_key)
        torch_output = model(**inputs).logits
        profiler.end(cpu_key)

        tt_attention_mask = torch.unsqueeze(inputs.attention_mask, 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, device)

        profiler.start(first_key)

        tt_output = tt_model(inputs.input_ids, tt_attention_mask).logits
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(inputs.input_ids, tt_attention_mask).logits
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_output

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    prep_perf_report(
        model_name="roberta",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    compile_time = first_iter_time - second_iter_time

    logger.info(f"roberta {comments} inference time: {second_iter_time}")
    logger.info(f"roberta compile time: {compile_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            0.405,
            17,
        ),
    ),
)
def test_perf_bare_metal(device, use_program_cache, expected_inference_time, expected_compile_time):
    run_perf_roberta(expected_inference_time, expected_compile_time, device)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            0.60,
            17.5,
        ),
    ),
)
def test_perf_virtual_machine(device, use_program_cache, expected_inference_time, expected_compile_time):
    run_perf_roberta(expected_inference_time, expected_compile_time, device)
