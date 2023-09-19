# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from datasets import load_dataset
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
import pytest
import tt_lib as ttl
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from test_bert_batch_dram import TtBertBatchDram

from tests.models.metal_BERT_large_15.model_config import get_model_config

from models.utility_functions import (
    enable_persistent_kernel_cache,
    enable_compilation_reports,
    enable_memory_reports,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
    disable_persistent_kernel_cache,
    profiler,
)
from models.utility_functions import prep_report
import pytest
from loguru import logger

BATCH_SIZE = 8
model_version = "phiyodr/bert-large-finetuned-squad2"
comments = "Large"
seq_len = 384
real_input = True
attention_mask = True
token_type_ids = True
model_config_str = "MIXED_PRECISION_BATCH8"


def run_perf_bert15(expected_inference_time, expected_compile_time, model_location_generator, device):
    model_config = get_model_config(model_config_str)
    model_name = str(model_location_generator(model_version, model_subdir = "Bert"))
    tokenizer_name = str(model_location_generator(model_version, model_subdir = "Bert"))

    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    HF_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    HF_model.eval()
    tt_model = TtBertBatchDram(HF_model.config, HF_model, device, model_config)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    context = BATCH_SIZE * [
        "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    ]
    question = BATCH_SIZE * ["What discipline did Winkelmann create?"]
    inputs = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=attention_mask,
        return_token_type_ids=token_type_ids,
        return_tensors="pt",
    )
    tt_input = tt_model.model_preprocessing(**inputs)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_out = HF_model(**inputs)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(1, *tt_input)
        ttl.device.Synchronize()
        profiler.end(first_key, force_enable=True)
        del tt_output
        tt_input = tt_model.model_preprocessing(**inputs)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(1, *tt_input)
        ttl.device.Synchronize()
        profiler.end(second_key, force_enable=True)
        del tt_output

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report(
        model_name="bert15",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time
    )
    compile_time = first_iter_time - second_iter_time
    logger.info(f"bert15 inference time: {second_iter_time}")
    logger.info(f"bert15 compile time: {compile_time}")


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    ([0.15, 11],),
)
def test_perf_virtual_machine(use_program_cache, expected_inference_time, expected_compile_time, model_location_generator, device):
    run_perf_bert15(expected_inference_time, expected_compile_time, model_location_generator, device)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    ([0.08, 8.5],),
)
def test_perf_bare_metal(use_program_cache, expected_inference_time, expected_compile_time, model_location_generator, device):
    run_perf_bert15(expected_inference_time, expected_compile_time, model_location_generator, device)
