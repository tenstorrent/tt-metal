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
import pytest
import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, profiler
from utility_functions_new import disable_persistent_kernel_cache, enable_persistent_kernel_cache
from utility_functions_new import prep_report

from transformers import BloomForCausalLM, BloomTokenizerFast
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger

import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_model as bloom_model


BATCH_SIZE = 1


def run_perf_bloom(expected_inference_time, expected_compile_time):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    model_name = "bigscience/bloom-560m"
    tokenizer_name = "bigscience/bloom-560m"
    comments = "560M"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    HF_model_top = BloomForCausalLM.from_pretrained(
        model_name, torchscript=False
    )
    HF_model_top.eval()

    config = HF_model_top.config

    state_dict = HF_model_top.state_dict()
    base_address = "transformer"

    tt_model = bloom_model.TtBloomModel(config, state_dict, base_address, device)
    HF_model = HF_model_top.transformer

    # Prepare input
    tokenizer = BloomTokenizerFast.from_pretrained(tokenizer_name)
    inputs = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."

    tokenized = tokenizer(inputs, return_tensors="pt")
    input_ids = tokenized.input_ids

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = HF_model.forward(input_ids)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model.forward(device, input_ids)
        tt_lib.device.Synchronize()
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model.forward(device, input_ids)
        tt_lib.device.Synchronize()
        profiler.end(second_key)
        del tt_output


    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    tt_lib.device.CloseDevice(device)

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    prep_report(
        model_name="bloom",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time
    )

    logger.info(f"bloom {comments} inference time: {second_iter_time}")
    logger.info(f"bloom {comments} compile time: {compile_time}")

    assert second_iter_time < expected_inference_time, f"bloom {comments} is too slow"
    assert compile_time < expected_compile_time, f"bloom {comments} compile time is too slow"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (1.4,
         16,
        ),
    ),
)
def test_perf_bare_metal(use_program_cache, expected_inference_time, expected_compile_time):
    run_perf_bloom(expected_inference_time, expected_compile_time)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (8,
         17,
        ),
    ),
)
def test_perf_virtual_machine(use_program_cache, expected_inference_time, expected_compile_time):
    run_perf_bloom(expected_inference_time, expected_compile_time)
