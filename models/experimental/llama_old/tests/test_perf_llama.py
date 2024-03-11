# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from models.experimental.llama_old.tt.llama_model import TtLlamaModel
import tt_lib
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    torch_to_tt_tensor_rm,
    Profiler,
)
from models.perf.perf_utils import prep_perf_report


BATCH_SIZE = 1


def run_perf_llama(expected_inference_time, expected_compile_time, device):
    model_version = "baffo32/decapoda-research-llama-7B-hf"
    tokenizer_version = "hf-internal-testing/llama-tokenizer"
    batch = 1
    seq_len = 64
    num_decoders = 2
    max_position_embeddings = 2048
    on_weka = False
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # get only llama model (no linear layer at the end)
    llama_model = hugging_face_reference_model.get_decoder()

    batch = BATCH_SIZE
    seq_len = seq_len
    if 1:
        llama_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        oneseq = [torch.arange(seq_len)] * batch
        llama_input = torch.stack(oneseq)
        llama_input = llama_input.reshape(batch, seq_len)

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    num_decoders = num_decoders
    base_url = "model.layers"
    max_position_embeddings = max_position_embeddings

    tt_llama_model = TtLlamaModel(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders,
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        pytorch_out = llama_model(llama_input)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_out = tt_llama_model(llama_input)
        tt_lib.device.Synchronize(device)
        profiler.end(first_key)
        del tt_out

        enable_persistent_kernel_cache()
        profiler.start(second_key)
        tt_out = tt_llama_model(llama_input)
        tt_lib.device.Synchronize(device)
        profiler.end(second_key)
        del tt_out

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    prep_perf_report(
        model_name="llama",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="7B",
        inference_time_cpu=cpu_time,
    )

    logger.info(f"llama 7B inference time: {second_iter_time}")
    logger.info(f"llama 7B compile time: {compile_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    ((0.150, 10),),
)
def test_perf_bare_metal(device, use_program_cache, expected_inference_time, expected_compile_time):
    run_perf_llama(expected_inference_time, expected_compile_time, device)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    ((0.3, 11),),
)
def test_perf_virtual_machine(device, use_program_cache, expected_inference_time, expected_compile_time):
    run_perf_llama(expected_inference_time, expected_compile_time, device)
