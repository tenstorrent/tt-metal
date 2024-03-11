# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from loguru import logger
import tt_lib
import json
from pathlib import Path

from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    torch_to_tt_tensor_rm,
    Profiler,
)
from models.perf.perf_utils import prep_perf_report
from models.experimental.mistral.reference.model import Transformer
from models.experimental.mistral.tt.mistral_transformer import TtTransformer
from models.experimental.mistral.mistral_utils import generate as generate_tt
from models.experimental.mistral.reference.tokenizer import generate as generate_torch
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.reference.tokenizer import Tokenizer

BATCH_SIZE = 1


def run_perf_mistral(expected_inference_time, expected_compile_time, device, model_location_generator):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    comments = "Mistral"
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    prompts = [
        "A Man is sitting on the roof ",
    ]

    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    tokenizer = Tokenizer(str(Path(mistral_path) / "tokenizer.model"))
    base_address = f""
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    model_args.max_batch_size = 1
    model_args.n_layers = 32
    max_tokens = 2
    max_batch_size = 1

    pytorch_model = Transformer.from_folder(
        Path(mistral_path), n_layers=32, max_batch_size=max_batch_size, is_whole_model=True
    )

    tt_cache_path = "/mnt/MLPerf/tt_dnn-models/tt/Mistral/"
    tt_model = TtTransformer(
        args=model_args,
        device=device,
        base_address=base_address,
        tt_cache_path=tt_cache_path,
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output, _logprobs = generate_torch(
            prompts,
            pytorch_model,
            tokenizer,
            max_tokens=max_tokens,
        )
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = generate_tt(
            prompts,
            tt_model,
            tokenizer,
            max_tokens=max_tokens,
        )
        tt_lib.device.Synchronize(device)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = generate_tt(
            prompts,
            tt_model,
            tokenizer,
            max_tokens=max_tokens,
        )
        tt_lib.device.Synchronize(device)
        profiler.end(second_key)
        del tt_output

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    prep_perf_report(
        model_name="Mistral",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"Mistral inference time: {second_iter_time}")
    logger.info(f"Mitstral compile time: {compile_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            411,
            11.2,
        ),
    ),
)
def test_perf_bare_metal(
    device, use_program_cache, expected_inference_time, expected_compile_time, model_location_generator
):
    run_perf_mistral(expected_inference_time, expected_compile_time, device, model_location_generator)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            411,
            11.2,
        ),
    ),
)
def test_perf_virtual_machine(
    device, use_program_cache, expected_inference_time, expected_compile_time, model_location_generator
):
    run_perf_mistral(expected_inference_time, expected_compile_time, device, model_location_generator)
