# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import ttnn
import torch
import pytest
import transformers
from loguru import logger
from datasets import load_dataset

from models.utility_functions import is_grayskull, is_wormhole_b0
from models.perf.perf_utils import prep_perf_report
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.whisper.tt import ttnn_optimized_functional_whisper
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


def get_expected_times(functional_whisper):
    return {
        ttnn_optimized_functional_whisper: (10.47, 9.5) if is_grayskull() else (23.4, 6.4),
    }[functional_whisper]


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("model_name", ["openai/whisper-base"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("functional_whisper", [ttnn_optimized_functional_whisper])
def test_performance(reset_seeds, use_program_cache, device, model_name, batch_size, functional_whisper):
    # Run TT Model
    tt_model_name = f"ttnn_{model_name}_optimized"
    config = transformers.WhisperConfig.from_pretrained(model_name)
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(model_name)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(
        [ds[i]["audio"]["array"] for i in range(batch_size)], sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs.input_features
    decoder_input_ids = torch.tensor([[1, 1]] * batch_size) * config.decoder_start_token_id
    model = transformers.WhisperModel.from_pretrained(model_name).eval()
    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=functional_whisper.convert_to_ttnn,
        custom_preprocessor=functional_whisper.custom_preprocessor,
        device=device,
    )

    durations = []
    for _ in range(2):
        (input_embeds, decoder_hidden_states, decoder_attention_mask) = functional_whisper.preprocess_inputs(
            config=config,
            input_features=input_features,
            input_ids=decoder_input_ids,
            attention_mask=None,
            parameters=ttnn_parameters,
            device=device,
            whisper_memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull else ttnn.L1_MEMORY_CONFIG,
        )
        start = time.time()
        last_hidden_state = functional_whisper.whisper(
            config,
            device,
            input_embeds,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            parameters=ttnn_parameters,
            whisper_memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull else ttnn.L1_MEMORY_CONFIG,
        )
        tt_output = ttnn.to_torch(last_hidden_state)
        end = time.time()
        duration = end - start
        durations.append(duration)

    inference_and_compile_time, inference_time, *_ = durations
    expected_compile_time, expected_inference_time = get_expected_times(functional_whisper)

    prep_perf_report(
        model_name=tt_model_name,
        batch_size=batch_size,
        inference_and_compile_time=durations[0],
        inference_time=durations[1],
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )
    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")
