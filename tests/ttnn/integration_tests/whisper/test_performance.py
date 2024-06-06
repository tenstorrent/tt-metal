# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.functional_whisper.tt import ttnn_functional_whisper, ttnn_optimized_functional_whisper
from transformers import AutoFeatureExtractor, WhisperModel, WhisperConfig
from datasets import load_dataset
import torch
from ttnn.model_preprocessing import preprocess_model_parameters
from loguru import logger
from models.utility_functions import skip_for_wormhole_b0
from models.perf.perf_utils import prep_perf_report
import time
import ttnn


def get_expected_times(functional_whisper):
    return {
        ttnn_functional_whisper: (11, 4.16),
        ttnn_optimized_functional_whisper: (1.2, 1.35),
    }[functional_whisper]


@skip_for_wormhole_b0(reason_str="Not tested on single WH")
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["openai/whisper-base"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [500])
@pytest.mark.parametrize("functional_whisper", [ttnn_functional_whisper, ttnn_optimized_functional_whisper])
def test_performance(device, use_program_cache, model_name, batch_size, sequence_size, functional_whisper):
    config = WhisperConfig.from_pretrained(model_name)

    # Run TT Model
    if functional_whisper == ttnn_functional_whisper:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_whisper == ttnn_optimized_functional_whisper:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_t5: {functional_whisper}")

    config = WhisperConfig.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id

    attention_mask = None

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: WhisperModel.from_pretrained(model_name).eval(),
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
            attention_mask=attention_mask,
            parameters=parameters,
            device=device,
        )

        start = time.time()
        tt_output = functional_whisper.whisper(
            config,
            input_embeds,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            parameters=parameters,
        )
        tt_output = ttnn.to_torch(tt_output)
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
