# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from datasets import load_dataset
from loguru import logger
from transformers import AutoFeatureExtractor, WhisperConfig, WhisperModel
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.whisper.tt import ttnn_optimized_functional_whisper
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import WHISPER_L1_SMALL_SIZE, init_kv_cache
from models.perf.perf_utils import prep_perf_report


def get_expected_times(model_name):
    """
    Returns expected compile time and inference time.
    """
    return {
        "openai/whisper-base": (18.0, 0.039),
        "distil-whisper/distil-large-v3": (15.3, 0.236),
    }[model_name]


def run_performance(
    device,
    model_name,
    device_batch_size,
    decoder_sequence_size,
    use_kv_cache,
    functional_whisper,
):
    batch_size = device.get_num_devices() * device_batch_size
    config = WhisperConfig.from_pretrained(model_name)
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
    # Run TT Model
    if functional_whisper == ttnn_optimized_functional_whisper:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_whisper: {functional_whisper}")

    config = WhisperConfig.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    decoder_input_ids = torch.ones(1, decoder_sequence_size).type(torch.int32) * config.decoder_start_token_id
    input_features = input_features.repeat(batch_size, 1, 1)
    decoder_input_ids = decoder_input_ids.repeat(batch_size, 1)
    attention_mask = None

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: WhisperModel.from_pretrained(model_name).eval(),
        convert_to_ttnn=functional_whisper.convert_to_ttnn,
        custom_preprocessor=functional_whisper.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    if use_kv_cache:
        kv_cache = init_kv_cache(
            config, device, max_batch_size=device_batch_size, max_seq_len=512, weights_mesh_mapper=weights_mesh_mapper
        )
        current_decode_pos = ttnn.from_torch(
            torch.zeros(batch_size), device=device, dtype=ttnn.int32, mesh_mapper=inputs_mesh_mapper
        )

    durations = []
    for _ in range(10):
        (input_embeds, decoder_hidden_states, decoder_attention_mask) = functional_whisper.preprocess_inputs(
            config=config,
            input_features=input_features,
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            parameters=parameters,
            device=device,
            inputs_mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
        )

        start = time.time()
        tt_output = functional_whisper.whisper(
            config,
            input_embeds,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            kv_cache=kv_cache if use_kv_cache else None,
            current_decode_pos=current_decode_pos if use_kv_cache else None,
            parameters=parameters,
        )
        end = time.time()
        tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)

        duration = end - start
        durations.append(duration)

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(model_name)
    prep_perf_report(
        model_name=tt_model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time (encoder + decoder): {inference_time}")
    logger.info(
        f"ttnn_whisper_v3_batch_size: {batch_size}, One inference iteration time (sec): {inference_time}, frames per sec: {round(batch_size /inference_time)}"
    )


@skip_for_grayskull()
@pytest.mark.parametrize("model_name", ["openai/whisper-base", "distil-whisper/distil-large-v3"])
@pytest.mark.parametrize("device_batch_size", [1])
@pytest.mark.parametrize("decoder_sequence_size", [1])
@pytest.mark.parametrize("use_kv_cache", [True])
@pytest.mark.parametrize("functional_whisper", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_whisper_v3_e2e_performance(
    device, model_name, device_batch_size, decoder_sequence_size, use_kv_cache, functional_whisper
):
    return run_performance(
        device, model_name, device_batch_size, decoder_sequence_size, use_kv_cache, functional_whisper
    )


@skip_for_grayskull()
@pytest.mark.parametrize("model_name", ["openai/whisper-base", "distil-whisper/distil-large-v3"])
@pytest.mark.parametrize("device_batch_size", [1])
@pytest.mark.parametrize("decoder_sequence_size", [1])
@pytest.mark.parametrize("use_kv_cache", [True])
@pytest.mark.parametrize("functional_whisper", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_whisper_v3_e2e_performance_dp(
    mesh_device, model_name, device_batch_size, decoder_sequence_size, use_kv_cache, functional_whisper
):
    return run_performance(
        mesh_device, model_name, device_batch_size, decoder_sequence_size, use_kv_cache, functional_whisper
    )
