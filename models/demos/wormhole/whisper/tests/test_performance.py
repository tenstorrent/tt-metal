# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import time
import torch
import pytest
import transformers
from loguru import logger
from datasets import load_dataset
from models.perf.perf_utils import prep_perf_report
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.whisper.tt import ttnn_optimized_functional_whisper
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_grayskull, run_for_wormhole_b0


def get_expected_times(functional_whisper):
    return {ttnn_optimized_functional_whisper: (43.84, 10.9)}[functional_whisper]


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["openai/whisper-base"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [500])
@pytest.mark.parametrize("functional_whisper", [ttnn_optimized_functional_whisper])
def test_performance(mesh_device, use_program_cache, model_name, batch_size, sequence_size, functional_whisper):
    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = batch_size * 2 if mesh_device_flag else batch_size

    config = transformers.WhisperConfig.from_pretrained(model_name)
    tt_model_name = f"ttnn_{model_name}_optimized"
    config = transformers.WhisperConfig.from_pretrained(model_name)
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(model_name)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(
        [ds[i]["audio"]["array"] for i in range(batch_size)], sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs.input_features
    decoder_input_ids = torch.tensor([[1, 1]] * batch_size) * config.decoder_start_token_id
    model = transformers.WhisperModel.from_pretrained(model_name)

    attention_mask = None

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        ttnn_parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: model,
            convert_to_ttnn=functional_whisper.convert_to_ttnn,
            custom_preprocessor=functional_whisper.custom_preprocessor,
            device=mesh_device,
        )

    durations = []
    for _ in range(2):
        (input_embeds, decoder_hidden_states, decoder_attention_mask) = functional_whisper.preprocess_inputs(
            config=config,
            input_features=input_features,
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            parameters=ttnn_parameters,
            device=mesh_device,
            whisper_memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull else ttnn.L1_MEMORY_CONFIG,
            weights_mesh_mapper=weights_mesh_mapper,
            inputs_mesh_mapper=inputs_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )

        start = time.time()
        tt_output = functional_whisper.whisper(
            config,
            mesh_device,
            input_embeds,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            parameters=ttnn_parameters,
            whisper_memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull else ttnn.L1_MEMORY_CONFIG,
            output_mesh_composer=output_mesh_composer,
            inputs_mesh_mapper=inputs_mesh_mapper,
        )
        tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)
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


@run_for_wormhole_b0()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test",
    [
        [
            8,
            "silicon_arch_name=wormhole_b0-silicon_arch_wormhole_b0=True-ttnn_model=models.demos.wormhole.whisper.tt.ttnn_optimized_functional_whisper-model_name=openai/whisper-base-batch_size=8",
        ],
    ],
)
def test_perf_device_bare_metal(batch_size, test):
    subdir = "ttnn_whisper"
    num_iterations = 1
    margin = 0.03
    expected_perf = 35.97

    command = (
        f"pytest tests/ttnn/integration_tests/whisper/test_ttnn_optimized_functional_whisper_wh.py::test_ttnn_whisper"
    )
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = batch_size * 2 if mesh_device_flag else batch_size

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=False)
    prep_device_perf_report(
        model_name=f"ttnn_whisper_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
