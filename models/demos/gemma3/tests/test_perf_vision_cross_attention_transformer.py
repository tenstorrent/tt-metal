# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from models.demos.gemma3.tt.model_config import ModelArgs, determine_device_name
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.ccl import TT_CCL

THRESHOLD_PERCENT = 5

SAVE_NEW_PERF_TARGETS = False
TARGETS_JSON_FILENAME = (
    "models/demos/gemma3/tests/perf_targets/targets_test_perf_vision_cross_attention_transformer.json"
)


@pytest.mark.parametrize("device_params", [{"fabric_config": True, "l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("nr_forward_iterations", [15])
def test_perf_gemma_vision(mesh_device, batch_size, nr_forward_iterations):
    profiler = BenchmarkProfiler()

    logger.info("Started profiling")
    profiler.start("total_run")
    run_model(
        mesh_device=mesh_device,
        batch_size=batch_size,
        profiler=profiler,
        nr_forward_iterations=nr_forward_iterations,
    )
    profiler.end("total_run")
    logger.info("Ended profiling")

    inference_measurements = [profiler.get_duration("model_forward_inference", i) for i in range(nr_forward_iterations)]
    inference_mean = sum(inference_measurements) / len(inference_measurements)

    measurement_keys = {k for _, k in profiler.start_times.keys()}

    measurements = dict()
    for k in measurement_keys:
        measurements[k] = profiler.get_duration(k) if k != "model_forward_inference" else inference_mean
        logger.info(f"measurement {k}: {measurements[k]}")

    model_name = get_model_name()

    targets = load_targets(TARGETS_JSON_FILENAME, device_type=determine_device_name(mesh_device), model_name=model_name)

    if SAVE_NEW_PERF_TARGETS:
        helper_write_to_json(
            determine_device_name(mesh_device),
            measurements["model_forward_inference"],
            output_filename=TARGETS_JSON_FILENAME,
            model_name=model_name,
        )

    upper_threshold = targets["model_forward_inference"] * (1 + THRESHOLD_PERCENT / 100)
    lower_threshold = targets["model_forward_inference"] * (1 - THRESHOLD_PERCENT / 100)
    assert lower_threshold < inference_mean, "Failed because it's too fast"
    assert inference_mean < upper_threshold, "Failed because it's too slow"


def helper_write_to_json(device_type, measurements, output_filename, model_name):
    """
    This function reads the file /output_filename/ and updates it with the new measurements. For example if the file has measurements for N150 it will overwrite them with the new measurements.
    """

    with open(output_filename, "r") as f:
        file_dict = json.load(f)

    if file_dict.get(model_name) is None:
        file_dict[model_name] = dict()
    if file_dict[model_name].get(device_type) is None:
        file_dict[model_name][device_type] = dict()

    file_dict[model_name][device_type] = {"model_forward_inference": measurements}

    with open(output_filename, "w") as f:
        json.dump(file_dict, f, indent=4)


def run_model(mesh_device, batch_size, profiler, nr_forward_iterations):
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading")

    image_size = model_args.vision_chunk_size
    in_channels = model_args.vision_in_channels

    input_tensor = torch.rand((batch_size, in_channels, image_size, image_size))

    profiler.start("weight_transfer_to_device_and_model_initialization")
    model = TtGemmaTransformerVision(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix="model.vision_tower.vision_model.",
        dtype=dtype,
        configuration=model_args,
        return_intermediate=False,
    )
    profiler.end("weight_transfer_to_device_and_model_initialization")

    ttnn.synchronize_device(mesh_device)
    profiler.start("model_forward_compile")
    test_output = model(input_tensor)
    ttnn.synchronize_device(mesh_device)
    profiler.end("model_forward_compile")

    for cur_inference_iteration in range(nr_forward_iterations):
        profiler.start("model_forward_inference", cur_inference_iteration)
        test_output = model(input_tensor)
        ttnn.synchronize_device(mesh_device)
        profiler.end("model_forward_inference", cur_inference_iteration)

    profiler.start("postprocessing_and_transfer")
    out = ttnn.from_device(test_output)

    tt_output_torch = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[0, :, :, :]
    profiler.end("postprocessing_and_transfer")

    return tt_output_torch


<<<<<<< HEAD
def load_targets(filename, device_type, model_name):
=======
def load_targets(filename, device_type):
>>>>>>> cb25b73173 (Mstojko/test vision cross attention transformer benchmark (#29699))
    if not os.path.exists(filename):
        logger.warning(f"Expected outputs file {filename} does not exist. Skipping loading targets.")
        return []

    with open(filename, "r") as f:
        try:
            targets = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filename}: {e}. Returning empty list.")
            return []

<<<<<<< HEAD
    dict_targets = targets[model_name][device_type]

    return dict_targets


def get_model_name():
    full_name = os.getenv("HF_MODEL")

    if "gemma" not in full_name:
        raise ValueError(f"Unsupported model name: {full_name}")

    if "4b" in full_name:
        return "gemma-3-4b-it"
    elif "27b" in full_name:
        return "gemma-3-27b-it"
    else:
        raise ValueError(f"Unsupported model name: {full_name}")
=======
    dict_targets = targets[device_type]

    return dict_targets
>>>>>>> cb25b73173 (Mstojko/test vision cross attention transformer benchmark (#29699))
