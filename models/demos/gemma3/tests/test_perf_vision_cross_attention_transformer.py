# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import re

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


def strip_trailing_number(s: str) -> str:
    return re.sub(r"\d+$", "", s)


@pytest.mark.parametrize("device_params", [{"fabric_config": True, "l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("nr_forward_iterations", [2])
def test_perf_gemma_vision(mesh_device, batch_size, nr_forward_iterations, nr_e2e_iterations):
    profiler = BenchmarkProfiler()
    key_model_forward = "model_forward"

    logger.info("Started profiling")
    profiler.start("total_run")
    run_model(
        mesh_device=mesh_device,
        batch_size=batch_size,
        profiler=profiler,
        nr_forward_iterations=nr_forward_iterations,
    )
    profiler.end("total_run")

    for key in keys_without_inference:
        measurements[key] = [profiler.get_duration(key, i) for i in range(nr_e2e_iterations)]

    measurements[key_model_forward + "_inference"] = [
        profiler.get_duration(key_model_forward + "_inference", i) for i in range(nr_forward_iterations - 1)
    ]
    logger.info("Ended profiling")

    measurements_summarised = (
        dict()
    )  # a mean, median, or similar function of all the measurements done for that one specific metric

    for key, val in measurements.items():
        mean = sum(val) / len(val)
        measurements_summarised[key] = mean

    if SAVE_NEW_PERF_TARGETS:
        helper_write_to_json(determine_device_name(mesh_device), measurements_summarised)

    targets = load_targets(
        TARGETS_JSON_FILENAME,
        device_type=determine_device_name(mesh_device),
    )

    for key in measurements_summarised:
        logger.info(f"measurement {key}: {measurements_summarised[key]}")

    upper_threshold = targets["model_forward_inference"] * (1 + THRESHOLD_PERCENT / 100)
    lower_threshold = targets["model_forward_inference"] * (1 - THRESHOLD_PERCENT / 100)
    measured_value = measurements_summarised[key]
    assert lower_threshold < measured_value < upper_threshold


def helper_write_to_json(device_type, measurements, output_filename=None):
    """
    This function is used to help you to faster generate the .json where all the measurements are stored
    """

    if output_filename is None:
        output_filename = "tmp_measurements_" + device_type + ".json"

    # Wrap the measurements dict inside another dict keyed by device_type
    data = {device_type: measurements}

    # Write JSON file with indentation for readability
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)


def run_model(mesh_device, batch_size, profiler, cur_e2e_iteration, nr_forward_iterations):
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    profiler.start("weight_loading", cur_e2e_iteration)
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading", cur_e2e_iteration)

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
    profiler.end("weight_transfer_to_device_and_model_initialization", cur_e2e_iteration)

    ttnn.synchronize_device(mesh_device)
    profiler.start("model_forward_compile")
    test_output = model(input_tensor)
    ttnn.synchronize_device(mesh_device)
    profiler.end("model_forward_compile")

    for cur_inference_iteration in range(nr_forward_iterations):
        ttnn.synchronize_device(mesh_device)
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


def load_targets(filename, device_type):
    if not os.path.exists(filename):
        logger.warning(f"Expected outputs file {filename} does not exist. Skipping loading targets.")
        return []

    with open(filename, "r") as f:
        try:
            targets = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filename}: {e}. Returning empty list.")
            return []

    dict_targets = targets[device_type]

    return dict_targets
