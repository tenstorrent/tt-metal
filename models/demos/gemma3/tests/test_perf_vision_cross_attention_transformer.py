# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import re

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from models.demos.gemma3.tt.model_config import ModelArgs, determine_device_name
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.ccl import TT_CCL

THRESHOLD_PERCENT = 5

SAVE_NEW_PERF_TARGETS_AND_DEBUG_PRINT = False


def strip_trailing_number(s: str) -> str:
    return re.sub(r"\d+$", "", s)


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"fabric_config": True, "l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("nr_forward_iterations", [15])
@pytest.mark.parametrize("nr_e2e_iterations", [1])
def test_perf_gemma_vision(mesh_device, batch_size, nr_forward_iterations, nr_e2e_iterations):
    profiler = BenchmarkProfiler()
    key_model_forward = "model_forward"

    logger.info("Started profiling")
    for cur_e2e_iteration in range(nr_e2e_iterations):
        if cur_e2e_iteration == 0:
            # // cache = skontaj gde god se koristi nr_forward_iterations i onaj all caps i svi ti da li je tacno to taj koji treba da se koristi, takodje dodati u parametre ovaj e2e shit
            nr_forward_iterations_actual = nr_forward_iterations
        else:
            nr_forward_iterations_actual = 1

        profiler.start("total_run" + str(cur_e2e_iteration), iteration=cur_e2e_iteration)
        run_model(
            mesh_device=mesh_device,
            batch_size=batch_size,
            profiler=profiler,
            cur_e2e_iteration=cur_e2e_iteration,
            nr_forward_iterations=nr_forward_iterations_actual,
            key_model_forward=key_model_forward,
        )
        profiler.end("total_run" + str(cur_e2e_iteration), iteration=cur_e2e_iteration)

    measurements = dict()
    keys_without_inference_with_nr = set(
        [k for _, k in profiler.start_times.keys() if not k.startswith(key_model_forward)]
    )
    keys_without_inference = set(strip_trailing_number(k) for k in keys_without_inference_with_nr)

    for key in keys_without_inference:
        measurements[key] = [profiler.get_duration(key + str(i), i) for i in range(nr_e2e_iterations)]

    measurements[key_model_forward + "_inference"] = [
        profiler.get_duration(key_model_forward + "_inference" + str(i), i) for i in range(nr_forward_iterations - 1)
    ]
    logger.info("Ended profiling")

    measurements_summarised = (
        dict()
    )  # a mean, median, or similar function of all the measurements done for that one specific metric

    for key, val in measurements.items():
        mean = sum(val) / len(val)
        measurements_summarised[key] = mean

    if SAVE_NEW_PERF_TARGETS_AND_DEBUG_PRINT:
        for key, val in measurements.items():
            mean = measurements_summarised[key]
            std = (sum([(i - mean) ** 2 for i in val]) / len(val)) ** (0.5)
            std_as_percent = std / mean * 100
            print("measurements for", key, val)
            print("stats for", key, f": {mean=} {std=} {std_as_percent=}%")
            print()
            print()
            print()

        helper_write_to_json(determine_device_name(mesh_device), measurements_summarised)

    targets = load_targets(
        "models/demos/gemma3/tests/perf_targets/targets_test_perf_vision_cross_attention_transformer.json",
        device_type=determine_device_name(mesh_device),
    )

    for key in measurements_summarised:
        logger.info(f"measurement {key}: {measurements_summarised[key]}")

    threshold = targets["model_forward_inference"] * (1 + THRESHOLD_PERCENT / 100)
    measured_value = measurements_summarised[key]
    assert measured_value < threshold


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


def run_model(mesh_device, batch_size, profiler, cur_e2e_iteration, nr_forward_iterations, key_model_forward):
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    profiler.start("weight_loading" + str(cur_e2e_iteration), cur_e2e_iteration)
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading" + str(cur_e2e_iteration), cur_e2e_iteration)

    image_size = model_args.vision_chunk_size
    in_channels = model_args.vision_in_channels

    input_tensor = torch.rand((batch_size, in_channels, image_size, image_size))

    profiler.start("weight_transfer_to_device_and_model_initialization" + str(cur_e2e_iteration), cur_e2e_iteration)
    model = TtGemmaTransformerVision(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix="model.vision_tower.vision_model.",
        dtype=dtype,
        configuration=model_args,
        return_intermediate=False,
    )
    profiler.end("weight_transfer_to_device_and_model_initialization" + str(cur_e2e_iteration), cur_e2e_iteration)

    for cur_forward_iteration in range(nr_forward_iterations):
        if cur_forward_iteration == 0:
            str_for_profiler = key_model_forward + "_compile" + str(cur_e2e_iteration)
            index_for_profiler = cur_e2e_iteration
        else:
            str_for_profiler = (
                key_model_forward + "_inference" + str(cur_forward_iteration - 1)
            )  # i-1 instead of i because we want to be able to iterate from 0 so its more clean
            index_for_profiler = cur_forward_iteration - 1

        ttnn.synchronize_device(mesh_device)
        profiler.start(str_for_profiler, index_for_profiler)

        test_output = model(input_tensor)

        ttnn.synchronize_device(mesh_device)
        profiler.end(str_for_profiler, index_for_profiler)

    profiler.start("postprocessing_and_transfer" + str(cur_e2e_iteration), cur_e2e_iteration)
    out = ttnn.from_device(test_output)

    tt_output_torch = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[0, :, :, :]
    profiler.end("postprocessing_and_transfer" + str(cur_e2e_iteration), cur_e2e_iteration)

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
