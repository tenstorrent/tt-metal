# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from models.demos.gemma3.tt.model_config import ModelArgs
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.ccl import TT_CCL

NR_ITER_E2E = 1
NR_FORWARD_ITERATIONS = 15
THRESHOLD_PERCENT = 0.1
TEST_FORWARD_INFERENCE_ONLY = True

DEBUG_SAVE_AND_PRINT = True


class BenchmarkProfilerWrapper:
    def __init__(self, device):
        self.device = device
        self.profiler_backbone = BenchmarkProfiler()

    def start(self, *args, **kwargs):
        ttnn.synchronize_device(self.device)
        self.profiler_backbone.start(*args, **kwargs)

    def end(self, *args, **kwargs):
        ttnn.synchronize_device(self.device)
        self.profiler_backbone.end(*args, **kwargs)

    def get_duration(self, *args, **kwargs):
        return self.profiler_backbone.get_duration(*args, **kwargs)


def set_hf_model_env():
    assert os.environ.get("HF_MODEL") is None, "This test will set it depending on the device being run"
    os_mesh_device = os.environ["MESH_DEVICE"]
    if os_mesh_device in ["N150", "N300"]:
        os.environ["HF_MODEL"] = "google/gemma-3-4b-it"
    elif os_mesh_device == "T3K":
        os.environ["HF_MODEL"] = "google/gemma-3-27b-it"
    else:
        assert False, "Unknown model"


# copied most of pytest parameters from https://github.com/tenstorrent/tt-metal/blob/1566d9ae155c4aba5432f874d375dfbae5d551cd/models/demos/gemma3/tests/test_vision_cross_attention_transformer.py#L30C5-L30C22
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"fabric_config": True, "l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("bsz", [1])
def test_perf_gemma_vision(
    mesh_device,
    bsz,
):
    set_hf_model_env()

    profiler = BenchmarkProfilerWrapper(device=mesh_device)

    keys_e2e = ["total_run", "model_load_and_initialization", "postprocessing_and_transfer"]
    key_model_forward = "model_forward"

    logger.info("Started profiling")
    for cur_e2e_iteration in range(NR_ITER_E2E):
        if cur_e2e_iteration == 0:
            nr_forward_iterations_actual = NR_FORWARD_ITERATIONS
        else:
            nr_forward_iterations_actual = 1

        profiler.start("total_run" + str(cur_e2e_iteration), iteration=cur_e2e_iteration)
        run_model(
            mesh_device=mesh_device,
            bsz=bsz,
            profiler=profiler,
            cur_e2e_iteration=cur_e2e_iteration,
            nr_forward_iterations=nr_forward_iterations_actual,
            key_model_forward=key_model_forward,
        )
        profiler.end("total_run" + str(cur_e2e_iteration), iteration=cur_e2e_iteration)

    measurements = dict()
    for key in keys_e2e + [key_model_forward + "_compile"]:
        measurements[key] = [profiler.get_duration(key + str(i), i) for i in range(NR_ITER_E2E)]
    measurements[key_model_forward + "_inference"] = [
        profiler.get_duration(key_model_forward + "_inference" + str(i), i) for i in range(NR_FORWARD_ITERATIONS - 1)
    ]
    logger.info("Ended profiling")

    measurements_summarised = (
        dict()
    )  # a mean, median, or similar function of all the measurements done for that one specific metric
    for key, val in measurements.items():
        mean = sum(val) / len(val)
        measurements_summarised[key] = mean

        # The idea is to look only at larger ones as they are only relevant for upper threshold. Also we square them so outliers have more impact
        # We can perhaps change this threshold at load time when the test is being run (eg. multiply it by 2)
        values_above = [p for p in val if p >= mean]
        average_squared_above = sum([p**2 for p in values_above]) / len(values_above)
        measurements_summarised[key + "_threshold"] = average_squared_above ** (0.5)

    if DEBUG_SAVE_AND_PRINT:
        for key, val in measurements.items():
            mean = measurements_summarised[key]
            std = (sum([(i - mean) ** 2 for i in val]) / len(val)) ** (0.5)
            std_as_percent = std / mean * 100
            print("measurements for", key, val)
            print("stats for", key, f": {mean=} {std=} {std_as_percent=}%")
            print()
            print()
            print()

        helper_write_to_json(os.environ.get("MESH_DEVICE"), measurements_summarised)

    targets = load_targets(
        "models/demos/gemma3/tests/targets_test_benchmark_vision_cross_attention_transformer.json",
        os.environ.get("MESH_DEVICE"),
    )

    if TEST_FORWARD_INFERENCE_ONLY:
        metric_keys_to_test = ["model_forward_inference"]
    else:
        metric_keys_to_test = [k for k in targets.keys() if not k.endswith("_threshold")]

    for key in metric_keys_to_test:
        threshold = targets[key] * (1 + THRESHOLD_PERCENT / 100)
        measured_value = measurements_summarised[key]
        assert measured_value < threshold


def get_image_features(vision_tower, projector, input_tensor):
    """
    Get image features from the vision tower and projector.
    """
    vision_token = vision_tower(input_tensor).last_hidden_state
    image_features = projector(vision_token)
    return image_features


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


def run_model(mesh_device, bsz, profiler, cur_e2e_iteration, nr_forward_iterations, key_model_forward):
    # copied large part of the function from https://github.com/tenstorrent/tt-metal/blob/1566d9ae155c4aba5432f874d375dfbae5d551cd/models/demos/gemma3/tests/test_vision_cross_attention_transformer.py#L30C5-L30C22

    profiler.start("model_load_and_initialization" + str(cur_e2e_iteration), cur_e2e_iteration)
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    image_size = model_args.vision_chunk_size
    in_channels = model_args.vision_in_channels

    input_tensor = torch.rand((bsz, in_channels, image_size, image_size))

    test_gemma_vision = TtGemmaTransformerVision(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix="model.vision_tower.vision_model.",
        dtype=dtype,
        configuration=model_args,
        return_intermediate=False,
    )

    profiler.end("model_load_and_initialization" + str(cur_e2e_iteration), cur_e2e_iteration)

    for cur_forward_iteration in range(nr_forward_iterations):
        if cur_forward_iteration == 0:
            str_for_profiler = key_model_forward + "_compile" + str(cur_e2e_iteration)
            index_for_profiler = cur_e2e_iteration
        else:
            str_for_profiler = (
                key_model_forward + "_inference" + str(cur_forward_iteration - 1)
            )  # i-1 instead of i because we want to be able to iterate from 0 so its more clean
            index_for_profiler = cur_forward_iteration - 1

        profiler.start(str_for_profiler, index_for_profiler)
        test_output = test_gemma_vision(input_tensor)
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

    dict_targets = targets["targets"][device_type]

    return dict_targets
