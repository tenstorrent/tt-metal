# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from models.demos.gemma3.tt.model_config import ModelArgs
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.ccl import TT_CCL


class BenchmarkProfilerWrapper:
    def __init__(self):
        self.profiler_backbone = BenchmarkProfiler()

    def start(self, *args, **kwargs):
        # syncr
        self.profiler_backbone.start(*args, **kwargs)

    def end(self, *args, **kwargs):
        # syncr
        self.profiler_backbone.end(*args, **kwargs)

    def get_duration(self, *args, **kwargs):
        return self.profiler_backbone.get_duration(*args, **kwargs)


# from models.common.utility_functions import (
#     disable_persistent_kernel_cache,
#     enable_persistent_kernel_cache,
# Profiler,
# )


# copied most of pytest parameters from https://github.com/tenstorrent/tt-metal/blob/1566d9ae155c4aba5432f874d375dfbae5d551cd/models/demos/gemma3/tests/test_vision_cross_attention_transformer.py#L30C5-L30C22
@pytest.mark.parametrize(
    "expected_time1TODO, expected_time2TODO",
    (
        (
            69.0,
            420.0,
        ),
    ),
)
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
    reset_seeds,
    bsz,
    expected_time1TODO,
    expected_time2TODO,
):
    # vrv ne koristiti
    # profiler = Profiler()
    # benchmark_data = BenchmarkData()

    # logger.info(f"Start profiler")
    # profiler = BenchmarkProfiler()

    profiler = BenchmarkProfilerWrapper()
    first_key = "first_iter"  # TODO menjaj
    second_key = "second_iter"  # TODO menjaj
    cpu_key = "ref_key"  # TODO menjaj

    nr_iter = 10
    keys = ["total_run", "model_load_and_initialization", "model_forward", "postprocessing_and_transfer"]

    for i in range(nr_iter):
        profiler.start("total_run" + str(i), iteration=i)
        run_model(mesh_device=mesh_device, bsz=bsz, profiler=profiler, iteration=i)
        profiler.end("total_run" + str(i), iteration=i)

    for key in keys:
        measurements = [profiler.get_duration(key + str(i), i) for i in range(nr_iter)]
        mean = sum(measurements) / len(measurements)
        std = sum([(i - mean) ** 2 for i in measurements]) ** (0.5)
        std_as_percent = std / mean * 100
        print("measurements for", key, measurements)
        print("stats for", key, f": {mean=} {std=} {std_as_percent=}%")
        print()
        print()
        print()

    print()

    # prep_perf_report(
    #     "trocr",
    #     BATCH_SIZE,
    #     first_iter_time, #?
    #     second_iter_time, #?
    #     expected_compile_time,
    #     expected_inference_time,
    #     "causal_llm", # TODO menjaj
    #     cpu_time,
    # )
    # # compile_time = first_iter_time - second_iter_time

    # logger.info(f"trocr inference time: {second_iter_time}") # TODO menjaj
    # logger.info(f"trocr compile time: {compile_time}") # TODO menjaj
    # assert second_iter_time < expected_inference_time, "trocr is too slow"
    # assert compile_time < expected_compile_time, "trocr compile time is too slow"

    # zelim ovakav format za assertove: assert (expected - measured) / expected * 0.1


def get_image_features(vision_tower, projector, input_tensor):
    """
    Get image features from the vision tower and projector.
    """
    vision_token = vision_tower(input_tensor).last_hidden_state
    image_features = projector(vision_token)
    return image_features


"""
def xxxx_test():
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    logger.info(f"Start profiler")
    profiler = BenchmarkProfiler()
    profiler.start("run")



    # svidja mi se ovaj pattern, mozda koristiti ovo za inpute mozda ne ali pattern dobar
    logger.info(f"Reading inputs...")
    profiler.start("loading_inputs")
    if len(input_prompts) == 1:  # Manual input
        input_prompts = input_prompts * global_batch_size
    else:  # Inputs from file
        input_prompts = load_inputs(input_prompts, global_batch_size, input_prompts)
    profiler.end("loading_inputs")

"""

# ============================================================


"""
takodje dobar pattern:
profiler.start(f"preprocess_prefill_inputs", iteration=batch_idx)
profiler.end(f"preprocess_prefill_inputs", iteration=batch_idx)



ovo se cepa brda puta
benchmark_data.add_measurement(profiler, 0, step_name, f"ttft_estimate_80l_{galaxy_type}", ttft_estimate_80l)

    Required Parameters:
    - profiler: BenchmarkProfiler instance to get timing data from
    - iteration: int - The benchmark iteration number
    - step_name: str - Name of the profiled step (must exist in profiler)
    - name: str - Unique identifier for this measurement
    - value: float - The measurement value to record

    Optional Parameters:
    - step_warm_up_num_iterations: int - Number of warm-up iterations for the step
    - target: float - Target performance value for comparison
    - device_power: float - Device power consumption during measurement
    - device_temperature: float - Device temperature during measurement

"""


def run_model(mesh_device, bsz, profiler, iteration):
    # copied large part of the function from https://github.com/tenstorrent/tt-metal/blob/1566d9ae155c4aba5432f874d375dfbae5d551cd/models/demos/gemma3/tests/test_vision_cross_attention_transformer.py#L30C5-L30C22

    profiler.start("model_load_and_initialization" + str(iteration), iteration)
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    vision_first_layer_prefix = "model.vision_tower.vision_model."
    vision_partial_state_dict = {
        k[len(vision_first_layer_prefix) :]: v
        for k, v in state_dict.items()
        if (k.startswith(vision_first_layer_prefix))
    }

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

    profiler.end("model_load_and_initialization" + str(iteration), iteration)

    profiler.start("model_forward" + str(iteration), iteration)
    test_output = test_gemma_vision(input_tensor)
    profiler.end("model_forward" + str(iteration), iteration)

    profiler.start("postprocessing_and_transfer" + str(iteration), iteration)
    out = ttnn.from_device(test_output)

    tt_output_torch = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[0, :, :, :]
    profiler.end("postprocessing_and_transfer" + str(iteration), iteration)

    return tt_output_torch
