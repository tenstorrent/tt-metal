# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from models.demos.gemma3.tt.model_config import ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL


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
def test_gemma_vision(
    mesh_device,
    reset_seeds,
    bsz,
):
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

    test_output = test_gemma_vision(input_tensor)

    out = ttnn.from_device(test_output)

    tt_output_torch = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[0, :, :, :]

    # TODO - neki assert
    print("adf")
    # assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"


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
