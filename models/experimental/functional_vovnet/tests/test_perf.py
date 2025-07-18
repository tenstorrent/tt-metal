# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import time
import timm
import torch
import pytest
import torch.nn.functional as F
from loguru import logger
from models.perf.perf_utils import prep_perf_report
from models.experimental.functional_vovnet.tt.vovnet import TtVoVNet
from models.utility_functions import run_for_wormhole_b0
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.experimental.functional_vovnet.tt.model_preprocessing import custom_preprocessor


def get_expected_times(name):
    base = {"vovnet": (77.95, 0.035)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vovnet(device):
    disable_persistent_kernel_cache()
    torch_input = torch.rand(1, 3, 224, 224)
    core_grid = ttnn.CoreGrid(y=8, x=8)
    n, c, h, w = torch_input.shape
    num_cores = core_grid.x * core_grid.y
    shard_h = (n * w * h + num_cores - 1) // num_cores
    grid_size = core_grid
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    torch_input = torch_input.permute(0, 2, 3, 1)
    torch_input = torch_input.reshape(1, 1, h * w * n, c)
    min_channels = 16
    if c < min_channels:
        padding_c = min_channels - c
    torch_input = F.pad(torch_input, (0, padding_c), "constant", 0)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_input = tt_input.to(device, input_mem_config)

    batch_size = torch_input.shape[0]
    torch_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True).eval()

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    ttnn_model = TtVoVNet(device=device, parameters=parameters, base_address="")

    durations = []

    for i in range(2):
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_input = tt_input.to(device, input_mem_config)
        start = time.time()
        ttnn_model_output = ttnn_model.forward(tt_input)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("vovnet")

    prep_perf_report(
        model_name="models/experimental/functional_vovnet",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 123],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_vovnet(batch_size, expected_perf):
    subdir = "ttnn_vovnet"
    num_iterations = 1
    margin = 0.03

    command = f"pytest tests/ttnn/integration_tests/vovnet/test_tt_vovnet.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=False)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_vovnet{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
