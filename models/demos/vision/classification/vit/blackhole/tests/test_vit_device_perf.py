# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.vision.classification.vit.common.tests.vit_helper_funcs import card_target, device_grid_width
from models.demos.vision.classification.vit.common.tests.vit_test_infra import create_test_infra
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

try:
    pass

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


# Device kernel throughput at batch 10 by card type prefix and device grid width. The model's grid width is
# the largest divisor of 12 that fits the device grid, so an 11-wide grid (P100, or P150 with two Tensix
# columns disabled) runs on 6 core columns and a 12- or 13-wide one on 12.
EXPECTED_KERNEL_SAMPLES_PER_SEC = {
    ("p100", 11): 3135,
    ("p150", 11): 3135,
    ("p150", 12): 4500,
    ("p150", 13): 4500,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [10])
def test_vit_device_ops(
    device,
    batch_size,
):
    torch.manual_seed(0)

    test_infra = create_test_infra(device, batch_size, use_random_input_tensor=True)

    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)

    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res)

    # include initial reshard in device perf test
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    output_tensor = test_infra.run()

    # include final s2i in device perf test
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)

    ttnn.synchronize_device(device)


@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.models_device_performance_bare_metal
# Three profiled runs of the model take close to the default 300 s.
@pytest.mark.timeout(600)
def test_vit_perf_device(batch_size):
    grid_x = device_grid_width()
    expected_kernel_samples_per_sec = card_target(EXPECTED_KERNEL_SAMPLES_PER_SEC, grid_x)
    if expected_kernel_samples_per_sec is None:
        pytest.skip(f"No device kernel throughput target for this card on a {grid_x}-wide grid")

    command = f"pytest models/demos/vision/classification/vit/blackhole/tests/test_vit_device_perf.py::test_vit_device_ops[{batch_size}-device_params0]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    post_processed_results = run_device_perf(command, subdir="vit", num_iterations=3, cols=cols, batch_size=batch_size)

    expected_results = check_device_perf(
        post_processed_results,
        margin=0.03,
        expected_perf_cols={inference_time_key: expected_kernel_samples_per_sec},
        assert_on_fail=True,
    )
    prep_device_perf_report(
        model_name=f"vit-{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
