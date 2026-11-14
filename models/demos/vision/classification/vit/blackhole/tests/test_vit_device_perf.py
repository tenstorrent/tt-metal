# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.vision.classification.vit.common.tests.vit_test_infra import create_test_infra
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

try:
    pass

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


# Device kernel throughput at batch 10 by card type prefix. The model's grid width is the largest divisor of
# 12 that fits the card, so P100's 11-wide grid runs on 6 core columns and P150's on 12.
EXPECTED_KERNEL_SAMPLES_PER_SEC = {
    "p100": 3135,
    "p150": 4500,
}


def card_type():
    """The card type the kernel driver reports, e.g. "p100a", or None without one.

    Read from sysfs because asking ttnn would open the cluster in this process and keep it, and the
    profiled run of test_vit_device_ops started below then waits on the device forever.
    """
    paths = sorted(Path("/sys/class/tenstorrent").glob("*/tt_card_type"))
    return paths[0].read_text().strip() if paths else None


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
    card = card_type()
    expected = [
        target for prefix, target in EXPECTED_KERNEL_SAMPLES_PER_SEC.items() if card and card.startswith(prefix)
    ]
    if not expected:
        pytest.skip(f"No device kernel throughput target for card type {card}")
    expected_kernel_samples_per_sec = expected[0]

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
