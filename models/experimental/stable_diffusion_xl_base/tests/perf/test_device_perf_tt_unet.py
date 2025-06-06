# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from models.utility_functions import is_wormhole_b0
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "device_time",
    ((549526000),),
)
def test_stable_diffusion_device_perf(device_time):
    subdir = "ttnn_stable_diffusion_xl_base"
    margin = 0.03
    batch = 1
    iterations = 1
    command = f"pytest models/experimental/stable_diffusion_xl_base/tests/pcc/test_module_tt_unet.py --profiler"
    cols = ["DEVICE KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    expected_perf_cols = {inference_time_key: device_time}

    if is_wormhole_b0():
        wh_arch_yaml_backup = None
        if "WH_ARCH_YAML" in os.environ:
            wh_arch_yaml_backup = os.environ["WH_ARCH_YAML"]
        os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

    post_processed_results = run_device_perf(command, subdir, iterations, cols, batch, has_signposts=True)
    print(post_processed_results)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"stable_diffusion_xl_base_{batch}batch",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

    if is_wormhole_b0():
        if wh_arch_yaml_backup is not None:
            os.environ["WH_ARCH_YAML"] = wh_arch_yaml_backup
        else:
            del os.environ["WH_ARCH_YAML"]
