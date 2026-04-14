# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.rvc.runner.performant_runner import RVCRunner
from models.demos.rvc.runner.performant_runner_infra import RVCInferenceConfig


def _require_assets_env() -> None:
    for env_key in ("RVC_CONFIGS_DIR", "RVC_ASSETS_DIR"):
        env_val = os.getenv(env_key)
        if not env_val:
            pytest.skip(f"{env_key} is not set; perf test requires model/config assets.")
        if not Path(env_val).exists():
            pytest.skip(f"{env_key} path does not exist: {env_val}")


def run_rvc_e2e_inference(device) -> None:
    _require_assets_env()
    model = RVCRunner()
    inference_config = RVCInferenceConfig(num_secs=3.0)

    model.initialize_inference(device, {"inference": inference_config})
    torch_input_tensor, _ = model.test_infra.setup_l1_sharded_input(device)
    inference_iter_count = 10

    t0 = time.time()
    for _ in range(inference_iter_count):
        _ = model.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    model.release_inference()

    inference_time_avg = round((t1 - t0) / inference_iter_count, 6)
    logger.info(
        f"ttnn_rvc. One inference iteration time (sec): {inference_time_avg}, num_input_samples: {torch_input_tensor.shape[0]}"
    )


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65384}], indirect=True)
def test_rvc_e2e(device):
    run_rvc_e2e_inference(device)
