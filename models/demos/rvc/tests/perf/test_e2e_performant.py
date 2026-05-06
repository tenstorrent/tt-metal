# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.rvc.runner.performant_runner import RVCInferenceConfig, RVCRunner


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
    input_duration_sec = 11.0
    inference_config = RVCInferenceConfig(num_secs=input_duration_sec)
    batch_size = device.get_num_devices()

    model.initialize_inference(
        device,
        {"inference": inference_config},
        batch_size=batch_size,
        validation=False,
        performance_runner=True,
    )
    torch_input_tensor = torch.randn(
        (batch_size, int(input_duration_sec * model.ttnn_pipeline.sr)), dtype=torch.float32
    )
    inference_iter_count = 3

    t0 = time.time()
    output = None
    for _ in range(inference_iter_count):
        output = model.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()

    inference_time_avg = round((t1 - t0) / inference_iter_count, 6)
    input_duration_sec = torch_input_tensor.shape[1] / model.ttnn_pipeline.sr
    output_duration_sec = output.shape[1] / model.ttnn_pipeline.tgt_sr
    output_batch_size = output.shape[0]
    aggregate_output_duration_sec = output_batch_size * output_duration_sec
    rtf = inference_time_avg / aggregate_output_duration_sec if aggregate_output_duration_sec > 0 else float("inf")
    logger.info(
        "ttnn_rvc. "
        f"One inference iteration time (sec): {inference_time_avg}, "
        f"input_duration_sec: {input_duration_sec:.6f}, "
        f"output_duration_sec: {output_duration_sec:.6f}, "
        f"aggregate_output_duration_sec: {aggregate_output_duration_sec:.6f}, "
        f"rtf: {rtf:.6f}, "
        f"batch_size: {output_batch_size}, "
        f"num_input_samples: {torch_input_tensor.shape[1]}, "
        f"output_shape: {tuple(output.shape)}"
    )


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 64192, "trace_region_size": 15079936, "num_command_queues": 2}], indirect=True
)
def test_rvc_e2e(device):
    run_rvc_e2e_inference(device)


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 64192, "trace_region_size": 15079936, "num_command_queues": 2}], indirect=True
)
def test_rvc_e2e_dp(mesh_device):
    run_rvc_e2e_inference(mesh_device)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 64192}], indirect=True)
def test_rvc_e2e_no_sync(device):
    _require_assets_env()
    model = RVCRunner()
    input_duration_sec = 11.0
    inference_config = RVCInferenceConfig(num_secs=input_duration_sec)
    batch_size = device.get_num_devices()

    model.initialize_inference(
        device,
        {"inference": inference_config},
        batch_size=batch_size,
        validation=False,
        # performance_runner=True,
    )
    torch_input_tensor = torch.randn(
        (batch_size, int(input_duration_sec * model.ttnn_pipeline.sr)), dtype=torch.float32
    )

    output = model.run(torch_input_tensor)
