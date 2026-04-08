# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Logged batch_size = batch_size_per_device * device.get_num_devices().
# - Single-device ``device`` fixture: batch > 16 typically OOMs at L1 trace capture for this path (640² YOLOv8s).
# - Global 32 total: use mesh DP tests with batch_size_per_device=1 and 32 devices (not batch 32 on 1 chip).

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov8s.common import YOLOV8S_L1_SMALL_SIZE
from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

# Before from_host_shards matched mesh size, batch 32 on 1 device failed earlier; fixing shards exposes L1 OOM here.
_SINGLE_DEVICE_MAX_BATCH_YOLOV8S_PERFORMANT = 16


def _skip_if_single_device_batch_exceeds_l1(device, batch_size_per_device: int) -> None:
    if device.get_num_devices() != 1:
        return
    if batch_size_per_device > _SINGLE_DEVICE_MAX_BATCH_YOLOV8S_PERFORMANT:
        pytest.skip(
            f"Single-device batch {batch_size_per_device} exceeds typical L1 budget for YOLOv8s trace+2CQ @640 "
            f"(>{_SINGLE_DEVICE_MAX_BATCH_YOLOV8S_PERFORMANT} → OOM in to_memory_config during trace capture). "
            "For global batch 32 run mesh DP with batch_size_per_device=1 × 32 devices."
        )


def run_yolov8s(
    device,
    batch_size_per_device,
    model_location_generator,
    *,
    split_host_device_timing: bool = False,
):
    """
    Default (split_host_device_timing=False): 10× performant_runner.run(); one log line — full per-iter
    time (host prep + H2D + reshard + trace + sync each call), same semantics as yolo_dp_mesh_infer.

    If split_host_device_timing=True: three log lines —
      (1) avg push+trace per iter (host prep via prepare_host_input is done once, not each iter),
      (2) FPS_est_run ≈ prep_once + (1) to match run() cost per iteration,
      (3) H2D vs reshard+trace+sync and FPS_compute_only (H2D excluded from that timer).
    """
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    performant_runner = YOLOv8sPerformantRunner(
        device,
        batch_size,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
        model_location_generator=model_location_generator,
    )

    input_shape = (batch_size, 3, 640, 640)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    if use_signpost:
        signpost(header="start")

    if split_host_device_timing:
        t_prep0 = time.perf_counter()
        tt_host = performant_runner.prepare_host_input(torch_input_tensor)
        prep_once_sec = time.perf_counter() - t_prep0
        h2d_secs: list[float] = []
        compute_secs: list[float] = []
        for _ in range(10):
            t_h0 = time.perf_counter()
            performant_runner.push_host_input_to_device_dram(tt_host)
            t_h1 = time.perf_counter()
            performant_runner.execute_reshard_and_trace()
            t_c1 = time.perf_counter()
            h2d_secs.append(t_h1 - t_h0)
            compute_secs.append(t_c1 - t_h1)
        push_trace_avg = round(sum(h2d_secs + compute_secs) / 10, 6)
        inference_time_avg = push_trace_avg
        compute_only_avg = round(sum(compute_secs) / 10, 6)
        h2d_only_avg = round(sum(h2d_secs) / 10, 6)
        # performant_runner.run() calls prepare_host_input every iteration; we only prepare once.
        full_run_equivalent_avg = round(prep_once_sec + push_trace_avg, 6)
    else:
        t0 = time.perf_counter()
        for _ in range(10):
            _ = performant_runner.run(torch_input_tensor)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        inference_time_avg = round((t1 - t0) / 10, 6)
        compute_only_avg = None
        h2d_only_avg = None

    if use_signpost:
        signpost(header="stop")

    performant_runner.release()
    if split_host_device_timing and compute_only_avg is not None:
        logger.info(
            f"Model: ttnn_yolov8s - batch_size: {batch_size}. "
            f"Avg push+trace per iter (sec): {inference_time_avg}, FPS: {round(batch_size / inference_time_avg)} "
            f"— host prep (_setup_l1_sharded_input) done once before loop, not each iter (unlike run())."
        )
        logger.info(
            f"Model: ttnn_yolov8s - batch_size: {batch_size}. "
            f"Estimated full run() per iter (prep_once + avg push+trace): {full_run_equivalent_avg} sec, "
            f"FPS_est_run: {round(batch_size / full_run_equivalent_avg)} "
            f"(prep_once={round(prep_once_sec, 6)}s)."
        )
        logger.info(
            f"Model: ttnn_yolov8s (device after H2D) - avg host→DRAM (sec): {h2d_only_avg}, "
            f"avg reshard+trace+sync (sec): {compute_only_avg}, "
            f"FPS_compute_only: {round(batch_size / compute_only_avg)}"
        )
    else:
        logger.info(
            f"Model: ttnn_yolov8s - batch_size: {batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round((batch_size) / inference_time_avg)}"
        )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8s_trace_2cqs_inference(
    device,
    batch_size_per_device,
    model_location_generator,
):
    _skip_if_single_device_batch_exceeds_l1(device, batch_size_per_device)
    run_yolov8s(
        device,
        batch_size_per_device,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    (1, 32),
)
def test_run_yolov8s_trace_2cqs_inference_fps_without_h2d(
    device,
    batch_size_per_device,
    model_location_generator,
):
    """Same runner; second log line uses FPS with host→DRAM time excluded (see performant_runner phases)."""
    _skip_if_single_device_batch_exceeds_l1(device, batch_size_per_device)
    run_yolov8s(
        device,
        batch_size_per_device,
        model_location_generator,
        split_host_device_timing=True,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8s_trace_2cqs_dp_inference(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
):
    run_yolov8s(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8s_trace_2cqs_dp_inference_fps_without_h2d(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
):
    """
    Same as DP inference on the session mesh; logs FPS_compute_only (reshard + trace + sync per iter).
    Host→DRAM copy still runs each iteration; it is only excluded from that timing window.
    """
    run_yolov8s(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
        split_host_device_timing=True,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((8, 4), id="galaxy_8x4_32_devices")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8s_trace_2cqs_dp_galaxy_8x4(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
):
    """
    MeshShape(8, 4) = 32 devices (Wormhole Galaxy), DP: batch_size_per_device=1 → global batch 32.
    Same Trace+2CQ performant runner as DP inference; skips if fewer than 32 devices (conftest).
    """
    run_yolov8s(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((8, 4), id="galaxy_8x4_32_devices")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8s_trace_2cqs_dp_galaxy_8x4_fps_without_h2d(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
):
    """8×4 mesh (32 devices), 1 per device → global bs=32; FPS_compute_only excludes host→DRAM from the timed segment."""
    run_yolov8s(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
        split_host_device_timing=True,
    )
