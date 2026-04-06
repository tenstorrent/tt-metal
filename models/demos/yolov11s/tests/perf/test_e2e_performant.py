# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov11s.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11s.runner.performant_runner import YOLOv11PerformantRunner
from models.demos.yolov11s.runner.performant_runner_infra import YOLOv11PerformanceRunnerInfra

_skip_e2e_on_unsupported_arch = not (is_wormhole_b0() or is_blackhole())

# Blackhole often needs more L1 / trace than Wormhole for the same traced YOLOv11s graph.
_E2E_L1_SMALL = 24576 if is_blackhole() else YOLOV11_L1_SMALL_SIZE
_E2E_TRACE_SINGLE = 23887872 if is_blackhole() else 6434816
_E2E_TRACE_DP = 23887872

_yolov11_e2e_l1_shard_patch_applied = False


def _apply_yolov11_e2e_wormhole_or_blackhole_sharding():
    """E2E tests only: allow Blackhole in L1 height-shard setup (matches WH 8x8) without relying on demo.py."""

    global _yolov11_e2e_l1_shard_patch_applied
    if _yolov11_e2e_l1_shard_patch_applied:
        return

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=16):
        if is_wormhole_b0() or is_blackhole():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            raise RuntimeError("Unsupported device: YOLOv11 e2e performant supports Wormhole B0 and Blackhole only.")

        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        if c < min_channels:
            c = min_channels
        elif c % min_channels != 0:
            c = ((c // min_channels) + 1) * min_channels

        n = n // self.num_devices if n // self.num_devices != 0 else n
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            core_grid,
            ttnn.ShardStrategy.HEIGHT,
        )
        assert torch_input_tensor.ndim == 4, "Expected input tensor to have shape (BS, C, H, W)"

        input_tensor = [torch_input_tensor[i].unsqueeze(0) for i in range(torch_input_tensor.shape[0])]
        tt_inputs_host = ttnn.from_host_shards(
            [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for t in input_tensor],
            device.shape,
        )
        return tt_inputs_host, input_mem_config

    YOLOv11PerformanceRunnerInfra._setup_l1_sharded_input = _setup_l1_sharded_input
    _yolov11_e2e_l1_shard_patch_applied = True


try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def run_yolov11_inference(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    _apply_yolov11_e2e_wormhole_or_blackhole_sharding()
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    performant_runner = YOLOv11PerformantRunner(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
        inputs_mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        outputs_mesh_composer=outputs_mesh_composer,
    )

    input_shape = (batch_size, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    if use_signpost:
        signpost(header="start")
    t0 = time.time()
    for _ in range(100):
        _ = performant_runner.run(torch_input_tensor=torch_input_tensor)
    ttnn.synchronize_device(device)
    t1 = time.time()
    if use_signpost:
        signpost(header="stop")
    performant_runner.release()
    inference_time_avg = round((t1 - t0) / 100, 6)
    logger.info(
        f"Model: ttnn_yolov11s - batch_size: {batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size / inference_time_avg)}"
    )


@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.skipif(
    _skip_e2e_on_unsupported_arch,
    reason="YOLOv11 e2e performant runs on Wormhole B0 or Blackhole only.",
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": _E2E_L1_SMALL, "trace_region_size": _E2E_TRACE_SINGLE, "num_command_queues": 2}],
    indirect=True,
)
def test_e2e_performant(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    reset_seeds,
):
    run_yolov11_inference(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
    )


@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 640),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.skipif(
    _skip_e2e_on_unsupported_arch,
    reason="YOLOv11 e2e performant runs on Wormhole B0 or Blackhole only.",
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": _E2E_L1_SMALL, "trace_region_size": _E2E_TRACE_DP, "num_command_queues": 2}],
    indirect=True,
)
def test_e2e_performant_dp(
    mesh_device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    reset_seeds,
):
    run_yolov11_inference(
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )
