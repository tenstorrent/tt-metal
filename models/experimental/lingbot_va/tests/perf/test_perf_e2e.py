# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""E2E perf: ``TtLingbotVA`` + ``tt_cnn`` pipeline with 2 command queues and **no** trace."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

_tt_metal_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(_tt_metal_root) not in sys.path:
    sys.path.insert(0, str(_tt_metal_root))

from models.experimental.lingbot_va.tests.demo.demo import build_infer_message
from models.experimental.lingbot_va.tests.mesh_utils import mesh_shape_request_param
from models.experimental.lingbot_va.tt.tt_lingbot_va import TtLingbotVA
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints"
OBS_H, OBS_W = 256, 320


def _make_message():
    rng = np.random.default_rng(42)
    cam_high = rng.integers(0, 256, size=(OBS_H, OBS_W, 3), dtype=np.uint8)
    cam_left = rng.integers(0, 256, size=(OBS_H, OBS_W, 3), dtype=np.uint8)
    cam_right = rng.integers(0, 256, size=(OBS_H, OBS_W, 3), dtype=np.uint8)
    return build_infer_message(
        cam_high=cam_high,
        cam_left_wrist=cam_left,
        cam_right_wrist=cam_right,
        prompt="Lift the cup from the table",
    )


def _host_input_tensor_for_pipeline(mesh_device):
    """Small TILE host tensor; pipeline only needs stable shape/shard (real data is in ``TtLingbotVA``)."""
    torch.manual_seed(0)
    host_torch = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    host = ttnn.from_torch(host_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    image_shape = host_torch.shape
    width = image_shape[-1]
    volume = image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3]
    physical_height = volume // width
    dram_grid_size = mesh_device.dram_grid_size()
    max_cores = dram_grid_size.x
    dram_cores = 1
    for cores in range(max_cores, 0, -1):
        if width % cores == 0 and (width // cores) % 32 == 0:
            dram_cores = cores
            break
    shard_width = width // dram_cores
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))}),
        [physical_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )
    l1_input_memory_config = ttnn.create_sharded_memory_config(
        shape=(physical_height, shard_width),
        core_grid=ttnn.CoreGrid(y=1, x=dram_cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return host, dram_input_memory_config, l1_input_memory_config


@pytest.mark.parametrize(
    "mesh_device",
    [mesh_shape_request_param()],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "trace_region_size": 7800000,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [4])
@pytest.mark.parametrize(
    "batch_size, expected_compile_time, expected_throughput_fps",
    [(1, 0.75, 5.68)],
)
@pytest.mark.timeout(600)
def test_perf_lingbot_va_e2e_2cq_trace(
    mesh_device,
    num_iterations,
    batch_size,
    expected_compile_time,
    expected_throughput_fps,
):
    ckpt = Path(CHECKPOINT_PATH).resolve()
    if not ckpt.is_dir():
        pytest.skip(f"Checkpoint dir not found: {ckpt}")

    message = _make_message()

    tt_model = TtLingbotVA.prepare(
        checkpoint_path=ckpt,
        message=message,
        mesh_device=mesh_device,
        num_inference_steps=1,
        action_num_inference_steps=1,
        frame_chunk_size=1,
    )

    image_host, dram_input_memory_config, l1_input_memory_config = _host_input_tensor_for_pipeline(mesh_device)

    pipe_cfg = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
    pipeline = create_pipeline_from_config(
        pipe_cfg,
        tt_model,
        mesh_device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    host_inputs = [image_host] * num_iterations

    start = time.time()
    pipeline.compile(image_host)
    end = time.time()
    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    start = time.time()
    outputs = pipeline.enqueue(host_inputs).pop_all()
    end = time.time()
    pipeline.cleanup()

    assert outputs is not None
    inference_time = (end - start) / num_iterations
    logger.info("Average model time={:.2f} ms", 1000.0 * inference_time)
    logger.info("Average model performance={:.2f} fps", num_iterations * batch_size / (end - start))

    prep_perf_report(
        model_name="lingbot_va-2cq-trace",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"batch_{batch_size}",
    )
