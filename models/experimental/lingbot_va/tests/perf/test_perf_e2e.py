# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""E2E-style perf: TT-CNN pipeline with two command queues and tracing disabled.

Uses ``PipelineConfig`` / ``create_pipeline_from_config`` (same pattern as
``models/experimental/pi0/tests/perf/test_perf_e2e.py``): host tensor → DRAM → L1 → model,
with overlapped input transfer (``MultiCQModelOverlappedInputExecutor``).

The callable is ``ttnn.identity`` on a representative image-shaped tensor; full Lingbot
``run_inference`` (text encoder, VAE, transformer stack) is covered in
``test_lingbot_va`` and the demo.

Env (optional): ``LINGBOT_VA_E2E_NUM_ITERS`` (default 32), compile/throughput expectations
for ``prep_perf_report`` (``LINGBOT_VA_EXPECTED_COMPILE_TIME_S``,
``LINGBOT_VA_EXPECTED_THROUGHPUT_FPS``).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from models.tt_dit.utils.test import line_params

_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Match pi0 perf: 2 CQ, no trace; fabric consistent with other Lingbot mesh tests.
LINGBOT_PERF_DEVICE_PARAMS = {
    **line_params,
    "l1_small_size": 16384,
    "trace_region_size": 0,
    "num_command_queues": 2,
}

# Representative input (same spatial layout as pi0 SigLIP images in pi0 perf test).
_INPUT_NC_HW = (1, 3, 224, 224)


def _build_width_sharded_memory_configs(
    image_shape: tuple[int, ...], mesh_device: ttnn.MeshDevice
) -> tuple[ttnn.MemoryConfig, ttnn.MemoryConfig]:
    dram_grid_size = mesh_device.dram_grid_size()
    width = image_shape[-1]
    volume = image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3]
    physical_height = volume // width
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
    return dram_input_memory_config, l1_input_memory_config


def _create_host_image_tensor(mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    torch_img = torch.randn(*_INPUT_NC_HW, dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_img,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ).cpu()


@pytest.mark.timeout(0)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [LINGBOT_PERF_DEVICE_PARAMS], indirect=True)
def test_e2e_perf(mesh_device: ttnn.MeshDevice):
    num_iterations = int(os.environ.get("LINGBOT_VA_E2E_NUM_ITERS", "32"))
    batch_size = 1
    expected_compile_time = float(os.environ.get("LINGBOT_VA_EXPECTED_COMPILE_TIME_S", "30.0"))
    expected_throughput = float(os.environ.get("LINGBOT_VA_EXPECTED_THROUGHPUT_FPS", "5.0"))

    mesh_device.enable_program_cache()

    def run_model(l1_input: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.identity(l1_input)

    dram_input_memory_config, l1_input_memory_config = _build_width_sharded_memory_configs(_INPUT_NC_HW, mesh_device)

    pipeline_config = PipelineConfig(
        use_trace=False,
        num_command_queues=2,
        all_transfers_on_separate_command_queue=False,
    )
    pipeline = create_pipeline_from_config(
        pipeline_config,
        run_model,
        mesh_device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    image_host = _create_host_image_tensor(mesh_device)
    host_inputs = [image_host] * num_iterations

    compile_start = time.time()
    pipeline.compile(image_host)
    compile_time = time.time() - compile_start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    enqueue_start = time.time()
    pipeline.enqueue(host_inputs).pop_all()
    enqueue_end = time.time()
    pipeline.cleanup()

    inference_time = (enqueue_end - enqueue_start) / num_iterations
    throughput_fps = batch_size / inference_time

    logger.info("Average model time={:.2f} ms", 1000.0 * inference_time)
    logger.info("Average model performance={:.4f} fps", throughput_fps)

    prep_perf_report(
        model_name="lingbot-va-2cq-pipeline",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput,
        comments=f"batch_{batch_size}_iters_{num_iterations}_2cq_no_trace_identity",
    )
