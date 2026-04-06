# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""E2E perf: staged ``TtLingbotVA`` + ``tt_cnn`` pipeline (2 CQs, overlapped H2D).

Each ``forward`` / ``__call__`` runs TT text encoder → VAE encode → Wan transformer (loads and frees
each stage in order). Random inputs are fixed in ``TtLingbotVA.__init__`` to demo-compatible shapes; the
pipeline’s host tensor is a dummy shape for ``compile`` / ``enqueue`` only.

**Why not ``use_trace=True`` here:** ``MultiCQTracedModelOverlappedInputExecutor.compile`` runs the model
once to build the output schema, then calls ``begin_trace_capture`` and runs the model **again** inside
the capture window. A staged forward that loads whole submodels onto the mesh (text encoder, VAE,
transformer) issues device writes during that second run, which triggers
``TT_FATAL: Writes are not supported during trace capture``. Trace is only viable when the traced
``model()`` body is a fixed compute graph without dynamic load/free of weights. Use
``MultiCQModelOverlappedInputExecutor`` (``use_trace=False``) for this test.

``num_iterations=1`` → one ``compile`` forward plus one ``enqueue`` forward.
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

_tt_metal_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(_tt_metal_root) not in sys.path:
    sys.path.insert(0, str(_tt_metal_root))

from models.experimental.lingbot_va.tests.mesh_utils import mesh_shape_request_param
from models.experimental.lingbot_va.tests.perf.tt_lingbot_va_perf import TtLingbotVA
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints"


def _mesh_device_param_for_e2e() -> tuple[int, int] | int:
    """Pytest ``mesh_device`` indirect param: one physical mesh when single-chip inference is requested."""
    if os.environ.get("LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH", "").strip().lower() in ("1", "true", "yes"):
        return (1, 1)
    return mesh_shape_request_param()


def _host_input_tensor_for_pipeline(mesh_device):
    """Small TILE host tensor; pipeline only needs stable shape/shard (``TtLingbotVA`` ignores its values)."""
    torch.manual_seed(0)
    host_torch = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    host = ttnn.from_torch(host_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    shp = host_torch.shape
    width = shp[-1]
    volume = shp[0] * shp[1] * shp[2] * shp[3]
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
    [_mesh_device_param_for_e2e()],
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
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize(
    "batch_size, expected_compile_time, expected_throughput_fps",
    [(1, 0.75, 5.68)],
)
@pytest.mark.timeout(600)
def test_perf_lingbot_va_e2e_2cq_staged(
    mesh_device,
    num_iterations,
    batch_size,
    expected_compile_time,
    expected_throughput_fps,
):
    ckpt = Path(CHECKPOINT_PATH).resolve()
    if not ckpt.is_dir():
        pytest.skip(f"Checkpoint dir not found: {ckpt}")

    # ``frame_chunk_size=2`` matches the previous ``prepare`` default; random tensors are fixed in ``__init__``.
    tt_model = TtLingbotVA(ckpt, mesh_device, frame_chunk_size=2)

    work_mesh = tt_model.models["mesh_device"]
    image_host, dram_input_memory_config, l1_input_memory_config = _host_input_tensor_for_pipeline(work_mesh)

    pipe_cfg = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
    pipeline = create_pipeline_from_config(
        pipe_cfg,
        tt_model,
        work_mesh,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    host_inputs = [image_host] * num_iterations

    t0 = time.time()
    pipeline.compile(image_host)
    compile_time = time.time() - t0

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    t1 = time.time()
    outputs = pipeline.enqueue(host_inputs).pop_all()
    elapsed = time.time() - t1
    pipeline.cleanup()

    assert outputs is not None
    inference_time = elapsed / num_iterations
    logger.info("Average model time={:.2f} ms", 1000.0 * inference_time)
    logger.info("Average model performance={:.2f} fps", num_iterations * batch_size / elapsed)

    prep_perf_report(
        model_name="lingbot_va-2cq-staged",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"batch_{batch_size}",
    )
