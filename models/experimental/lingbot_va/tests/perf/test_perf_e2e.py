# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""E2E perf: ``TtLingbotVA`` + ``tt_cnn`` pipeline with 2 command queues.

When ``use_trace`` is False, the model runs full prepared :func:`demo.run_inference` per iteration.
When True, the pipeline uses TTNN trace capture and a single ``WanTransformer3DModel`` video forward per iteration.
"""

from __future__ import annotations

import os
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
from models.experimental.lingbot_va.tests.download_pretrained_weights import setup_checkpoint_root_for_tests
from models.experimental.lingbot_va.tests.mesh_utils import mesh_shape_request_param

setup_checkpoint_root_for_tests()
from models.experimental.lingbot_va.tests.perf.tt_lingbot_va_perf import TtLingbotVA
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

# Absolute from tt-metal root: ``TtLingbotVA.prepare`` / ``run_inference`` chdir to the Lingbot-VA package
# root, so a cwd-relative path breaks the second parametrized case (e.g. use_trace=True).
CHECKPOINT_DIR = (_tt_metal_root / "models/experimental/lingbot_va/reference/checkpoints").resolve()
OBS_H, OBS_W = 256, 320


def _perf_csv_row_strings(
    model_name: str,
    batch_size: int,
    inference_and_compile_time: float,
    inference_time: float,
    expected_compile_time: float,
    expected_inference_time: float,
    comments: str,
    inference_time_cpu: float | None = None,
) -> dict[str, str]:
    """Mirror ``prep_perf_report`` string columns for logging only (keeps ``perf_utils`` unchanged)."""
    compile_time = inference_and_compile_time - inference_time
    device_throughput = "{:.4f}".format(batch_size * (1 / inference_time))
    cpu_tp = batch_size * (1 / inference_time_cpu) if inference_time_cpu else "unknown"
    cpu_throughput = "{:.4f}".format(cpu_tp) if not isinstance(cpu_tp, str) else cpu_tp
    return {
        "Model": model_name,
        "Setting": comments,
        "Batch": str(batch_size),
        "First Run (sec)": "{:.2f}".format(inference_and_compile_time),
        "Second Run (sec)": "{:.2f}".format(inference_time),
        "Compile Time (sec)": "{:.2f}".format(compile_time),
        "Expected Compile Time (sec)": "{:.2f}".format(expected_compile_time),
        "Inference Time (sec)": "{:.4f}".format(inference_time),
        "Expected Inference Time (sec)": "{:.4f}".format(expected_inference_time),
        "Throughput (batch*inf/sec)": device_throughput,
        "Inference Time CPU (sec)": "{:.4f}".format(inference_time_cpu) if inference_time_cpu else "unknown",
        "Throughput CPU (batch*inf/sec)": cpu_throughput,
    }


def _mesh_device_param_for_e2e() -> tuple[int, int] | int:
    """Pytest ``mesh_device`` indirect param: one physical mesh when single-chip inference is requested."""
    if os.environ.get("LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH", "").strip().lower() in ("1", "true", "yes"):
        return (1, 1)
    return mesh_shape_request_param()


def _make_message():
    rng = np.random.default_rng(42)
    cam_high, cam_left, cam_right = (rng.integers(0, 256, size=(OBS_H, OBS_W, 3), dtype=np.uint8) for _ in range(3))
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
@pytest.mark.parametrize("num_iterations", [4])
@pytest.mark.parametrize(
    "use_trace, batch_size, expected_compile_time, expected_throughput_fps",
    [
        # Baseline compile wall (pipeline.compile) from N150 wormhole_b0 sample runs; see README E2E perf.
        (False, 1, 3.14, 1.02),
        (True, 1, 0.47, 5.68),
    ],
)
@pytest.mark.timeout(600)
def test_perf_lingbot_va_e2e_2cq(
    mesh_device,
    num_iterations,
    use_trace,
    batch_size,
    expected_compile_time,
    expected_throughput_fps,
):
    ckpt = CHECKPOINT_DIR
    if not ckpt.is_dir():
        pytest.skip(f"Checkpoint dir not found: {ckpt}")

    message = _make_message()

    tt_model = TtLingbotVA.prepare(
        checkpoint_path=ckpt,
        message=message,
        mesh_device=mesh_device,
        num_inference_steps=1,
        action_num_inference_steps=1,
        use_trace=use_trace,
    )

    # Must match models["mesh_device"] (same mesh when env opens (1,1); else (1,1) submesh inside multi-chip).
    work_mesh = tt_model.models["mesh_device"]
    image_host, dram_input_memory_config, l1_input_memory_config = _host_input_tensor_for_pipeline(work_mesh)

    pipe_cfg = PipelineConfig(use_trace=use_trace, num_command_queues=2, all_transfers_on_separate_command_queue=False)
    pipeline = create_pipeline_from_config(
        pipe_cfg,
        tt_model,
        work_mesh,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    # Same host tensor each iteration; pipeline treats inputs as read-only for this perf case.
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
    # First run (compile + one steady-state iter) for perf CSV: matches prep_perf_report "Compile Time" = first_run - infer.
    first_run_wall_s = compile_time + inference_time
    throughput_batch_per_s = num_iterations * batch_size / elapsed
    expected_inference_time_s = batch_size / expected_throughput_fps
    model_name = f"lingbot_va-2cq-trace_{'on' if use_trace else 'off'}"
    comments = f"batch_{batch_size}_trace_{use_trace}"

    perf_row = _perf_csv_row_strings(
        model_name=model_name,
        batch_size=batch_size,
        inference_and_compile_time=first_run_wall_s,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time_s,
        comments=comments,
        inference_time_cpu=None,
    )

    csv_name = f"perf_{model_name.replace('/', '_')}_{comments}_{time.strftime('%Y_%m_%d')}.csv"
    mesh_shape = tuple(work_mesh.shape)
    num_devices = work_mesh.get_num_devices()

    logger.info("======== Lingbot-VA E2E perf (full report) ========")
    logger.info(
        "Pipeline context: use_trace={} num_iterations={} mesh_shape={} num_devices={}",
        use_trace,
        num_iterations,
        mesh_shape,
        num_devices,
    )
    logger.info(
        "Pipeline wall: compile_s={:.4f} enqueue_total_s={:.4f} avg_iter_s={:.4f} avg_iter_ms={:.2f}",
        compile_time,
        elapsed,
        inference_time,
        inference_time * 1000.0,
    )
    logger.info("Pipeline throughput: batch*iter/s={:.4f} (fps_equiv)", throughput_batch_per_s)
    logger.info(
        "Expected baselines: expected_compile_s={} expected_throughput_fps={} expected_inference_s={:.4f}",
        expected_compile_time,
        expected_throughput_fps,
        expected_inference_time_s,
    )
    logger.info("prep_perf_report CSV row (all columns, same formatting as written to CSV):")
    for key in sorted(perf_row.keys()):
        logger.info("  {:<38} {}", key, perf_row[key])
    logger.info("Derived checks: reported_compile_s={:.4f} (first_run - avg_infer)", first_run_wall_s - inference_time)
    logger.info("Derived checks: reported_device_throughput={}", perf_row.get("Throughput (batch*inf/sec)", ""))
    logger.info(
        "CPU columns (not measured this test): Inference Time CPU={} Throughput CPU={}",
        perf_row["Inference Time CPU (sec)"],
        perf_row["Throughput CPU (batch*inf/sec)"],
    )
    logger.info("Artifacts: perf CSV filename (cwd)={}", csv_name)
    logger.info("====================================================")

    prep_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        inference_and_compile_time=first_run_wall_s,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time_s,
        comments=comments,
    )
