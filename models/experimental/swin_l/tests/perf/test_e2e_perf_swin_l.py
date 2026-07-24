# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end performance tests for Swin-L backbone on TT device (trace + 2CQ)."""

import os
import time
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.swin_l.common import (
    SWIN_L_DEPTHS,
    SWIN_L_EMBED_DIM,
    SWIN_L_NUM_HEADS,
    SWIN_L_WINDOW_SIZE,
)
from models.experimental.swin_l.tt.model_preprocessing import compute_attn_masks, load_backbone_weights
from models.experimental.swin_l.tt.tt_backbone import TtSwinLBackbone
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)

_CHECKPOINT_CANDIDATES = [
    "dino_5scale_swin_l.pth",
    "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth",
]


def _get_swin_l_checkpoint_path():
    """Returns the path to an existing Swin-L checkpoint file, or empty string if none found."""
    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    ckpt_dir = base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
    for name in _CHECKPOINT_CANDIDATES:
        path = ckpt_dir / name
        if path.is_file():
            return str(path)
    return ""


def _create_swin_l_pipeline_model(ttnn_model, batch_size, input_h, padded_input_w, actual_input_w):
    """Returns a pipeline run function that preprocesses L1 input, runs the backbone, and returns the last feature map."""

    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE
        reshaped = ttnn.reshape(l1_input_tensor, (batch_size, 3, input_h, padded_input_w))
        model_input = ttnn.to_memory_config(reshaped, ttnn.DRAM_MEMORY_CONFIG)
        for t in (reshaped, l1_input_tensor):
            if t.is_allocated():
                ttnn.deallocate(t)

        if padded_input_w != actual_input_w:
            model_input = ttnn.slice(
                model_input,
                [0, 0, 0, 0],
                [batch_size, 3, input_h, actual_input_w],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        features = ttnn_model(model_input)
        last = features[-1]
        for f in features[:-1]:
            if f.is_allocated():
                ttnn.deallocate(f)
        return last

    return run


def _get_l1_input_memory_config(host_input):
    """Builds height-sharded L1 memory config for pipeline input based on tensor shape and core grid."""
    height, width = host_input.shape[-2], host_input.shape[-1]
    core_grid = ttnn.CoreGrid(x=8, y=1)
    if height % core_grid.num_cores != 0:
        core_grid = ttnn.CoreGrid(x=4, y=1)
    return ttnn.create_sharded_memory_config(
        shape=(height // core_grid.num_cores, width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@run_for_wormhole_b0()
@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 10000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [8])
@pytest.mark.parametrize("batch_size, expected_compile_time, expected_throughput_fps", [(1, 7.4, 1.9)])
def test_swin_l_backbone_e2e_perf_trace_2cq(
    device, num_iterations, batch_size, expected_compile_time, expected_throughput_fps
):
    """Measures compile time and inference throughput for Swin-L backbone with trace and 2 command queues."""
    input_h, actual_input_w = 800, 1333
    padded_input_w = ((actual_input_w + 31) // 32) * 32

    ckpt_path = _get_swin_l_checkpoint_path()
    if not ckpt_path:
        pytest.skip(
            "Checkpoint not found. Download DINO-5scale Swin-L checkpoint under "
            "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
        )

    logger.info("Loading Swin-L TTNN parameters...")
    parameters = load_backbone_weights(
        ckpt_path,
        device,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
    )
    attn_masks = compute_attn_masks(input_h, actual_input_w, 4, SWIN_L_WINDOW_SIZE, device)
    ttnn_model = TtSwinLBackbone(
        device,
        parameters,
        embed_dim=SWIN_L_EMBED_DIM,
        depths=tuple(SWIN_L_DEPTHS),
        num_heads=tuple(SWIN_L_NUM_HEADS),
        window_size=SWIN_L_WINDOW_SIZE,
        attn_masks=attn_masks,
    )

    host_input_nchw = ttnn.from_torch(
        torch.rand(batch_size, 3, input_h, padded_input_w, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=None,
    )
    host_input = ttnn.reshape(host_input_nchw, (1, 1, batch_size * 3 * input_h, padded_input_w))
    dram_config = get_memory_config_for_persistent_dram_tensor(
        host_input.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )
    l1_config = _get_l1_input_memory_config(host_input)

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=_create_swin_l_pipeline_model(ttnn_model, batch_size, input_h, padded_input_w, actual_input_w),
        device=device,
        dram_input_memory_config=dram_config,
        l1_input_memory_config=l1_config,
    )

    logger.info("Compiling Swin-L e2e perf pipeline (trace + 2CQ)...")
    start = time.perf_counter()
    pipeline.compile(host_input)
    compile_and_first_run_time = time.perf_counter() - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)
    inputs = [host_input] * num_iterations

    logger.info(f"Running Swin-L e2e perf for {num_iterations} iterations...")
    start = time.perf_counter()
    _ = pipeline.enqueue(inputs).pop_all()
    inference_time = (time.perf_counter() - start) / num_iterations

    pipeline.cleanup()

    fps = batch_size / inference_time
    logger.info(f"Swin-L average inference time: {inference_time:.4f} s, throughput: {fps:.2f} FPS")

    prep_perf_report(
        model_name="ttnn_swin_l_backbone_trace_2cq",
        batch_size=batch_size,
        inference_and_compile_time=compile_and_first_run_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"{input_h}x{actual_input_w}_batch{batch_size}",
    )
