# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
from loguru import logger
import os
import urllib.request

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.petr.tt.tt_petr import ttnn_PETR
from models.experimental.petr.reference.petr import PETR
from models.experimental.petr.tt.model_preprocessing import get_parameters
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
)

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def create_petr_pipeline_model(ttnn_model, modified_batch_img_metas, batch_size, num_cams, original_shape, full_shape):
    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE
        assert l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1

        l1_input_interleaved = ttnn.to_memory_config(l1_input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        l1_input_reshaped = ttnn.reshape(l1_input_interleaved, original_shape)
        l1_input_5d = ttnn.reshape(l1_input_reshaped, full_shape)
        ttnn_inputs = {"imgs": l1_input_5d}
        output = ttnn_model.predict(ttnn_inputs, modified_batch_img_metas, skip_post_processing=True)
        return [output["all_cls_scores"], output["all_bbox_preds"]]

    return run


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "trace_region_size": 50000000,
            "num_command_queues": 1,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch, expected_compile_time, expected_throughput_fps",
    [(1, 1.2, 0.8)],
)
def test_petr_e2e_perf(
    num_iterations,
    batch,
    expected_compile_time,
    expected_throughput_fps,
    device,
    reset_seeds,
):
    torch.manual_seed(42)
    B, num_cams, C, H, W = 1, 1, 3, 320, 800
    inputs = {"imgs": torch.randn(B, num_cams, C, H, W, dtype=torch.float32)}

    scale_x = 800 / 1600
    scale_y = 320 / 900
    intrinsics_original = [
        [1266.417203046554, 0.0, 816.2670197447984],
        [0.0, 1266.417203046554, 491.50706579294757],
        [0.0, 0.0, 1.0],
    ]
    intrinsics = [
        [
            intrinsics_original[i][j] * (scale_x if j == 0 or j == 2 else scale_y if i == 1 or i == 2 else 1)
            for j in range(3)
        ]
        for i in range(3)
    ]

    modified_batch_img_metas = [
        {
            "img_shape": (H, W),
            "pad_shape": (H, W),
            "cam2img": [intrinsics],
            "lidar2cam": [torch.eye(4).numpy().tolist()],
        }
    ]

    torch_model = PETR(use_grid_mask=True)

    weights_url = (
        "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/petr/petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    )
    resources_dir = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
    weights_path = os.path.abspath(os.path.join(resources_dir, "petr_vovnet_gridmask_p4_800x320-e2191752.pth"))

    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
    if not os.path.exists(weights_path):
        logger.info(f"Downloading PETR weights from {weights_url} ...")
        urllib.request.urlretrieve(weights_url, weights_path)
        logger.info(f"Weights downloaded to {weights_path}")

    weights_state_dict = torch.load(weights_path, weights_only=False)["state_dict"]
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    logger.info("Preprocessing model parameters...")
    parameters, query_embedding_input = get_parameters(torch_model, device)

    logger.info("Creating TTNN model...")
    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    logger.info("Running torch model for reference...")
    torch_output = torch_model.predict(inputs, modified_batch_img_metas, skip_post_processing=True)

    logger.info("Preparing input tensors...")
    input_shape = inputs["imgs"].shape
    logger.info(f"Input shape: {input_shape}")

    if len(input_shape) == 5:
        B, num_cams, C, H, W = input_shape
    else:
        B = 1
        num_cams = input_shape[0]
        C, H, W = input_shape[1:]

    torch_input_reshaped = inputs["imgs"].reshape(B * num_cams, C, H, W)

    original_shape = torch_input_reshaped.shape
    torch_input_2d = torch_input_reshaped.reshape(-1, W)
    logger.info(f"Flattened input shape for pipeline: {torch_input_2d.shape}")

    ttnn_input_tensor = ttnn.from_torch(torch_input_2d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    logger.info("Creating L1 input memory config...")
    height, width = ttnn_input_tensor.shape
    max_cores_x = 8
    max_cores_y = 7
    max_cores = max_cores_x * max_cores_y

    for num_cores in range(max_cores, 0, -1):
        if height % num_cores == 0:
            shard_height = height // num_cores
            if shard_height % 32 == 0:
                break

    if num_cores <= max_cores_x:
        l1_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    else:
        cores_y = (num_cores + max_cores_x - 1) // max_cores_x
        cores_x = max_cores_x
        l1_core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores_x - 1, cores_y - 1))}
        )

    l1_shard_shape = (height // num_cores, width)
    l1_shard_spec = ttnn.ShardSpec(l1_core_grid, l1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    l1_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec
    )

    logger.info("Creating DRAM input memory config...")
    dram_grid_size = device.dram_grid_size()
    dram_height, dram_width = ttnn_input_tensor.shape

    for num_dram_cores in range(dram_grid_size.x, 0, -1):
        if dram_height % num_dram_cores == 0:
            dram_shard_height = dram_height // num_dram_cores
            if dram_shard_height % 32 == 0:
                break

    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_cores - 1, 0))}),
        [dram_shard_height, dram_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    logger.info("Creating pipeline model wrapper...")
    full_shape = (B, num_cams, C, H, W)
    run_model = create_petr_pipeline_model(
        ttnn_model, modified_batch_img_metas, B, num_cams, original_shape, full_shape
    )

    logger.info("Creating pipeline...")
    pipeline = create_pipeline_from_config(
        device=device,
        model=run_model,
        config=PipelineConfig(
            use_trace=False,
            num_command_queues=1,
            all_transfers_on_separate_command_queue=False,
        ),
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    input_tensors = [ttnn_input_tensor] * num_iterations

    logger.info("Compiling pipeline...")
    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    end = time.time()
    compile_time = end - start
    logger.info(f"Compilation time: {compile_time:.2f} seconds")

    logger.info("Preallocating output tensors...")
    pipeline.preallocate_output_tensors_on_host(num_iterations)

    if use_signpost:
        signpost(header="start")

    logger.info(f"Running {num_iterations} inference iterations...")
    start = time.time()
    outputs = pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    if use_signpost:
        signpost(header="stop")

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    fps = batch / inference_time

    logger.info(f"Average inference time: {inference_time*1000:.2f} ms")
    logger.info(f"Throughput: {fps:.2f} FPS")

    prep_perf_report(
        model_name="ttnn_petr-trace-2cq",
        batch_size=batch,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch / expected_throughput_fps,
        comments=f"batch_{batch}-E2E with 2CQ and trace",
    )
