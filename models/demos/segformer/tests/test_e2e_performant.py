# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

from models.demos.segformer.runner.performant_runner import SegformerTrace2CQ
from models.demos.segformer.tt.common import get_mesh_mappers
from models.utility_functions import run_for_wormhole_b0


def run_segformer_trace_2cqs_inference(
    device,
    batch_size,
    model_location_generator,
):
    segformer_trace_2cq = SegformerTrace2CQ()
    inputs_mapper, _, output_composer = get_mesh_mappers(device)
    segformer_trace_2cq.initialize_segformer_trace_2cqs_inference(
        device, model_location_generator, batch_size, mesh_mapper=inputs_mapper, mesh_composer=output_composer
    )
    input_shape = (batch_size, 3, 512, 512)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    inference_iter_count = 1
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        t0 = time.time()
        output = segformer_trace_2cq.run(torch_input_tensor)
        t1 = time.time()
        inference_time_iter.append(t1 - t0)
    segformer_trace_2cq.release_segformer_trace_2cqs_inference()
    inference_time_avg = round(sum(inference_time_iter) / len(inference_time_iter), 6)
    logger.info(
        f"ttnn_segformer_512x512_batch_size_{batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
@pytest.mark.models_performance_bare_metal
def test_segformer_e2e(device, batch_size, model_location_generator):
    run_segformer_trace_2cqs_inference(device, batch_size, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    ((2),),
)
@pytest.mark.models_performance_bare_metal
def test_segformer_e2e_dp(mesh_device, batch_size, model_location_generator):
    run_segformer_trace_2cqs_inference(mesh_device, batch_size, model_location_generator)
