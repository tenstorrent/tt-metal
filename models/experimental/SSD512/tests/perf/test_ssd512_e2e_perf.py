# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.SSD512.tests.perf.performant_infra import SSD512PerformantTestInfra
from models.experimental.SSD512.common import load_torch_model, SSD512_L1_SMALL_SIZE
from models.experimental.SSD512.tt.tt_ssd import TtSSD
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from models.common.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SSD512_L1_SMALL_SIZE, "trace_region_size": 10000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize("batch_size, size, expected_compile_time, expected_throughput_fps", [(1, 512, 25.4, 66.5)])
@pytest.mark.models_performance_bare_metal
def test_ssd512_e2e_performant(
    device,
    num_iterations,
    batch_size,
    size,
    expected_compile_time,
    expected_throughput_fps,
    reset_seeds,
    model_location_generator,
):
    torch_model = load_torch_model(phase="test", size=size)
    torch_input = torch.randn(batch_size, 3, size, size, dtype=torch.float32)

    ttnn_model = TtSSD(torch_model, torch_input, device, batch_size)
    ttnn.synchronize_device(device)

    infra = SSD512PerformantTestInfra(device, ttnn_model, dtype=ttnn.bfloat16)
    pipeline_model = infra

    ttnn_input_tensor, l1_input_memory_config, dram_input_memory_config = infra.create_pipeline_memory_configs(
        torch_input
    )

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=pipeline_model,
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=l1_input_memory_config,
    )

    input_tensors = [ttnn_input_tensor] * num_iterations

    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    end = time.time()
    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    start = time.time()
    outputs = pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time:.2f} ms")
    logger.info(f"Average model performance={num_iterations * batch_size / (end - start):.2f} fps")

    prep_perf_report(
        model_name="ssd512-trace-2cq",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput_fps,
        comments=f"batch_{batch_size}-size_{size}",
    )
