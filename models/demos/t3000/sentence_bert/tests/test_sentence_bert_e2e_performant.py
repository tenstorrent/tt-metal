# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.demos.sentence_bert.runner.performant_runner import SentenceBERTPerformantRunner
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "act_dtype, weight_dtype",
    ((ttnn.bfloat16, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize("device_batch_size, sequence_length", [(8, 384)])
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant_sentencebert_data_parallel(
    mesh_device, device_batch_size, sequence_length, act_dtype, weight_dtype
):
    batch_size = device_batch_size * mesh_device.get_num_devices()
    performant_runner = SentenceBERTPerformantRunner(
        device=mesh_device,
        device_batch_size=device_batch_size,
        sequence_length=sequence_length,
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
    )
    performant_runner._capture_sentencebert_trace_2cqs()
    inference_times = []
    for _ in range(10):
        t0 = time.time()
        _ = performant_runner.run()
        t1 = time.time()
        inference_times.append(t1 - t0)

    performant_runner.release()

    inference_time_avg = round(sum(inference_times) / len(inference_times), 6)
    logger.info(
        f"ttnn_sentencebert_batch_size: {batch_size}, One inference iteration time (sec): {inference_time_avg}, Sentence per sec: {round(batch_size/inference_time_avg)}"
    )
