# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.utility_functions import run_for_wormhole_b0

from models.demos.bert_tiny.tests.perf_e2e_bert_tiny import run_perf_bert_tiny


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1332224}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time, sequence_size",
    ((8, 0.004, 30, 128),),
)
def test_perf_trace_2cqs(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    sequence_size,
):
    run_perf_bert_tiny(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        sequence_size,
        device,
        "mrm8488/bert-tiny-finetuned-squadv2",
    )
