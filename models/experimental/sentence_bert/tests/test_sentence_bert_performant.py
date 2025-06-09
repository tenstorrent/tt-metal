# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.sentence_bert.tests.sentence_bert_performant import (
    run_sentence_bert_inference,
    run_sentence_bert_trace_inference,
    run_sentence_bert_trace_2cqs_inference,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize("device_batch_size, sequence_length", [(8, 384)])
def test_run_sentence_bert_inference(
    device,
    device_batch_size,
    sequence_length,
    use_program_cache,
):
    run_sentence_bert_inference(device, device_batch_size, sequence_length)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 3686400}], indirect=True)
@pytest.mark.parametrize("device_batch_size, sequence_length", [(8, 384)])
def test_run_sentence_bert_trace_inference(
    device,
    device_batch_size,
    sequence_length,
    use_program_cache,
):
    run_sentence_bert_trace_inference(device, device_batch_size, sequence_length)


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6397952, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("device_batch_size, sequence_length", [(8, 384)])
def test_run_sentence_bert_trace_2cq_inference(
    device,
    device_batch_size,
    sequence_length,
    use_program_cache,
):
    run_sentence_bert_trace_2cqs_inference(
        device=device, device_batch_size=device_batch_size, sequence_length=sequence_length
    )
