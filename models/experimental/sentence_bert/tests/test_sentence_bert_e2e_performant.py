# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import time
import torch
import transformers
from loguru import logger
from models.utility_functions import run_for_wormhole_b0
from models.experimental.sentence_bert.tests.sentence_bert_e2e_performant import SentenceBERTrace2CQ
from models.experimental.sentence_bert.reference.sentence_bert import custom_extended_mask
from models.experimental.sentence_bert.ttnn.common import preprocess_inputs


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size, sequence_length", [(8, 384)])
@pytest.mark.parametrize(
    "inputs",
    [["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [8, 384]]],
)
def test_run_sentence_bert_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    inputs,
    sequence_length,
    model_location_generator,
):
    config = transformers.BertConfig.from_pretrained(inputs[0])
    input_ids = torch.randint(low=0, high=config.vocab_size - 1, size=inputs[1], dtype=torch.int64)
    # torch.save(input_ids,"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/dumps/input1")
    attention_mask = torch.ones(inputs[1][0], inputs[1][1])
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = torch.zeros(inputs[1], dtype=torch.int64)
    # position_ids = torch.arange(0, inputs[1][1], dtype=torch.int64).unsqueeze(dim=0)
    position_ids = create_position_ids_from_input_ids(input_ids=input_ids, padding_idx=config.pad_token_id)

    sentence_bert_trace_2cq = SentenceBERTrace2CQ()

    sentence_bert_trace_2cq.initialize_sentence_bert_trace_2cqs_inference(
        device,
        input_ids,
        extended_mask,
        token_type_ids,
        position_ids,
        device_batch_size=batch_size,
        weight_dtype=ttnn.bfloat8_b,
        sequence_length=sequence_length,
    )
    ttnn.synchronize_device(device)
    inference_iter_count = 10
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        t0 = time.time()
        # print("iter is", iter)
        # output = sentence_bert_trace_2cq.execute_sentence_bert_trace_2cqs_inference(iter)
        (
            sentence_bert_trace_2cq.ttnn_input_ids,
            sentence_bert_trace_2cq.ttnn_token_type_ids,
            sentence_bert_trace_2cq.ttnn_position_ids,
            sentence_bert_trace_2cq.ttnn_attention_mask,
        ) = preprocess_inputs(input_ids, token_type_ids, position_ids, extended_mask, device)
        output = sentence_bert_trace_2cq.run(input_ids, iter)
        ttnn.synchronize_device(device)
        t1 = time.time()
        # print("iter time is", t1 - t0)
        inference_time_iter.append(t1 - t0)
    sentence_bert_trace_2cq.release_sentence_bert_trace_2cqs_inference()
    inference_time_avg = round(sum(inference_time_iter) / len(inference_time_iter), 6)
    logger.info(
        f"ttnn_sentence_bert inference iteration time (sec): {inference_time_avg}, Sentence per sec: {round(batch_size/inference_time_avg)}"
    )
