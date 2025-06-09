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
from models.experimental.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.experimental.sentence_bert.ttnn.ttnn_sentence_bert_model import TtnnSentenceBertModel
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.sentence_bert.ttnn.common import custom_preprocessor, preprocess_inputs


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
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
    sentence_bert_trace_2cq = SentenceBERTrace2CQ()

    sentence_bert_trace_2cq.initialize_sentence_bert_trace_2cqs_inference(
        device, sequence_length=sequence_length, device_batch_size=batch_size, weight_dtype=ttnn.bfloat8_b
    )

    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).eval()
    config = transformers.BertConfig.from_pretrained(inputs[0])
    input_ids = torch.randint(low=0, high=config.vocab_size - 1, size=inputs[1], dtype=torch.int64)
    attention_mask = torch.randint(0, inputs[1][0], size=inputs[1], dtype=torch.int64)
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = torch.zeros(inputs[1], dtype=torch.int64)
    position_ids = torch.arange(0, inputs[1][1], dtype=torch.int64).unsqueeze(dim=0)
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_module = TtnnSentenceBertModel(parameters=parameters, config=config)
    ttnn_input_ids, ttnn_token_type_ids, ttnn_position_ids, ttnn_attention_mask = preprocess_inputs(
        input_ids, token_type_ids, position_ids, extended_mask, device
    )
    ttnn_input_ids = ttnn.from_device(ttnn_input_ids)
    inference_iter_count = 10
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        t0 = time.time()
        output = sentence_bert_trace_2cq.execute_sentence_bert_trace_2cqs_inference(ttnn_input_ids)
        t1 = time.time()
        inference_time_iter.append(t1 - t0)
    sentence_bert_trace_2cq.release_sentence_bert_trace_2cqs_inference()
    inference_time_avg = round(sum(inference_time_iter) / len(inference_time_iter), 6)
    logger.info(
        f"ttnn_sentence_bert inference iteration time (sec): {inference_time_avg}, Sentence per sec: {round(batch_size/inference_time_avg)}"
    )
