# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
import transformers
import ttnn
import pytest
import torch
from loguru import logger
from models.utility_functions import run_for_wormhole_b0
from models.experimental.sentence_bert.tests.sentence_bert_performant_runner import SentenceBERTPerformantRunner
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "act_dtype, weight_dtype",
    ((ttnn.bfloat16, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize("batch_size, sequence_length", [(8, 384)])
@pytest.mark.parametrize(
    "inputs",
    [
        [
            "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            [8, 384],
            [
                "Yarın tatil yapacağım, ailemle beraber doğada vakit geçireceğiz, yürüyüşler yapıp, keşifler yapacağız, çok keyifli bir tatil olacak.",
                "Yarın tatilde olacağım, ailemle birlikte şehir dışına çıkacağız, doğal güzellikleri keşfedecek ve eğlenceli zaman geçireceğiz.",
                "Yarın tatil planım var, ailemle doğa yürüyüşlerine çıkıp, yeni yerler keşfedeceğiz, harika bir tatil olacak.",
                "Yarın tatil için yola çıkacağız, ailemle birlikte sakin bir yerlerde vakit geçirip, doğa aktiviteleri yapacağız.",
                "Yarın tatilde olacağım, ailemle birlikte doğal alanlarda gezi yapıp, yeni yerler keşfedeceğiz, eğlenceli bir tatil geçireceğiz.",
                "Yarın tatilde olacağım, ailemle birlikte şehir dışında birkaç gün geçirip, doğa ile iç içe olacağız.",
                "Yarın tatil için yola çıkıyoruz, ailemle birlikte doğada keşif yapıp, eğlenceli etkinliklere katılacağız.",
                "Yarın tatilde olacağım, ailemle doğada yürüyüş yapıp, yeni yerler keşfederek harika bir zaman geçireceğiz.",
            ],
        ]
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_e2e_performant_sentencebert(
    device, batch_size, sequence_length, act_dtype, weight_dtype, inputs, use_program_cache
):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).eval()
    config = transformers.BertConfig.from_pretrained(inputs[0])
    tokenizer = transformers.AutoTokenizer.from_pretrained(inputs[0])
    encoded_input = tokenizer(inputs[2], padding="max_length", max_length=384, truncation=True, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = encoded_input["token_type_ids"]
    position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(
        input_ids, attention_mask=extended_mask, token_type_ids=token_type_ids, position_ids=position_ids
    )
    performant_runner = SentenceBERTPerformantRunner(
        device=device,
        device_batch_size=batch_size,
        sequence_length=sequence_length,
        act_dtype=act_dtype,
        weight_dtype=weight_dtype,
        model_name=inputs[0],
        input_ids=input_ids,
        extended_mask=extended_mask,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
    )
    print("before")
    print(
        assert_with_pcc(
            reference_out.last_hidden_state, performant_runner.runner_infra.torch_output.last_hidden_state, 0.0
        )
    )
    performant_runner._capture_sentencebert_trace_2cqs()
    print(
        "afetr capture",
    )
    print(
        assert_with_pcc(
            reference_out.last_hidden_state, performant_runner.runner_infra.torch_output.last_hidden_state, 0.0
        )
    )
    inference_times = []
    for _ in range(10):
        t0 = time.time()
        print("iter is", _)
        _ = performant_runner.run(input_ids)
        performant_runner.runner_infra.input_ids = input_ids
        performant_runner.runner_infra.extended_mask = extended_mask
        performant_runner.runner_infra.token_type_ids = token_type_ids
        performant_runner.runner_infra.position_ids = position_ids

        print(assert_with_pcc(reference_out.last_hidden_state, ttnn.to_torch(_).squeeze(dim=1), 0.0))
        t1 = time.time()
        inference_times.append(t1 - t0)

    performant_runner.release()

    inference_time_avg = round(sum(inference_times) / len(inference_times), 6)
    logger.info(
        f"ttnn_sentencebert_batch_size: {batch_size}, One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
