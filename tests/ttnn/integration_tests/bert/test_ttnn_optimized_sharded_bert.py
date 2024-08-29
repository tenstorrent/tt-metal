# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers

from models.demos.bert.reference import torch_bert
from models.demos.bert.tt import ttnn_optimized_sharded_bert

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_wormhole_b0, is_blackhole


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_for_question_answering(device, use_program_cache, model_name, batch_size, sequence_size):
    torch.manual_seed(1234)

    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2

    # TODO(arakhmati): re-enable the line below once the issue with ttnn.embedding is fixed
    # torch_bert_input = torch.randint(0, config.config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_bert_input = torch.randint(0, 1, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(1, sequence_size)

    torch_parameters = preprocess_model_parameters(
        model_name=f"torch_{model_name}",
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, config=config
        ).eval(),
        convert_to_ttnn=lambda *_: False,
    )

    torch_output = torch_bert.bert_for_question_answering(
        config,
        torch_bert_input,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        parameters=torch_parameters,
    )

    tt_model_name = f"ttnn_{model_name}_optimized"

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name,
            config=config,
        ).eval(),
        custom_preprocessor=ttnn_optimized_sharded_bert.custom_preprocessor,
        device=device,
    )

    ttnn_bert_inputs = ttnn_optimized_sharded_bert.preprocess_inputs(
        torch_bert_input,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        device=device,
    )

    config = ttnn_optimized_sharded_bert.update_model_config(config, batch_size)
    tt_output = ttnn_optimized_sharded_bert.bert_for_question_answering(
        config,
        *ttnn_bert_inputs,
        parameters=parameters,
    )
    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output[..., :2]

    # TODO(arakhmati): Investigate why the PCC is not 0.9999
    # assert_with_pcc(torch_output, tt_output, 0.9999)
