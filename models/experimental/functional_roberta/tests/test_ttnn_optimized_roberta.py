# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import tt_lib
import transformers

from models.demos.bert.tt import ttnn_optimized_bert, ttnn_bert
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0

from transformers import RobertaForQuestionAnswering, RobertaConfig


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["deepset/roberta-large-squad2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_roberta_for_question_answering(device, use_program_cache, reset_seeds, model_name, batch_size, sequence_size):
    config = RobertaConfig.from_pretrained(model_name)
    model = RobertaForQuestionAnswering.from_pretrained(model_name)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(batch_size, sequence_size)

    torch_output = model(
        input_ids=input_ids,
        attention_mask=torch_attention_mask,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
    )
    torch_output_start_logits = torch_output.start_logits
    torch_output_end_logits = torch_output.end_logits

    tt_model_name = f"ttnn_{model_name}_optimized"

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.RobertaForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=ttnn_optimized_bert.custom_preprocessor,
        device=device,
    )

    ttnn_roberta_inputs = ttnn_optimized_bert.preprocess_inputs(
        input_ids,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        device=device,
    )

    tt_output = ttnn_optimized_bert.bert_for_question_answering(
        config,
        *ttnn_roberta_inputs,
        parameters=parameters,
        name="roberta",
    )
    tt_output = ttnn.to_torch(tt_output)

    tt_output_start_logits = tt_output[..., :, 0]
    tt_output_end_logits = tt_output[..., :, 1]

    assert_with_pcc(torch_output_start_logits, tt_output_start_logits, 0.4505)
    assert_with_pcc(torch_output_end_logits, tt_output_end_logits, 0.4590)
