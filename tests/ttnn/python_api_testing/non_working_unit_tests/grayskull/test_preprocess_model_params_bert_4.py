# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback
import transformers

# from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops

from ttnn.model_preprocessing import preprocess_model, preprocess_model_parameters
from models.demos.bert.tt import ttnn_bert
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_preprocessing_model_params_bert_4_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    model_name = "phiyodr/bert-large-finetuned-squad2"

    # set parameters
    batch_size = input_shape[0][0]
    sequence_size = input_shape[0][1]
    num_hidden_layers = 1

    # get torch model
    config = transformers.BertConfig.from_pretrained(model_name)
    config.position_embedding_type = "none"
    if num_hidden_layers is not None:
        config.num_hidden_layers = num_hidden_layers
    else:
        pytest.skip("Test mismatches when the default number of hidden layers is used")
    model = transformers.BertForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    # set inputs
    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = None

    try:
        # run model
        torch_output = model(
            torch_input_ids,
            token_type_ids=torch_token_type_ids,
            position_ids=torch_position_ids,
            attention_mask=torch_attention_mask,
        )

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
        )

        ttnn_bert_inputs = ttnn_bert.preprocess_inputs(
            torch_input_ids,
            torch_token_type_ids,
            torch_position_ids,
            torch_attention_mask,
            device=device,
        )
        output = ttnn_bert.bert_for_question_answering(
            config,
            *ttnn_bert_inputs,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)
        start_logits = output[..., 0]
        end_logits = output[..., 1]

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert_with_pcc(torch_output.start_logits, start_logits, 0.997)
    assert_with_pcc(torch_output.end_logits, end_logits, 0.996)


test_sweep_args = [
    (
        [(4, 1216, 1024)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        8687804,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_preprocessing_model_params_bert_4(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    run_preprocessing_model_params_bert_4_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
    )
