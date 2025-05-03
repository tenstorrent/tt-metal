# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers

from ttnn.model_preprocessing import preprocess_model_parameters

from models.demos.bert.reference import torch_bert
from models.utility_functions import torch_random, is_wormhole_b0, is_blackhole

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_attention(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch.ones(1, sequence_size)
    torch_output, *_ = model(torch_hidden_states, attention_mask=torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_bert.bert_attention(
        config,
        torch_hidden_states,
        attention_mask=torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_intermediate(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertIntermediate(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_bert.bert_intermediate(
        torch_hidden_states,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_output(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertOutput(config).eval()

    torch_intermediate = torch_random(
        (batch_size, sequence_size, config.intermediate_size), -0.1, 0.1, dtype=torch.float32
    )
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_intermediate, torch_residual)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_bert.bert_output(
        config,
        torch_intermediate,
        torch_residual,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


class BertFeedForward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = transformers.models.bert.modeling_bert.BertIntermediate(config)
        self.output = transformers.models.bert.modeling_bert.BertOutput(config)

    def forward(self, hidden_states):
        intermediate = self.intermediate(hidden_states)
        hidden_states = self.output(intermediate, hidden_states)
        return hidden_states


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_feedforward(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = BertFeedForward(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_bert.bert_feedforward(
        config,
        torch_hidden_states,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_layer(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertLayer(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch.ones(1, sequence_size)
    torch_output, *_ = model(torch_hidden_states, attention_mask=torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_bert.bert_layer(
        config,
        torch_hidden_states,
        attention_mask=torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_encoder(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertEncoder(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch.ones(1, sequence_size)
    torch_output = model(torch_hidden_states, attention_mask=torch_attention_mask).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_bert.bert_encoder(
        config,
        torch_hidden_states,
        attention_mask=torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.BertModel.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.ones((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.ones((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(1, sequence_size)
    torch_output = model(
        torch_input_ids,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
        attention_mask=torch_attention_mask,
    ).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_bert.bert(
        config,
        torch_input_ids,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_for_question_answering(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.BertForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.ones((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.ones((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(1, sequence_size)
    torch_output = model(
        torch_input_ids,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
        attention_mask=torch_attention_mask,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_bert.bert_for_question_answering(
        config,
        torch_input_ids,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        parameters=parameters,
    )
    start_logits = output[..., 0]
    end_logits = output[..., 1]

    assert_with_pcc(torch_output.start_logits, start_logits)
    assert_with_pcc(torch_output.end_logits, end_logits)
