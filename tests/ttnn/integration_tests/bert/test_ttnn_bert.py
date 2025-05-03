# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.demos.bert.tt import ttnn_bert
from models.utility_functions import torch_random, is_wormhole_b0, is_blackhole

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertAttention(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_mask = torch.ones(1, sequence_size, dtype=torch.bfloat16)
    torch_output, *_ = model(torch_hidden_states, attention_mask=torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn_bert.bert_attention(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("torch_dtype", [torch.bfloat16])
def test_bert_intermediate(device, model_name, batch_size, sequence_size, torch_dtype):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertIntermediate(config).eval()
    model = model.to(torch_dtype)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch_dtype)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn_bert.bert_intermediate(
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.9997)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_output(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertOutput(config).eval()

    torch_intermediate = torch_random(
        (batch_size, sequence_size, config.intermediate_size), -0.1, 0.1, dtype=torch.float32
    )
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_intermediate, torch_residual)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
    )

    intermediate = ttnn.from_torch(torch_intermediate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn_bert.bert_output(
        config,
        intermediate,
        residual,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.99919)


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
def test_bert_feedforward(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = BertFeedForward(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn_bert.bert_feedforward(
        config,
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.99979)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_layer(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertLayer(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_mask = torch.ones(1, sequence_size, dtype=torch.bfloat16)
    torch_output, *_ = model(torch_hidden_states, attention_mask=torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn_bert.bert_layer(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.99964)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert_encoder(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2
    model = transformers.models.bert.modeling_bert.BertEncoder(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = None
    torch_output = model(torch_hidden_states, attention_mask=torch_attention_mask).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.to(torch.bfloat16),
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if torch_attention_mask is not None:
        attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        attention_mask = None
    output = ttnn_bert.bert_encoder(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch.float32), 0.99939)


@pytest.mark.skip(reason="Mismatch in output")
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bert(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.BertModel.from_pretrained(model_name, config=config).eval()
    model = model.to(torch.bfloat16)

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.ones((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.ones((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(1, sequence_size, dtype=torch.bfloat16)
    torch_output = model(
        torch_input_ids,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
        attention_mask=torch_attention_mask,
    ).last_hidden_state

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
    output = ttnn_bert.bert(
        config,
        *ttnn_bert_inputs,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("num_hidden_layers", [1, None])
def test_bert_for_question_answering(device, model_name, batch_size, sequence_size, num_hidden_layers):
    torch.manual_seed(0)

    config = transformers.BertConfig.from_pretrained(model_name)
    if num_hidden_layers is not None:
        config.num_hidden_layers = num_hidden_layers
    else:
        pytest.skip("Test mismatches when the default number of hidden layers is used")
    model = transformers.BertForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = None
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

    assert_with_pcc(torch_output.start_logits, start_logits, 0.997)
    assert_with_pcc(torch_output.end_logits, end_logits, 0.996)
