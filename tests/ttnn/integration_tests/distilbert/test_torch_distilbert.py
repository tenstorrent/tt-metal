# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers
from transformers import (
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
)
from transformers import AutoTokenizer

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_distilbert.reference import torch_distilbert

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [768])
def test_distilbert_attention(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(model_name)
    model = HF_model.distilbert.transformer.layer[0].attention
    config = HF_model.config

    hidden_states = torch.rand((batch_size, 19, 768))
    attention_mask = torch.rand(batch_size, 19)
    torch_output, *_ = model(hidden_states, hidden_states, hidden_states, attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_distilbert.attention(
        config,
        hidden_states,
        mask=attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.999)


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [768])
def test_distilbert_feedforward(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(model_name)
    LAYER_INDEX = 0
    model = HF_model.distilbert.transformer.layer[LAYER_INDEX].ffn
    config = HF_model.config

    hidden_states = torch.rand((batch_size, 19, 768))
    torch_output = model(hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_distilbert.ffn(
        config,
        hidden_states,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [768])
def test_distilbert_transformer_block(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(model_name)
    LAYER_INDEX = 0
    model = HF_model.distilbert.transformer.layer[LAYER_INDEX]
    config = HF_model.config

    hidden_states = torch.rand((batch_size, 19, 768))
    attention_mask = torch.rand(batch_size, 19)

    torch_output = model(hidden_states, attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_distilbert.transformer_block(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output[0], output, 0.9999)


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [768])
def test_distilbert_transformer(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(model_name)

    model = HF_model.distilbert.transformer
    config = HF_model.config

    hidden_states = torch.rand((batch_size, 19, 768))
    attention_mask = torch.rand(batch_size, 19)
    head_mask = [None, None, None, None, None, None]

    torch_output = model(hidden_states, attention_mask, head_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_distilbert.transformer(
        config,
        hidden_states,
        attention_mask,
        head_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output[0], output, 0.9999)


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [768])
def test_distilbert_model(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(model_name)

    model = HF_model.distilbert
    config = HF_model.config

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )

    inputs = tokenizer(question, context, return_tensors="pt")
    position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
    torch_output = model(**inputs)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_distilbert.distilbert(
        config,
        inputs.input_ids,
        inputs.attention_mask,
        position_ids=position_ids,
        parameters=parameters,
    )

    assert_with_pcc(torch_output[0], output, 0.9999)


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [768])
def test_distilbert_qa(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(model_name)

    model = HF_model
    config = HF_model.config

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )

    inputs = tokenizer(question, context, return_tensors="pt")
    position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
    torch_output = model(**inputs)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_distilbert.distilbert_for_question_answering(
        config,
        inputs.input_ids,
        inputs.attention_mask,
        position_ids=position_ids,
        parameters=parameters,
    )

    start_logits, end_logits = output.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()

    assert_with_pcc(torch_output.start_logits, start_logits, 0.99)
    assert_with_pcc(torch_output.end_logits, end_logits, 0.99)
