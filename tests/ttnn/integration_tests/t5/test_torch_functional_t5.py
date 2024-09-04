# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers

from models.demos.grayskull.t5.reference import torch_functional_t5 as functional_t5
from models.utility_functions import torch_random
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_norm(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerNorm(config.d_model).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = functional_t5.t5_layer_norm(config, torch_hidden_states, weight=parameters.weight)

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_dense_act_dense(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5DenseActDense(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = functional_t5.t5_dense_act_dense(config, torch_hidden_states, parameters)

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_dense_gated_act_dense(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5DenseGatedActDense(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = functional_t5.t5_dense_gated_act_dense(config, torch_hidden_states, parameters)

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_ff(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerFF(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = functional_t5.t5_layer_ff(config, torch_hidden_states, parameters)

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_attention(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5Attention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output, _ = functional_t5.t5_attention(config, torch_hidden_states, is_decoder=False, parameters=parameters)

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_self_attention(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerSelfAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output, _ = functional_t5.t5_layer_self_attention(
        config, torch_hidden_states, is_decoder=False, parameters=parameters
    )

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_cross_attention(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerCrossAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_key_value_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states, torch_key_value_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output, _ = functional_t5.t5_layer_cross_attention(
        config, torch_hidden_states, torch_key_value_states, is_decoder=False, parameters=parameters
    )

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_block_encoder(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5Block(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output, _, _ = functional_t5.t5_block(config, torch_hidden_states, is_decoder=False, parameters=parameters)

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_block_decoder(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    config.is_decoder = True
    model = transformers.models.t5.modeling_t5.T5Block(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_encoder_hidden_states = torch_random(
        (batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32
    )
    torch_output, *_ = model(torch_hidden_states, encoder_hidden_states=torch_encoder_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output, _, _ = functional_t5.t5_block(
        config,
        torch_hidden_states,
        encoder_hidden_states=torch_encoder_hidden_states,
        is_decoder=False,
        parameters=parameters,
    )

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_stack_encoder(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    config.use_cache = False  # Can't use cache when running as encoder
    shared_embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
    model = transformers.models.t5.modeling_t5.T5Stack(config, shared_embedding).eval()

    torch_input_ids = torch_random((batch_size, sequence_size), 0, config.vocab_size, dtype=torch.int64)
    torch_output = model(torch_input_ids).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = functional_t5.t5_stack(
        config,
        torch_input_ids,
        shared_embedding_weight=shared_embedding.weight,
        parameters=parameters,
    )

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_stack_decoder(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    config.is_decoder = True
    shared_embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
    model = transformers.models.t5.modeling_t5.T5Stack(config, shared_embedding).eval()

    torch_input_ids = torch_random((batch_size, sequence_size), 0, config.vocab_size, dtype=torch.int64)
    torch_encoder_hidden_states = torch_random(
        (batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32
    )
    torch_output = model(torch_input_ids, encoder_hidden_states=torch_encoder_hidden_states).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = functional_t5.t5_stack(
        config,
        torch_input_ids,
        encoder_hidden_states=torch_encoder_hidden_states,
        shared_embedding_weight=shared_embedding.weight,
        parameters=parameters,
    )

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_for_conditional_generation(model_name, batch_size, sequence_size):
    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_name).eval()

    torch_input_ids = torch_random((batch_size, sequence_size), 0, config.vocab_size, dtype=torch.int64)
    torch_decoder_input_ids = torch_random((batch_size, sequence_size), 0, config.vocab_size, dtype=torch.int64)
    torch_output = model(torch_input_ids, decoder_input_ids=torch_decoder_input_ids).logits

    parameters = preprocess_model_parameters(
        model_name=f"torch_{model_name}",
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output, *_ = functional_t5.t5_for_conditional_generation(
        config,
        torch_input_ids,
        torch_decoder_input_ids,
        parameters=parameters,
    )

    assert torch.allclose(torch_output, output)
    assert_with_pcc(torch_output, output)
