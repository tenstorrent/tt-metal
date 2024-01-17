# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers

from models.experimental.functional_t5.tt import ttnn_functional_t5 as functional_t5
from models.utility_functions import torch_random, skip_for_wormhole_b0
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_norm(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerNorm(config.d_model).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states)
    hidden_states = ttnn.to_device(hidden_states, device)
    output = functional_t5.t5_layer_norm(config, hidden_states, weight=parameters.weight)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_dense_act_dense(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5DenseActDense(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states)
    hidden_states = ttnn.to_device(hidden_states, device)
    output = functional_t5.t5_dense_act_dense(config, hidden_states, parameters)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9992)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_dense_gated_act_dense(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5DenseGatedActDense(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states)
    hidden_states = ttnn.to_device(hidden_states, device)
    output = functional_t5.t5_dense_gated_act_dense(config, hidden_states, parameters)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9991)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_ff(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerFF(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states)
    hidden_states = ttnn.to_device(hidden_states, device)
    output = functional_t5.t5_layer_ff(config, hidden_states, parameters)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.99906)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5Attention(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states)
    hidden_states = ttnn.to_device(hidden_states, device)
    output = functional_t5.t5_attention(config, hidden_states, parameters=parameters)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.998)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_self_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerSelfAttention(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states)
    hidden_states = ttnn.to_device(hidden_states, device)
    output = functional_t5.t5_layer_self_attention(config, hidden_states, parameters=parameters)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.99828)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_cross_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerCrossAttention(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    torch_key_value_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output, *_ = model(torch_hidden_states, torch_key_value_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states)
    hidden_states = ttnn.to_device(hidden_states, device)
    key_value_states = ttnn.from_torch(torch_key_value_states)
    key_value_states = ttnn.to_device(key_value_states, device)
    output = functional_t5.t5_layer_cross_attention(config, hidden_states, key_value_states, parameters=parameters)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9983)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_block_encoder(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5Block(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states)
    hidden_states = ttnn.to_device(hidden_states, device)
    output = functional_t5.t5_block(config, hidden_states, parameters=parameters)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9973)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_block_decoder(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    config.is_decoder = True
    model = transformers.models.t5.modeling_t5.T5Block(config).eval()
    model = model.to(torch.bfloat16)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    torch_encoder_hidden_states = torch_random(
        (batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16
    )
    torch_output, *_ = model(torch_hidden_states, encoder_hidden_states=torch_encoder_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states)
    hidden_states = ttnn.to_device(hidden_states, device)
    encoder_hidden_states = ttnn.from_torch(torch_encoder_hidden_states)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device)
    output = functional_t5.t5_block(
        config,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        parameters=parameters,
    )
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9973)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_stack_encoder(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    config.use_cache = False  # Can't use cache when running as encoder
    shared_embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
    model = transformers.models.t5.modeling_t5.T5Stack(config, shared_embedding).eval()
    model = model.to(torch.bfloat16)

    torch_input_ids = torch_random((batch_size, sequence_size), 0, config.vocab_size, dtype=torch.int64)
    torch_output = model(torch_input_ids).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )
    shared_embedding = preprocess_model_parameters(initialize_model=lambda: shared_embedding, device=device)

    input_ids = ttnn.from_torch(torch_input_ids)
    input_ids = ttnn.to_device(input_ids, device)
    output = functional_t5.t5_stack(
        config,
        input_ids,
        shared_embedding_weight=shared_embedding.weight,
        parameters=parameters,
    )
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.996)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_stack_decoder(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    config.is_decoder = True
    shared_embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
    model = transformers.models.t5.modeling_t5.T5Stack(config, shared_embedding).eval()
    model = model.to(torch.bfloat16)

    torch_input_ids = torch_random((batch_size, sequence_size), 0, config.vocab_size, dtype=torch.int64)
    torch_encoder_hidden_states = torch_random(
        (batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16
    )
    torch_output = model(torch_input_ids, encoder_hidden_states=torch_encoder_hidden_states).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )
    shared_embedding = preprocess_model_parameters(initialize_model=lambda: shared_embedding, device=device)

    input_ids = ttnn.from_torch(torch_input_ids)
    input_ids = ttnn.to_device(input_ids, device)
    encoder_hidden_states = ttnn.from_torch(torch_encoder_hidden_states)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device)
    output = functional_t5.t5_stack(
        config,
        input_ids,
        encoder_hidden_states=encoder_hidden_states,
        shared_embedding_weight=shared_embedding.weight,
        parameters=parameters,
    )
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9959)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_for_conditional_generation(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_name).eval()

    torch_input_ids = torch_random((batch_size, sequence_size), 0, 1, dtype=torch.int64)
    torch_decoder_input_ids = torch_random((batch_size, sequence_size), 0, 1, dtype=torch.int64)
    torch_output = model(torch_input_ids, decoder_input_ids=torch_decoder_input_ids).logits

    parameters = preprocess_model_parameters(
        f"ttnn_{model_name}",
        initialize_model=lambda: model,
        custom_preprocessor=functional_t5.custom_preprocessor,
        device=device,
    )

    input_ids = ttnn.from_torch(torch_input_ids)
    input_ids = ttnn.to_device(input_ids, device)
    decoder_input_ids = ttnn.from_torch(torch_decoder_input_ids)
    decoder_input_ids = ttnn.to_device(decoder_input_ids, device)
    output, *_ = functional_t5.t5_for_conditional_generation(
        config,
        input_ids,
        decoder_input_ids,
        parameters=parameters,
    )
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    # TODO(arakhmati): debug why pcc is low
    assert_with_pcc(torch_output, output, pcc=0.9987)
