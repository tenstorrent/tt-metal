# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers

from models.demos.grayskull.t5.tt import ttnn_optimized_functional_t5 as functional_t5
from models.utility_functions import torch_random, is_wormhole_b0, is_blackhole
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters


@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_norm(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerNorm(config.d_model).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = functional_t5.t5_layer_norm(config, hidden_states, weight=parameters.weight)
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.9999


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_dense_act_dense(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5DenseActDense(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = functional_t5.t5_dense_act_dense(config, hidden_states, parameters)
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.99811


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_dense_gated_act_dense(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5DenseGatedActDense(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = functional_t5.t5_dense_gated_act_dense(config, hidden_states, parameters)
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.99907


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_ff(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerFF(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = functional_t5.t5_layer_ff(config, hidden_states, parameters)
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.9979


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5Attention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=functional_t5.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output, _ = functional_t5.t5_attention(config, hidden_states, is_decoder=False, parameters=parameters)
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.9990


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_self_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerSelfAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output, _ = functional_t5.t5_layer_self_attention(
        config,
        hidden_states,
        is_decoder=False,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.997


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_layer_cross_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5LayerCrossAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_key_value_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states, torch_key_value_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=functional_t5.custom_preprocessor,
        device=device,
        prefix="EncDecAttention",
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    key_value_states = ttnn.from_torch(
        torch_key_value_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    output, _ = functional_t5.t5_layer_cross_attention(
        config,
        hidden_states,
        key_value_states,
        is_decoder=False,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.9999


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_block_encoder(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.models.t5.modeling_t5.T5Block(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output, _, _ = functional_t5.t5_block(
        config,
        hidden_states,
        is_decoder=False,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.9935


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_block_decoder(device, model_name, batch_size, sequence_size):
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
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    encoder_hidden_states = ttnn.from_torch(
        torch_encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    output, _, _ = functional_t5.t5_block(
        config,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        is_decoder=False,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.9936


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_stack_encoder(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    config.use_cache = False  # Can't use cache when running as encoder
    shared_embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
    model = transformers.models.t5.modeling_t5.T5Stack(config, shared_embedding).eval()

    torch_input_ids = torch_random((batch_size, sequence_size), 0, config.vocab_size, dtype=torch.int64)
    torch_output = model(torch_input_ids).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )
    shared_embedding = preprocess_model_parameters(initialize_model=lambda: shared_embedding, device=device)

    input_ids = ttnn.from_torch(torch_input_ids, device=device)
    output = functional_t5.t5_stack(
        config,
        input_ids,
        shared_embedding_weight=shared_embedding.weight,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.9944


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_stack_decoder(device, model_name, batch_size, sequence_size):
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
        initialize_model=lambda: model, custom_preprocessor=functional_t5.custom_preprocessor, device=device
    )
    shared_embedding = preprocess_model_parameters(initialize_model=lambda: shared_embedding, device=device)

    input_ids = ttnn.from_torch(torch_input_ids, device=device)
    encoder_hidden_states = ttnn.from_torch(
        torch_encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    output = functional_t5.t5_stack(
        config,
        input_ids,
        encoder_hidden_states=encoder_hidden_states,
        shared_embedding_weight=shared_embedding.weight,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.993


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["t5-small", "google/flan-t5-small"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [128])
def test_t5_for_conditional_generation(device, model_name, batch_size, sequence_size):
    pytest.skip("Issue 9555: seeing PCC issues if running this in same process as encoder/decoder")
    torch.manual_seed(0)

    config = transformers.T5Config.from_pretrained(model_name)
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_name).eval()

    torch_input_ids = torch_random((batch_size, sequence_size), 0, config.vocab_size, dtype=torch.int64)
    torch_decoder_input_ids = torch_random((batch_size, sequence_size), 0, config.vocab_size, dtype=torch.int64)
    torch_output = model(torch_input_ids, decoder_input_ids=torch_decoder_input_ids).logits

    parameters = preprocess_model_parameters(
        model_name=f"ttnn_{model_name}_optimized",
        initialize_model=lambda: model,
        custom_preprocessor=functional_t5.custom_preprocessor,
        device=device,
    )

    input_ids = ttnn.from_torch(torch_input_ids, device=device)
    decoder_input_ids = ttnn.from_torch(torch_decoder_input_ids, device=device)
    output, *_ = functional_t5.t5_for_conditional_generation(
        config,
        input_ids,
        decoder_input_ids,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.952
