# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.functional_whisper.reference import torch_functional_whisper
import transformers
from transformers import AutoFeatureExtractor, WhisperModel, WhisperConfig
from datasets import load_dataset
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from ttnn.model_preprocessing import preprocess_model_parameters, make_parameter_dict

# MODEL_NAME = "openai/whisper-base"
MODEL_NAME = "openai/whisper-tiny.en"


@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [1500])
@pytest.mark.parametrize("use_key_value_states", [False, True])
def test_whisper_attention(model_name, batch_size, sequence_size, use_key_value_states):
    torch.manual_seed(0)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = (
        transformers.models.whisper.modeling_whisper.WhisperAttention(
            embed_dim=config.d_model, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout
        )
        .to(torch.bfloat16)
        .eval()
    )
    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    if use_key_value_states:
        key_value_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.bfloat16)
    else:
        key_value_states = None
    torch_output = model(torch_hidden_states, key_value_states=key_value_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
        prefix="encoder_attn" if use_key_value_states else "",
    )

    attention_mask = None
    tt_output = torch_functional_whisper.whisper_attention(
        config,
        torch_hidden_states,
        attention_mask,
        key_value_states=key_value_states,
        parameters=parameters,
    )
    assert_with_pcc(torch_output[0], tt_output)


@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [1500])
def test_encoder_layer(model_name, batch_size, sequence_size):
    torch.manual_seed(0)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperEncoderLayer(config).to(torch.bfloat16).eval()

    embed_dim = config.d_model
    torch_hidden_states = torch_random((batch_size, sequence_size, embed_dim), -0.1, 0.1, dtype=torch.bfloat16)

    attention_mask = None
    layer_head_mask = None
    torch_output = model(torch_hidden_states, attention_mask, layer_head_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
    )

    output = torch_functional_whisper.encoder_layer(config, torch_hidden_states, parameters=parameters)
    assert_with_pcc(torch_output[0], output, pcc=0.9987)


@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("feature_size", [80])
@pytest.mark.parametrize("sequence_length", [3000])
def test_encoder(model_name, batch_size, feature_size, sequence_length):
    torch.manual_seed(0)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperEncoder(config).to(torch.bfloat16).eval()

    torch_hidden_states = torch_random((batch_size, feature_size, sequence_length), -0.1, 0.1, dtype=torch.bfloat16)

    attention_mask = None
    head_mask = None
    torch_output = model(torch_hidden_states, attention_mask, head_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
    )

    inputs_embeds = torch_functional_whisper.preprocess_encoder_inputs(
        input_features=torch_hidden_states,
        parameters=parameters,
    )

    output = torch_functional_whisper.encoder(config, inputs_embeds, parameters=parameters)

    assert_with_pcc(torch_output[0], output, pcc=0.997)


@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [1500])
def test_decoder_layer(model_name, batch_size, sequence_size):
    torch.manual_seed(0)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperDecoderLayer(config).to(torch.bfloat16).eval()
    model = model.to(torch.bfloat16)

    embed_dim = config.d_model
    torch_hidden_states = torch_random((batch_size, 32, embed_dim), -0.1, 0.1, dtype=torch.bfloat16)

    torch_encoder_hidden_states = torch_random((batch_size, sequence_size, embed_dim), -0.1, 0.1, dtype=torch.bfloat16)

    attention_mask = torch_random((batch_size, 1, 32, 32), -0.1, 0.1, dtype=torch.bfloat16)
    layer_head_mask = None
    cross_attn_layer_head_mask = None
    torch_output = model(torch_hidden_states, attention_mask, layer_head_mask, cross_attn_layer_head_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
    )

    output = torch_functional_whisper.decoder_layer(
        config, torch_hidden_states, attention_mask, torch_encoder_hidden_states, parameters=parameters
    )

    assert_with_pcc(torch_output[0], output, 0.94)


@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [1500])
def test_decoder(model_name, batch_size, sequence_size):
    torch.manual_seed(0)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperDecoder(config).to(torch.bfloat16).eval()

    embed_dim = config.d_model

    torch_encoder_hidden_states = torch_random((batch_size, sequence_size, embed_dim), -0.1, 0.1, dtype=torch.bfloat16)

    decoder_input_ids = torch.ones(1, 32).type(torch.int32) * config.decoder_start_token_id

    attention_mask = None
    head_mask = None
    cross_attn_layer_head_mask = None
    torch_output = model(
        decoder_input_ids, attention_mask, torch_encoder_hidden_states, head_mask, cross_attn_layer_head_mask
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
    )

    (decoder_hidden_states, decoder_attention_mask) = torch_functional_whisper.preprocess_decoder_inputs(
        decoder_input_ids, attention_mask, parameters=parameters
    )

    output = torch_functional_whisper.decoder(
        config,
        hidden_states=decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=torch_encoder_hidden_states,
        parameters=parameters,
    )

    assert_with_pcc(torch_output[0], output, pcc=0.9988)


# Verify that the torch functional model matches exactly the default model.
def test_torch_whisper():
    model_name = "openai/whisper-base"
    config = WhisperConfig.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    dtype_to_use = torch.bfloat16
    input_features = inputs.input_features.type(dtype_to_use)
    decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id

    model = WhisperModel.from_pretrained(model_name).to(dtype_to_use).eval()
    expected_last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state

    parameters = preprocess_model_parameters(
        f"torch_functional_whisper_not_contiguous_{dtype_to_use}",
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
    )

    attention_mask = None
    (input_embeds, decoder_hidden_states, decoder_attention_mask) = torch_functional_whisper.preprocess_inputs(
        input_features=input_features, input_ids=decoder_input_ids, attention_mask=attention_mask, parameters=parameters
    )

    expected_last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
    # original_last_hidden_state = torch_functional_whisper.whisper_original(input_features, decoder_input_ids, parameters, embed_dim, num_heads)
    last_hidden_state = torch_functional_whisper.whisper(
        config,
        input_embeds,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        parameters=parameters,
    )
    assert_with_pcc(expected_last_hidden_state, last_hidden_state, pcc=0.99837)
