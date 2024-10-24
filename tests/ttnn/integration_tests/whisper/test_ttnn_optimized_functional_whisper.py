# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import transformers
from datasets import load_dataset
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.whisper.reference import torch_functional_whisper
from models.demos.wormhole.whisper.tt import ttnn_optimized_functional_whisper
from models.utility_functions import torch_random, is_grayskull, is_wormhole_b0

MODEL_NAME = "openai/whisper-base"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("sequence_size", [1500])
@pytest.mark.parametrize("use_key_value_states", [False, True])
def test_whisper_attention(
    mesh_device, ttnn_model, model_name, batch_size, sequence_size, use_key_value_states, reset_seeds
):
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperAttention(
        embed_dim=config.d_model, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout
    ).eval()
    torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
    tt_model_name = f"ttnn_{model_name}_optimized"

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        ttnn_parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: model,
            convert_to_ttnn=ttnn_model.convert_to_ttnn,
            custom_preprocessor=ttnn_model.custom_preprocessor,
            prefix="encoder_attn" if use_key_value_states else "",
            device=mesh_device,
        )

    ttnn_hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )
    if use_key_value_states:
        torch_key_value_states = torch_random(
            (batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32
        )
        ttnn_key_value_states = ttnn.from_torch(
            torch_key_value_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=inputs_mesh_mapper,
        )
    else:
        torch_key_value_states = None
        ttnn_key_value_states = None

    torch_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
        prefix="encoder_attn" if use_key_value_states else "",
    )

    torch_attention_mask = None

    torch_output = torch_functional_whisper.whisper_attention(
        config,
        torch_hidden_states,
        torch_attention_mask,
        key_value_states=torch_key_value_states,
        parameters=torch_parameters,
    )

    attention_mask = None
    output = ttnn_model.whisper_attention(
        config,
        mesh_device,
        ttnn_hidden_states,
        attention_mask,
        key_value_states=ttnn_key_value_states,
        parameters=ttnn_parameters,
        whisper_memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull else ttnn.L1_MEMORY_CONFIG,
        output_mesh_composer=output_mesh_composer,
        inputs_mesh_mapper=inputs_mesh_mapper,
    )

    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
    assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("sequence_size", [1500])
def test_encoder_layer(mesh_device, ttnn_model, model_name, batch_size, sequence_size, reset_seeds):
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperEncoderLayer(config)
    model = model

    embed_dim = config.d_model
    torch_hidden_states = torch_random((batch_size, sequence_size, embed_dim), -0.1, 0.1, dtype=torch.float32)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
    )
    torch_output = torch_functional_whisper.encoder_layer(config, torch_hidden_states, parameters=parameters)

    tt_model_name = f"ttnn_{model_name}_optimized"

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        ttnn_parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: model,
            convert_to_ttnn=ttnn_model.convert_to_ttnn,
            custom_preprocessor=ttnn_model.custom_preprocessor,
            device=mesh_device,
        )

    model = model.eval()
    ttnn_hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )

    output = ttnn_model.encoder_layer(
        config,
        mesh_device,
        ttnn_hidden_states,
        parameters=ttnn_parameters,
        whisper_memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull else ttnn.L1_MEMORY_CONFIG,
        output_mesh_composer=output_mesh_composer,
        inputs_mesh_mapper=inputs_mesh_mapper,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output, output, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("feature_size", [80])
@pytest.mark.parametrize("sequence_length", [3000])
def test_encoder(mesh_device, ttnn_model, model_name, batch_size, feature_size, sequence_length, reset_seeds):
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperEncoder(config)

    torch_input_features = torch_random((batch_size, feature_size, sequence_length), -0.1, 0.1, dtype=torch.float32)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
    )

    def convert_to_ttnn(model, name):
        return name not in [
            "conv1",
            "conv2",
            "embed_positions",
        ]

    inputs_embeds = torch_functional_whisper.preprocess_encoder_inputs(
        input_features=torch_input_features,
        parameters=parameters,
    )

    torch_output = torch_functional_whisper.encoder(config, inputs_embeds, parameters=parameters)

    tt_model_name = f"ttnn_{model_name}_optimized"

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        ttnn_parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: model,
            convert_to_ttnn=convert_to_ttnn,
            custom_preprocessor=ttnn_model.custom_preprocessor,
            device=mesh_device,
        )

    ttnn_parameters.embed_positions.weight = ttnn_parameters.embed_positions.weight.unsqueeze(0)
    ttnn_parameters.embed_positions.weight = ttnn.from_torch(
        ttnn_parameters.embed_positions.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )

    model = model.eval()
    input_embeds = ttnn_model.preprocess_encoder_inputs(
        input_features=torch_input_features,
        parameters=ttnn_parameters,
        device=mesh_device,
        whisper_memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull else ttnn.L1_MEMORY_CONFIG,
        weights_mesh_mapper=weights_mesh_mapper,
        inputs_mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )
    input_embeds = ttnn.to_layout(input_embeds, ttnn.TILE_LAYOUT)
    input_embeds = ttnn.to_device(input_embeds, mesh_device)

    output = ttnn_model.encoder(
        config,
        mesh_device,
        input_embeds,
        parameters=ttnn_parameters,
        whisper_memory_config=ttnn.L1_MEMORY_CONFIG,
        output_mesh_composer=output_mesh_composer,
        inputs_mesh_mapper=inputs_mesh_mapper,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("sequence_size", [1500])
def test_decoder_layer(mesh_device, reset_seeds, ttnn_model, model_name, batch_size, sequence_size):
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperDecoderLayer(config).eval()
    num_heads = config.encoder_attention_heads
    embed_dim = config.d_model
    torch_hidden_states = torch_random((batch_size, 2, embed_dim), -0.1, 0.1, dtype=torch.float32)

    torch_encoder_hidden_states = torch_random((batch_size, sequence_size, embed_dim), -0.1, 0.1, dtype=torch.float32)

    attention_mask = torch_random((batch_size, 1, 2, 2), -0.1, 0.1, dtype=torch.float32)
    # Putting num_heads in the channel because the add does not support broadcasting outside of the h and w dimensions.
    attention_mask = attention_mask.expand(-1, num_heads, -1, -1)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
    )

    torch_output = torch_functional_whisper.decoder_layer(
        config, torch_hidden_states, attention_mask, torch_encoder_hidden_states, parameters=parameters
    )
    tt_model_name = f"ttnn_{model_name}_optimized"
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        ttnn_parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: model,
            convert_to_ttnn=ttnn_model.convert_to_ttnn,
            custom_preprocessor=ttnn_model.custom_preprocessor,
            device=mesh_device,
        )

    model = model.eval()
    ttnn_hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
    )

    ttnn_attention_mask = ttnn.from_torch(
        attention_mask, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=inputs_mesh_mapper, layout=ttnn.TILE_LAYOUT
    )

    ttnn_encoder_hidden_states = ttnn.from_torch(
        torch_encoder_hidden_states,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
    )

    output = ttnn_model.decoder_layer(
        config,
        mesh_device,
        ttnn_hidden_states,
        ttnn_attention_mask,
        ttnn_encoder_hidden_states,
        parameters=ttnn_parameters,
        whisper_memory_config=ttnn.L1_MEMORY_CONFIG,
        output_mesh_composer=output_mesh_composer,
        inputs_mesh_mapper=inputs_mesh_mapper,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("sequence_size", [1500])
def test_decoder(mesh_device, ttnn_model, model_name, batch_size, sequence_size, reset_seeds):
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperDecoder(config).eval()

    embed_dim = config.d_model

    torch_encoder_hidden_states = torch_random((batch_size, sequence_size, embed_dim), -0.1, 0.1, dtype=torch.float32)

    decoder_input_ids = torch.tensor([[1, 1]] * batch_size) * config.decoder_start_token_id

    attention_mask = None

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
    )

    (decoder_hidden_states, decoder_attention_mask) = torch_functional_whisper.preprocess_decoder_inputs(
        decoder_input_ids, attention_mask, parameters=parameters
    )

    torch_output = torch_functional_whisper.decoder(
        config,
        hidden_states=decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=torch_encoder_hidden_states,
        parameters=parameters,
    )

    tt_model_name = f"ttnn_{model_name}_optimized"

    def convert_to_ttnn(model, name):
        return name not in ["conv1", "conv2", "embed_positions", "embed_tokens"]

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        ttnn_parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: model,
            convert_to_ttnn=convert_to_ttnn,
            custom_preprocessor=ttnn_model.custom_preprocessor,
            device=mesh_device,
        )
    model = model.eval()

    ttnn_decoder_input_ids = ttnn.from_torch(decoder_input_ids, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)
    ttnn_decoder_input_ids = ttnn.to_device(ttnn_decoder_input_ids, mesh_device)

    ttnn_encoder_hidden_states = ttnn.from_torch(
        torch_encoder_hidden_states, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper
    )
    ttnn_encoder_hidden_states = ttnn.to_layout(ttnn_encoder_hidden_states, ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.to_device(ttnn_encoder_hidden_states, mesh_device)

    (decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_decoder_inputs(
        config,
        decoder_input_ids,
        attention_mask,
        parameters=ttnn_parameters,
        device=mesh_device,
        inputs_mesh_mapper=inputs_mesh_mapper,
    )

    output = ttnn_model.decoder(
        config,
        device=mesh_device,
        hidden_states=decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        parameters=ttnn_parameters,
        whisper_memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull else ttnn.L1_MEMORY_CONFIG,
        output_mesh_composer=output_mesh_composer,
        inputs_mesh_mapper=inputs_mesh_mapper,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output, output, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
def test_ttnn_whisper(tmp_path, mesh_device, model_name, ttnn_model, batch_size, reset_seeds):
    config = transformers.WhisperConfig.from_pretrained(model_name)
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(model_name)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(
        [ds[i]["audio"]["array"] for i in range(batch_size)], sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs.input_features
    decoder_input_ids = torch.tensor([[1, 1]] * batch_size) * config.decoder_start_token_id

    model = transformers.WhisperModel.from_pretrained(model_name)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model.eval(),
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_whisper.custom_preprocessor,
    )

    (encoder_hidden_states, decoder_hidden_states, decoder_attention_mask) = torch_functional_whisper.preprocess_inputs(
        input_features=input_features,
        input_ids=decoder_input_ids,
        attention_mask=None,
        parameters=parameters,
    )

    torch_last_hidden_state = torch_functional_whisper.whisper(
        config,
        encoder_hidden_states,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        parameters=parameters,
    )

    tt_model_name = f"ttnn_{model_name}_optimized"
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        ttnn_parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: model,
            convert_to_ttnn=ttnn_model.convert_to_ttnn,
            custom_preprocessor=ttnn_model.custom_preprocessor,
            device=mesh_device,
        )

    model = model.eval()

    (input_embeds, decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_inputs(
        config=config,
        input_features=input_features,
        input_ids=decoder_input_ids,
        attention_mask=None,
        parameters=ttnn_parameters,
        device=mesh_device,
        whisper_memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull else ttnn.L1_MEMORY_CONFIG,
        weights_mesh_mapper=weights_mesh_mapper,
        inputs_mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    last_hidden_state = ttnn_model.whisper(
        config,
        mesh_device,
        input_embeds,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        parameters=ttnn_parameters,
        whisper_memory_config=ttnn.DRAM_MEMORY_CONFIG if is_grayskull else ttnn.L1_MEMORY_CONFIG,
        output_mesh_composer=output_mesh_composer,
        inputs_mesh_mapper=inputs_mesh_mapper,
    )

    last_hidden_state = ttnn.to_torch(last_hidden_state, mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_last_hidden_state, last_hidden_state, 0.857)
