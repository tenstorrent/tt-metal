# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from datasets import load_dataset
from loguru import logger
from transformers import AutoFeatureExtractor, EncoderDecoderCache, WhisperConfig, WhisperModel
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import is_blackhole, torch_random
from models.demos.audio.whisper.tt import ttnn_optimized_functional_whisper
from models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper import WHISPER_L1_SMALL_SIZE, init_kv_cache
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc

# MODEL_NAME = "openai/whisper-base"
MODEL_NAME = "distil-whisper/distil-large-v3"


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size_per_device", [1, 2])
@pytest.mark.parametrize(
    "sequence_size, use_encoder_states, use_attn_mask, use_kv_cache",
    (
        [1500, False, False, False],
        [32, False, True, False],
        [1, False, False, True],
        [1, True, False, False],
    ),
    ids=[
        "encoder_attn",
        "decoder_self_attn",
        "decoder_self_attn_kv_cache",
        "decoder_cross_attn",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_whisper_attention(
    mesh_device,
    ttnn_model,
    model_name,
    batch_size_per_device,
    sequence_size,
    use_encoder_states,
    use_attn_mask,
    use_kv_cache,
):
    torch.manual_seed(0)
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    is_decode = use_encoder_states or use_attn_mask or use_kv_cache
    model = transformers.models.whisper.modeling_whisper.WhisperAttention(
        embed_dim=config.d_model,
        num_heads=config.encoder_attention_heads,
        dropout=config.attention_dropout,
        is_decoder=is_decode,
        layer_idx=0,
        config=config,
    ).eval()

    if use_encoder_states:
        torch_encoder_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
        ttnn_encoder_states = ttnn.from_torch(
            torch_encoder_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=input_mesh_mapper,
        )
    else:
        torch_encoder_states = None
        ttnn_encoder_states = None

    if use_attn_mask:
        num_heads = config.decoder_attention_heads
        torch_attention_mask = torch_random(
            (batch_size, 1, sequence_size, sequence_size), -0.1, 0.1, dtype=torch.float32
        )
        ttnn_attention_mask = torch_attention_mask.expand(-1, num_heads, -1, -1)
        ttnn_attention_mask = ttnn.from_torch(
            ttnn_attention_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=input_mesh_mapper,
        )
    else:
        torch_attention_mask = None
        ttnn_attention_mask = None

    past_key_values = None
    if use_kv_cache:
        past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
        kv_cache, _ = init_kv_cache(
            config,
            mesh_device,
            max_seq_len=512,
            n_layers=1,
            weights_mesh_mapper=weights_mesh_mapper,
        )
        kv_cache = kv_cache[batch_size_per_device][0]  # Get first layer's cache for attention test
        current_decode_pos = ttnn.from_torch(
            torch.zeros(batch_size), device=mesh_device, dtype=ttnn.int32, mesh_mapper=input_mesh_mapper
        )
        generation_length = 5
    else:
        generation_length = 1

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
        prefix="encoder_attn" if use_encoder_states else "",
    )

    expec_out_pcc, expec_k_cache_pcc, expec_v_cache_pcc = 0.999, 0.999, 0.999
    does_pass = True
    for i in range(generation_length):
        torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
        ttnn_hidden_states = ttnn.from_torch(
            torch_hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=input_mesh_mapper,
        )

        torch_output, _, present_key_values = model(
            torch_hidden_states,
            attention_mask=torch_attention_mask,
            key_value_states=torch_encoder_states,
            past_key_value=past_key_values,
        )
        if is_decode and use_kv_cache:
            past_keys = present_key_values.key_cache
            past_values = present_key_values.value_cache

        output = ttnn_model.whisper_attention(
            config,
            ttnn_hidden_states,
            ttnn_attention_mask,
            is_decode=(not use_encoder_states),
            encoder_hidden_states=ttnn_encoder_states,
            kv_cache=kv_cache if use_kv_cache else None,
            current_decode_pos=current_decode_pos if use_kv_cache else None,
            parameters=ttnn_parameters,
        )
        output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
        # 4D to 3D
        if len(output.shape) == 4:
            output = output.squeeze(1)

        pcc_passed, output_pcc = comp_pcc(torch_output, output, expec_out_pcc)
        logger.info(f"[pos={i}] Output PCC: {output_pcc}")
        if not pcc_passed:
            does_pass = False
            logger.warning(f"[pos={i}] Output PCC {output_pcc} is lower than {expec_out_pcc}")

        if use_kv_cache:
            k_cache = ttnn.to_torch(kv_cache[0], mesh_composer=output_mesh_composer)[:, :, : (i + 1), :]
            v_cache = ttnn.to_torch(kv_cache[1], mesh_composer=output_mesh_composer)[:, :, : (i + 1), :]

            pcc_passed, k_cache_pcc = comp_pcc(past_keys[0], k_cache, expec_k_cache_pcc)
            logger.info(f"[pos={i}] K Cache PCC: {k_cache_pcc}")
            if not pcc_passed:
                does_pass = False
                logger.warning(f"[pos={i}] K Cache PCC {k_cache_pcc} is lower than {expec_k_cache_pcc}")

            pcc_passed, v_cache_pcc = comp_pcc(past_values[0], v_cache, expec_v_cache_pcc)
            logger.info(f"[pos={i}] V Cache PCC: {v_cache_pcc}")
            if not pcc_passed:
                does_pass = False
                logger.warning(f"[pos={i}] V Cache PCC {v_cache_pcc} is lower than {expec_v_cache_pcc}")

            # Update current decode pos
            ttnn.plus_one(current_decode_pos)

    if does_pass:
        logger.info("All PCC checks passed!")
    else:
        assert does_pass, f"PCC is lower than expected for some of the outputs. Check warnings!"


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size_per_device", [1, 2])
@pytest.mark.parametrize("sequence_size", [1500])
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_encoder_layer(mesh_device, ttnn_model, model_name, batch_size_per_device, sequence_size):
    torch.manual_seed(0)
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperEncoderLayer(config).eval()

    embed_dim = config.d_model
    torch_hidden_states = torch_random((batch_size, sequence_size, embed_dim), -0.1, 0.1, dtype=torch.float32)

    torch_output = model(torch_hidden_states, attention_mask=None, layer_head_mask=None)[0]

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )
    ttnn_hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    output = ttnn_model.encoder_layer(config, ttnn_hidden_states, parameters=ttnn_parameters)
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)

    # 4D to 3D and unpadding
    if len(output.shape) == 4:
        output = output.squeeze(1)
    output = output[:, :sequence_size, :]

    _, pcc_message = assert_with_pcc(torch_output, output, pcc=0.999)
    logger.info(f"Output PCC: {pcc_message}")


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size_per_device", [1, 2])
@pytest.mark.parametrize("sequence_length", [3000])
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_encoder(mesh_device, ttnn_model, model_name, batch_size_per_device, sequence_length):
    torch.manual_seed(0)
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperEncoder(config).eval()

    feature_size = config.num_mel_bins
    torch_input_features = torch_random((batch_size, feature_size, sequence_length), -0.1, 0.1, dtype=torch.float32)

    torch_output = model(torch_input_features).last_hidden_state

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
        prefix="encoder",
        device=mesh_device,
    )

    input_embeds = ttnn_model.preprocess_encoder_inputs(
        config=config,
        input_features=torch_input_features.unsqueeze(1),
        parameters=ttnn_parameters,
        device=mesh_device,
        weights_mesh_mapper=weights_mesh_mapper,
        input_mesh_mapper=input_mesh_mapper,
    )
    output = ttnn_model.encoder(config, input_embeds, parameters=ttnn_parameters)
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)

    # 4D to 3D and unpadding
    if len(output.shape) == 4:
        output = output.squeeze(1)
    sequence_size = torch_output.shape[-2]
    output = output[:, :sequence_size, :]

    _, pcc_message = assert_with_pcc(torch_output, output, 0.998)
    logger.info(f"Output PCC: {pcc_message}")


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size_per_device", [1, 2])
@pytest.mark.parametrize("encoder_sequence_size", [1500])
@pytest.mark.parametrize(
    "decoder_sequence_size, use_kv_cache",
    (
        [32, False],
        [1, True],
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_decoder_layer(
    mesh_device,
    ttnn_model,
    model_name,
    batch_size_per_device,
    encoder_sequence_size,
    decoder_sequence_size,
    use_kv_cache,
):
    torch.manual_seed(0)
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperDecoderLayer(config).eval()

    num_heads = config.encoder_attention_heads
    embed_dim = config.d_model
    torch_hidden_states = torch_random((batch_size, decoder_sequence_size, embed_dim), -0.1, 0.1, dtype=torch.float32)

    torch_encoder_hidden_states = torch_random(
        (batch_size, encoder_sequence_size, embed_dim), -0.1, 0.1, dtype=torch.float32
    )

    attention_mask = torch_random(
        (batch_size, 1, decoder_sequence_size, decoder_sequence_size), -0.1, 0.1, dtype=torch.float32
    )

    torch_output = model(torch_hidden_states, attention_mask, torch_encoder_hidden_states)[0]

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )
    ttnn_hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )
    attention_mask = attention_mask.expand(-1, num_heads, -1, -1)
    ttnn_attention_mask = ttnn.from_torch(
        attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=input_mesh_mapper
    )
    ttnn_encoder_hidden_states = ttnn.from_torch(
        torch_encoder_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=input_mesh_mapper,
    )

    if use_kv_cache:
        kv_cache, cross_attn_cache = init_kv_cache(
            config,
            mesh_device,
            max_seq_len=512,
            n_layers=1,
            weights_mesh_mapper=weights_mesh_mapper,
        )
        kv_cache = kv_cache[batch_size_per_device][0]
        cross_attn_cache = cross_attn_cache[batch_size_per_device][0]
        current_decode_pos = ttnn.from_torch(
            torch.zeros(batch_size), device=mesh_device, dtype=ttnn.int32, mesh_mapper=input_mesh_mapper
        )

    output = ttnn_model.decoder_layer(
        config,
        ttnn_hidden_states,
        ttnn_attention_mask,
        ttnn_encoder_hidden_states,
        kv_cache=kv_cache if use_kv_cache else None,
        cross_attn_cache=cross_attn_cache if use_kv_cache else None,
        cross_attn_cache_valid=False,
        current_decode_pos=current_decode_pos if use_kv_cache else None,
        parameters=ttnn_parameters,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
    # 4D to 3D
    if len(output.shape) == 4:
        output = output.squeeze(1)

    _, pcc_message = assert_with_pcc(torch_output, output, 0.999)
    logger.info(f"Output PCC: {pcc_message}")


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size_per_device", [1, 2])
@pytest.mark.parametrize("encoder_sequence_size", [1500])
@pytest.mark.parametrize(
    "decoder_sequence_size, use_kv_cache",
    (
        [32, False],
        [1, True],
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_decoder(
    mesh_device,
    ttnn_model,
    model_name,
    batch_size_per_device,
    encoder_sequence_size,
    decoder_sequence_size,
    use_kv_cache,
):
    torch.manual_seed(0)
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperDecoder(config).eval()
    embed_dim = config.d_model

    torch_encoder_hidden_states = torch_random(
        (batch_size, encoder_sequence_size, embed_dim), -0.1, 0.1, dtype=torch.float32
    )

    decoder_input_ids = torch.ones(batch_size, decoder_sequence_size).type(torch.int32) * config.decoder_start_token_id

    attention_mask = None

    torch_output = model(
        decoder_input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=torch_encoder_hidden_states,
    ).last_hidden_state

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
        prefix="decoder",
    )
    ttnn_decoder_input_ids = ttnn.from_torch(decoder_input_ids, dtype=ttnn.bfloat16, mesh_mapper=input_mesh_mapper)
    ttnn_decoder_input_ids = ttnn.to_device(ttnn_decoder_input_ids, mesh_device)

    ttnn_encoder_hidden_states = ttnn.from_torch(
        torch_encoder_hidden_states, dtype=ttnn.bfloat16, mesh_mapper=input_mesh_mapper
    )
    ttnn_encoder_hidden_states = ttnn.to_layout(ttnn_encoder_hidden_states, ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.to_device(ttnn_encoder_hidden_states, mesh_device)

    (decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_decoder_inputs(
        config,
        decoder_input_ids,
        attention_mask,
        parameters=ttnn_parameters,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
    )

    if use_kv_cache:
        kv_cache, cross_attn_cache = init_kv_cache(
            config,
            mesh_device,
            max_seq_len=512,
            weights_mesh_mapper=weights_mesh_mapper,
        )
        current_decode_pos = ttnn.from_torch(
            torch.zeros(batch_size), device=mesh_device, dtype=ttnn.int32, mesh_mapper=input_mesh_mapper
        )

    output = ttnn_model.decoder(
        config,
        hidden_states=decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        kv_cache=kv_cache[batch_size_per_device] if use_kv_cache else None,
        cross_attn_cache=cross_attn_cache[batch_size_per_device] if use_kv_cache else None,
        cross_attn_cache_valid=False,
        current_decode_pos=current_decode_pos if use_kv_cache else None,
        parameters=ttnn_parameters,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)

    # 4D to 3D
    if len(output.shape) == 4:
        output = output.squeeze(1)

    _, pcc_message = assert_with_pcc(torch_output, output, pcc=0.999)
    logger.info(f"Output PCC: {pcc_message}")


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize(
    "decoder_sequence_size, use_kv_cache",
    (
        [32, False],
        [1, True],
    ),
)
@pytest.mark.parametrize("batch_size_per_device", [1, 2])
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_ttnn_whisper(
    tmp_path, mesh_device, ttnn_model, model_name, decoder_sequence_size, use_kv_cache, batch_size_per_device
):
    torch.manual_seed(0)
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    config = WhisperConfig.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    input_features = input_features.repeat(batch_size, 1, 1)
    decoder_input_ids = torch.ones(batch_size, decoder_sequence_size).type(torch.int32) * config.decoder_start_token_id
    attention_mask = None
    model = WhisperModel.from_pretrained(model_name).eval()

    expected_last_hidden_state = model(
        input_features,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    ).last_hidden_state

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )

    (input_embeds, decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_inputs(
        config=config,
        input_features=input_features.unsqueeze(1),
        input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        parameters=ttnn_parameters,
        device=mesh_device,
        input_mesh_mapper=input_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
    )

    if use_kv_cache:
        kv_cache, cross_attn_cache = init_kv_cache(
            config,
            mesh_device,
            max_seq_len=512,
            weights_mesh_mapper=weights_mesh_mapper,
        )
        current_decode_pos = ttnn.from_torch(
            torch.zeros(batch_size), device=mesh_device, dtype=ttnn.int32, mesh_mapper=input_mesh_mapper
        )

    last_hidden_state = ttnn_model.whisper(
        config,
        input_embeds,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        kv_cache=kv_cache[batch_size_per_device] if use_kv_cache else None,
        cross_attn_cache=cross_attn_cache[batch_size_per_device] if use_kv_cache else None,
        current_decode_pos=current_decode_pos if use_kv_cache else None,
        parameters=ttnn_parameters,
    )
    last_hidden_state = ttnn.to_torch(last_hidden_state, mesh_composer=output_mesh_composer)

    # 4D to 3D
    if len(last_hidden_state.shape) == 4:
        last_hidden_state = last_hidden_state.squeeze(1)

    if is_blackhole():
        expec_out_pcc = 0.990
    else:  # wormhole_b0
        expec_out_pcc = 0.991
    _, pcc_message = assert_with_pcc(expected_last_hidden_state, last_hidden_state, expec_out_pcc)
    logger.info(f"Output PCC: {pcc_message}")


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size_per_device", [1, 2])
@pytest.mark.parametrize("encoder_sequence_size", [1500])
@pytest.mark.parametrize("num_decode_iterations", [5])
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE, "trace_region_size": 1000000}], indirect=True
)
def test_traced_decoder_executor(
    mesh_device,
    ttnn_model,
    model_name,
    batch_size_per_device,
    encoder_sequence_size,
    num_decode_iterations,
):
    """
    Test that traced decoder execution produces correct outputs compared to HF reference model.

    This test:
    1. Sets up decoder with KV cache and cross-attention cache
    2. Runs first iteration to populate cross-attention cache (non-traced)
    3. Captures the trace after first iteration
    4. Runs subsequent iterations using traced execution
    5. Compares traced outputs with HF reference model outputs
    """
    torch.manual_seed(0)
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    embed_dim = config.d_model

    # Create encoder hidden states (simulating encoder output)
    torch_encoder_hidden_states = torch_random(
        (batch_size, encoder_sequence_size, embed_dim), -0.1, 0.1, dtype=torch.float32
    )
    ttnn_encoder_hidden_states = ttnn.from_torch(
        torch_encoder_hidden_states, dtype=ttnn.bfloat16, mesh_mapper=input_mesh_mapper
    )
    ttnn_encoder_hidden_states = ttnn.to_layout(ttnn_encoder_hidden_states, ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.to_device(ttnn_encoder_hidden_states, mesh_device)

    # Create HF reference model
    hf_model = transformers.models.whisper.modeling_whisper.WhisperDecoder(config).eval()
    hf_past_key_values = EncoderDecoderCache.from_legacy_cache(None)

    # Preprocess model parameters for TTNN
    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: hf_model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
        prefix="decoder",
    )

    # Initialize KV cache and cross-attention cache for TTNN
    kv_cache, cross_attn_cache = init_kv_cache(
        config,
        mesh_device,
        max_seq_len=512,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    current_decode_pos = ttnn.from_torch(
        torch.zeros(batch_size), device=mesh_device, dtype=ttnn.int32, mesh_mapper=input_mesh_mapper
    )

    # Trace management variables
    trace_id_decoder = None
    trace_input_decoder = None
    trace_compiled = False
    cross_attn_cache_valid = False

    for i in range(num_decode_iterations):
        # Create decoder input for this iteration
        decoder_input_ids = torch.ones(batch_size, 1).type(torch.int32) * (config.decoder_start_token_id + i)

        # Run HF reference model
        hf_output = hf_model(
            decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=torch_encoder_hidden_states,
            past_key_values=hf_past_key_values,
            use_cache=True,
        )
        hf_reference_output = hf_output.last_hidden_state
        hf_past_key_values = hf_output.past_key_values

        # Preprocess decoder inputs for TTNN
        decoder_hidden_states, _ = ttnn_model.preprocess_decoder_inputs(
            config,
            decoder_input_ids,
            attention_mask=None,
            parameters=ttnn_parameters,
            device=mesh_device,
            decode_pos=i,
            create_attention_mask=False,
            input_mesh_mapper=input_mesh_mapper,
        )

        if trace_compiled and trace_id_decoder is not None:
            # Use traced execution
            # Copy new input to persistent L1 tensor
            trace_input_decoder = ttnn.to_memory_config(
                decoder_hidden_states,
                ttnn.L1_MEMORY_CONFIG,
                output_tensor=trace_input_decoder,
            )
            # Execute trace
            ttnn.execute_trace(mesh_device, trace_id_decoder, cq_id=0, blocking=True)
            # traced_output = trace_output_decoder
        else:
            # Non-traced execution (first iteration populates cross-attention cache)
            traced_output = ttnn_model.decoder(
                config,
                hidden_states=decoder_hidden_states,
                decoder_attention_mask=None,
                encoder_hidden_states=ttnn_encoder_hidden_states,
                kv_cache=kv_cache[batch_size_per_device],
                cross_attn_cache=cross_attn_cache[batch_size_per_device],
                cross_attn_cache_valid=cross_attn_cache_valid,
                current_decode_pos=current_decode_pos,
                parameters=ttnn_parameters,
            )

            # After first iteration, cross_attn_cache is populated
            if i == 0:
                cross_attn_cache_valid = True

            # After first iteration, capture the trace
            if i == 0:
                logger.info("Capturing decoder trace after first iteration")

                # Create a decoder function that captures the required parameters
                def traced_decoder_fn(hidden_states):
                    return ttnn_model.decoder(
                        config,
                        hidden_states=hidden_states,
                        decoder_attention_mask=None,
                        encoder_hidden_states=ttnn_encoder_hidden_states,
                        kv_cache=kv_cache[batch_size_per_device],
                        cross_attn_cache=cross_attn_cache[batch_size_per_device],
                        cross_attn_cache_valid=True,  # Cache is now populated
                        current_decode_pos=current_decode_pos,
                        parameters=ttnn_parameters,
                    )

                # Move input to L1 for trace capture
                l1_memory_config = ttnn.L1_MEMORY_CONFIG
                l1_input = ttnn.to_memory_config(decoder_hidden_states, l1_memory_config)

                # Compile run
                compile_output = traced_decoder_fn(l1_input)
                ttnn.deallocate(compile_output, force=True)
                ttnn.deallocate(l1_input)
                logger.info("Decoder trace compile run complete")

                # Allocate L1 input for trace capture
                trace_input_decoder = ttnn.to_memory_config(decoder_hidden_states, l1_memory_config)

                # Capture trace
                trace_id_decoder = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                traced_output = traced_decoder_fn(trace_input_decoder)

                ttnn.end_trace_capture(mesh_device, trace_id_decoder, cq_id=0)
                ttnn.synchronize_device(mesh_device)

                trace_compiled = True
                logger.info("Decoder trace capture complete")

        traced_output_torch = ttnn.to_torch(traced_output, mesh_composer=output_mesh_composer)
        ttnn.plus_one(current_decode_pos)

        # Compare traced output with HF reference output
        pcc_passed, pcc_message = comp_pcc(hf_reference_output, traced_output_torch, 0.999)
        logger.info(f"[iteration={i}] Traced vs HF Reference PCC: {pcc_message}")

        if not pcc_passed:
            # Cleanup before assertion
            if trace_id_decoder is not None:
                ttnn.release_trace(mesh_device, trace_id_decoder)
            assert pcc_passed, f"[iteration={i}] PCC check failed: {pcc_message}"

    # Cleanup trace
    if trace_id_decoder is not None:
        ttnn.release_trace(mesh_device, trace_id_decoder)
        logger.info("Trace released successfully")

    logger.info(f"All {num_decode_iterations} iterations passed PCC checks!")
