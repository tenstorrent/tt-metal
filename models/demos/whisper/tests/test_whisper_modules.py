# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from datasets import load_dataset
from loguru import logger
from transformers import AutoFeatureExtractor, WhisperConfig, WhisperModel
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.whisper.tt import ttnn_optimized_functional_whisper
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import WHISPER_L1_SMALL_SIZE, init_kv_cache
from models.utility_functions import is_blackhole, torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc

# MODEL_NAME = "openai/whisper-base"
MODEL_NAME = "distil-whisper/distil-large-v3"


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [1])
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
    device,
    ttnn_model,
    model_name,
    batch_size,
    sequence_size,
    use_encoder_states,
    use_attn_mask,
    use_kv_cache,
    use_program_cache,
):
    torch.manual_seed(0)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    is_decode = use_encoder_states or use_attn_mask or use_kv_cache
    model = transformers.models.whisper.modeling_whisper.WhisperAttention(
        embed_dim=config.d_model,
        num_heads=config.encoder_attention_heads,
        dropout=config.attention_dropout,
        is_decoder=is_decode,
    ).eval()

    if use_encoder_states:
        torch_encoder_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
        ttnn_encoder_states = ttnn.from_torch(
            torch_encoder_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
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
            ttnn_attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
    else:
        torch_attention_mask = None
        ttnn_attention_mask = None

    past_key_values = None
    if use_kv_cache:
        kv_cache = init_kv_cache(config, device, max_batch_size=batch_size, max_seq_len=512, n_layers=1)[0]
        current_decode_pos = ttnn.from_torch(torch.zeros(batch_size), device=device, dtype=ttnn.int32)
        generation_length = 5
    else:
        generation_length = 1

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: True,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
        prefix="encoder_attn" if use_encoder_states else "",
    )

    expec_out_pcc, expec_k_cache_pcc, expec_v_cache_pcc = 0.999, 0.999, 0.999
    does_pass = True
    for i in range(generation_length):
        torch_hidden_states = torch_random((batch_size, sequence_size, config.d_model), -0.1, 0.1, dtype=torch.float32)
        ttnn_hidden_states = ttnn.from_torch(
            torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        torch_output, _, past_key_values = model(
            torch_hidden_states,
            attention_mask=torch_attention_mask,
            key_value_states=torch_encoder_states,
            past_key_value=past_key_values,
        )
        if is_decode and use_kv_cache:
            past_keys = past_key_values[0]
            past_values = past_key_values[1]

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
        output = ttnn.to_torch(output)

        pcc_passed, output_pcc = comp_pcc(torch_output, output, expec_out_pcc)
        logger.info(f"[pos={i}] Output PCC: {output_pcc}")
        if not pcc_passed:
            does_pass = False
            logger.warning(f"[pos={i}] Output PCC {output_pcc} is lower than {expec_out_pcc}")

        if use_kv_cache:
            k_cache = ttnn.to_torch(kv_cache[0])[:, :, : (i + 1), :]
            v_cache = ttnn.to_torch(kv_cache[1])[:, :, : (i + 1), :]

            pcc_passed, k_cache_pcc = comp_pcc(past_keys, k_cache, expec_k_cache_pcc)
            logger.info(f"[pos={i}] K Cache PCC: {k_cache_pcc}")
            if not pcc_passed:
                does_pass = False
                logger.warning(f"[pos={i}] K Cache PCC {k_cache_pcc} is lower than {expec_k_cache_pcc}")

            pcc_passed, v_cache_pcc = comp_pcc(past_values, v_cache, expec_v_cache_pcc)
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
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [1500])
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_encoder_layer(device, ttnn_model, model_name, batch_size, sequence_size, use_program_cache):
    torch.manual_seed(0)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperEncoderLayer(config).eval()

    embed_dim = config.d_model
    torch_hidden_states = torch_random((batch_size, sequence_size, embed_dim), -0.1, 0.1, dtype=torch.float32)

    torch_output = model(torch_hidden_states, attention_mask=None, layer_head_mask=None)[0]

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )
    ttnn_hidden_states = ttnn.from_torch(
        torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    output = ttnn_model.encoder_layer(config, ttnn_hidden_states, parameters=ttnn_parameters)
    output = ttnn.to_torch(output)

    _, pcc_message = assert_with_pcc(torch_output, output, pcc=0.999)
    logger.info(f"Output PCC: {pcc_message}")


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [3000])
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_encoder(device, ttnn_model, model_name, batch_size, sequence_length, use_program_cache):
    torch.manual_seed(0)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperEncoder(config).eval()

    feature_size = config.num_mel_bins
    torch_input_features = torch_random((batch_size, feature_size, sequence_length), -0.1, 0.1, dtype=torch.float32)

    torch_output = model(torch_input_features).last_hidden_state

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        prefix="encoder",
        device=device,
    )

    input_embeds = ttnn_model.preprocess_encoder_inputs(
        config=config,
        input_features=torch_input_features,
        parameters=ttnn_parameters,
        device=device,
    )
    output = ttnn_model.encoder(config, input_embeds, parameters=ttnn_parameters)
    output = ttnn.to_torch(output)

    _, pcc_message = assert_with_pcc(torch_output, output, 0.998)
    logger.info(f"Output PCC: {pcc_message}")


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [1])
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
    device,
    ttnn_model,
    model_name,
    batch_size,
    encoder_sequence_size,
    decoder_sequence_size,
    use_kv_cache,
    use_program_cache,
):
    torch.manual_seed(0)
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
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )
    ttnn_hidden_states = ttnn.from_torch(
        torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    attention_mask = attention_mask.expand(-1, num_heads, -1, -1)
    ttnn_attention_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_encoder_hidden_states = ttnn.from_torch(
        torch_encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    if use_kv_cache:
        kv_cache = init_kv_cache(config, device, max_batch_size=batch_size, max_seq_len=512, n_layers=1)[0]
        current_decode_pos = ttnn.from_torch(torch.zeros(batch_size), device=device, dtype=ttnn.int32)

    output = ttnn_model.decoder_layer(
        config,
        ttnn_hidden_states,
        ttnn_attention_mask,
        ttnn_encoder_hidden_states,
        kv_cache=kv_cache if use_kv_cache else None,
        current_decode_pos=current_decode_pos if use_kv_cache else None,
        parameters=ttnn_parameters,
    )
    output = ttnn.to_torch(output)

    _, pcc_message = assert_with_pcc(torch_output, output, 0.999)
    logger.info(f"Output PCC: {pcc_message}")


@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_whisper])
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [1])
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
    device,
    ttnn_model,
    model_name,
    batch_size,
    encoder_sequence_size,
    decoder_sequence_size,
    use_kv_cache,
    use_program_cache,
):
    torch.manual_seed(0)
    config = transformers.WhisperConfig.from_pretrained(model_name)
    model = transformers.models.whisper.modeling_whisper.WhisperDecoder(config).eval()
    embed_dim = config.d_model

    torch_encoder_hidden_states = torch_random(
        (batch_size, encoder_sequence_size, embed_dim), -0.1, 0.1, dtype=torch.float32
    )

    decoder_input_ids = torch.ones(1, decoder_sequence_size).type(torch.int32) * config.decoder_start_token_id

    attention_mask = None

    torch_output = model(
        decoder_input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=torch_encoder_hidden_states,
    ).last_hidden_state

    ttnn_parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
        prefix="decoder",
    )
    ttnn_decoder_input_ids = ttnn.from_torch(decoder_input_ids, dtype=ttnn.bfloat16)
    ttnn_decoder_input_ids = ttnn.to_device(ttnn_decoder_input_ids, device)

    ttnn_encoder_hidden_states = ttnn.from_torch(torch_encoder_hidden_states, dtype=ttnn.bfloat16)
    ttnn_encoder_hidden_states = ttnn.to_layout(ttnn_encoder_hidden_states, ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.to_device(ttnn_encoder_hidden_states, device)

    (decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_decoder_inputs(
        config, decoder_input_ids, attention_mask, parameters=ttnn_parameters, device=device
    )

    if use_kv_cache:
        kv_cache = init_kv_cache(config, device, max_batch_size=batch_size, max_seq_len=512)
        current_decode_pos = ttnn.from_torch(torch.zeros(batch_size), device=device, dtype=ttnn.int32)

    output = ttnn_model.decoder(
        config,
        hidden_states=decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        kv_cache=kv_cache if use_kv_cache else None,
        current_decode_pos=current_decode_pos if use_kv_cache else None,
        parameters=ttnn_parameters,
    )
    output = ttnn.to_torch(output)

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
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_ttnn_whisper(tmp_path, device, ttnn_model, model_name, decoder_sequence_size, use_kv_cache, use_program_cache):
    torch.manual_seed(0)
    config = WhisperConfig.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    decoder_input_ids = torch.ones(1, decoder_sequence_size).type(torch.int32) * config.decoder_start_token_id

    batch_size = 1
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
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )

    (input_embeds, decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_inputs(
        config=config,
        input_features=input_features,
        input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        parameters=ttnn_parameters,
        device=device,
    )

    if use_kv_cache:
        kv_cache = init_kv_cache(config, device, max_batch_size=batch_size, max_seq_len=512)
        current_decode_pos = ttnn.from_torch(torch.zeros(batch_size), device=device, dtype=ttnn.int32)

    last_hidden_state = ttnn_model.whisper(
        config,
        input_embeds,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        kv_cache=kv_cache if use_kv_cache else None,
        current_decode_pos=current_decode_pos if use_kv_cache else None,
        parameters=ttnn_parameters,
    )
    last_hidden_state = ttnn.to_torch(last_hidden_state)

    if is_blackhole():
        expec_out_pcc = 0.990
    else:  # wormhole_b0
        expec_out_pcc = 0.991
    _, pcc_message = assert_with_pcc(expected_last_hidden_state, last_hidden_state, expec_out_pcc)
    logger.info(f"Output PCC: {pcc_message}")
