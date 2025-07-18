# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
import ttnn
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ..tt.clip_encoder import TtCLIPTextTransformer, TtCLIPTextTransformerParameters, TtCLIPConfig
from ..tt.utils import assert_quality


@pytest.mark.parametrize(
    "model_name",
    [
        "large",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(("use_program_cache"), [False, True])
def test_clip_encoder(*, device: ttnn.Device, use_program_cache: bool, model_name: str) -> None:
    if use_program_cache:
        device.enable_program_cache()

    model_name_checkpoint = f"stabilityai/stable-diffusion-3.5-{model_name}"

    hf_model_1 = CLIPTextModelWithProjection.from_pretrained(
        model_name_checkpoint, subfolder="text_encoder", local_files_only=True
    )
    hf_model_2 = CLIPTextModelWithProjection.from_pretrained(
        model_name_checkpoint, subfolder="text_encoder_2", local_files_only=True
    )
    tokenizer_1 = CLIPTokenizer.from_pretrained(model_name_checkpoint, subfolder="tokenizer", local_files_only=True)
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_name_checkpoint, subfolder="tokenizer_2", local_files_only=True)

    hf_model_1.eval()
    hf_model_2.eval()

    # Debug: Print config values for both models
    logger.info("=== HuggingFace Model 1 Config ===")
    logger.info(f"vocab_size: {hf_model_1.config.vocab_size}")
    logger.info(f"hidden_size: {hf_model_1.config.hidden_size}")
    logger.info(f"intermediate_size: {hf_model_1.config.intermediate_size}")
    logger.info(f"num_attention_heads: {hf_model_1.config.num_attention_heads}")
    logger.info(f"num_hidden_layers: {hf_model_1.config.num_hidden_layers}")
    logger.info(f"layer_norm_eps: {hf_model_1.config.layer_norm_eps}")
    logger.info(f"attention_dropout: {hf_model_1.config.attention_dropout}")
    logger.info(f"hidden_act: {hf_model_1.config.hidden_act}")
    logger.info(f"Full config: {hf_model_1.config}")

    logger.info("=== HuggingFace Model 2 Config ===")
    logger.info(f"vocab_size: {hf_model_2.config.vocab_size}")
    logger.info(f"hidden_size: {hf_model_2.config.hidden_size}")
    logger.info(f"intermediate_size: {hf_model_2.config.intermediate_size}")
    logger.info(f"num_attention_heads: {hf_model_2.config.num_attention_heads}")
    logger.info(f"num_hidden_layers: {hf_model_2.config.num_hidden_layers}")
    logger.info(f"layer_norm_eps: {hf_model_2.config.layer_norm_eps}")
    logger.info(f"attention_dropout: {hf_model_2.config.attention_dropout}")
    logger.info(f"hidden_act: {hf_model_2.config.hidden_act}")
    logger.info(f"Full config: {hf_model_2.config}")

    # test text encoder 1
    logger.info("testing text encoder 1...")
    start_time = time.time()
    parameters_1 = TtCLIPTextTransformerParameters.from_torch(
        hf_model_1.state_dict(), device=device, dtype=ttnn.bfloat16
    )

    config_1 = TtCLIPConfig(
        vocab_size=hf_model_1.config.vocab_size,
        d_model=hf_model_1.config.hidden_size,
        d_ff=hf_model_1.config.intermediate_size,
        num_heads=hf_model_1.config.num_attention_heads,
        num_layers=hf_model_1.config.num_hidden_layers,
        max_position_embeddings=77,
        layer_norm_eps=hf_model_1.config.layer_norm_eps,
        attention_dropout=hf_model_1.config.attention_dropout,
        hidden_act=hf_model_1.config.hidden_act,
    )
    logger.info("=== TtCLIPConfig 1 ===")
    logger.info(f"vocab_size: {config_1.vocab_size}")
    logger.info(f"d_model: {config_1.d_model}")
    logger.info(f"d_ff: {config_1.d_ff}")
    logger.info(f"num_heads: {config_1.num_heads}")
    logger.info(f"num_layers: {config_1.num_layers}")
    logger.info(f"max_position_embeddings: {config_1.max_position_embeddings}")
    logger.info(f"layer_norm_eps: {config_1.layer_norm_eps}")
    logger.info(f"attention_dropout: {config_1.attention_dropout}")
    logger.info(f"hidden_act: {config_1.hidden_act}")

    tt_model_1 = TtCLIPTextTransformer(parameters_1, config_1)
    logger.info(f"text encoder 1 creation time: {time.time() - start_time}")

    test_text = "A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"

    hf_inputs_1 = tokenizer_1(test_text, padding=True, truncation=True, max_length=77, return_tensors="pt")
    tt_tokens_1 = ttnn.from_torch(hf_inputs_1.input_ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = time.time()
    with torch.no_grad():
        hf_output_1 = hf_model_1(**hf_inputs_1)
        sequence_output_1 = hf_output_1.last_hidden_state
        pooled_output_1 = hf_output_1.text_embeds
    logger.info(f"text encoder 1 CPU runtime: {time.time() - start_time}")

    # debug
    logger.info(f"HF text encoder 1 sequence output shape: {sequence_output_1.shape}")
    logger.info(f"HF text encoder 1 pooled output shape: {pooled_output_1.shape}")
    logger.info(
        f"HF text encoder 1 sequence output mean: {sequence_output_1.mean():.6f}, std: {sequence_output_1.std():.6f}"
    )
    logger.info(f"HF text encoder 1 pooled output mean: {pooled_output_1.mean():.6f}, std: {pooled_output_1.std():.6f}")

    logger.info("compiling text encoder 1...")
    tt_model_1(tt_tokens_1, device)

    logger.info("executing text encoder 1...")
    start_time = time.time()
    tt_sequence_output_1, tt_pooled_output_1 = tt_model_1(tt_tokens_1, device)
    logger.info(f"text encoder 1 TT-NN runtime: {time.time() - start_time}")
    logger.info("text encoder 1 done...")

    tt_sequence_output_torch_1 = ttnn.to_torch(tt_sequence_output_1)
    tt_pooled_output_torch_1 = ttnn.to_torch(tt_pooled_output_1)

    # debug
    logger.info(f"TT text encoder 1 sequence output shape: {tt_sequence_output_torch_1.shape}")
    logger.info(f"TT text encoder 1 pooled output shape: {tt_pooled_output_torch_1.shape}")
    logger.info(
        f"TT text encoder 1 sequence output mean: {tt_sequence_output_torch_1.mean():.6f}, std: {tt_sequence_output_torch_1.std():.6f}"
    )
    logger.info(
        f"TT text encoder 1 pooled output mean: {tt_pooled_output_torch_1.mean():.6f}, std: {tt_pooled_output_torch_1.std():.6f}"
    )

    assert sequence_output_1.shape == tt_sequence_output_torch_1.shape
    assert pooled_output_1.shape == tt_pooled_output_torch_1.shape

    assert_quality(sequence_output_1, tt_sequence_output_torch_1, pcc=0.99)
    assert_quality(pooled_output_1, tt_pooled_output_torch_1, pcc=0.99)

    # test text encoder 2
    logger.info("testing text encoder 2...")
    start_time = time.time()
    parameters_2 = TtCLIPTextTransformerParameters.from_torch(
        hf_model_2.state_dict(), device=device, dtype=ttnn.bfloat16
    )

    config_2 = TtCLIPConfig(
        vocab_size=hf_model_2.config.vocab_size,
        d_model=hf_model_2.config.hidden_size,
        d_ff=hf_model_2.config.intermediate_size,
        num_heads=hf_model_2.config.num_attention_heads,
        num_layers=hf_model_2.config.num_hidden_layers,
        max_position_embeddings=77,
        layer_norm_eps=hf_model_2.config.layer_norm_eps,
        attention_dropout=hf_model_2.config.attention_dropout,
        hidden_act=hf_model_2.config.hidden_act,
    )
    logger.info("=== TtCLIPConfig 2 ===")
    logger.info(f"vocab_size: {config_2.vocab_size}")
    logger.info(f"d_model: {config_2.d_model}")
    logger.info(f"d_ff: {config_2.d_ff}")
    logger.info(f"num_heads: {config_2.num_heads}")
    logger.info(f"num_layers: {config_2.num_layers}")
    logger.info(f"max_position_embeddings: {config_2.max_position_embeddings}")
    logger.info(f"layer_norm_eps: {config_2.layer_norm_eps}")
    logger.info(f"attention_dropout: {config_2.attention_dropout}")
    logger.info(f"hidden_act: {config_2.hidden_act}")

    tt_model_2 = TtCLIPTextTransformer(parameters_2, config_2)
    logger.info(f"text encoder 2 creation time: {time.time() - start_time}")

    hf_inputs_2 = tokenizer_2(test_text, padding=True, truncation=True, max_length=77, return_tensors="pt")
    tt_tokens_2 = ttnn.from_torch(hf_inputs_2.input_ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = time.time()
    with torch.no_grad():
        hf_output_2 = hf_model_2(**hf_inputs_2)
        sequence_output_2 = hf_output_2.last_hidden_state
        pooled_output_2 = hf_output_2.text_embeds
    logger.info(f"text encoder 2 CPU runtime: {time.time() - start_time}")

    # debug
    logger.info(f"HF text encoder 2 sequence output shape: {sequence_output_2.shape}")
    logger.info(f"HF text encoder 2 pooled output shape: {pooled_output_2.shape}")
    logger.info(
        f"HF text encoder 2 sequence output mean: {sequence_output_2.mean():.6f}, std: {sequence_output_2.std():.6f}"
    )
    logger.info(f"HF text encoder 2 pooled output mean: {pooled_output_2.mean():.6f}, std: {pooled_output_2.std():.6f}")

    logger.info("compiling text encoder 2...")
    tt_model_2(tt_tokens_2, device)

    logger.info("executing text encoder 2...")
    start_time = time.time()
    tt_sequence_output_2, tt_pooled_output_2 = tt_model_2(tt_tokens_2, device)
    logger.info(f"text encoder 2 TT-NN runtime: {time.time() - start_time}")
    logger.info("text encoder 2 done...")

    tt_sequence_output_torch_2 = ttnn.to_torch(tt_sequence_output_2)
    tt_pooled_output_torch_2 = ttnn.to_torch(tt_pooled_output_2)

    # debug
    logger.info(f"TT text encoder 2 sequence output shape: {tt_sequence_output_torch_2.shape}")
    logger.info(f"TT text encoder 2 pooled output shape: {tt_pooled_output_torch_2.shape}")
    logger.info(
        f"TT text encoder 2 sequence output mean: {tt_sequence_output_torch_2.mean():.6f}, std: {tt_sequence_output_torch_2.std():.6f}"
    )
    logger.info(
        f"TT text encoder 2 pooled output mean: {tt_pooled_output_torch_2.mean():.6f}, std: {tt_pooled_output_torch_2.std():.6f}"
    )

    assert sequence_output_2.shape == tt_sequence_output_torch_2.shape
    assert pooled_output_2.shape == tt_pooled_output_torch_2.shape

    assert_quality(sequence_output_2, tt_sequence_output_torch_2, pcc=0.99)
    assert_quality(pooled_output_2, tt_pooled_output_torch_2, pcc=0.99)
