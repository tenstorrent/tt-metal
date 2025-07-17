# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
import ttnn
from loguru import logger
from transformers import CLIPTextModel, CLIPTokenizer

from ..tt.clip_encoder import TtCLIPTextTransformer, TtCLIPTextTransformerParameters, TtCLIPConfig
from ..tt.utils import assert_quality


@pytest.mark.parametrize(
    "model_name",
    [
        "openai/clip-vit-base-patch32",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(("use_program_cache"), [False, True])
def test_clip_encoder(*, device: ttnn.Device, use_program_cache: bool, model_name: str) -> None:
    if use_program_cache:
        device.enable_program_cache()

    hf_model = CLIPTextModel.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    hf_model.eval()

    start_time = time.time()
    parameters = TtCLIPTextTransformerParameters.from_torch(hf_model.state_dict(), device=device, dtype=ttnn.bfloat16)
    tt_model = TtCLIPTextTransformer(
        parameters,
        TtCLIPConfig(
            vocab_size=hf_model.config.vocab_size,
            d_model=hf_model.config.hidden_size,
            d_ff=hf_model.config.intermediate_size,
            num_heads=hf_model.config.num_attention_heads,
            num_layers=hf_model.config.num_hidden_layers,
            max_position_embeddings=77,
            layer_norm_eps=hf_model.config.layer_norm_eps,
            attention_dropout=hf_model.config.attention_dropout,
        ),
    )
    logger.info(f"model creation time: {time.time() - start_time}")

    test_text = "A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"

    hf_inputs = tokenizer(test_text, padding=True, truncation=True, max_length=77, return_tensors="pt")

    tt_tokens = ttnn.from_torch(hf_inputs.input_ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = time.time()
    with torch.no_grad():
        hf_output = hf_model(**hf_inputs)
        sequence_output = hf_output.last_hidden_state
        pooled_output = hf_output.pooler_output
    logger.info(f"CPU runtime: {time.time() - start_time}")

    # debug
    logger.info(f"HF sequence output shape: {sequence_output.shape}")
    logger.info(f"HF pooled output shape: {pooled_output.shape}")
    logger.info(f"HF sequence output mean: {sequence_output.mean():.6f}, std: {sequence_output.std():.6f}")
    logger.info(f"HF pooled output mean: {pooled_output.mean():.6f}, std: {pooled_output.std():.6f}")

    logger.info("compiling...")
    tt_model(tt_tokens, device)

    logger.info("executing...")
    start_time = time.time()
    tt_sequence_output, tt_pooled_output = tt_model(tt_tokens, device)
    logger.info(f"TT-NN runtime: {time.time() - start_time}")
    logger.info("done...")

    tt_sequence_output_torch = ttnn.to_torch(tt_sequence_output)
    tt_pooled_output_torch = ttnn.to_torch(tt_pooled_output)

    # debug
    logger.info(f"TT sequence output shape: {tt_sequence_output_torch.shape}")
    logger.info(f"TT pooled output shape: {tt_pooled_output_torch.shape}")
    logger.info(
        f"TT sequence output mean: {tt_sequence_output_torch.mean():.6f}, std: {tt_sequence_output_torch.std():.6f}"
    )
    logger.info(f"TT pooled output mean: {tt_pooled_output_torch.mean():.6f}, std: {tt_pooled_output_torch.std():.6f}")

    # check shapes
    assert sequence_output.shape == tt_sequence_output_torch.shape
    assert pooled_output.shape == tt_pooled_output_torch.shape

    # check quality
    assert_quality(sequence_output, tt_sequence_output_torch, pcc=0.99)
    assert_quality(pooled_output, tt_pooled_output_torch, pcc=0.99)
