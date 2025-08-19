# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
import ttnn
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPTextModel

from models.experimental.stable_diffusion_35_large.tt.clip_encoder import (
    TtCLIPTextTransformer,
    TtCLIPTextTransformerParameters,
    TtCLIPConfig,
)
from models.experimental.stable_diffusion_35_large.tt.utils import assert_quality


@pytest.mark.parametrize(
    "clip_path, tokenizer_path, expected_pcc",
    [
        ("text_encoder", "tokenizer", 0.99),
        ("text_encoder_2", "tokenizer_2", 0.98),
    ],
    ids=["encoder_1", "encoder_2"],
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["n150"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SDXL_L1_SMALL_SIZE}],
    indirect=["device_params"],
)
def test_clip_encoder(
    *, mesh_device: ttnn.Device, clip_path: str, tokenizer_path: str, expected_pcc: float, is_ci_env
) -> None:
    model_name_checkpoint = f"stabilityai/stable-diffusion-xl-base-1.0"

    has_projection = clip_path == "text_encoder_2"  # text encoder 2 has text projection, text encoder 1 does not

    if has_projection:
        hf_model = CLIPTextModelWithProjection.from_pretrained(
            model_name_checkpoint, subfolder=clip_path, local_files_only=is_ci_env
        )
    else:
        hf_model = CLIPTextModel.from_pretrained(model_name_checkpoint, subfolder=clip_path, local_files_only=is_ci_env)
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name_checkpoint, subfolder=tokenizer_path, local_files_only=is_ci_env
    )

    hf_model.eval()

    # debug
    logger.info("=== HuggingFace Model 1 Config ===")
    logger.info(f"vocab_size: {hf_model.config.vocab_size}")
    logger.info(f"hidden_size: {hf_model.config.hidden_size}")
    logger.info(f"intermediate_size: {hf_model.config.intermediate_size}")
    logger.info(f"num_attention_heads: {hf_model.config.num_attention_heads}")
    logger.info(f"num_hidden_layers: {hf_model.config.num_hidden_layers}")
    logger.info(f"layer_norm_eps: {hf_model.config.layer_norm_eps}")
    logger.info(f"attention_dropout: {hf_model.config.attention_dropout}")
    logger.info(f"hidden_act: {hf_model.config.hidden_act}")
    logger.info(f"Full config: {hf_model.config}")

    # test text encoder 1
    logger.info("testing text encoder 1...")
    start_time = time.time()
    parameters = TtCLIPTextTransformerParameters.from_torch(
        hf_model.state_dict(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        parallel_manager=None,
        has_text_projection=has_projection,
    )

    config = TtCLIPConfig(
        vocab_size=hf_model.config.vocab_size,
        d_model=hf_model.config.hidden_size,
        d_ff=hf_model.config.intermediate_size,
        num_heads=hf_model.config.num_attention_heads,
        num_layers=hf_model.config.num_hidden_layers,
        max_position_embeddings=77,
        layer_norm_eps=hf_model.config.layer_norm_eps,
        attention_dropout=hf_model.config.attention_dropout,
        hidden_act=hf_model.config.hidden_act,
    )

    tt_model = TtCLIPTextTransformer(parameters, config)
    logger.info(f"text encoder creation time: {time.time() - start_time}")

    # cannot use randn tensor, since HF tokenizer appends a specific eos token syntax
    test_text = "A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"

    hf_inputs = tokenizer(test_text, padding=True, truncation=True, max_length=77, return_tensors="pt")
    tt_tokens = ttnn.from_torch(
        hf_inputs.input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    start_time = time.time()
    with torch.no_grad():
        hf_output = hf_model(**hf_inputs, output_hidden_states=True)
        sequence_output = hf_output.hidden_states[-2]
        pooled_output = hf_output.text_embeds if has_projection else hf_output.pooler_output
    logger.info(f"text encoder 1 CPU runtime: {time.time() - start_time}")

    # debug
    logger.info(f"HF text encoder 1 sequence output shape: {sequence_output.shape}")
    logger.info(f"HF text encoder 1 pooled output shape: {pooled_output.shape}")
    logger.info(
        f"HF text encoder 1 sequence output mean: {sequence_output.mean():.6f}, std: {sequence_output.std():.6f}"
    )
    logger.info(f"HF text encoder 1 pooled output mean: {pooled_output.mean():.6f}, std: {pooled_output.std():.6f}")

    logger.info("compiling text encoder...")
    tt_model(tt_tokens, mesh_device, parallel_manager=None)

    logger.info("executing text encoder...")
    start_time = time.time()
    eos_token_id = hf_model.config.eos_token_id

    tt_sequence_output, tt_projected_output = tt_model(tt_tokens, mesh_device, eos_token_id, parallel_manager=None)

    logger.info(f"text encoder TT-NN runtime: {time.time() - start_time}")
    logger.info("text encoder done...")

    tt_sequence_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_sequence_output.hidden_states[-2])[0])
    tt_projected_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_projected_output)[0])

    # debug
    logger.info(f"TT text encoder sequence output shape: {tt_sequence_output_torch.shape}")
    logger.info(f"TT text encoder pooled output shape: {tt_projected_output_torch.shape}")
    logger.info(
        f"TT text encoder sequence output mean: {tt_sequence_output_torch.mean():.6f}, std: {tt_sequence_output_torch.std():.6f}"
    )
    logger.info(
        f"TT text encoder pooled output mean: {tt_projected_output_torch.mean():.6f}, std: {tt_projected_output_torch.std():.6f}"
    )

    assert sequence_output.shape == tt_sequence_output_torch.shape
    assert pooled_output.shape == tt_projected_output_torch.shape

    # For some reason, this has pcc 10 both here and in sd3.5 large when max_length padding is used
    assert_quality(sequence_output, tt_sequence_output_torch, pcc=expected_pcc)
    assert_quality(pooled_output, tt_projected_output_torch, pcc=expected_pcc)
