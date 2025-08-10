# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
from loguru import logger
from models.experimental.stable_diffusion_xl_base.tt.encoders.tt.tt_clip_text_transformer import TtClipTextTransformer
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from diffusers import DiffusionPipeline
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, hidden_size, encoder_id, num_layers, num_attention_heads, pcc",
    [
        ((1, 77), 768, 1, 12, 12, 0.90),
        ((1, 77), 1280, 2, 32, 20, 0.957),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@torch.no_grad()
def test_encoder(device, input_shape, hidden_size, encoder_id, num_layers, num_attention_heads, pcc):
    torch.manual_seed(0)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    text_encoder = pipe.text_encoder if encoder_id == 1 else pipe.text_encoder_2
    tokenizer = pipe.tokenizer if encoder_id == 1 else pipe.tokenizer_2
    text_encoder.eval()
    state_dict = text_encoder.state_dict()
    torch_transformer = text_encoder.text_model

    model_config = ModelOptimisations()
    tt_transformer = TtClipTextTransformer(
        device,
        state_dict,
        "text_model",
        model_config,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        num_encoder_layers=num_layers,
    )

    test_text = "A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"
    hf_inputs = tokenizer(
        test_text,
        padding="max_length",  # Pad to max_length
        max_length=77,  # CLIP's sequence length
        truncation=True,  # Handle long texts
        return_tensors="pt",
    )
    tt_tokens = ttnn.from_torch(
        hf_inputs.input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    print("hf_inputs.input_ids shape = ", hf_inputs.input_ids.shape)
    torch_pooled_output = torch_transformer(hf_inputs.input_ids)[1]

    print("tt_tokens shape = ", tt_tokens.shape)
    tt_pooled_output = tt_transformer.forward(tt_tokens)
    tt_pooled_output = ttnn.to_torch(tt_pooled_output)

    del pipe, tt_transformer
    gc.collect()

    _, pcc_message = assert_with_pcc(tt_pooled_output, torch_pooled_output, pcc)
    logger.info(f"PCC is: {pcc_message}")
