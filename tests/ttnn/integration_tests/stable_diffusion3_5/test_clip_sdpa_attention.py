# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from diffusers import StableDiffusion3Pipeline
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.clip_sdpa_attention import CLIPSdpaAttention
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_sdpa_attention import ttnn_CLIPSdpaAttention


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_clip_sdpa_attention(device, reset_seeds):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )
    clip_attention_config = pipe.text_encoder.config

    reference_model = CLIPSdpaAttention(clip_attention_config).to(dtype=torch.bfloat16)

    reference_model.eval()

    hidden_states = torch.randn(1, 77, 768, dtype=torch.bfloat16)
    attention_mask = None
    causal_attention_mask = torch.randn(1, 1, 77, 77, dtype=torch.bfloat16)
    output_attentions = False

    parameters = preprocess_model_parameters(initialize_model=lambda: reference_model, device=device)

    torch_output = reference_model(hidden_states, attention_mask, causal_attention_mask, output_attentions)

    ttnn_model = ttnn_CLIPSdpaAttention(clip_attention_config)

    ttnn_hidden_states = ttnn.from_torch(
        hidden_states,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_causal_attention_mask = ttnn.from_torch(
        causal_attention_mask,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = ttnn_model(ttnn_hidden_states, None, ttnn_causal_attention_mask, False, parameters)

    assert_with_pcc(torch_output[0], ttnn.to_torch(ttnn_output[0]), pcc=0.99)
