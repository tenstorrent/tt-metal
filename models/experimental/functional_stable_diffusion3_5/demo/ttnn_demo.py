# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.integration_tests.stable_diffusion3_5.test_ttnn_sd3_transformer_2d_model import (
    create_custom_preprocessor,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_sd3_transformer_2d_model import (
    ttnn_SD3Transformer2DModel,
)

from models.experimental.functional_stable_diffusion3_5.demo.ttnn_pipeline import ttnnStableDiffusion3Pipeline
from diffusers import StableDiffusion3Pipeline

import ttnn
import torch
import pytest
from models.experimental.functional_stable_diffusion3_5.reference.sd3_transformer_2d_model import SD3Transformer2DModel


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_demo(device, reset_seeds):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )
    config = pipe.transformer.config

    reference_model = SD3Transformer2DModel(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=24,
        attention_head_dim=64,
        num_attention_heads=24,
        joint_attention_dim=4096,
        caption_projection_dim=1536,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=384,
        dual_attention_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        qk_norm="rms_norm",
        config=config,
    )
    reference_model.load_state_dict(pipe.transformer.state_dict())
    reference_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    parameters["pos_embed"]["proj"]["weight"] = ttnn.from_device(parameters["pos_embed"]["proj"]["weight"])
    parameters["pos_embed"]["proj"]["bias"] = ttnn.from_device(parameters["pos_embed"]["proj"]["bias"])

    ttnn_model = ttnn_SD3Transformer2DModel(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=24,
        attention_head_dim=64,
        num_attention_heads=24,
        joint_attention_dim=4096,
        caption_projection_dim=1536,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=384,
        dual_attention_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        qk_norm="rms_norm",
        config=config,
        parameters=parameters,
    )

    ttnn_pipe = ttnnStableDiffusion3Pipeline(
        ttnn_model, pipe.scheduler, pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.text_encoder_2, pipe.tokenizer_2
    )
    # image = pipe(
    #     "A capybara holding a sign that reads Hello World",
    #     num_inference_steps=1,
    #     guidance_scale=4.5,
    # ).images[0]
    # image.save("capybara_without_tokenizer3_iteration1.png")

    image = ttnn_pipe(
        "A capybara holding a sign that reads Hello World",
        num_inference_steps=40,
        guidance_scale=4.5,
        parameters_transformer=parameters,
        device_ttnn=device,
    ).images[0]
    image.save("ttnn_capybara.png")
