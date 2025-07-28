# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tt.encoders.tt.tt_clip_mlp import TtClipMLP
from diffusers import DiffusionPipeline
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shape, encoder_id",
    [
        ((77, 768), 1),
        ((77, 1280), 2),
    ],
)
def test_mlp(device, input_shape, encoder_id, is_ci_env, reset_seeds):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    text_encoder = pipe.text_encoder if encoder_id == 1 else pipe.text_encoder_2
    text_encoder.eval()
    state_dict = text_encoder.state_dict()

    torch_mlp = text_encoder.text_model.encoder.layers[0].mlp

    model_config = ModelOptimisations()
    tt_mlp = TtClipMLP(
        device,
        state_dict,
        f"text_model.encoder.layers.0.mlp",
        model_config,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_mlp(torch_input_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output_tensor = tt_mlp.forward(ttnn_input_tensor)
    output_tensor = ttnn.to_torch(ttnn_output_tensor).squeeze()

    del pipe
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"PCC is {pcc_message}")
