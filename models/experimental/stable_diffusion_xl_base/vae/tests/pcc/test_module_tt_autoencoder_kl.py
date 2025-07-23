# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from diffusers import AutoencoderKL
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random

from loguru import logger


@torch.no_grad()
@pytest.mark.parametrize(
    "input_shape, pcc",
    [
        ((1, 4, 128, 128), 0.89),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_vae(device, input_shape, pcc, is_ci_env, reset_seeds):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        subfolder="vae",
        local_files_only=is_ci_env,
    )
    vae.eval()
    state_dict = vae.state_dict()

    logger.info("Loading weights to device")
    model_config = ModelOptimisations()
    tt_vae = TtAutoencoderKL(device, state_dict, model_config)
    logger.info("Loaded weights")
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)

    logger.info("Running reference model")
    torch_output_tensor = vae.decode(torch_input_tensor, return_dict=False)[0]
    logger.info("Torch model done")

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    logger.info("Running TT model")
    output_tensor = tt_vae.forward(ttnn_input_tensor, [B, C, H, W])
    logger.info("TT model done")

    del vae
    gc.collect()

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)
