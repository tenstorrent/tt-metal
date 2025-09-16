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
from models.common.utility_functions import torch_random

from loguru import logger


@torch.no_grad()
@pytest.mark.parametrize(
    "input_shape, pcc, vae_block",
    [
        ((1, 4, 128, 128), 0.89, "decoder"),
        ((1, 3, 1024, 1024), 0.967, "encoder"),
    ],
    ids=("test_decode", "test_encode"),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_vae(device, input_shape, vae_block, pcc, is_ci_env, reset_seeds, is_ci_v2_env, model_location_generator):
    model_location = model_location_generator(
        "stable-diffusion-xl-base-1.0/vae", download_if_ci_v2=True, ci_v2_timeout_in_s=1800
    )
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0" if not is_ci_v2_env else model_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env or is_ci_v2_env,
        subfolder="vae" if not is_ci_v2_env else None,
    )
    vae.eval()
    state_dict = vae.state_dict()

    logger.info("Loading weights to device")
    model_config = ModelOptimisations()
    tt_vae = TtAutoencoderKL(device, state_dict, model_config)
    logger.info("Loaded weights")
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)

    logger.info("Running reference model")
    if vae_block == "encoder":
        torch_output_tensor = vae.encode(torch_input_tensor, return_dict=False)[0]
    else:
        torch_output_tensor = vae.decode(torch_input_tensor, return_dict=False)[0]
    logger.info("Torch model done")

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    logger.info("Running TT model")
    if vae_block == "encoder":
        output_tensor = tt_vae.encode(ttnn_input_tensor, [B, C, H, W])

        output_tensor = output_tensor.latent_dist.sample()
        torch_output_tensor = torch_output_tensor.sample()
    else:
        output_tensor, [C, H, W] = tt_vae.decode(ttnn_input_tensor, [B, C, H, W])

        output_tensor = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)).float()
        output_tensor = output_tensor.reshape(B, H, W, C)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    logger.info("TT model done")

    del vae
    gc.collect()

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)
