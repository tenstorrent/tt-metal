# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_downblock2d import TtDownEncoderBlock2D
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from diffusers import AutoencoderKL
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, block_id, pcc",
    [
        ((1, 128, 1024, 1024), 0, 0.998),
        ((1, 128, 512, 512), 1, 0.996),
        ((1, 256, 256, 256), 2, 0.999),
        ((1, 512, 128, 128), 3, 0.999),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_downblock2d(device, block_id, input_shape, pcc, is_ci_env, reset_seeds):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        subfolder="vae",
        local_files_only=is_ci_env,
    )
    vae.eval()
    state_dict = vae.state_dict()

    torch_downblock = vae.encoder.down_blocks[block_id]

    model_config = ModelOptimisations()
    tt_downblock = TtDownEncoderBlock2D(
        device,
        state_dict,
        f"encoder.down_blocks.{block_id}",
        model_config=model_config,
        has_downsample=block_id < 3,
        has_shortcut=block_id > 0 and block_id < 3,
    )
    logger.info("Loaded weights")

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_downblock(torch_input_tensor)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    ttnn_output_tensor, output_shape = tt_downblock.forward(ttnn_input_tensor, [B, C, H, W])

    output_tensor = ttnn.to_torch(ttnn_output_tensor)

    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del vae, tt_downblock
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")
