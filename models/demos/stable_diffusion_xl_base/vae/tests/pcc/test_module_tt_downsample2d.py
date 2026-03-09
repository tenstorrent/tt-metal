# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch
from diffusers import AutoencoderKL
from loguru import logger

import ttnn
from models.common.utility_functions import torch_random, is_blackhole
from models.demos.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE
from models.demos.stable_diffusion_xl_base.tt.sdxl_utility import from_channel_last_ttnn, to_channel_last_ttnn
from models.demos.stable_diffusion_xl_base.vae.tt.model_configs import load_vae_model_optimisations
from models.demos.stable_diffusion_xl_base.vae.tt.tt_downsample2d import TtDownsample2D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, down_block_id",
    [
        # 1024x1024 image resolution
        ((1024, 1024), (1, 128, 1024, 1024), 0),
        ((1024, 1024), (1, 256, 512, 512), 1),
        ((1024, 1024), (1, 512, 256, 256), 2),
        # 512x512 image resolution
        ((512, 512), (1, 128, 512, 512), 0),
        ((512, 512), (1, 256, 256, 256), 1),
        ((512, 512), (1, 512, 128, 128), 2),
    ],
)
@pytest.mark.parametrize("stride", [(2, 2)])
@pytest.mark.parametrize("padding", [(0, 1, 0, 1)])
@pytest.mark.parametrize("dilation", [(1, 1)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_downsample2d(
    device,
    image_resolution,
    input_shape,
    down_block_id,
    stride,
    padding,
    dilation,
    debug_mode,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_vae_location,
    reset_seeds,
):
    if image_resolution == (512, 512) and is_blackhole():
        pytest.skip("512x512 not supported on Blackhole")
    vae = AutoencoderKL.from_pretrained(
        sdxl_base_vae_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
        subfolder=None if is_ci_v2_env else "vae",
    )
    vae.eval()
    state_dict = vae.state_dict()

    torch_downsample = vae.encoder.down_blocks[down_block_id].downsamplers[0]
    groups = 1

    model_config = load_vae_model_optimisations(image_resolution)
    tt_downsample = TtDownsample2D(
        device,
        state_dict,
        f"encoder.down_blocks.{down_block_id}.downsamplers.0",
        stride,
        padding,
        dilation,
        groups,
        model_config=model_config,
        debug_mode=debug_mode,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_downsample(torch_input_tensor)

    ttnn_input_tensor = to_channel_last_ttnn(
        torch_input_tensor, ttnn.bfloat16, device, ttnn.DRAM_MEMORY_CONFIG, ttnn.TILE_LAYOUT
    )

    ttnn_output_tensor, output_shape = tt_downsample.forward(ttnn_input_tensor, input_shape)

    output_tensor = from_channel_last_ttnn(
        ttnn_output_tensor, [input_shape[0], output_shape[1], output_shape[2], output_shape[0]]
    )

    del vae, tt_downsample
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"PCC is {pcc_message}")
