# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_upsample2d import TtUpsample2D
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from diffusers import AutoencoderKL
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    to_channel_last_ttnn,
    from_channel_last_ttnn,
)


@pytest.mark.parametrize(
    "input_shape, up_block_id", [((1, 512, 128, 128), 0), ((1, 512, 256, 256), 1), ((1, 256, 512, 512), 2)]
)
@pytest.mark.parametrize("stride", [(1, 1)])
@pytest.mark.parametrize("padding", [(1, 1)])
@pytest.mark.parametrize("dilation", [(1, 1)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 16384}], indirect=True)
def test_vae_upsample2d(device, input_shape, up_block_id, stride, padding, dilation, use_program_cache, reset_seeds):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, subfolder="vae"
    )
    vae.eval()
    state_dict = vae.state_dict()

    torch_upsample = vae.decoder.up_blocks[up_block_id].upsamplers[0]
    groups = 1

    model_config = ModelOptimisations()
    tt_upsample = TtUpsample2D(
        device,
        state_dict,
        f"decoder.up_blocks.{up_block_id}.upsamplers.0",
        model_config,
        stride,
        padding,
        dilation,
        groups,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_upsample(torch_input_tensor)

    ttnn_input_tensor = to_channel_last_ttnn(
        torch_input_tensor, ttnn.bfloat16, device, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn_output_tensor, output_shape = tt_upsample.forward(ttnn_input_tensor)
    output_tensor = from_channel_last_ttnn(
        ttnn_output_tensor, [input_shape[0], output_shape[1], output_shape[2], output_shape[0]]
    )

    del vae
    gc.collect()

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
