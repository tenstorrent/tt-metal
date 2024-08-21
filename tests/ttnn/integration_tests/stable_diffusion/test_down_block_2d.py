# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from diffusers import StableDiffusionPipeline

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0

from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor

from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_downblock_2d import downblock2d
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    pre_process_input,
    post_process_output,
)


@skip_for_grayskull()
@skip_for_wormhole_b0(reason_str="#10923: Hangs on init")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 1280, 8, 8),
    ],
)
@pytest.mark.parametrize(
    "temb_shape",
    [
        (1, 1, 2, 1280),
    ],
)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
def test_down_block_2d_512x512(input_shape, temb_shape, device, model_name, reset_seeds):
    torch.manual_seed(0)
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    unet_downblock = pipe.unet.down_blocks[3]
    unet_resnet_downblock_module_list = unet_downblock.resnets

    N, in_channels, H, W = input_shape
    _, _, _, temb_channels = temb_shape

    torch_hidden_states = torch.randn(input_shape, dtype=torch.float32)
    temb = torch.randn(temb_shape, dtype=torch.float32)

    torch_output, torch_output_states = unet_downblock(torch_hidden_states, temb.squeeze(0).squeeze(0))

    ttnn_hidden_states = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16), device),
        layout=ttnn.TILE_LAYOUT,
    )
    temb = temb.permute(2, 0, 1, 3)  # pre-permute temb
    ttnn_temb = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(temb, dtype=ttnn.bfloat16), device),
        layout=ttnn.TILE_LAYOUT,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    parameters = parameters.down_blocks[3]
    reader_patterns_cache = {}
    model = downblock2d(device, parameters, reader_patterns_cache, N, H, W, compute_kernel_config)
    ttnn_hidden_states = pre_process_input(device, ttnn_hidden_states)
    ttnn_out, ttnn_output_states = model(
        temb=ttnn_temb,
        hidden_states=ttnn_hidden_states,
        in_channels=in_channels,
        out_channels=in_channels,
        temb_channels=temb_channels,
        dropout=0.0,
        num_layers=2,
        resnet_eps=1e-05,
        resnet_time_scale_shift="default",
        resnet_act_fn="silu",
        resnet_groups=32,
        resnet_pre_norm=True,
        output_scale_factor=1.0,
        add_downsample=False,
        downsample_padding=1,
    )

    ttnn_output = post_process_output(device, ttnn_out, N, H, W, in_channels)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output_torch, 0.97)
