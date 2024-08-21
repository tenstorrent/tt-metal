# # SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# # SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from torch import nn
from diffusers import StableDiffusionPipeline

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import torch_random
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_downsample_2d import downsample_2d
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    pre_process_input,
    post_process_output,
)


@skip_for_grayskull()
@skip_for_wormhole_b0(reason_str="#10923: Hangs on init")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index",
    [
        (2, 320, 64, 64, 0),
        (2, 640, 32, 32, 1),
        (2, 1280, 16, 16, 2),
    ],
)
def test_downsample_2d_512x512(device, model_name, batch_size, in_channels, input_height, input_width, index):
    input_shape = batch_size, in_channels, input_height, input_width
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    unet_downblock = pipe.unet.down_blocks[index]
    ref_model = unet_downblock.downsamplers[0]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.down_blocks[index].downsamplers[0]

    torch_hidden_states = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)

    torch_output = ref_model(torch_hidden_states)

    ttnn_hidden_state = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.ROW_MAJOR_LAYOUT
    )
    reader_patterns_cache = {}

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    model = downsample_2d(
        device, parameters, reader_patterns_cache, batch_size, input_height, input_width, compute_kernel_config
    )
    ttnn_hidden_state = pre_process_input(device, ttnn_hidden_state)
    ttnn_output = model(
        in_channels=in_channels,
        hidden_states=ttnn_hidden_state,
        use_conv=True,
        out_channels=in_channels,
        padding=1,
    )

    ttnn_output = post_process_output(device, ttnn_output, batch_size, input_height // 2, input_width // 2, in_channels)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)
