# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_upsample_nearest_2d import upsample_nearest2d
from tt_lib.fallback_ops import fallback_ops
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_utility_functions import (
    run_ttnn_conv_with_pre_and_post_tensor_formatting,
)

config_override = {
    (320, 320, 64, 64): {"act_block_h": 64},
    (640, 640, 32, 32): {"act_block_h": 64},
    (640, 1920, 32, 32): {"act_block_h": 32},
    (640, 1280, 32, 32): {"act_block_h": 32},
    (1280, 1920, 16, 16): {"act_block_h": 32},
    (1280, 1280, 32, 32): {"act_block_h": 32},
    (320, 960, 64, 64): {"act_block_h": 32},
    (640, 960, 32, 32): {"act_block_h": 32},
    (320, 640, 64, 64): {"act_block_h": 32},
    (640, 640, 64, 64): {"act_block_h": 64},
}


def upsample2d(device, input, parameters, in_channels, out_channels, scale_factor=2.0, reader_patterns_cache=None):
    conv_on_device = reader_patterns_cache is not None
    tt_out = upsample_nearest2d(input, scale_factor)

    weight = ttnn.to_layout(parameters.conv.weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_layout(parameters.conv.bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)
    if conv_on_device:
        batch_size = tt_out.shape[0]
        input_height = tt_out.shape[2]
        input_width = tt_out.shape[3]
        out_channels = weight.shape[0]
        in_channels = weight.shape[1]
        # breakpoint()
        bias = torch.reshape(bias, (1, 1, 1, out_channels))
        tt_weight_tensor = ttnn.from_torch(weight, ttnn.float32)
        tt_bias_tensor = ttnn.from_torch(bias, ttnn.float32)
        conv_config_override = {}
        if (out_channels, in_channels, input_height, input_width) in config_override:
            conv_config_override = config_override[(out_channels, in_channels, input_height, input_width)]
        conv = ttnn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dtype=ttnn.bfloat8_b,
            device=device,
            use_1d_systolic_array=False,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=tt_weight_tensor,
            bias=tt_bias_tensor,
            math_fidelity=ttnn.MathFidelity.LoFi,
            weights_dtype=ttnn.bfloat8_b,
            conv_blocking_and_parallelization_config_override=conv_config_override,
            use_shallow_conv_variant=False,
            enable_auto_formatting=True,
            deallocate_activation=True,
        )
        tt_out = run_ttnn_conv_with_pre_and_post_tensor_formatting(
            device, conv, tt_out, batch_size, input_height, input_width, out_channels
        )
    else:
        tt_out = ttnn.from_device(tt_out)
        tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
        tt_out = ttnn.to_torch(tt_out)
        tt_out = torch_to_tt_tensor_rm(tt_out, device)
        weight = torch_to_tt_tensor_rm(weight, device, put_on_device=False)
        bias = torch_to_tt_tensor_rm(bias, device, put_on_device=False)

        conv = fallback_ops.Conv2d(
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        tt_out = conv(tt_out)
        torch_out = tt_to_torch_tensor(tt_out)
        tt_out = ttnn.from_torch(torch_out, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    return tt_out
