# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_upsample_nearest_2d import upsample_nearest2d
from tt_lib.fallback_ops import fallback_ops
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
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


class upsample2d:
    def __init__(self, device, parameters, reader_patterns_cache, batch_size, input_height, input_width):
        self.device = device
        self.parameters = parameters
        weight = ttnn.to_layout(parameters.conv.weight, layout=ttnn.ROW_MAJOR_LAYOUT)
        weight = ttnn.to_torch(weight)
        weight = torch.permute(weight, (2, 3, 0, 1))
        bias = ttnn.to_layout(parameters.conv.bias, layout=ttnn.ROW_MAJOR_LAYOUT)
        bias = ttnn.to_torch(bias)

        self.scale_factor = 2
        input_height = input_height * self.scale_factor
        input_width = input_width * self.scale_factor

        out_channels = weight.shape[0]
        in_channels = weight.shape[1]
        # breakpoint()
        bias = torch.reshape(bias, (1, 1, 1, out_channels))
        tt_weight_tensor = ttnn.from_torch(weight, ttnn.float32)
        tt_bias_tensor = ttnn.from_torch(bias, ttnn.float32)
        conv_config_override = {}
        if (out_channels, in_channels, input_height, input_width) in config_override:
            conv_config_override = config_override[(out_channels, in_channels, input_height, input_width)]
        self.conv = ttnn.Conv2d(
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
            # enable_auto_formatting=True,
            deallocate_activation=True,
        )
        self.output_height = self.conv.output_height
        self.output_width = self.conv.output_width

    def __call__(self, input, in_channels, out_channels):
        if input.layout == ttnn.TILE_LAYOUT:
            input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
        tt_out = upsample_nearest2d(input, self.scale_factor)
        del input
        tt_out = ttnn.reshape(tt_out, (1, 1, tt_out.shape[0] * tt_out.shape[1] * tt_out.shape[2], tt_out.shape[3]))
        if ttnn.get_memory_config(tt_out) != self.conv.conv.input_sharded_memory_config:
            tt_out = ttnn.to_memory_config(tt_out, self.conv.conv.input_sharded_memory_config)
        tt_out = self.conv(tt_out)
        # tt_out = run_ttnn_conv_with_pre_and_post_tensor_formatting(
        #     self.device,
        #     self.conv,
        #     tt_out,
        #     self.conv.batch_size,
        #     self.conv.input_height,
        #     self.conv.input_width,
        #     self.conv.out_channels,
        # )

        return tt_out
