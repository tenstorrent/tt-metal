# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.mobileNetV3.tt.ttnn_invertedResidual import InvertedResidualConfig
from functools import partial

reduce_divider = 1
dilation = 1

bneck_conf = partial(InvertedResidualConfig, width_mult=1.0)
adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=1.0)

# parameters for inverted_residual_setting:
#   input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation, input_height, input_width
inverted_residual_setting = [
    bneck_conf(16, 3, 16, 16, True, "RE", 2, 1, input_height=112, input_width=112),
    bneck_conf(16, 3, 72, 24, False, "RE", 2, 1, input_height=56, input_width=56),
    bneck_conf(24, 3, 88, 24, False, "RE", 1, 1, input_height=28, input_width=28),
    bneck_conf(24, 5, 96, 40, True, "HS", 2, 1, input_height=28, input_width=28),
    bneck_conf(40, 5, 240, 40, True, "HS", 1, 1, input_height=14, input_width=14),
    bneck_conf(40, 5, 240, 40, True, "HS", 1, 1, input_height=14, input_width=14),
    bneck_conf(40, 5, 120, 48, True, "HS", 1, 1, input_height=14, input_width=14),
    bneck_conf(48, 5, 144, 48, True, "HS", 1, 1, input_height=14, input_width=14),
    bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation, input_height=14, input_width=14),
    # fmt: off
    bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation, input_height=7, input_width=7),
    bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation, input_height=7, input_width=7),
    # fmt: on
]

last_channel = adjust_channels(1024 // reduce_divider)
