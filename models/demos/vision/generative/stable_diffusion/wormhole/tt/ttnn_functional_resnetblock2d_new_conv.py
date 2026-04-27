// File: models/demos/vision/generative/stable_diffusion/wormhole/tt/ttnn_functional_resnetblock2d_new_conv.py
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import torch

import ttnn
from models.demos.vision.generative.stable_diffusion.wormhole.sd_helper_funcs import (
    reshard_for_output_channels_divisibility,
)
from models.demos.vision.generative.stable_diffusion.wormhole.tt.ttnn_functional_utility_functions import (
    get_default_compute_config,
    permute_conv_parameters,
    weight_to_bfp8,
)

config_override = {
    (320, 320, 64, 64): {"act_block_h": 32 * 16},
    (640, 1920, 32, 32): {"act_block_h": 32 * 4},
    (640, 1280, 32, 32): {"act_block_h": 32 * 4},
    (320, 960, 64, 64): {"act_block_h": 32 * 4},
    (320, 640, 64, 64): {"act_block_h": 32 * 8},
    (640, 640, 32, 32): {"act_block_h": 32 * 4},
}