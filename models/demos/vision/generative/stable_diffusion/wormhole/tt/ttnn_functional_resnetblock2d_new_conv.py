// File: models/demos/vision/generative/stable_diffusion/wormhole/tt/ttnn_functional_resnetblock2d_new_conv.py
# SPDX-FileCopyrightText: 2024 Tenstorrent Inc.

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

# This file is a test file for model stable diffusion 1.4.
# No changes are required for the stable diffusion 1.4 model.
# Any modifications should be scoped to the relevant model and issue.
# Refer to the specific model's requirements before making changes to this test file.

config_override = {}