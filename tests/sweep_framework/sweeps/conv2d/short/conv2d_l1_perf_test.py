# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import os
import itertools
import random
import torch

import ttnn
import pytest

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED

BF8 = ttnn.bfloat8_b
BFP16 = ttnn.bfloat16

TL = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT


# Suite 1: Height sharded cases with mostly BF8 and TL
all_test_cases_suite_1 = [
    # { batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, input_dtype, output_dtype, input_layout, output_layout, sharding_scheme }
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 16 - PASSING
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 17 - PASSING
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 22 - PASSING
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 23 - PASSING
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BFP16, TL, RM, HS),  # index 25 - PASSING
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 26 - PASSING
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 27 - PASSING
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, RM, HS),  # index 29 - PASSING
    # From test_conv_features (5x5 kernel variations)
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 34 - PASSING
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 35 - PASSING
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BFP16, TL, RM, HS),  # index 37 - PASSING
    # From test_conv_dilation - HEIGHT_SHARDED cases with different spatial dimensions
    (1, 32, 32, 64, 64, 3, 3, 1, 1, 2, 2, 2, 2, 1, BF8, BFP16, TL, TL, HS),  # index 38 - PASSING
    (1, 32, 32, 64, 64, 3, 3, 1, 1, 3, 3, 3, 3, 1, BF8, BFP16, TL, TL, HS),  # index 39 - PASSING
    (1, 32, 32, 128, 64, 3, 3, 1, 1, 2, 2, 2, 2, 1, BF8, BFP16, TL, TL, HS),  # index 40 - PASSING
    (1, 32, 32, 128, 64, 3, 3, 1, 1, 3, 3, 3, 3, 1, BF8, BFP16, TL, TL, HS),  # index 41 - PASSING
    # From test_conv_for_segformer_512x512 - SegFormer specific cases
    (1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 46 - PASSING
    (1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 47 - PASSING
    # Additional HEIGHT_SHARDED test cases from test_new_conv2d.py
    # From test_conv_dilation - more variations with different dilation and padding
    (1, 32, 32, 64, 64, 3, 3, 1, 1, 2, 2, 2, 2, 1, BF8, BF8, TL, TL, HS),
    (1, 32, 32, 64, 64, 3, 3, 1, 1, 3, 3, 3, 3, 1, BF8, BF8, TL, TL, HS),
    (1, 32, 32, 128, 64, 3, 3, 1, 1, 2, 2, 2, 2, 1, BF8, BF8, TL, TL, HS),
    (1, 32, 32, 128, 64, 3, 3, 1, 1, 3, 3, 3, 3, 1, BF8, BF8, TL, TL, HS),
    # From test_conv_features - additional variations
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    # From test_segformer - variations of efficient self-attention
    (1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 160, 160, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 160, 160, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    # Same cases as above but with different dtype combinations
    (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (2, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    # From test_halo_reshard_conv - adapted to HEIGHT_SHARDED
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BF8, BF8, TL, TL, HS),
    # From test_sd_conv adapted to HEIGHT_SHARDED
    (1, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (2, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (2, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    # From test_vanilla_unet adapted to HEIGHT_SHARDED
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 64, 64, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 128, 64, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 128, 128, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    # Additional combinations with varying data types
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 64, 64, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 128, 64, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 128, 128, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    # More variations with ROW_MAJOR_LAYOUT
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, RM, HS),
    # Kernel size 7x7 test cases
    (1, 3, 32, 512, 512, 7, 7, 4, 4, 3, 3, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 3, 32, 512, 512, 7, 7, 4, 4, 3, 3, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 3, 32, 512, 512, 7, 7, 2, 2, 3, 3, 1, 1, 1, BF8, BF8, TL, TL, HS),
    # More test cases with varied groups
    (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 128, BF8, BF8, TL, TL, HS),
    (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 256, BF8, BF8, TL, TL, HS),
    # Variations with same params but different dtypes
    (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 128, BFP16, BFP16, TL, TL, HS),
    (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 256, BFP16, BFP16, TL, TL, HS),
    # Mixed dtype combinations
    (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 128, BF8, BFP16, TL, TL, HS),
    (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 256, BF8, BFP16, TL, TL, HS),
    # Mixed dtype combinations (opposite)
    (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 128, BFP16, BF8, TL, TL, HS),
    (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 256, BFP16, BF8, TL, TL, HS),
    # More spatial dimensions with HEIGHT_SHARDED from test_unet_conv tests
    (1, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    # More variations with different dtypes
    (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 64, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 64, 192, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 64, 256, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 64, 320, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 128, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 128, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 128, 192, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 128, 256, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 128, 320, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 320, 320, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    # Mixed dtype combinations (opposite)
    (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 64, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 64, 192, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 64, 256, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 64, 320, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 128, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 128, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 128, 192, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 128, 256, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 128, 320, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 320, 320, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    # More variations with different spatial dimensions
    (1, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),
    (1, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    # More variations with different batch sizes and channels
    (2, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (2, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (2, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (2, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),
    (2, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (2, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    # More variations with different kernel sizes
    (1, 32, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 64, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 32, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 64, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 32, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),
    (1, 64, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),
    (1, 32, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 64, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    # # More variations with different strides
    (1, 32, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 64, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 32, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 64, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 32, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),
    (1, 64, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),
    (1, 32, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 64, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    # More variations with different padding
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, HS),
    # More variations with different groups
    (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BF8, BF8, TL, TL, HS),
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BF8, BF8, TL, TL, HS),
    (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BFP16, TL, TL, HS),
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BFP16, TL, TL, HS),
    (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BF8, BFP16, TL, TL, HS),
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BF8, BFP16, TL, TL, HS),
    (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BF8, TL, TL, HS),
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BF8, TL, TL, HS),
    # More variations with different output layouts
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, RM, HS),
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, RM, HS),
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, RM, HS),
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, RM, HS),
]

# Suite 2: Row major input and output layouts for HS

all_test_cases_suite_2 = [
    # Format: (batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, input_dtype, output_dtype, input_layout, output_layout, sharding_scheme)
    # Representative grouped convolutions covering different group values and configurations
    # Small groups (2-4)
    (1, 224, 224, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BFP16, RM, RM, HS),
    (1, 240, 240, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 2, BFP16, BFP16, RM, RM, HS),
    (1, 72, 72, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, 3, BFP16, BFP16, RM, RM, HS),
    (1, 192, 192, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 4, BFP16, BFP16, RM, RM, HS),
    # Medium groups (5-16)
    (1, 120, 120, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 5, BFP16, BFP16, RM, RM, HS),
    (1, 48, 48, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, 6, BFP16, BFP16, RM, RM, HS),
    (1, 168, 168, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 7, BFP16, BFP16, RM, RM, HS),
    (1, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 8, BFP16, BFP16, RM, RM, HS),
    (1, 144, 144, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 9, BFP16, BFP16, RM, RM, HS),
    (1, 160, 160, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 10, BFP16, BFP16, RM, RM, HS),
    (1, 16, 16, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 16, BFP16, BFP16, RM, RM, HS),
    # Large groups (20-64)
    (1, 20, 20, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 20, BFP16, BFP16, RM, RM, HS),
    (1, 24, 24, 56, 56, 5, 5, 2, 2, 2, 2, 1, 1, 24, BFP16, BFP16, RM, RM, HS),
    (1, 32, 32, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 32, BFP16, BFP16, RM, RM, HS),
    (1, 64, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, 64, BFP16, BFP16, RM, RM, HS),
    (1, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, BFP16, BFP16, RM, RM, HS),
    # Very large groups (128+)
    (1, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 128, BFP16, BFP16, RM, RM, HS),
    (1, 144, 144, 65, 65, 3, 3, 1, 1, 1, 1, 1, 1, 144, BFP16, BFP16, RM, RM, HS),
    (1, 240, 240, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 240, BFP16, BFP16, RM, RM, HS),
    # Different dilation values
    (1, 128, 128, 28, 28, 3, 3, 1, 1, 2, 2, 2, 2, 1, BFP16, BFP16, RM, RM, HS),  # dilation 2,2
    (1, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 3, 3, 1, BFP16, BFP16, RM, RM, HS),  # dilation 3,3
    # Different kernel sizes (no 1x1)
    (1, 128, 128, 60, 80, 4, 4, 4, 4, 0, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # 4x4 kernel stride 4
    (1, 64, 3, 224, 224, 7, 7, 2, 2, 3, 3, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # 7x7 kernel stride 2
    (1, 96, 96, 28, 28, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # 5x5 kernel
    (1, 80, 80, 64, 64, 6, 6, 2, 2, 3, 3, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # 6x6 kernel
    (1, 112, 112, 48, 48, 8, 8, 2, 2, 4, 4, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # 8x8 kernel
    # Asymmetric kernels (excluding 1xN)
    (1, 256, 256, 14, 14, 5, 1, 1, 1, 2, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # 5x1 kernel
    (1, 192, 192, 17, 17, 7, 1, 1, 1, 3, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 192, 192, 24, 24, 2, 5, 1, 1, 0, 2, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # 2x5 kernel
    # Different padding configurations
    (1, 64, 64, 10, 10, 3, 3, 1, 1, 0, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # no padding
    (1, 256, 256, 32, 32, 3, 3, 1, 1, 0, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # no padding
    (1, 32, 32, 30, 30, 3, 3, 1, 1, 0, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # no padding
    (1, 128, 128, 48, 48, 5, 5, 1, 1, 3, 3, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # large padding
    (1, 160, 160, 32, 32, 7, 7, 1, 1, 4, 4, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # very large padding
    # Large input sizes
    (1, 128, 128, 180, 320, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # large asymmetric input
    (1, 128, 128, 90, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # medium asymmetric input
    (1, 96, 96, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # very large square input
    # Different channel configurations
    (1, 40, 102, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # odd channels
    (1, 34, 118, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # small odd channels
    (1, 68, 142, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # medium odd channels
    # Small input sizes (no 1x1 kernels)
    (1, 128, 128, 5, 5, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # small 5x5
    (1, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # small 8x8
    (1, 192, 192, 6, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # small 6x6
    # ResNet patterns
    (1, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # ResNet layer1
    (1, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # ResNet layer2
    (1, 256, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # ResNet downsampling
    (1, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # Custom large input
    (1, 96, 96, 64, 64, 13, 13, 3, 3, 6, 6, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # 13x13 kernel
    (1, 34, 118, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 46, 122, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 128, 128, 100, 136, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 64, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 32, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 16, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 192, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 32, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 32, 128, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 288, 144, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 28, 144, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 144, 144, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 58, 152, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 48, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 16, 16, 224, 224, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 32, 16, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 46, 172, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 18, 18, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 48, 192, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 224, 192, 35, 35, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 48, 192, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 384, 192, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 48, 192, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 40, 196, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 78, 218, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 68, 236, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 64, 24, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 14, 24, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 256, 256, 45, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 256, 256, 50, 68, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 18, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 160, 272, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 34, 276, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 16, 28, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 134, 296, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 32, 3, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 116, 304, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 58, 310, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 64, 32, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 128, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 64, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 64, 32, 147, 147, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 64, 32, 150, 150, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 32, 32, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 64, 32, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 96, 32, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 32, 32, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 20, 34, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 36, 36, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 68, 360, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 98, 368, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 14, 40, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 116, 428, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 16, 46, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
    (1, 128, 48, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),
]


# Function to run conv2d withut tensors
def run_perf_benchmark(
    input_spec,
    device,
):
    [
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        input_dtype,
        output_dtype,
        input_layout,
        output_layout,
        sharding_scheme,
    ] = input_spec

    print(f"Running test case {input_spec}")
    branch = os.environ.get("BRANCH", "latest")
    if branch == "old":
        print("Running on old branch")
    else:
        print("Running on latest branch")

    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, kernel_height, kernel_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()

    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()

    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )

    tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, BFP16)
    tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, BFP16)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=input_layout, dtype=input_dtype, device=device)
    # Build config parameters dynamically
    config_params = {
        "shard_layout": sharding_scheme,
        "output_layout": output_layout,
    }

    # Add dtype only for old branch
    if branch == "old":
        config_params["dtype"] = output_dtype
    if branch == "latest" and sharding_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        print("enabling weights double buffer")
        config_params["enable_weights_double_buffer"] = True

    conv_config = ttnn.Conv2dConfig(**config_params)
    # Build conv2d parameters
    conv2d_params = {
        "input_tensor": tt_input_tensor,
        "weight_tensor": tt_weight_tensor,
        "in_channels": input_channels,
        "out_channels": output_channels,
        "device": device,
        "bias_tensor": tt_bias_tensor,
        "kernel_size": (kernel_height, kernel_width),
        "stride": (stride_h, stride_w),
        "padding": (pad_h, pad_w),
        "dilation": (dilation_h, dilation_w),
        "batch_size": batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "groups": groups,
        "conv_config": conv_config,
        "return_output_dim": True,
        "return_weights_and_bias": True,
    }

    # Add dtype only for latest branch
    if branch == "latest":
        conv2d_params["dtype"] = output_dtype

    [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(**conv2d_params)

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    print("End of test case")
    return check_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=0.998)


# Create parameters structure for pytest - 7 separate suites
parameters_suite_1 = {"input_specs": all_test_cases_suite_1}
parameters_suite_2 = {"input_specs": all_test_cases_suite_2}
# parameters_suite_3 = {"input_specs": all_test_cases_suite_3}
# parameters_suite_4 = {"input_specs": all_test_cases_suite_4}
# parameters_suite_5 = {"input_specs": all_test_cases_suite_5}
# parameters_suite_6 = {"input_specs": all_test_cases_suite_6}
# parameters_suite_7 = {"input_specs": all_test_cases_suite_7}

# # Combined list for reference
# all_test_cases = (all_test_cases_suite_1 + all_test_cases_suite_2 + all_test_cases_suite_3 +
#                  all_test_cases_suite_4 + all_test_cases_suite_5 + all_test_cases_suite_6 + all_test_cases_suite_7)


# Test functions for each suite
@pytest.mark.parametrize("input_spec", parameters_suite_1["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_l1_perf_suite_1(device, input_spec):
    run_perf_benchmark(input_spec, device)


@pytest.mark.parametrize("input_spec", parameters_suite_2["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_l1_perf_suite_2(device, input_spec):
    run_perf_benchmark(input_spec, device)


# @pytest.mark.parametrize("input_spec", parameters_suite_3["input_specs"])
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# def test_conv2d_l1_perf_suite_3(device, input_spec):
#     run_perf_benchmark(input_spec, device)

# @pytest.mark.parametrize("input_spec", parameters_suite_4["input_specs"])
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# def test_conv2d_l1_perf_suite_4(device, input_spec):
#     run_perf_benchmark(input_spec, device)

# @pytest.mark.parametrize("input_spec", parameters_suite_5["input_specs"])
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# def test_conv2d_l1_perf_suite_5(device, input_spec):
#     run_perf_benchmark(input_spec, device)

# @pytest.mark.parametrize("input_spec", parameters_suite_6["input_specs"])
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# def test_conv2d_l1_perf_suite_6(device, input_spec):
#     run_perf_benchmark(input_spec, device)

# @pytest.mark.parametrize("input_spec", parameters_suite_7["input_specs"])
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# def test_conv2d_l1_perf_suite_7(device, input_spec):
#     run_perf_benchmark(input_spec, device)

if __name__ == "__main__":
    """
    Example usage for performance testing
    """
    print(f"Conv2d Performance Test Suite - Split into 2 suites")
    print(f"")
    print(f"Suite 1: {len(all_test_cases_suite_1)} test cases (TILE input/output layouts)")
    print(f"Suite 2: {len(all_test_cases_suite_2)} test cases (ROW_MAJOR input/output layouts)")
