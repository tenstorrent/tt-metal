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
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 0
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 1
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 2
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 3
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BFP16, TL, RM, HS),  # index 4
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, RM, HS),  # index 5
    # From test_conv_features (5x5 kernel variations)
    # From test_conv_dilation - HEIGHT_SHARDED cases with different spatial dimensions
    (1, 32, 32, 64, 64, 3, 3, 1, 1, 2, 2, 2, 2, 1, BF8, BFP16, TL, TL, HS),  # index 6
    (1, 32, 32, 64, 64, 3, 3, 1, 1, 3, 3, 3, 3, 1, BF8, BFP16, TL, TL, HS),  # index 7
    (1, 32, 32, 128, 64, 3, 3, 1, 1, 2, 2, 2, 2, 1, BF8, BFP16, TL, TL, HS),  # index 8
    (1, 32, 32, 128, 64, 3, 3, 1, 1, 3, 3, 3, 3, 1, BF8, BFP16, TL, TL, HS),  # index 9
    # From test_conv_for_segformer_512x512 - SegFormer specific cases
    (1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 10
    (1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 11
    # Additional HEIGHT_SHARDED test cases from test_new_conv2d.py
    # From test_conv_dilation - more variations with different dilation and padding
    (1, 32, 32, 64, 64, 3, 3, 1, 1, 2, 2, 2, 2, 1, BF8, BF8, TL, TL, HS),  # index 12
    (1, 32, 32, 64, 64, 3, 3, 1, 1, 3, 3, 3, 3, 1, BF8, BF8, TL, TL, HS),  # index 13
    (1, 32, 32, 128, 64, 3, 3, 1, 1, 2, 2, 2, 2, 1, BF8, BF8, TL, TL, HS),  # index 14
    (1, 32, 32, 128, 64, 3, 3, 1, 1, 3, 3, 3, 3, 1, BF8, BF8, TL, TL, HS),  # index 15
    # From test_conv_features - additional variations
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 16
    (2, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 17
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 18
    (2, 16, 16, 256, 256, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 19
    # From test_segformer - variations of efficient self-attention
    (1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 20
    (1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 21
    (1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 22
    (1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 23
    (1, 160, 160, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 24
    (1, 160, 160, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 25
    # Same cases as above but with different dtype combinations
    (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 26
    (2, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 27
    # From test_halo_reshard_conv - adapted to HEIGHT_SHARDED
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 28
    (1, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 29
    (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BF8, BF8, TL, TL, HS),  # index 30
    # From test_sd_conv adapted to HEIGHT_SHARDED
    (1, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 31
    (2, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 32
    # From test_vanilla_unet adapted to HEIGHT_SHARDED
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 33
    # (1, 64, 64, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS), # index 34
    (1, 128, 64, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 34
    (1, 128, 128, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 35
    # Additional combinations with varying data types
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 36
    # (1, 64, 64, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS), # index 37
    (1, 128, 64, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 37
    (1, 128, 128, 120, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 38
    # More variations with ROW_MAJOR_LAYOUT
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, RM, HS),  # index 39
    # Kernel size 7x7 test cases
    (1, 3, 32, 512, 512, 7, 7, 4, 4, 3, 3, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 40
    (1, 3, 32, 512, 512, 7, 7, 4, 4, 3, 3, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 41
    # (1, 3, 32, 512, 512, 7, 7, 2, 2, 3, 3, 1, 1, 1, BF8, BF8, TL, TL, HS), # index 42
    # More test cases with varied groups
    (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 128, BF8, BF8, TL, TL, HS),  # index 42
    (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 256, BF8, BF8, TL, TL, HS),  # index 43
    # Variations with same params but different dtypes
    (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 128, BFP16, BFP16, TL, TL, HS),  # index 44
    (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 256, BFP16, BFP16, TL, TL, HS),  # index 45
    # Mixed dtype combinations
    (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 128, BF8, BFP16, TL, TL, HS),  # index 46
    (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 256, BF8, BFP16, TL, TL, HS),  # index 47
    # Mixed dtype combinations (opposite)
    (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 128, BFP16, BF8, TL, TL, HS),  # index 48
    (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 256, BFP16, BF8, TL, TL, HS),  # index 49
    # More spatial dimensions with HEIGHT_SHARDED from test_unet_conv tests
    (1, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 50
    (1, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 51
    (1, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 52
    (1, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 53
    (1, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 54
    # More variations with different dtypes
    (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 55
    (1, 64, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 56
    (1, 64, 192, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 57
    (1, 64, 256, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 58
    (1, 64, 320, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 59
    (1, 128, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 60
    (1, 128, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 61
    (1, 128, 192, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 62
    (1, 128, 256, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 63
    (1, 128, 320, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 64
    # Mixed dtype combinations (opposite)
    (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 65
    (1, 64, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 66
    (1, 64, 192, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 67
    (1, 64, 256, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 68
    (1, 64, 320, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 69
    (1, 128, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 70
    (1, 128, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 71
    (1, 128, 192, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 72
    (1, 128, 256, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 73
    (1, 128, 320, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 74
    # More variations with different spatial dimensions
    (1, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 75
    (1, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 76
    (1, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 77
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 78
    (1, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 79
    (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 80
    # More variations with different kernel sizes
    (1, 32, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 81
    (1, 64, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 82
    (1, 32, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 83
    (1, 64, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 84
    (1, 32, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 85
    (1, 64, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 86
    (1, 32, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 87
    (1, 64, 32, 128, 128, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 88
    # # More variations with different strides
    (1, 32, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 89
    (1, 64, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 90
    (1, 32, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 91
    (1, 64, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 92
    (1, 32, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 93
    (1, 64, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 94
    (1, 32, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 95
    (1, 64, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 96
    # More variations with different padding
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 97
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 98
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 99
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 100
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 101
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 102
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 103
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 104
    # More variations with different groups
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BF8, BF8, TL, TL, HS),  # index 105
    (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BFP16, TL, TL, HS),  # index 106
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BFP16, TL, TL, HS),  # index 107
    (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BF8, BFP16, TL, TL, HS),  # index 108
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BF8, BFP16, TL, TL, HS),  # index 109
    (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BF8, TL, TL, HS),  # index 110
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BF8, TL, TL, HS),  # index 111
    # More variations with different output layouts
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, RM, HS),  # index 112
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, RM, HS),  # index 113
    (1, 32, 32, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, RM, HS),  # index 114
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, RM, HS),  # index 115
    ###############################################################################################
    #########################suite 2 test cases#################################################
    (1, 224, 224, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BFP16, RM, RM, HS),  # index 116
    (1, 240, 240, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 2, BFP16, BFP16, RM, RM, HS),  # index 117
    (1, 72, 72, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, 3, BFP16, BFP16, RM, RM, HS),  # index 118
    (1, 192, 192, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 4, BFP16, BFP16, RM, RM, HS),  # index 119
    # Medium groups (5-16)
    (1, 120, 120, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 5, BFP16, BFP16, RM, RM, HS),  # index 120
    (1, 48, 48, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, 6, BFP16, BFP16, RM, RM, HS),  # index 121
    (1, 168, 168, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 7, BFP16, BFP16, RM, RM, HS),  # index 122
    (1, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 8, BFP16, BFP16, RM, RM, HS),  # index 123
    (1, 144, 144, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 9, BFP16, BFP16, RM, RM, HS),  # index 124
    (1, 160, 160, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 10, BFP16, BFP16, RM, RM, HS),  # index 125
    (1, 16, 16, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 16, BFP16, BFP16, RM, RM, HS),  # index 126
    # Large groups (20-64)
    (1, 20, 20, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 20, BFP16, BFP16, RM, RM, HS),  # index 127
    (1, 24, 24, 56, 56, 5, 5, 2, 2, 2, 2, 1, 1, 24, BFP16, BFP16, RM, RM, HS),  # index 128
    (1, 32, 32, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 32, BFP16, BFP16, RM, RM, HS),  # index 129
    (1, 64, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, 64, BFP16, BFP16, RM, RM, HS),  # index 130
    (1, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 64, BFP16, BFP16, RM, RM, HS),  # index 131
    # Very large groups (128+)
    (1, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 128, BFP16, BFP16, RM, RM, HS),  # index 132
    (1, 144, 144, 65, 65, 3, 3, 1, 1, 1, 1, 1, 1, 144, BFP16, BFP16, RM, RM, HS),  # index 133
    (1, 240, 240, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 240, BFP16, BFP16, RM, RM, HS),  # index 134
    # Different dilation values
    (1, 128, 128, 28, 28, 3, 3, 1, 1, 2, 2, 2, 2, 1, BFP16, BFP16, RM, RM, HS),  # index 135
    (1, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 3, 3, 1, BFP16, BFP16, RM, RM, HS),  # index 136
    # Different kernel sizes (no 1x1)
    (1, 128, 128, 60, 80, 4, 4, 4, 4, 0, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 137
    (1, 64, 3, 224, 224, 7, 7, 2, 2, 3, 3, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 138
    (1, 96, 96, 28, 28, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 139
    (1, 80, 80, 64, 64, 6, 6, 2, 2, 3, 3, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 140
    (1, 112, 112, 48, 48, 8, 8, 2, 2, 4, 4, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 141
    # Asymmetric kernels (excluding 1xN)
    (1, 256, 256, 14, 14, 5, 1, 1, 1, 2, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 142
    (1, 192, 192, 17, 17, 7, 1, 1, 1, 3, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 143
    (1, 192, 192, 24, 24, 2, 5, 1, 1, 0, 2, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 144
    # Different padding configurations
    (1, 64, 64, 10, 10, 3, 3, 1, 1, 0, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 145
    (1, 256, 256, 32, 32, 3, 3, 1, 1, 0, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 146
    (1, 32, 32, 30, 30, 3, 3, 1, 1, 0, 0, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 147
    (1, 128, 128, 48, 48, 5, 5, 1, 1, 3, 3, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 148
    (1, 160, 160, 32, 32, 7, 7, 1, 1, 4, 4, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 149
    # Large input sizes
    (1, 128, 128, 90, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 150
    (1, 96, 96, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 151
    # Different channel configurations
    (1, 40, 102, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 152
    (1, 34, 118, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 153
    (1, 68, 142, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 154
    # Small input sizes (no 1x1 kernels)
    (1, 128, 128, 5, 5, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 155
    (1, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 156
    (1, 192, 192, 6, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 157
    # ResNet patterns
    (1, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 158
    (1, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 159
    (1, 256, 128, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 160
    (1, 32, 32, 240, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 161
    (1, 96, 96, 64, 64, 13, 13, 3, 3, 6, 6, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 162
    (1, 46, 122, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 163
    (1, 128, 128, 100, 136, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 164
    (1, 64, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 165
    (1, 32, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 166
    (1, 16, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 167
    (1, 192, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 168
    (1, 32, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 169
    (1, 32, 128, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 170
    (1, 288, 144, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 171
    (1, 28, 144, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 172
    (1, 144, 144, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 173
    (1, 58, 152, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 174
    (1, 48, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 175
    (1, 16, 16, 224, 224, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 176
    (1, 32, 16, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 177
    (1, 46, 172, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 178
    (1, 18, 18, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 179
    (1, 48, 192, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 180
    (1, 224, 192, 35, 35, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 181
    (1, 48, 192, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 182
    (1, 384, 192, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 183
    (1, 48, 192, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 184
    (1, 40, 196, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 185
    (1, 78, 218, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 186
    (1, 68, 236, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 187
    (1, 64, 24, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 188
    (1, 14, 24, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 189
    (1, 256, 256, 45, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 190
    (1, 256, 256, 50, 68, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 191
    (1, 18, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 192
    (1, 160, 272, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 193
    (1, 34, 276, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 194
    (1, 16, 28, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 195
    (1, 134, 296, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 196
    (1, 32, 3, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 197
    (1, 116, 304, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 198
    (1, 58, 310, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 199
    (1, 64, 32, 112, 112, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 200
    (1, 64, 32, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 201
    (1, 128, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 202
    (1, 64, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 203
    (1, 64, 32, 147, 147, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 204
    (1, 64, 32, 150, 150, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 205
    (1, 32, 32, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 206
    (1, 64, 32, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 207
    (1, 96, 32, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 208
    (1, 32, 32, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 209
    (1, 20, 34, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 210
    (1, 36, 36, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 211
    (1, 68, 360, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 212
    (1, 98, 368, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 213
    (1, 14, 40, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 214
    (1, 116, 428, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 215
    (1, 16, 46, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 216
    (1, 128, 48, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 217
    #############################################################################################
    #########################suite 3 test cases##################################################
    #########################block sharded cases#################################################
    (2, 128, 128, 32, 32, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BF8, TL, TL, BS),  # index 218
    (2, 128, 128, 32, 32, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 219
    # From test_conv_dilation - explicit BLOCK_SHARDED with dilation
    (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, BS),  # index 220
    (2, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 221
    (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 222
    (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 223
    (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, BS),  # index 224
    (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 225
    (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 226
    (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, BS),  # index 227
    (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, BS),  # index 228
    (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 229
    (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 230
    (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 231
    (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 232
    (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 233
    (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 234
    (2, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 235
    # From test_sd_conv - Stable Diffusion cases
    (1, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, BS),  # index 236
    (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, BS),  # index 237
    (1, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, BS),  # index 238
    (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 239
    # #############################################################################################
    (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 240
    (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 241
    (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 242
    (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 243
    (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 244
    (1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 245
    (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 246
    (8, 256, 256, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 247
    ###############################################################################################
    #########################Regression test cases#################################################
    (2, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, BS),  # index 248
    (2, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, BS),  # index 249
    (2, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 250
    (2, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 251
    (2, 128, 128, 32, 32, 5, 5, 2, 2, 2, 2, 1, 1, 1, BF8, BFP16, TL, TL, BS),  # index 252
    (2, 128, 128, 32, 32, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 253
    # From test_conv_features_multi_device - explicit BLOCK_SHARDED with groups
    (2, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 2, BF8, BF8, TL, TL, BS),  # index 254
    (2, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 2, BF8, BFP16, TL, TL, BS),  # index 255
    (2, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 2, BFP16, BF8, TL, TL, BS),  # index 256
    (2, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 2, BFP16, BFP16, TL, TL, BS),  # index 257
    # From test_conv_dilation - explicit BLOCK_SHARDED with dilation
    (1, 128, 128, 8, 8, 3, 3, 1, 1, 2, 2, 2, 2, 1, BFP16, BFP16, TL, TL, BS),  # index 258
    (1, 128, 128, 8, 8, 3, 3, 1, 1, 3, 3, 3, 3, 1, BFP16, BFP16, TL, TL, BS),  # index 259
    (1, 128, 128, 16, 16, 3, 3, 1, 1, 2, 2, 2, 2, 1, BFP16, BFP16, TL, TL, BS),  # index 260
    (1, 128, 128, 16, 16, 3, 3, 1, 1, 3, 3, 3, 3, 1, BFP16, BFP16, TL, TL, BS),  # index 261
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 2, 2, 2, 2, 1, BFP16, BFP16, TL, TL, BS),  # index 262
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 3, 3, 3, 3, 1, BFP16, BFP16, TL, TL, BS),  # index 263
    # From test_halo_reshard_conv - BLOCK_SHARDED
    (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 264
    (1, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 265
    (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, BS),  # index 266
    # From test_sd_conv_wh - Stable Diffusion cases
    (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 267
    (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 268
    (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 269
    (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 270
    (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 271
    (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 272
    (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, BS),  # index 273
    #################################################################################################
    # Suite 4: WIDTH_SHARDED test cases from test_new_conv2d.py
    # From test_conv_features - Basic WIDTH_SHARDED test cases with 3x3 kernels
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, WS),  # index 274
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 275
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, RM, TL, WS),  # index 276
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, RM, TL, WS),  # index 277
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, RM, WS),  # index 278
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, WS),  # index 279
    # From test_conv_features - 5x5 kernels with WIDTH_SHARDED
    (2, 256, 256, 8, 8, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, WS),  # index 280
    (2, 256, 256, 8, 8, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 281
    (2, 256, 256, 8, 8, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BF8, RM, TL, WS),  # index 282
    (2, 256, 256, 8, 8, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BFP16, RM, TL, WS),  # index 283
    (2, 256, 256, 8, 8, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BFP16, TL, RM, WS),  # index 284
    (2, 256, 256, 8, 8, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BFP16, RM, RM, WS),  # index 285
    # From test_conv_features_multi_device - WIDTH_SHARDED with different groups
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 2, BFP16, BF8, TL, TL, WS),  # index 286
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 2, BFP16, BFP16, TL, TL, WS),  # index 287
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 2, BFP16, BF8, RM, TL, WS),  # index 288
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 2, BFP16, BFP16, RM, TL, WS),  # index 289
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 2, BFP16, BFP16, TL, RM, WS),  # index 290
    (2, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 2, BFP16, BFP16, RM, RM, WS),  # index 291
    # From test_conv_dilation - WIDTH_SHARDED with dilation=2
    (1, 768, 768, 16, 16, 3, 3, 1, 1, 2, 2, 2, 2, 1, BFP16, BFP16, TL, TL, WS),  # index 292
    (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 2, 2, 2, 2, 1, BFP16, BFP16, TL, TL, WS),  # index 293
    (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 2, 2, 2, 2, 1, BFP16, BFP16, TL, TL, WS),  # index 294
    (1, 1280, 2560, 8, 8, 3, 3, 1, 1, 2, 2, 2, 2, 1, BFP16, BFP16, TL, TL, WS),  # index 295
    # From test_conv_dilation - WIDTH_SHARDED with dilation=3
    (1, 768, 768, 16, 16, 3, 3, 1, 1, 3, 3, 3, 3, 1, BFP16, BFP16, TL, TL, WS),  # index 296
    (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 3, 3, 3, 3, 1, BFP16, BFP16, TL, TL, WS),  # index 297
    (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 3, 3, 3, 3, 1, BFP16, BFP16, TL, TL, WS),  # index 298
    (1, 1280, 2560, 8, 8, 3, 3, 1, 1, 3, 3, 3, 3, 1, BFP16, BFP16, TL, TL, WS),  # index 299
    # Additional variations with different batch sizes - scaling WIDTH_SHARDED base cases
    (1, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, WS),  # index 300
    (1, 256, 256, 8, 8, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 301
    (1, 256, 256, 8, 8, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BF8, TL, TL, WS),  # index 302
    (1, 256, 256, 8, 8, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 303
    # WIDTH_SHARDED with different spatial dimensions - scaled down versions
    (1, 256, 256, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 304
    (1, 256, 256, 16, 16, 5, 5, 2, 2, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 305
    # WIDTH_SHARDED with stride=1 variations
    (1, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 306
    (1, 256, 256, 8, 8, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 307
    # WIDTH_SHARDED with larger spatial dimensions
    (1, 256, 256, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 308
    # WIDTH_SHARDED with mixed layout combinations for RM support
    (1, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, RM, TL, WS),  # index 309
    (1, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, TL, WS),  # index 310
    (1, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, RM, WS),  # index 311
    (1, 256, 256, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, WS),  # index 312
    (1, 960, 960, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 960, BFP16, BFP16, TL, TL, WS),  # index 313
    (1, 672, 672, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 672, BFP16, BFP16, TL, TL, WS),  # index 314
    (1, 960, 960, 32, 32, 1, 5, 1, 1, 0, 2, 1, 1, 960, BFP16, BFP16, TL, TL, WS),  # index 315
    (1, 960, 960, 32, 32, 5, 1, 1, 1, 2, 0, 1, 1, 960, BFP16, BFP16, TL, TL, WS),  # index 316
    (1, 672, 672, 32, 32, 1, 5, 1, 1, 0, 2, 1, 1, 672, BFP16, BFP16, TL, TL, WS),  # index 317
    (1, 672, 672, 32, 32, 5, 1, 1, 1, 2, 0, 1, 1, 672, BFP16, BFP16, TL, TL, WS),  # index 318
    (1, 480, 480, 32, 32, 1, 5, 1, 1, 0, 2, 1, 1, 480, BFP16, BFP16, TL, TL, WS),  # index 319
    (1, 480, 480, 32, 32, 5, 1, 1, 1, 2, 0, 1, 1, 480, BFP16, BFP16, TL, TL, WS),  # index 320
    (1, 240, 240, 32, 32, 1, 5, 1, 1, 0, 2, 1, 1, 240, BFP16, BFP16, TL, TL, WS),  # index 321
    (1, 240, 240, 32, 32, 5, 1, 1, 1, 2, 0, 1, 1, 240, BFP16, BFP16, TL, TL, WS),  # index 322
    # Smaller spatial dimensions for performance testing
    (1, 960, 960, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 960, BFP16, BFP16, TL, TL, WS),  # index 323
    (1, 672, 672, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 672, BFP16, BFP16, TL, TL, WS),  # index 324
    (1, 576, 576, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 576, BFP16, BFP16, TL, TL, WS),  # index 325
    (1, 480, 480, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 480, BFP16, BFP16, TL, TL, WS),  # index 326
    # Mixed data type combinations for WIDTH_SHARDED
    (1, 960, 960, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 960, BFP16, BF8, TL, TL, WS),  # index 327
    (1, 672, 672, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 672, BFP16, BF8, TL, TL, WS),  # index 328
    (1, 576, 576, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 576, BFP16, BF8, TL, TL, WS),  # index 329
    (1, 480, 480, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 480, BFP16, BF8, TL, TL, WS),  # index 330
    (1, 1024, 1024, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1024, BFP16, BFP16, TL, TL, WS),  # index 331
    (1, 768, 768, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 768, BFP16, BFP16, TL, TL, WS),  # index 332
    # Additional test cases from test_new_conv2d.py and conv2d_short_sweep.py
    # Varying parameters other than WIDTH_SHARDED
    # Different spatial dimensions - small 8x8 (removing duplicates from Suite 4)
    (2, 128, 128, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 333
    # Different spatial dimensions - medium 64x64
    (1, 512, 256, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 334
    (1, 256, 512, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 335
    # Different kernel sizes with asymmetric combinations
    (1, 320, 320, 32, 32, 1, 5, 1, 1, 0, 2, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 336
    (1, 320, 320, 32, 32, 5, 1, 1, 1, 2, 0, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 337
    (1, 256, 256, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, WS),  # index 338
    (1, 192, 192, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, WS),  # index 339
    (1, 336, 336, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 340
    #############################################################################################
    #########################misc test cases########################################################
    (1, 64, 64, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 341
    (1, 128, 128, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 342
    (1, 64, 64, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 343
    (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 344
    (1, 64, 64, 64, 64, 5, 5, 1, 1, 2, 2, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 345
    (1, 64, 64, 64, 64, 7, 7, 1, 1, 3, 3, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 346
    (1, 64, 64, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, HS),  # index 347
    (1, 64, 64, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, HS),  # index 348
    (1, 64, 64, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, RM, HS),  # index 349
    (1, 64, 64, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, TL, HS),  # index 350
    (1, 64, 64, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 64, BFP16, BFP16, TL, TL, HS),  # index 351
    (1, 64, 64, 64, 64, 3, 3, 1, 1, 2, 2, 2, 2, 1, BFP16, BFP16, TL, TL, HS),  # index 352
    (1, 512, 512, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 353
    ############################################################################################3
    #########################################################################################333##
    ####################333suite 6##############################################################
    (1, 128, 512, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 354
    (1, 256, 512, 30, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 355
    (1, 512, 512, 30, 40, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 356
    # ===== MISSING BF8->BF8 CONFIGURATIONS =====
    (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, HS),  # index 357 - UNet large channels
    # ===== MISSING RM LAYOUT CONFIGURATIONS =====
    # Mixed input/output layout combinations
    (2, 128, 256, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, TL, HS),  # index 358 - RM input, TL output
    (2, 128, 256, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, RM, HS),  # index 359 - TL input, RM output
    (2, 128, 256, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 360 - Both RM
    (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, TL, HS),  # index 361 - UNet RM input
    (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, BS),  # index 362 - Large channels both RM
    # ===== model K CONFIGURATIONS (with dilation) =====
    (1, 48, 32, 252, 252, 3, 3, 1, 1, 0, 0, 2, 2, 1, BFP16, BFP16, TL, TL, HS),  # index 363 - Segformer dilation=2
    (1, 56, 48, 248, 248, 3, 3, 1, 1, 0, 0, 4, 4, 1, BFP16, BFP16, TL, TL, HS),  # index 364 - Segformer dilation=4
    (1, 64, 56, 240, 240, 3, 3, 1, 1, 0, 0, 8, 8, 1, BFP16, BFP16, TL, TL, HS),  # index 365 - Segformer dilation=8
    # ===== RESNET50 CONFIGURATIONS (stride=2, higher batch) =====
    (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 366 - ResNet50 stride=2 batch=8
    (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, HS),  # index 367 - ResNet50 batch=20
    # ===== UNET CONFIGURATIONS (large channels) =====
    (1, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 368 - UNet asymmetric channels
    (1, 640, 1280, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, BS),  # index 369 - UNet asymmetric channels
    (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, RM, HS),  # index 370 - UNet mixed + RM
    # ===== GROUP/DEPTHWISE CONVOLUTIONS =====
    (1, 128, 128, 60, 80, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BFP16, TL, TL, HS),  # index 371 - Group conv groups=2
    (1, 128, 128, 60, 80, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BFP16, RM, TL, HS),  # index 372 - Group conv RM input
    (1, 128, 128, 60, 80, 3, 3, 1, 1, 1, 1, 1, 1, 2, BFP16, BFP16, TL, RM, HS),  # index 373 - Group conv RM output
    # ===== LARGE CHANNEL STRESS TESTS =====
    (1, 1024, 1024, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, WS),  # index 374 - 1024 channels
    # ===== CONV_FEATURES CONFIGURATIONS (mixed types) =====
    (2, 768, 32, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, RM, RM, HS),  # index 375 - Extreme asymmetric
]


# Suite 3: BLOCK_SHARDED test cases
all_test_cases_suite_3 = [
    # Format: (batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, input_dtype, output_dtype, input_layout, output_layout, sharding_scheme)
    # From test_conv_features - explicit BLOCK_SHARDED with various kernels/dtypes
]

all_test_cases_suite_4 = [
    # Format: (batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, input_dtype, output_dtype, input_layout, output_layout, sharding_scheme)
]

all_test_cases_suite_5 = [
    # Format: (batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, input_dtype, output_dtype, input_layout, output_layout, sharding_scheme)
]

import hashlib
import sys


def hash_input_spec(input_spec):
    return hashlib.sha256(str(input_spec).encode()).hexdigest()


hash_to_input_spec = {}


# Function to run conv2d
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

    input_spec_hash = hash_input_spec(input_spec)
    if input_spec_hash in hash_to_input_spec:
        print(f"Skipping duplicate test case: {input_spec}")
        sys.exit(0)
    hash_to_input_spec[input_spec_hash] = input_spec

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

    if branch == "latest" and sharding_scheme == "BS":
        print(f"enabling full inner dim")
        config_params["full_inner_dim"] = True

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


all_test_cases_suite_7_latest = [
    # { batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    # input_dtype, output_dtype, input_layout, output_layout, act_block_h_override, act_block_w_div, enable_act_double_buffer, enable_weights_double_buffer, sharding_scheme }
    # sdxl test cases
    (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, BS),  # index 0
    (1, 1280, 1280, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 256, 1, True, True, BS),  # index 1
    (1, 640, 1280, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 128, 1, True, True, BS),  # index 2
    (1, 1280, 1920, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, BS),  # index 3
    # (1, 640, 1920, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 64, 1, True, True, BS), # index 4
    (1, 1280, 2560, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, BS),  # index 4
    (1, 640, 320, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 512, 1, True, True, BS),  # index 5
    (1, 320, 320, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 256, 1, True, True, BS),  # index 6
    (1, 1280, 640, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, BS),  # index 7
    (1, 640, 640, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 128, 1, True, True, BS),  # index 8
    (1, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, BS),  # index 9
    # (1, 320, 640, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 64, 1, True, True, BS), # index 11
    (1, 640, 960, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 128, 1, True, True, BS),  # index 10
    # # (1, 320, 960, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 256, 1, True, True, BS), # index 13 on new branch also it needs channel spliting.
    # # Stride 2x2 cases for downsampling performance
    (1, 320, 320, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 512, 1, True, True, BS),  # index 11
    (1, 640, 640, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, BS),  # index 12
    # # Special channel cases - very low output/input channels
    (1, 4, 320, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, 128, 1, True, False, HS),  # index 13
    (1, 320, 4, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, 256, 1, False, False, HS),  # index 14
    # Additional test cases -  from models
    # FP32 ResNet50 configurations for precision benchmarking
    (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, 512, 1, True, True, BS),  # index 15
    (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, 512, 1, True, True, BS),  # index 16
    (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 512, 1, True, True, BS),  # index 17
    (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, 0, 1, True, True, BS),  # index 18
    (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, 0, 1, True, True, BS),  # index 19
    (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 0, 1, True, True, BS),  # index 20
    (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, 0, 1, True, True, BS),  # index 21
    (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, 0, 1, True, True, BS),  # index 22
    (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 0, 1, True, True, BS),  # index 23
    # Stable Diffusion extreme channel configurations - tests memory bandwidth limits
    (2, 320, 2560, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 0, 1, True, True, WS),  # index 24
    (2, 320, 2560, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, WS),  # index 25
    (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 0, 1, True, True, WS),  # index 26
    (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 256, 1, True, True, WS),  # index 27
    (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 0, 1, True, True, WS),  # index 28
    (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, WS),  # index 29
    (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 512, 1, True, True, WS),  # index 30
    (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 512, 1, True, True, WS),  # index 31
    (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 512, 1, True, True, WS),  # index 32
    (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 512, 1, True, True, WS),  # index 33
    (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 512, 1, True, True, WS),  # index 34
    (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 512, 1, True, True, WS),  # index 35
    # (2, 320, 960, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 512, 1, True, True, WS), # index 39
    # (2, 320, 960, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 128, 1, True, True, WS), # index 40
    # (2, 320, 640, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 512, 1, True, True, WS), # index 41
    # (2, 320, 640, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 128, 1, True, True, WS), # index 42
]


all_test_cases_suite_7_old = [
    # { batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    # input_dtype, output_dtype, input_layout, output_layout, act_block_h_override, act_block_w_div, enable_act_double_buffer, enable_weights_double_buffer, sharding_scheme }
    # sdxl test cases
    (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 64, 1, True, True, BS),  # index 0
    (1, 1280, 1280, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 64, 1, False, False, BS),  # index 1
    (1, 640, 1280, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 32, 1, True, True, BS),  # index 2
    (1, 1280, 1920, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 32, 1, False, False, BS),  # index 3
    (1, 1280, 2560, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 32, 1, False, False, BS),  # index 4
    (1, 640, 320, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 256, 1, False, False, BS),  # index 5
    (1, 320, 320, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 128, 1, True, True, BS),  # index 6
    (1, 1280, 640, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, BS),  # index 7
    (1, 640, 640, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 32, 1, True, True, BS),  # index 8
    (1, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 128, 1, True, True, BS),  # index 9
    (1, 640, 960, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 32, 1, True, True, BS),  # index 10
    # # Stride 2x2 cases
    (1, 320, 320, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 128, 1, True, True, BS),  # index 11
    (1, 640, 640, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, BS),  # index 12
    # # Special channel cases - very low output/input channels
    (1, 4, 320, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, 64, 1, True, True, HS),  # index 13
    (1, 320, 4, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, 256, 1, True, True, HS),  # index 14
    # Additional test cases -  from models
    # FP32 ResNet50 configurations
    (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, 256, 1, True, True, BS),  # index 15
    (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, 512, 1, True, True, BS),  # index 16
    (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 512, 1, True, True, BS),  # index 17
    (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, 256, 1, True, True, BS),  # index 18
    (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, 256, 1, True, True, BS),  # index 19
    (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 256, 1, True, True, BS),  # index 20
    (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BFP16, TL, TL, 256, 1, True, True, BS),  # index 21
    (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BFP16, BF8, TL, TL, 0, 1, True, True, BS),  # index 22
    (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 0, 1, True, True, BS),  # index 23
    # # Stable Diffusion extreme channel configurations
    (2, 320, 2560, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 0, 1, True, True, WS),  # index 24
    (2, 320, 2560, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, WS),  # index 25
    (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 256, 1, True, True, WS),  # index 26
    (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 256, 1, True, True, WS),  # index 27
    (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 0, 1, True, True, WS),  # index 28
    (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 0, 1, True, True, WS),  # index 29
    (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 512, 1, True, True, WS),  # index 30
    (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 512, 1, True, True, WS),  # index 31
    (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 512, 1, True, True, WS),  # index 32
    (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 512, 1, True, True, WS),  # index 33
    (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BF8, TL, TL, 512, 1, True, True, WS),  # index 34
    (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, BF8, BFP16, TL, TL, 512, 1, True, True, WS),  # index 35
]


def run_perf_with_large_tensors(
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
        act_block_h_override,
        act_block_w_div,
        enable_act_double_buffer,
        enable_weights_double_buffer,
        sharding_scheme,
    ] = input_spec

    input_spec_hash = hash_input_spec(input_spec)
    if input_spec_hash in hash_to_input_spec:
        print(f"Skipping duplicate test case: {input_spec}")
        sys.exit(0)
    hash_to_input_spec[input_spec_hash] = input_spec

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

    initial_weight_dtype = ttnn.float32 if input_dtype == ttnn.bfloat8_b else input_dtype
    initial_bias_dtype = ttnn.float32 if input_dtype == ttnn.bfloat8_b else BFP16

    tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, initial_weight_dtype)
    tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, initial_bias_dtype)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=input_layout, dtype=ttnn.bfloat16, device=device)
    # Build config parameters dynamically
    config_params = {
        "shard_layout": sharding_scheme,
        "output_layout": output_layout,
    }

    config_params["enable_act_double_buffer"] = enable_act_double_buffer
    config_params["enable_weights_double_buffer"] = enable_weights_double_buffer
    config_params["act_block_h_override"] = act_block_h_override
    config_params["act_block_w_div"] = act_block_w_div
    config_params["deallocate_activation"] = True
    config_params["weights_dtype"] = input_dtype

    # Add dtype only for old branch
    if branch == "old":
        config_params["dtype"] = output_dtype

    if branch == "latest" and sharding_scheme == "BS":
        print(f"enabling full inner dim")
        config_params["full_inner_dim"] = True

    print(config_params)
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
# parameters_suite_2 = {"input_specs": all_test_cases_suite_2}
# parameters_suite_3 = {"input_specs": all_test_cases_suite_3}
# parameters_suite_4 = {"input_specs": all_test_cases_suite_4}
# parameters_suite_5 = {"input_specs": all_test_cases_suite_5}
# parameters_suite_6 = {"input_specs": all_test_cases_suite_6}

# Choose suite 7 based on BRANCH environment variable
branch = os.environ.get("BRANCH", "latest")
if branch == "old":
    parameters_suite_7 = {"input_specs": all_test_cases_suite_7_old}
else:
    parameters_suite_7 = {"input_specs": all_test_cases_suite_7_latest}

# # Combined list for reference
# all_test_cases = (all_test_cases_suite_1 + all_test_cases_suite_2 + all_test_cases_suite_3 +
#                  all_test_cases_suite_4 + all_test_cases_suite_5 + all_test_cases_suite_6 + all_test_cases_suite_7)


# Test functions for each suite
@pytest.mark.parametrize("input_spec", parameters_suite_1["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_l1_perf_suite_1(device, input_spec):
    run_perf_benchmark(input_spec, device)


# @pytest.mark.parametrize("input_spec", parameters_suite_2["input_specs"])
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# def test_conv2d_l1_perf_suite_2(device, input_spec):
#     run_perf_benchmark(input_spec, device)


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


@pytest.mark.parametrize("input_spec", parameters_suite_7["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_l1_perf_suite_7(device, input_spec):
    run_perf_with_large_tensors(input_spec, device)


if __name__ == "__main__":
    """
    Example usage for performance testing
    """
    print(f"Conv2d Performance Test Suite - Split into 7 suites")
    print(f"")
    print(f"Suite 1: {len(all_test_cases_suite_1)} test cases (HEIGHT_SHARDED)")
    # print(f"Suite 2: {len(all_test_cases_suite_2)} test cases (HEIGHT_SHARDED with ROW_MAJOR layouts)")
    # print(f"Suite 3: {len(all_test_cases_suite_3)} test cases (BLOCK_SHARDED)")
    # print(f"Suite 4: {len(all_test_cases_suite_4)} test cases (WIDTH_SHARDED)")
    # print(f"Suite 5: {len(all_test_cases_suite_5)} test cases (Split Reader Optimization)")
    # print(f"Suite 6: {len(all_test_cases_suite_6)} test cases (Missing from test_new_conv2d.py)")
    print(f"Suite 7: {len(all_test_cases_suite_7)} test cases (UNet 1024x1024 high-performance configurations)")
