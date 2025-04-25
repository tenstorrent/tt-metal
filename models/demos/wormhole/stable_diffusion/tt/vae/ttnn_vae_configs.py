# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

MIDBLOCK_RESNET_NORM_NUM_BLOCKS = [
    (1, 1),  # resnet 0
    (1, 1),  # resnet 1
]

UPBLOCK_RESNET_NORM_NUM_BLOCKS = [
    [
        (1, 1),  # upblock 0, resnet 0
        (1, 1),  # upblock 0, resnet 1
        (1, 1),  # upblock 0, resnet 2
    ],
    [
        (1, 1),  # upblock 1, resnet 0
        (1, 1),  # upblock 1, resnet 1
        (1, 1),  # upblock 1, resnet 2
    ],
    [
        (4, 4),  # upblock 2, resnet 0
        (4, 4),  # upblock 2, resnet 1
        (4, 16),  # upblock 2, resnet 2
    ],
    [
        (16, 16),  # upblock 3, resnet 0
        (16, 16),  # upblock 3, resnet 1
        (16, 16),  # upblock 3, resnet 2
    ],
]

# each resnet block has two different conv layers
# each connv layer has in and out channel split factor
# (conv1_in_ch_split_factor, conv2_in_ch_split_factor), (conv2_in_ch_split_factor, conv2_out_ch_split_factor)

MIDBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS = [
    ((1, 1), (1, 1)),  # resnet 0
    ((1, 1), (1, 1)),  # resnet 1
]
UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS = [
    [
        ((1, 1), (1, 1)),  # upblock 0, resnet 0
        ((1, 1), (1, 1)),  # upblock 0, resnet 1
        ((1, 1), (1, 1)),  # upblock 0, resnet 2
    ],
    [
        ((1, 1), (1, 1)),  # upblock 1, resnet 0
        ((1, 1), (1, 1)),  # upblock 1, resnet 1
        ((1, 1), (1, 1)),  # upblock 1, resnet 2
    ],
    [
        ((4, 2), (2, 2)),  # upblock 2, resnet 0
        ((1, 1), (1, 1)),  # upblock 2, resnet 1
        ((1, 1), (1, 1)),  # upblock 2, resnet 2
    ],
    [
        ((8, 2), (4, 2)),  # upblock 3, resnet 0
        ((4, 2), (4, 2)),  # upblock 3, resnet 1
        ((4, 2), (4, 2)),  # upblock 3, resnet 2
    ],
]

UPBLOCK_UPSAMPLE_CONV_CHANNEL_SPLIT_FACTORS = [
    (1, 1),  # upblock 0
    (4, 2),  # upblock 1
    (4, 4),  # upblock 2
    (1, 1),  # upblock 3 (no upsample here)
]


GROUPNORM_EPSILON = 1e-6
GROUPNORM_GROUPS = 32


def get_default_conv_config():
    return ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        activation="",
        deallocate_activation=True,
    )


def get_default_compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
