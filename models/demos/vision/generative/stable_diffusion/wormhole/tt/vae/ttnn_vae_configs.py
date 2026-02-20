# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
        (32, 32),  # upblock 3, resnet 0
        (32, 32),  # upblock 3, resnet 1
        (32, 32),  # upblock 3, resnet 2
    ],
]

GROUPNORM_EPSILON = 1e-6
GROUPNORM_GROUPS = 32
GROUPNORM_DECODER_NUM_BLOCKS = 32


def get_default_conv_config():
    return ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        deallocate_activation=True,
    )


def get_default_conv_output_dtype():
    return ttnn.bfloat16


def get_default_compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
