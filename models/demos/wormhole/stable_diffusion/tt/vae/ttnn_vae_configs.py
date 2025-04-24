# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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
# each connv layer has in channel split factor
MIDBLOCK_RESNET_CONV_IN_CHANNEL_SPLIT_FACTORS = [
    (1, 1),  # resnet 0
    (1, 1),  # resnet 1
]

UPBLOCK_RESNET_CONV_IN_CHANNEL_SPLIT_FACTORS = [
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
        (2, 1),  # upblock 2, resnet 0
        (1, 1),  # upblock 2, resnet 1
        (1, 1),  # upblock 2, resnet 2
    ],
    [
        (8, 4),  # upblock 3, resnet 0
        (4, 4),  # upblock 3, resnet 1
        (4, 4),  # upblock 3, resnet 2
    ],
]

UPBLOCK_UPSAMPLE_CONV_CHANNEL_SPLIT_FACTORS = [
    (1, 1),  # upblock 0
    (2, 2),  # upblock 1
    (4, 2),  # upblock 2
    (1, 1),  # upblock 3 (no upsample here)
]
