# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import pytest
import ttnn
from models.utility_functions import comp_pcc
from models.demos.resnet.tt.metalResnetBlock50 import (
    compute_conv_output_shape,
    resnet50_1x1_conv_as_matmul,
    resnet50_optimized_conv,
    _nearest_32,
    format_tensor,
)

# hardcoding matmul config for 1x1 convs
# key: mm act height, mm act width, mm weight width
hardcoded_matmul_config_conv = {
    1: {
        (3136, 64, 64): {
            "compute_with_storage_grid_size": (2, 2),
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 49,
            "per_core_N": 1,
        },
        (3136, 64, 256): {
            "compute_with_storage_grid_size": (4, 2),
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 49,
            "per_core_N": 2,
        },
        (3136, 256, 64): {
            "compute_with_storage_grid_size": (2, 7),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 14,
            "per_core_N": 1,
        },
        (3136, 256, 128): {
            "compute_with_storage_grid_size": (4, 7),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 14,
            "per_core_N": 1,
        },
        (800, 128, 512): {
            "compute_with_storage_grid_size": (4, 2),
            "in0_block_w": 4,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 13,
            "per_core_N": 4,
        },
        (800, 512, 128): {
            "compute_with_storage_grid_size": (4, 4),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 7,
            "per_core_N": 1,
        },
        (800, 512, 256): {
            "compute_with_storage_grid_size": (8, 4),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 7,
            "per_core_N": 1,
        },
        (224, 256, 1024): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 4,
        },
        (224, 1024, 256): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 32,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 1,
        },
        (224, 1024, 512): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 32,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 2,
        },
        (64, 512, 2048): {
            "compute_with_storage_grid_size": (8, 2),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 8,
        },
        (64, 2048, 512): {
            "compute_with_storage_grid_size": (8, 2),
            "in0_block_w": 64,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 2,
        },
    },
    2: {
        (6272, 64, 64): {
            "compute_with_storage_grid_size": (2, 4),
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 49,
            "per_core_N": 1,
        },
        (6272, 64, 256): {
            "compute_with_storage_grid_size": (4, 4),
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 49,
            "per_core_N": 2,
        },
        (6272, 256, 64): {
            "compute_with_storage_grid_size": (2, 9),  # (x,y)
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 22,  # across y
            "per_core_N": 1,  # across x
        },
        (6272, 256, 128): {
            "compute_with_storage_grid_size": (4, 9),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 22,
            "per_core_N": 1,
        },
        (1568, 128, 512): {
            "compute_with_storage_grid_size": (4, 4),
            "in0_block_w": 4,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 13,
            "per_core_N": 4,
        },
        (1568, 512, 128): {
            "compute_with_storage_grid_size": (4, 9),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 6,
            "per_core_N": 1,
        },
        (1568, 512, 256): {
            "compute_with_storage_grid_size": (8, 9),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 6,
            "per_core_N": 1,
        },
        (416, 256, 1024): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 8,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 2,
            "per_core_N": 4,
        },
        (416, 1024, 256): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 32,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 2,
            "per_core_N": 1,
        },
        (416, 1024, 512): {
            "compute_with_storage_grid_size": (8, 7),
            "in0_block_w": 32,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 2,
            "per_core_N": 2,
        },
        (128, 512, 2048): {
            "compute_with_storage_grid_size": (8, 4),
            "in0_block_w": 16,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 8,
        },
        (128, 2048, 512): {
            "compute_with_storage_grid_size": (8, 4),
            "in0_block_w": 64,
            "out_subblock_h": 1,
            "out_subblock_w": 1,
            "per_core_M": 1,
            "per_core_N": 2,
        },
    },
    8: {
        (25088, 64, 64): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=2,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=8,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (25088, 64, 256): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=8,
            per_core_N=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (25088, 256, 64): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=8,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (25088, 256, 128): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=8,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (6272, 128, 512): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=2,
            per_core_N=16,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (6272, 256, 512): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=2,
            per_core_N=16,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (6272, 512, 128): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=16,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=2,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (6272, 512, 256): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=2,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=20,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 256, 1024): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=5,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 256): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=4,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=5,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=5,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 512, 1024): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=5,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=7,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (416, 512, 2048): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=2,
            per_core_N=8,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (416, 1024, 2048): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=2,
            per_core_N=8,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (416, 2048, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=8,
            out_subblock_h=2,
            out_subblock_w=2,
            per_core_M=2,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
    },
    16: {
        (50176, 64, 64): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=2,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=16,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (50176, 64, 256): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=16,
            per_core_N=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (50176, 256, 64): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=16,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (50176, 256, 128): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=16,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (12544, 128, 512): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=4,
            per_core_N=16,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (12544, 256, 512): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=4,
            per_core_N=16,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (12544, 512, 128): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=16,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=4,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (12544, 512, 256): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=2,
            out_subblock_h=4,
            out_subblock_w=1,
            per_core_M=36,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (3136, 256, 1024): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=9,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (3136, 1024, 256): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=4,
            out_subblock_h=3,
            out_subblock_w=1,
            per_core_M=9,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (3136, 1024, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=4,
            out_subblock_h=3,
            out_subblock_w=2,
            per_core_M=9,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (3136, 512, 1024): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=9,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (3136, 1024, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(9, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=11,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (800, 512, 2048): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(9, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=3,
            per_core_N=8,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (800, 1024, 2048): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(9, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=3,
            per_core_N=8,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (800, 2048, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(9, 8),
            in0_block_w=8,
            out_subblock_h=3,
            out_subblock_w=2,
            per_core_M=3,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
    },
}

hardcoded_conv_blocking_and_parallelization_config = {
    1: {
        (3136, 64): [64 * 3, 64, 64, 64, 64, 64, (7, 7), 64, 64],
        (800, 128): [128, 32, 128, 32, 64, 32, (5, 5), 32, 128],
        (224, 256): [256, 32, 128, 32, 128, 32, (1, 7), 32, 256],
        (64, 512): [512, 32, 64, 32, 64, 32, (1, 2), 32, 512],
        # bypass convs
        (3136, 256): [128, 64, 64, 64, 64, 64, (7, 7), 64, 256],
        (800, 512): [256, 32, 64, 32, 64, 32, (5, 5), 32, 512],
        (224, 1024): [512, 32, 128, 32, 64, 32, (1, 7), 32, 1024],
        (64, 2048): [1024, 32, 128, 32, 64, 32, (1, 2), 32, 2048],
    },
    2: {
        (6272, 64): [64 * 3, 128, 64, 128, 64, 128, (7, 7), 128, 64],
        (1568, 128): [128, 32, 128, 32, 64, 32, (7, 7), 32, 128],
        (416, 256): [256, 64, 128, 64, 128, 64, (7, 1), 64, 256],
        (128, 512): [512, 32, 64, 32, 64, 32, (1, 4), 32, 512],
        # bypass convs
        (6272, 256): [128, 128, 64, 128, 64, 128, (7, 7), 128, 256],
        (1568, 512): [256, 32, 64, 32, 64, 32, (7, 7), 32, 512],
        (416, 1024): [512, 64, 128, 64, 64, 64, (7, 1), 64, 1024],
        (128, 2048): [1024, 64, 128, 64, 64, 64, (1, 2), 64, 2048],
    },
    8: {
        (25088, 64): [64 * 3, 256, 64, 128, 64, 256, (12, 9), 256, 64],
        (6272, 128): [128, 64, 128, 64, 128, 64, (12, 9), 64, 128],
        (1568, 256): [256, 160, 32, 160, 32, 160, (10, 8), 160, 32],
        (416, 512): [512, 64, 64, 64, 64, 64, (7, 8), 64, 64],
        # bypass convs
        (6272, 512): [256, 64, 512, 32, 256, 64, (12, 9), 64, 512],
        (1568, 1024): [512, 160, 128, 32, 128, 160, (10, 8), 160, 128],
        (416, 2048): [512, 64, 256, 32, 256, 64, (7, 8), 64, 256],
    },
    16: {
        (50176, 64): [64 * 3, 512, 64, 128, 64, 512, (12, 9), 512, 64],
        (12544, 128): [128, 128, 128, 64, 128, 128, (12, 9), 128, 128],
        (3136, 256): [256, 288, 32, 96, 32, 288, (11, 8), 288, 32],
        (800, 512): [512, 96, 64, 96, 64, 96, (9, 8), 96, 64],
        # bypass convs
        (12544, 512): [256, 128, 512, 32, 256, 128, (12, 9), 128, 512],
        (3136, 1024): [512, 288, 128, 32, 128, 288, (11, 8), 288, 128],
        (800, 2048): [512, 96, 256, 32, 256, 96, (9, 8), 96, 256],
    },
}


@pytest.mark.parametrize("N", [8, 16], ids=["batch_8", "batch_16"])
@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        # 1x1 convs in rn50
        (64, 64, 56, 56, 1, 1, 1, 1, 0, 0),
        (
            256,
            64,
            56,
            56,
            1,
            1,
            1,
            1,
            0,
            0,
        ),  # slow with new_matmul but less than bias computation time
        (64, 256, 56, 56, 1, 1, 1, 1, 0, 0),
        (64, 256, 56, 56, 1, 1, 1, 1, 0, 0),
        (128, 256, 56, 56, 1, 1, 1, 1, 0, 0),
        (512, 128, 28, 28, 1, 1, 1, 1, 0, 0),
        (128, 512, 28, 28, 1, 1, 1, 1, 0, 0),
        (256, 512, 28, 28, 1, 1, 1, 1, 0, 0),
        (1024, 256, 14, 14, 1, 1, 1, 1, 0, 0),
        (256, 1024, 14, 14, 1, 1, 1, 1, 0, 0),
        (512, 1024, 14, 14, 1, 1, 1, 1, 0, 0),
        (2048, 512, 7, 7, 1, 1, 1, 1, 0, 0),
        (
            512,
            2048,
            7,
            7,
            1,
            1,
            1,
            1,
            0,
            0,
        ),  # slightly slower with new matmul but less than old matmul + bias computation time
        # unique convs in rn50 (complete list)
        # layer1
        (64, 64, 56, 56, 3, 3, 1, 1, 1, 1),
        # layer2
        (512, 256, 56, 56, 1, 1, 2, 2, 0, 0),
        (128, 128, 56, 56, 3, 3, 2, 2, 1, 1),
        (128, 128, 28, 28, 3, 3, 1, 1, 1, 1),
        # layer3
        (256, 256, 28, 28, 3, 3, 2, 2, 1, 1),
        (1024, 512, 28, 28, 1, 1, 2, 2, 0, 0),
        (256, 256, 14, 14, 3, 3, 1, 1, 1, 1),
        # layer4
        (512, 512, 14, 14, 3, 3, 2, 2, 1, 1),
        (2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0),
        (512, 512, 7, 7, 3, 3, 1, 1, 1, 1),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["weights_BFLOAT16", "weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["output_BFLOAT16", "output_BFLOAT8_B"],
)
def test_resnet50_conv(
    device, use_program_cache, N, K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w, weights_dtype, output_dtype
):
    output_mem_config = ttnn.L1_MEMORY_CONFIG
    if (
        output_dtype == ttnn.bfloat16
        and N > 8
        and [K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w] == [1024, 512, 28, 28, 1, 1, 2, 2, 0, 0]
    ):
        pytest.skip("Config does not fit in L1")

    for i in range(1):  # increase num of iterations to test op caching
        assert C % 32 == 0
        assert K % 32 == 0
        torch.manual_seed(0)
        memory_config = (ttnn.DRAM_MEMORY_CONFIG if N > 8 else ttnn.L1_MEMORY_CONFIG,)

        conv_input_shape = [N, C, H, W]
        conv_weight_shape = [K, C, R, S]
        conv_bias_shape = [1, 1, 1, K]
        conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
        conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
        conv_input_shape_nhwc = conv_input_pyt_nhwc.shape
        conv_weight_pyt = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
        conv_bias_pyt = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()
        out_golden = torch.nn.functional.conv2d(
            conv_input_pyt,
            conv_weight_pyt,
            bias=conv_bias_pyt.reshape(-1),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
        )

        is_1x1_conv = R == 1 and S == 1 and stride_h == 1 and stride_w == 1 and pad_h == 0 and pad_w == 0

        conv_params = [K, C, R, S, stride_h, stride_w, pad_h, pad_w, 1, 1]
        conv_output_shape = compute_conv_output_shape(conv_params, conv_input_shape_nhwc)
        logger.info("Conv output shape - ", conv_output_shape)
        conv_as_mm_padded_act_height = _nearest_32(conv_output_shape[0] * conv_output_shape[1] * conv_output_shape[2])

        if is_1x1_conv:
            matmul_config = None
            assert (conv_as_mm_padded_act_height, C, K) in hardcoded_matmul_config_conv[N]
            logger.info("Setting matmul config for 1x1 conv")
            matmul_config = hardcoded_matmul_config_conv[N][(conv_as_mm_padded_act_height, C, K)]
            # 1x1 conv with stride 1 padding 0 is run using regular matmul
            conv = resnet50_1x1_conv_as_matmul(
                conv_weight_pyt.reshape(-1).tolist(),
                conv_params,
                device,
                conv_bias_pyt.reshape(-1).tolist(),
                matmul_config,
                output_mem_config=output_mem_config,
                weights_dtype=weights_dtype,
                output_dtype=output_dtype,
                # math_fidelity=ttnn.MathFidelity.HiFi4,
            )
        else:
            assert (conv_as_mm_padded_act_height, K) in hardcoded_conv_blocking_and_parallelization_config[N]
            conv_config = hardcoded_conv_blocking_and_parallelization_config[N][(conv_as_mm_padded_act_height, K)]
            [
                act_block_w_datums,
                act_block_h_datums,
                weight_block_w_datums,
                out_subblock_h_datums,
                out_subblock_w_datums,
                out_block_h_datums,
                grid_size,
                per_core_out_matrix_h,
                per_core_weight_matrix_w,
            ] = conv_config
            if R == 1 and S == 1:
                assert C % act_block_w_datums == 0
            else:
                assert act_block_w_datums == C or act_block_w_datums == C * S
            assert act_block_w_datums % 32 == 0
            assert act_block_h_datums % 32 == 0
            assert weight_block_w_datums % 32 == 0
            assert per_core_out_matrix_h % 32 == 0
            per_core_out_matrix_h_ntiles = (int)(per_core_out_matrix_h / 32)
            per_core_weight_matrix_w_ntiles = (int)(per_core_weight_matrix_w / 32)
            conv = resnet50_optimized_conv(
                conv_weight_pyt.reshape(-1).tolist(),
                conv_params,
                device,
                [act_block_h_datums, act_block_w_datums],
                [act_block_w_datums, weight_block_w_datums],
                [out_subblock_h_datums, out_subblock_w_datums],
                out_block_h_datums,
                grid_size,
                per_core_out_matrix_h_ntiles,
                per_core_weight_matrix_w_ntiles,
                conv_bias_pyt.reshape(-1).tolist(),
                output_mem_config=output_mem_config,
                weights_dtype=weights_dtype,
                output_dtype=output_dtype,
                math_fidelity=ttnn.MathFidelity.HiFi4,
            )

        conv_input_on_device = ttnn.Tensor(
            conv_input_pyt_nhwc.reshape(-1).tolist(),
            conv_input_pyt_nhwc.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        ).to(device, memory_config)

        if is_1x1_conv:
            # convert activation RM to tile layout
            conv_input_on_device = conv_input_on_device.reshape(
                1,
                1,
                conv_input_shape_nhwc[0] * conv_input_shape_nhwc[1] * conv_input_shape_nhwc[2],
                conv_input_shape_nhwc[3],
            )
            conv_input_on_device = format_tensor(conv_input_on_device, ttnn.TILE_LAYOUT, device, memory_config)

        output_on_device = conv(conv_input_on_device)
        if output_mem_config.is_sharded():
            output_on_device = ttnn.sharded_to_interleaved(output_on_device, memory_config)

        # convert tiled output to RM
        assert output_on_device.get_layout() == ttnn.TILE_LAYOUT
        output_on_device = format_tensor(output_on_device, ttnn.ROW_MAJOR_LAYOUT, device, memory_config)
        output_on_device = output_on_device.reshape(
            conv_output_shape[0],
            conv_output_shape[1],
            conv_output_shape[2],
            conv_output_shape[3],
        )

        # Copy to host and Compare against pytorch
        out = output_on_device.cpu()
        assert out.get_layout() == ttnn.ROW_MAJOR_LAYOUT

        out_result = out.to_torch()
        # NHWC to NCHW
        out_result = torch.transpose(out_result, 2, 3)
        out_result = torch.transpose(out_result, 1, 2)

        # Compare against golden
        assert out_result.shape == out_golden.shape
        passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
        logger.debug(f"Passing={passing_pcc}")
        logger.debug(f"Output pcc={output_pcc}")
        assert passing_pcc
