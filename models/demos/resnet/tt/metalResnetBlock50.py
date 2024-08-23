# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Callable
import time
import ttnn
import torch
import torch.nn as nn
import math
from models.demos.resnet.utils import fold_bn_to_conv_weights_bias
from models.utility_functions import tt2torch_tensor, torch2tt_tensor
from tt_lib.utils import pad_weight

from models.utility_functions import is_grayskull
from tt_lib.fused_ops.average_pool import run_avg_pool_on_device_wrapper as TtAvgPool
from tt_lib.fused_ops.max_pool import run_max_pool_on_device_wrapper as TtMaxPool
from tt_lib.fused_ops.max_pool import compute_max_pool_shape
from tt_lib.fused_ops.conv import (
    resnet50_first_conv,
    resnet50_1x1_conv_as_matmul,
    resnet50_optimized_conv,
    resnet50_1x1_conv_s2_as_downsample_and_matmul,
)
from ttnn.operations.conv.tt_py_composite_conv import (
    TTPyCompositeConv,
    SlidingWindowOpParamsWithParallelConfig,
)
from ttnn.operations.pool import TTPyMaxPool

from models.utility_functions import (
    _nearest_32,
    pad_and_fold_conv_activation_for_unity_stride,
    pad_and_fold_conv_filters_for_unity_stride,
)

hardcoded_matmul_config_linear = {
    8: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    ),
    16: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    ),
    20: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    ),
}


def ResnetLinear(
    in_features: int,
    out_features: int,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor] = None,
    transpose: bool = True,
    output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
    model_config=None,
    device=None,
    batch_size=None,
):
    """
    Returns a function for linear operation in resnet with bias.
    """
    if bias is not None:
        assert bias.get_legacy_shape()[-1] == out_features, "bias shape is not as expected"
        if device is not None:
            bias = bias.to(device)

    if transpose:
        assert weight.get_legacy_shape() == [1, 1, out_features, in_features], "weight does not have the expected shape"
        weight_T = ttnn.transpose(weight, -2, -1)
    else:
        assert weight.get_legacy_shape() == [
            1,
            1,
            in_features,
            out_features,
        ], "weight does not have the expected shape"
        weight_T = weight
    if device is not None:
        weight_T = weight_T.to(device)

    matmul_config = None
    if batch_size in hardcoded_matmul_config_linear and output_mem_config.is_sharded():
        matmul_config = hardcoded_matmul_config_linear[batch_size]

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=model_config["MATH_FIDELITY"],
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=model_config["MATH_FIDELITY"],
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    ## this uses the systolic 1d matmul with bias fused
    if matmul_config is None:

        def linear_(act):
            return ttnn.linear(act, weight_T, bias=bias, memory_config=output_mem_config)

    else:

        def linear_(act):
            return ttnn.linear(
                act,
                weight_T,
                bias=bias,
                program_config=matmul_config,
                memory_config=output_mem_config,
                dtype=model_config["ACTIVATIONS_DTYPE"],
                compute_kernel_config=compute_kernel_config,
            )

    return linear_


def do_nothing_op(x):
    return x


def _nearest_y(x, y):
    return math.ceil(x / y) * y


def format_tensor(x, target_layout, device, output_mem_config, pad_value=0.0):
    if x.get_layout() == target_layout:
        return x
    if x.get_layout() == ttnn.ROW_MAJOR_LAYOUT and target_layout == ttnn.TILE_LAYOUT:
        x_padded_shape = ttnn.experimental.tensor.pad_to_tile_shape(x.get_legacy_shape(), False, False, True, True)
        if x.get_legacy_shape() != x_padded_shape:
            return ttnn.experimental.tensor.format_input_tensor(
                x, device, x_padded_shape, pad_value, target_layout, output_mem_config
            )
        else:
            return ttnn.tilize(x, memory_config=output_mem_config, use_multicore=True)
    elif x.get_layout() == ttnn.TILE_LAYOUT and target_layout == ttnn.ROW_MAJOR_LAYOUT:
        if x.get_legacy_shape() != x.shape_without_padding():
            return ttnn.experimental.tensor.format_output_tensor(
                x, x.shape_without_padding(), device, target_layout, output_mem_config
            )
        else:
            return ttnn.untilize(x, memory_config=output_mem_config, use_multicore=True)
    else:
        assert False


# Local copy of unpad_from_zero to always set output to
def unpad_from_zero(x, desired_shape):
    if x.get_legacy_shape()[-1] == desired_shape[-1] and x.get_legacy_shape()[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = x.to(ttnn.ROW_MAJOR_LAYOUT)
        x = x.unpad(
            (0, 0, 0, 0),
            (
                desired_shape[0] - 1,
                desired_shape[1] - 1,
                desired_shape[2] - 1,
                desired_shape[3] - 1,
            ),
        )
        x = x.to_torch().to(torch.float)
    return x


def compute_conv_output_shape(conv_params, x_shape):
    H = x_shape[1]
    W = x_shape[2]
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    OH = ((int)((H - R + 2 * P_H) / U)) + 1
    OW = ((int)((W - S + 2 * P_W) / V)) + 1
    return [x_shape[0], OH, OW, K]


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
    20: {
        (62720, 64, 64): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=2,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=20,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (62720, 64, 256): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=20,
            per_core_N=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (62720, 256, 64): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=20,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (62720, 256, 128): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=20,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (15680, 128, 512): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=5,
            per_core_N=16,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (15680, 256, 512): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=5,
            per_core_N=16,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (15680, 512, 128): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(12, 9),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=5,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        ),
        (15680, 512, 256): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(12, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=41,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (3936, 256, 1024): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(12, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=11,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (3936, 1024, 256): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(12, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=11,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (3936, 1024, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(12, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=11,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (3936, 512, 1024): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(12, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=11,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (3936, 1024, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(12, 8),
            in0_block_w=4,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=12,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (992, 512, 2048): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=3,
            per_core_N=8,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (992, 1024, 2048): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=3,
            per_core_N=8,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (992, 2048, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
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
        (3136, 64): [64 * 3, 64, 64, 64, 64, 64, (7, 7), 64, 64, -1],
        (800, 128): [128 * 3, 32, 128, 32, 64, 32, (5, 5), 32, 128, -1],
        (224, 256): [256, 32, 128, 32, 128, 32, (1, 7), 32, 256, -1],
        (64, 512): [512, 32, 64, 32, 64, 32, (1, 2), 32, 512, -1],
        # bypass convs
        (3136, 256): [128, 64, 64, 64, 64, 64, (7, 7), 64, 256, -1],
        (800, 512): [256, 32, 64, 32, 64, 32, (5, 5), 32, 512, -1],
        (224, 1024): [512, 32, 128, 32, 64, 32, (1, 7), 32, 1024, -1],
        (64, 2048): [1024, 32, 128, 32, 64, 32, (1, 2), 32, 2048, -1],
    },
    2: {
        (6272, 64): [64 * 3, 128, 64, 128, 64, 128, (7, 7), 128, 64, -1],
        (1568, 128): [128 * 3, 32, 128, 32, 64, 32, (7, 7), 32, 128, -1],
        (416, 256): [256, 64, 128, 64, 128, 64, (7, 1), 64, 256, -1],
        (128, 512): [512, 32, 64, 32, 64, 32, (1, 4), 32, 512, -1],
        # bypass convs
        (6272, 256): [128, 128, 64, 128, 64, 128, (7, 7), 128, 256, -1],
        (1568, 512): [256, 32, 64, 32, 64, 32, (7, 7), 32, 512, -1],
        (416, 1024): [512, 64, 128, 64, 64, 64, (7, 1), 64, 1024, -1],
        (128, 2048): [1024, 64, 128, 64, 64, 64, (1, 2), 64, 2048, -1],
    },
    8: {
        (100352, 64): [16 * 4, 1024, 64, 128, 64, 1024, (12, 9), 1024, 64, 98],
        (25088, 64): [64 * 3, 256, 64, 128, 64, 256, (12, 9), 256, 64, 98],
        (6272, 128): [128 * 3, 64, 128, 64, 128, 64, (12, 9), 64, 128, 98],
        (1568, 256): [256, 160, 32, 160, 32, 160, (10, 8), 160, 32, 10],
        (416, 512): [512, 64, 64, 64, 64, 64, (7, 8), 64, 64, 7],
    },
    16: {
        (200704, 64): [16 * 4, 1024, 64, 128, 64, 2048, (12, 9), 2048, 64, 98],
        (50176, 64): [64 * 3, 256, 64, 128, 64, 512, (12, 9), 512, 64, 98],
        (12544, 128): [128 * 3, 128, 128, 64, 128, 128, (12, 9), 128, 128, 98],
        (3136, 256): [256, 288, 32, 96, 32, 288, (11, 8), 288, 32, 11],
        (800, 512): [512, 96, 64, 96, 64, 96, (9, 8), 96, 64, 9],
    },
    20: {
        (250880, 64): [16 * 4, 1280, 64, 128, 64, 2560, (12, 9), 2560, 64, 98],
        (62720, 64): [
            64 * 3,
            320,
            64,
            64,
            64,
            640,
            (12, 9),
            640,
            64,
            98,
        ],  # try actblock h = 320, subblock h = 64
        (15680, 128): [128 * 3, 160, 128, 32, 128, 160, (12, 9), 160, 128, 98],
        (3936, 256): [256, 352, 32, 32, 32, 352, (12, 8), 352, 32, 12],
        (992, 512): [512, 96, 64, 96, 64, 96, (11, 8), 96, 64, 11],
    },
}


class Bottleneck:
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device=None,
        state_dict=None,
        base_address=None,
        fold_batchnorm=False,
        downsample_conv_on_tt=None,
        norm_layer_after_downsample_conv_on_tt=None,
        downsample_params=[],
        storage_in_dram=True,
        input_shape=[],
        batch_size=1,
        sharded=None,
        out_sharded=False,
        use_downsample_op_and_mm_for_conv1x1_s2=False,
        model_config=None,
        conv_halo=False,
        conv_2d=False,
        reader_patterns_cache={},
    ) -> None:
        super().__init__()
        self.device = device
        self.state_dict = state_dict
        self.base_address = base_address
        self.fold_batchnorm = fold_batchnorm
        self.downsample_conv_on_tt = downsample_conv_on_tt
        self.norm_layer_after_downsample_conv_on_tt = norm_layer_after_downsample_conv_on_tt
        self.downsample_params = downsample_params
        self.storage_in_dram = storage_in_dram
        self.conv_halo = conv_halo
        self.model_config = model_config
        if self.storage_in_dram:
            self.memory_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            self.memory_config = ttnn.L1_MEMORY_CONFIG
        if sharded is not None:
            self.sharded_memory_config = ttnn.MemoryConfig(sharded, ttnn.BufferType.L1)
        else:
            self.sharded_memory_config = self.memory_config
        self.out_memory_config = self.sharded_memory_config if out_sharded else self.memory_config
        self.out_in_place = out_sharded
        if use_downsample_op_and_mm_for_conv1x1_s2:
            assert sharded

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        conv1_weight = state_dict[f"{base_address}.conv1.weight"]
        conv1_bias = None

        self.bn1 = norm_layer(width)
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address}.bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address}.bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn1.num_batches_tracked"], requires_grad=False
        )
        self.bn1.eval()

        conv2_weight = state_dict[f"{base_address}.conv2.weight"]
        conv2_bias = None

        self.bn2 = norm_layer(width)
        self.bn2.weight = nn.Parameter(state_dict[f"{self.base_address}.bn2.weight"])
        self.bn2.bias = nn.Parameter(state_dict[f"{self.base_address}.bn2.bias"])
        self.bn2.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_mean"])
        self.bn2.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_var"])
        self.bn2.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn2.num_batches_tracked"], requires_grad=False
        )
        self.bn2.eval()

        conv3_weight = state_dict[f"{base_address}.conv3.weight"]
        conv3_bias = None

        self.bn3 = norm_layer(planes * self.expansion)
        self.bn3.weight = nn.Parameter(state_dict[f"{self.base_address}.bn3.weight"])
        self.bn3.bias = nn.Parameter(state_dict[f"{self.base_address}.bn3.bias"])
        self.bn3.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn3.running_mean"])
        self.bn3.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn3.running_var"])
        self.bn3.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address}.bn3.num_batches_tracked"], requires_grad=False
        )
        self.bn3.eval()

        self.relu = ttnn.relu
        self.downsample = downsample
        self.stride = stride

        if self.fold_batchnorm:
            conv1_weight, conv1_bias = fold_bn_to_conv_weights_bias(conv1_weight, self.bn1)
            conv2_weight, conv2_bias = fold_bn_to_conv_weights_bias(conv2_weight, self.bn2)
            conv3_weight, conv3_bias = fold_bn_to_conv_weights_bias(conv3_weight, self.bn3)
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        self.module_input_shape = input_shape
        self.deallocate = True
        self.downsample_or_noop = self.downsample_conv_on_tt
        if self.downsample_or_noop is None:
            self.downsample_or_noop = do_nothing_op
            self.deallocate = False
        else:
            if (not use_downsample_op_and_mm_for_conv1x1_s2) and (
                self.downsample_params[2] != 1 or self.downsample_params[4] != 1 or self.downsample_params[6] != 0
            ):
                # this downsample conv requires row major input
                def downsample_conv_op_wrapper(op):
                    def downsample_conv_op_with_formatting(x):
                        x = format_tensor(x, ttnn.ROW_MAJOR_LAYOUT, self.device, self.memory_config)
                        x = x.reshape(
                            self.module_input_shape[0],
                            self.module_input_shape[1],
                            self.module_input_shape[2],
                            self.module_input_shape[3],
                        )
                        return op(x)

                    return downsample_conv_op_with_formatting

                self.downsample_or_noop = downsample_conv_op_wrapper(self.downsample_conv_on_tt)

        self.conv1_params = [width, inplanes, 1, 1, 1, 1, 0, 0, dilation, groups]
        self.conv1_output_shape = compute_conv_output_shape(self.conv1_params, input_shape)
        conv1_as_mm_padded_act_height = _nearest_32(
            self.conv1_output_shape[0] * self.conv1_output_shape[1] * self.conv1_output_shape[2]
        )
        assert (conv1_as_mm_padded_act_height, inplanes, width) in hardcoded_matmul_config_conv[batch_size]
        matmul_config = hardcoded_matmul_config_conv[batch_size][(conv1_as_mm_padded_act_height, inplanes, width)]
        # 1x1 conv with stride 1 padding 0 is run using regular matmul
        if is_grayskull():
            compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
            )
        else:
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

        untilize_out = False

        self.conv1 = resnet50_1x1_conv_as_matmul(
            conv1_weight.reshape(-1).tolist(),
            self.conv1_params,
            self.device,
            conv1_bias.tolist(),
            matmul_config,
            fuse_relu=True,
            output_mem_config=self.sharded_memory_config,
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            output_dtype=ttnn.bfloat16 if untilize_out else model_config["ACTIVATIONS_DTYPE"],
            compute_kernel_config=compute_kernel_config,
            untilize_out=untilize_out,
        )

        self.conv2_params = [width, width, 3, 3, stride, stride, 1, 1, dilation, groups]
        self.conv2_output_shape = compute_conv_output_shape(self.conv2_params, self.conv1_output_shape)
        conv2_output_padded_face_size = _nearest_32(
            self.conv2_output_shape[0] * self.conv2_output_shape[1] * self.conv2_output_shape[2]
        )
        assert (
            conv2_output_padded_face_size,
            width,
        ) in hardcoded_conv_blocking_and_parallelization_config[batch_size]
        [
            act_block_w,
            act_block_h,
            weight_block_w,
            out_subblock_h,
            out_subblock_w,
            out_block_h,
            grid_size,
            per_core_act_h,
            per_core_weight_w,
            num_cores_nhw,  # This number is only meaningful for batch 8, 16
        ] = hardcoded_conv_blocking_and_parallelization_config[batch_size][(conv2_output_padded_face_size, width)]

        assert per_core_act_h % 32 == 0
        per_core_act_h_ntiles = (int)(per_core_act_h / 32)
        per_core_weight_w_ntiles = (int)(per_core_weight_w / 32)
        # For sharded input, use new untilize_with_halo + conv infra
        if self.conv_halo:
            sliding_window_op_params = SlidingWindowOpParamsWithParallelConfig(
                stride_h=stride,
                stride_w=stride,
                pad_h=1,
                pad_w=1,
                window_h=3,
                window_w=3,
                batch_size=input_shape[0],
                input_h=input_shape[1],
                input_w=input_shape[2],
                num_cores_h=grid_size[1],
                num_cores_w=grid_size[0],
                num_cores_nhw=num_cores_nhw,
            )

            config_override = {
                "act_block_w": act_block_w,
                "act_block_h": act_block_h,
                "out_subblock_h": out_subblock_h,
                "out_subblock_w": out_subblock_w,
                "grid_size": grid_size,
                "per_core_out_matrix_height": per_core_act_h,
                "per_core_weight_matrix_width": per_core_weight_w,
                "num_cores_nhw": num_cores_nhw,
            }

            assert out_block_h == per_core_act_h, "out_block_h != per_core_act_h"
            assert weight_block_w == per_core_weight_w, "weight_block_w != per_core_weight_w"

            move_utwh_output = False
            if self.deallocate and (
                self.module_input_shape[0] == 20
                and self.module_input_shape[1] == 56
                and self.module_input_shape[3] == 256
            ):
                move_utwh_output = True
            tt_tensor_conv_weight = ttnn.Tensor(
                conv2_weight.reshape(-1).tolist(),
                conv2_weight.shape,
                (model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32),
                ttnn.ROW_MAJOR_LAYOUT,
            )
            tt_tensor_conv_bias = ttnn.Tensor(
                conv2_bias.tolist(),
                [1, 1, 1, conv2_bias.shape[-1]],
                (model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32),
                ttnn.ROW_MAJOR_LAYOUT,
            )
            self.conv2 = TTPyCompositeConv(
                sliding_window_op_params,
                tt_tensor_conv_weight,
                width,
                width,
                self.device,
                not conv_2d,
                reader_patterns_cache,
                conv_blocking_and_parallelization_config_override=config_override,
                bias=tt_tensor_conv_bias,
                fuse_relu=True,
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                output_dtype=model_config["ACTIVATIONS_DTYPE"],
                math_fidelity=model_config["MATH_FIDELITY"],
                move_utwh_output=move_utwh_output,
                deallocate_activation=True,
                compute_kernel_config=compute_kernel_config,
            )
        else:
            self.conv2 = resnet50_optimized_conv(
                conv2_weight.reshape(-1).tolist(),
                self.conv2_params,
                self.device,
                [act_block_h, act_block_w],
                [act_block_w, weight_block_w],
                [out_subblock_h, out_subblock_w],
                out_block_h,
                grid_size,
                per_core_act_h_ntiles,
                per_core_weight_w_ntiles,
                conv2_bias.tolist(),
                True,
                output_mem_config=self.sharded_memory_config,
                input_tensor_shape=self.conv1_output_shape,
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                output_dtype=model_config["ACTIVATIONS_DTYPE"],
                math_fidelity=model_config["MATH_FIDELITY"],
                act_c_num_blocks=1,
                compute_kernel_config=compute_kernel_config,
            )

        self.conv3_params = [planes * self.expansion, width, 1, 1, 1, 1, 0, 0, dilation, groups]
        self.conv3_output_shape = compute_conv_output_shape(self.conv3_params, self.conv2_output_shape)
        conv3_as_mm_padded_act_height = _nearest_32(
            self.conv3_output_shape[0] * self.conv3_output_shape[1] * self.conv3_output_shape[2]
        )
        matmul_config = None
        assert (
            conv3_as_mm_padded_act_height,
            width,
            planes * self.expansion,
        ) in hardcoded_matmul_config_conv[batch_size]
        # logger.info("Setting matmul config for 1x1 conv (third conv in module)")
        matmul_config = hardcoded_matmul_config_conv[batch_size][
            (conv3_as_mm_padded_act_height, width, planes * self.expansion)
        ]
        if is_grayskull():
            compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
            )
        else:
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

        # 1x1 conv with stride 1 padding 0 is run using regular matmul
        self.conv3 = resnet50_1x1_conv_as_matmul(
            conv3_weight.reshape(-1).tolist(),
            self.conv3_params,
            self.device,
            conv3_bias.tolist(),
            matmul_config,
            output_mem_config=self.sharded_memory_config,
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            output_dtype=model_config["ACTIVATIONS_DTYPE"],
            compute_kernel_config=compute_kernel_config,
        )
        self.conv3_output_shape = compute_conv_output_shape(self.conv3_params, self.conv2_output_shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # logger.info("This module input shape - ", self.module_input_shape)
        # conv1 is 1x1 conv
        # logger.info("Running conv1")
        out = self.conv1(x)

        if not (self.module_input_shape[1] == 56 and self.module_input_shape[3] == 64):
            ds_out = self.downsample_or_noop(x)
            if self.deallocate:
                x.deallocate()

        if not self.conv_halo:
            out = format_tensor(out, ttnn.ROW_MAJOR_LAYOUT, self.device, self.memory_config)
            out = out.reshape(
                self.conv1_output_shape[0],
                self.conv1_output_shape[1],
                self.conv1_output_shape[2],
                self.conv1_output_shape[3],
            )

        # logger.info("Running conv2")
        out = self.conv2(out)
        # conv3 is 1x1 conv
        # logger.info("Running conv3")
        out = self.conv3(out)

        if self.module_input_shape[1] == 56 and self.module_input_shape[3] == 64:
            ds_out = self.downsample_or_noop(x)
            if self.deallocate:
                x.deallocate()

        fused_activations = [ttnn.UnaryOpType.RELU]

        # logger.info("Running eltwise add")
        out = ttnn.add(
            out,
            ds_out,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            memory_config=self.out_memory_config,
            activations=fused_activations,
            output_tensor=out if self.out_in_place else None,
        )
        if self.module_input_shape[0] == 20 and self.module_input_shape[1] == 56 and self.module_input_shape[3] == 64:
            out = ttnn.move(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Bottleneck,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device=None,
        state_dict=None,
        base_address=None,
        fold_batchnorm=False,
        storage_in_dram=True,
        conv_input_face_shape_hw=[224, 224],
        batch_size=1,
        model_config=None,
        sharded=False,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_address_with_dot = base_address  # this is root layer, no dot is needed
        self.state_dict = state_dict
        self.fold_batchnorm = fold_batchnorm
        self.storage_in_dram = storage_in_dram
        self.conv_input_face_shape_hw = conv_input_face_shape_hw
        self.batch_size = batch_size
        self.sharded = sharded
        self.model_config = model_config
        self.reader_patterns_cache = {}
        if self.storage_in_dram:
            self.memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        else:
            self.memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        if sharded:
            self.height_sharded_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1
            )
            self.width_sharded_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1
            )
        else:
            self.height_sharded_memory_config = self.memory_config
            self.width_sharded_memory_config = self.memory_config
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        conv1_weight = state_dict[f"{self.base_address_with_dot}conv1.weight"]
        conv1_bias = None

        self.bn1 = norm_layer(self.inplanes)  # batch norm
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(
            state_dict[f"{self.base_address_with_dot}bn1.num_batches_tracked"], requires_grad=False
        )
        self.bn1.eval()

        if self.fold_batchnorm:
            conv1_weight, conv1_bias = fold_bn_to_conv_weights_bias(conv1_weight, self.bn1)
            self.bn1 = nn.Identity()

        self.conv1_params = [self.inplanes, 3, 7, 7, 2, 2, 3, 3, 1, groups]
        if batch_size == 1:
            act_block_h_datums = 256
            grid_size = (7, 7)
            per_core_act_h_ntiles = 8
        elif batch_size == 2:
            act_block_h_datums = 256
            grid_size = (7, 7)
            per_core_act_h_ntiles = 16
        elif batch_size == 8:
            # 7,7 multi core config triggers non-deterministic output
            # grid_size = (7,7)
            # per_core_act_h_ntiles = 64
            # act_block_h_datums = 256
            # grid_size = (7,8)
            # per_core_act_h_ntiles = 56
            act_block_h_datums = 1024
            grid_size = (12, 9)
            per_core_act_h_ntiles = 32
            self.layer_3_grid_size = (10, 8)
            self.layer_4_grid_size = (7, 8)
        elif batch_size == 16:
            # 7,7 multi core config triggers non-deterministic output
            # grid_size = (7,7)
            # per_core_act_h_ntiles = 64
            # act_block_h_datums = 256
            # grid_size = (7,8)
            # per_core_act_h_ntiles = 56
            act_block_h_datums = 2048
            grid_size = (12, 9)
            per_core_act_h_ntiles = 64
            self.layer_3_grid_size = (11, 8)
            self.layer_4_grid_size = (9, 8)
        elif batch_size == 20:
            act_block_h_datums = 2560
            grid_size = (12, 9)
            per_core_act_h_ntiles = 80
            self.layer_3_grid_size = (12, 8)
            self.layer_4_grid_size = (11, 8)
        self.end_grid_size = (8, 4)
        self.first_conv_num_cores_nhw = 98
        if sharded:
            self.fold_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(10, 7),
                    ),
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 8),
                        ttnn.CoreCoord(3, 8),
                    ),
                }
            )
            self.n_fold_cores = 92

            self.shard_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(11, 7),
                    ),
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 8),
                        ttnn.CoreCoord(1, 8),
                    ),
                }
            )
            self.n_dram_cores = 8
            self.dram_shard_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 0),
                    )
                }
            )

            self.folded_conv1_params = [self.inplanes, 16, 4, 4, 1, 1, 0, 0, 1, groups]
            first_conv_output_padded_nhw_size = _nearest_y(112 * 112 * batch_size, 98 * 32)
            first_conv_output_channels = 64
            assert (
                first_conv_output_padded_nhw_size,
                first_conv_output_channels,
            ) in hardcoded_conv_blocking_and_parallelization_config[batch_size]
            [
                act_block_w,
                act_block_h,
                weight_block_w,
                out_subblock_h,
                out_subblock_w,
                out_block_h,
                grid_size,
                per_core_act_h,
                per_core_weight_w,
                self.first_conv_num_cores_nhw,  # This number is only meaningful for batch 8, 16
            ] = hardcoded_conv_blocking_and_parallelization_config[batch_size][
                (first_conv_output_padded_nhw_size, first_conv_output_channels)
            ]

            sliding_window_op_params = SlidingWindowOpParamsWithParallelConfig(
                stride_h=1,
                stride_w=1,
                pad_h=0,
                pad_w=0,
                window_h=4,
                window_w=4,
                batch_size=batch_size,
                input_h=115,
                input_w=115,
                num_cores_h=grid_size[1],
                num_cores_w=grid_size[0],
                num_cores_nhw=self.first_conv_num_cores_nhw,
            )

            per_core_act_h_ntiles = (int)(per_core_act_h / 32)
            per_core_weight_w_ntiles = (int)(per_core_weight_w / 32)
            conv1_weight = pad_and_fold_conv_filters_for_unity_stride(conv1_weight, 2, 2)
            config_override = {
                "act_block_w": act_block_w,
                "act_block_h": act_block_h,
                "out_subblock_h": out_subblock_h,
                "out_subblock_w": out_subblock_w,
                "grid_size": grid_size,
                "per_core_out_matrix_height": per_core_act_h,
                "per_core_weight_matrix_width": per_core_weight_w,
                "num_cores_nhw": self.first_conv_num_cores_nhw,
            }

            assert out_block_h == per_core_act_h, "out_block_h != per_core_act_h"
            assert weight_block_w == per_core_weight_w, "weight_block_w != per_core_weight_w"

            tt_tensor_conv_weight = ttnn.Tensor(
                conv1_weight.reshape(-1).tolist(),
                conv1_weight.shape,
                (model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32),
                ttnn.ROW_MAJOR_LAYOUT,
            )
            tt_tensor_conv_bias = ttnn.Tensor(
                conv1_bias.tolist(),
                [1, 1, 1, conv1_bias.shape[-1]],
                (model_config["WEIGHTS_DTYPE"] if model_config["WEIGHTS_DTYPE"] != ttnn.bfloat8_b else ttnn.float32),
                ttnn.ROW_MAJOR_LAYOUT,
            )
            if is_grayskull():
                compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                    math_fidelity=model_config["MATH_FIDELITY"],
                    math_approx_mode=True,
                )
            else:
                compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=model_config["MATH_FIDELITY"],
                    math_approx_mode=True,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )
            self.conv1 = TTPyCompositeConv(
                sliding_window_op_params,
                tt_tensor_conv_weight,
                self.inplanes,
                16,
                self.device,
                True,
                self.reader_patterns_cache,
                conv_blocking_and_parallelization_config_override=config_override,
                bias=tt_tensor_conv_bias,
                fuse_relu=True,
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                output_dtype=model_config["ACTIVATIONS_DTYPE"],
                math_fidelity=model_config["MATH_FIDELITY"],
                use_shallow_conv_variant=True,
                deallocate_activation=True,
                padded_input_channels=16,
                compute_kernel_config=compute_kernel_config,
            )
            self.first_conv_op_params = sliding_window_op_params
        else:
            self.conv1 = resnet50_first_conv(
                conv1_weight.reshape(-1).tolist(),
                self.conv1_params,
                self.device,
                [act_block_h_datums, 32],
                [32, 64],
                [32, 64],
                act_block_h_datums,
                grid_size,
                per_core_act_h_ntiles,
                conv1_bias.tolist(),
                8,
                fuse_relu=True,
                out_mem_config=self.height_sharded_memory_config if sharded else None,
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                output_dtype=model_config["ACTIVATIONS_DTYPE"],
                math_fidelity=model_config["MATH_FIDELITY"],
            )
        self.conv1_output_shape = compute_conv_output_shape(
            self.conv1_params,
            [
                batch_size,
                self.conv_input_face_shape_hw[0],
                self.conv_input_face_shape_hw[1],
                self.inplanes,
            ],
        )
        self.relu = ttnn.relu

        self.maxpool_config_params = {"kernel_size": 3, "stride": 2, "pad": 1, "dilation": 1}
        self.max_pool_reader_patterns_cache = {}
        if sharded:
            max_pool_op_params = SlidingWindowOpParamsWithParallelConfig(
                stride_h=2,
                stride_w=2,
                pad_h=1,
                pad_w=1,
                window_h=3,
                window_w=3,
                batch_size=batch_size,
                input_h=self.conv1_output_shape[1],
                input_w=self.conv1_output_shape[2],
                num_cores_h=grid_size[1],
                num_cores_w=grid_size[0],
                num_cores_nhw=self.first_conv_num_cores_nhw,
            )
            self.maxpool = TTPyMaxPool(max_pool_op_params, self.device, self.max_pool_reader_patterns_cache)
        else:
            self.maxpool = TtMaxPool(
                self.device,
                self.conv1_output_shape[0],  ## in_n
                self.conv1_output_shape[1],  ## in_h
                self.conv1_output_shape[2],  ## in_w
                kernel_size=self.maxpool_config_params["kernel_size"],
                stride=self.maxpool_config_params["stride"],
                padding=self.maxpool_config_params["pad"],
                output_mem_config=self.height_sharded_memory_config,
                nblocks=1,
                channels_last=True,
                reshape_2d=True,
            )

        self.maxpool_output_shape = compute_max_pool_shape(3, 2, 1, self.conv1_output_shape)
        self.layer1, self.layer1_output_shape = self._make_layer(
            block,
            64,
            layers[0],
            name="layer1",
            state_dict=state_dict,
            layer_input_shape=self.maxpool_output_shape,
            batch_size=batch_size,
            sharded=ttnn.TensorMemoryLayout.HEIGHT_SHARDED if sharded else None,
            out_sharded=True,
            conv_halo=True if sharded else False,
            model_config=model_config,
        )
        self.layer2, self.layer2_output_shape = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            name="layer2",
            state_dict=state_dict,
            layer_input_shape=self.layer1_output_shape,
            batch_size=batch_size,
            sharded=ttnn.TensorMemoryLayout.HEIGHT_SHARDED if sharded else None,
            out_sharded=True,
            use_downsample_op_and_mm_for_conv1x1_s2=True if sharded else False,
            conv_halo=True if sharded else False,
            model_config=model_config,
        )
        self.layer3, self.layer3_output_shape = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            name="layer3",
            state_dict=state_dict,
            layer_input_shape=self.layer2_output_shape,
            batch_size=batch_size,
            sharded=ttnn.TensorMemoryLayout.BLOCK_SHARDED if sharded else None,
            out_sharded=True,
            use_downsample_op_and_mm_for_conv1x1_s2=True if sharded else False,
            model_config=model_config,
            conv_halo=True if sharded else False,
            conv_2d=True,
        )
        self.layer4, self.layer4_output_shape = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            name="layer4",
            state_dict=state_dict,
            layer_input_shape=self.layer3_output_shape,
            batch_size=batch_size,
            sharded=ttnn.TensorMemoryLayout.BLOCK_SHARDED if sharded else None,
            out_sharded=True,
            use_downsample_op_and_mm_for_conv1x1_s2=True if sharded else False,
            model_config=model_config,
            conv_halo=True if sharded else False,
            conv_2d=True,
        )

        # All modules in RN50 are unrolled here. One variable for each module. Only specific number of modules supported - layers MUST equal to [3, 4, 6, 3]
        assert layers == [3, 4, 6, 3]
        self.layer1_module1 = self.layer1[0]
        self.layer1_module2 = self.layer1[1]
        self.layer1_module3 = self.layer1[2]

        self.layer2_module1 = self.layer2[0]
        self.layer2_module2 = self.layer2[1]
        self.layer2_module3 = self.layer2[2]
        self.layer2_module4 = self.layer2[3]

        self.layer3_module1 = self.layer3[0]
        self.layer3_module2 = self.layer3[1]
        self.layer3_module3 = self.layer3[2]
        self.layer3_module4 = self.layer3[3]
        self.layer3_module5 = self.layer3[4]
        self.layer3_module6 = self.layer3[5]

        self.layer4_module1 = self.layer4[0]
        self.layer4_module2 = self.layer4[1]
        self.layer4_module3 = self.layer4[2]

        self.avgpool = TtAvgPool(self.device)

        fc_weight = pad_weight(state_dict[f"{self.base_address_with_dot}fc.weight"])
        fc_weight = torch.transpose(fc_weight, 3, 2)
        fc_weight = ttnn.Tensor(
            fc_weight.reshape(-1).tolist(),
            fc_weight.shape,
            model_config["WEIGHTS_DTYPE"],
            ttnn.ROW_MAJOR_LAYOUT,
        ).to(ttnn.TILE_LAYOUT)
        fc_bias = pad_weight(state_dict[f"{self.base_address_with_dot}fc.bias"])
        fc_bias = ttnn.Tensor(
            fc_bias.reshape(-1).tolist(),
            fc_bias.shape,
            model_config["WEIGHTS_DTYPE"],
            ttnn.ROW_MAJOR_LAYOUT,
        ).to(ttnn.TILE_LAYOUT)
        self.fc = ResnetLinear(
            512 * block.expansion,
            1024,
            fc_weight,
            fc_bias,
            transpose=False,
            output_mem_config=self.width_sharded_memory_config,
            model_config=model_config,
            device=self.device,
            batch_size=batch_size,
        )  # num_classes = 1000
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def __del__(self):
        # Need to clear global configs for each Resnet run
        self.reader_patterns_cache.clear()
        self.max_pool_reader_patterns_cache.clear()

    def _make_layer(
        self,
        block: Bottleneck,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        name: str = None,
        state_dict=None,
        layer_input_shape=[],
        batch_size=1,
        sharded=None,
        out_sharded=False,
        use_downsample_op_and_mm_for_conv1x1_s2=False,
        model_config=None,
        conv_halo=False,
        conv_2d=False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        self.downsample_conv_on_tt = None
        self.norm_layer_after_downsample_conv_on_tt = None
        if sharded is not None:
            self.ds_conv_output_memory_config = ttnn.MemoryConfig(sharded, ttnn.BufferType.L1)
        else:
            self.ds_conv_output_memory_config = self.memory_config
        if use_downsample_op_and_mm_for_conv1x1_s2:
            assert sharded
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            nl = norm_layer(planes * block.expansion)
            nl.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.weight"])
            nl.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.bias"])
            nl.running_mean = nn.Parameter(
                state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_mean"]
            )
            nl.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_var"])
            nl.num_batches_tracked = nn.Parameter(
                state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.num_batches_tracked"],
                requires_grad=False,
            )
            nl.eval()
            downsample_conv_weight = state_dict[f"{self.base_address_with_dot}{name}.0.downsample.0.weight"]
            downsample_conv_bias = None

            if self.fold_batchnorm:
                downsample_conv_weight, downsample_conv_bias = fold_bn_to_conv_weights_bias(downsample_conv_weight, nl)
                nl = nn.Identity()

            # With single buffered input CB, these shapes work -
            # hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_downsample_conv = {
            #     (3136, 256) : [128, 128, 128, 64] ,
            #     (800, 512) : [128, 128, 128, 64] ,
            #     (224, 1024) : [64, 128, 64, 64],
            #     (64, 2048) : [64, 128, 64, 64] ,
            # }

            downsample_output_channels = planes * block.expansion
            self.downsample_params = [
                downsample_output_channels,
                self.inplanes,
                1,
                1,
                stride,
                stride,
                0,
                0,
                self.dilation,
                1,
            ]
            self.downsample_conv_output_shape = compute_conv_output_shape(self.downsample_params, layer_input_shape)
            is_downsample_1x1_conv = stride == 1
            is_1x1_downsample_conv_sanity_check = (
                self.downsample_params[2] == 1
                and self.downsample_params[3] == 1
                and self.downsample_params[4] == 1
                and self.downsample_params[5] == 1
                and self.downsample_params[6] == 0
                and self.downsample_params[7] == 0
            )
            assert is_1x1_downsample_conv_sanity_check == is_downsample_1x1_conv
            downsample_output_padded_face_size = _nearest_32(
                self.downsample_conv_output_shape[0]
                * self.downsample_conv_output_shape[1]
                * self.downsample_conv_output_shape[2]
            )
            matmul_config = None

            if is_grayskull():
                compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                    math_fidelity=model_config["MATH_FIDELITY"],
                    math_approx_mode=True,
                )
            else:
                compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=model_config["MATH_FIDELITY"],
                    math_approx_mode=True,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )

            if is_downsample_1x1_conv:
                assert (
                    downsample_output_padded_face_size,
                    self.inplanes,
                    downsample_output_channels,
                ) in hardcoded_matmul_config_conv[batch_size]
                # logger.info("Setting matmul config for 1x1 conv (downsample stride 1 conv in module)")
                matmul_config = hardcoded_matmul_config_conv[batch_size][
                    (downsample_output_padded_face_size, self.inplanes, downsample_output_channels)
                ]
                self.downsample_conv_on_tt = resnet50_1x1_conv_as_matmul(
                    downsample_conv_weight.reshape(-1).tolist(),
                    self.downsample_params,
                    self.device,
                    downsample_conv_bias.tolist(),
                    matmul_config,
                    output_mem_config=self.ds_conv_output_memory_config,
                    weights_dtype=model_config["WEIGHTS_DTYPE"],
                    output_dtype=model_config["ACTIVATIONS_DTYPE"],
                    compute_kernel_config=compute_kernel_config,
                )
            elif use_downsample_op_and_mm_for_conv1x1_s2:
                assert (
                    downsample_output_padded_face_size,
                    self.inplanes,
                    downsample_output_channels,
                ) in hardcoded_matmul_config_conv[batch_size]
                matmul_config = hardcoded_matmul_config_conv[batch_size][
                    (downsample_output_padded_face_size, self.inplanes, downsample_output_channels)
                ]
                assert stride == 2
                downsample_op_params = [
                    batch_size,
                    layer_input_shape[1],
                    layer_input_shape[2],
                    stride,
                    stride,
                ]
                # logger.info("Calling ds op and matmul op, input shape - ", layer_input_shape)

                self.downsample_conv_on_tt = resnet50_1x1_conv_s2_as_downsample_and_matmul(
                    downsample_conv_weight.reshape(-1).tolist(),
                    self.downsample_params,
                    downsample_op_params,  # used by downsample op
                    self.device,
                    downsample_conv_bias.tolist(),
                    matmul_config,
                    self.ds_conv_output_memory_config,
                    weights_dtype=model_config["WEIGHTS_DTYPE"],
                    output_dtype=model_config["ACTIVATIONS_DTYPE"],
                    compute_kernel_config=compute_kernel_config,
                )
            else:
                assert (
                    downsample_output_padded_face_size,
                    downsample_output_channels,
                ) in hardcoded_conv_blocking_and_parallelization_config[batch_size]
                [
                    act_block_w_datums,
                    act_block_h_datums,
                    weight_block_w_datums,
                    out_subblock_h_datums,
                    out_subblock_w_datums,
                    out_block_h_datums,
                    grid_size,
                    per_core_act_h,
                    per_core_weight_w,
                    num_cores_nhw,  # This number is only meaningful for batch 8, 16
                ] = hardcoded_conv_blocking_and_parallelization_config[batch_size][
                    (downsample_output_padded_face_size, downsample_output_channels)
                ]
                assert per_core_act_h % 32 == 0
                per_core_act_h_ntiles = (int)(per_core_act_h / 32)
                per_core_weight_w_ntiles = (int)(per_core_weight_w / 32)
                assert self.inplanes % act_block_w_datums == 0
                self.downsample_conv_on_tt = resnet50_optimized_conv(
                    downsample_conv_weight.reshape(-1).tolist(),
                    self.downsample_params,
                    self.device,
                    [act_block_h_datums, act_block_w_datums],
                    [act_block_w_datums, weight_block_w_datums],
                    [out_subblock_h_datums, out_subblock_w_datums],
                    out_block_h_datums,
                    grid_size,
                    per_core_act_h_ntiles,
                    per_core_weight_w_ntiles,
                    downsample_conv_bias.tolist(),
                    output_mem_config=self.ds_conv_output_memory_config,
                    weights_dtype=model_config["WEIGHTS_DTYPE"],
                    output_dtype=model_config["ACTIVATIONS_DTYPE"],
                    math_fidelity=model_config["MATH_FIDELITY"],
                    compute_kernel_config=compute_kernel_config,
                )
            self.norm_layer_after_downsample_conv_on_tt = nl

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                device=self.device,
                state_dict=self.state_dict,
                base_address=f"{self.base_address_with_dot}{name}.0",
                fold_batchnorm=self.fold_batchnorm,
                downsample_conv_on_tt=self.downsample_conv_on_tt,
                norm_layer_after_downsample_conv_on_tt=self.norm_layer_after_downsample_conv_on_tt,
                downsample_params=self.downsample_params,
                storage_in_dram=self.storage_in_dram,
                input_shape=layer_input_shape,
                batch_size=batch_size,
                sharded=sharded,
                out_sharded=sharded is not None,
                use_downsample_op_and_mm_for_conv1x1_s2=use_downsample_op_and_mm_for_conv1x1_s2,
                model_config=model_config,
                conv_halo=conv_halo,
                conv_2d=conv_2d,
                reader_patterns_cache=self.reader_patterns_cache,
            )
        )
        self.inplanes = planes * block.expansion
        for block_num in range(1, blocks):
            previous_layer = layers[-1]
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    device=self.device,
                    state_dict=self.state_dict,
                    base_address=f"{self.base_address_with_dot}{name}.{block_num}",
                    fold_batchnorm=self.fold_batchnorm,
                    storage_in_dram=self.storage_in_dram,
                    input_shape=previous_layer.conv3_output_shape,
                    batch_size=batch_size,
                    sharded=sharded,
                    out_sharded=True if block_num != blocks - 1 else out_sharded,
                    model_config=model_config,
                    conv_halo=conv_halo,
                    conv_2d=conv_2d,
                    reader_patterns_cache=self.reader_patterns_cache,
                )
            )
        last_layer_shape = layers[-1].conv3_output_shape
        return layers, last_layer_shape

    def preprocessing(self, x: torch.Tensor) -> ttnn.Tensor:
        if self.sharded:
            x = pad_and_fold_conv_activation_for_unity_stride(x, 3, 3, 2, 2)
            x = torch.permute(x, (0, 2, 3, 1))
            x = x.reshape(
                1,
                1,
                x.shape[0] * x.shape[1] * x.shape[2],
                x.shape[3],
            )
            input_size_to_shard_evenly = _nearest_y(x.shape[2], self.first_conv_num_cores_nhw * 32)
            x = torch.nn.functional.pad(x, (0, 0, 0, input_size_to_shard_evenly - x.shape[2], 0, 0))

            x = ttnn.Tensor(x, ttnn.bfloat16)
        else:
            extra_padding_for_32B_alignment = 25
            x = torch.nn.functional.pad(x, (3, 4 + extra_padding_for_32B_alignment, 3, 3, 0, 1))
            x = torch.permute(x, (0, 2, 3, 1))
            x = ttnn.Tensor(x, ttnn.bfloat16)
        return x

    def preprocessing_with_fold(self, x: torch.Tensor) -> ttnn.Tensor:
        if not self.sharded:
            raise ValueError("Preprocessing is only supported for sharded model")

        stride_h = 2
        stride_w = 2

        # NCWH -> NWHC
        x = torch.permute(x, (0, 2, 3, 1))

        # pad to 230x230x4
        C = _nearest_y(x.shape[3], 4)
        x = torch.nn.functional.pad(x, (0, C - x.shape[3], 3, 3, 3, 3))

        # reshape to [N, H, W / stride_w, C * stride_w]. It's a no-op in torch, and this way we only need to fold on height.
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // stride_w, x.shape[3] * stride_w)

        NHW = x.shape[0] * x.shape[1] * x.shape[2]
        NHW_even = _nearest_y(NHW // stride_h, self.first_conv_num_cores_nhw * 32)

        shard_spec = ttnn.ShardSpec(
            self.fold_grid,
            [NHW // self.n_fold_cores, x.shape[3]],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        x = torch2tt_tensor(
            x,
            self.device,
            tt_layout=ttnn.ROW_MAJOR_LAYOUT,
            tt_memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                shard_spec,
            ),
        )

        # fold for unity stride on device
        x = ttnn.fold(x, stride_h=stride_h, stride_w=1)

        shard_shape = [
            NHW_even // self.first_conv_num_cores_nhw,
            x.get_legacy_shape()[3],
        ]

        x = ttnn.reshard(
            x,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    self.shard_grid,
                    shard_shape,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        )

        return x

    def forward(self, x: ttnn.Tensor, write_event=None, op_event=None, final_out_mem_config=None) -> ttnn.Tensor:
        x_in = None
        if not self.sharded:
            original_A_cl_host_shape = x.get_legacy_shape()
            x = x.reshape(
                x.get_legacy_shape()[0],
                x.get_legacy_shape()[1],
                1,
                x.get_legacy_shape()[2] * x.get_legacy_shape()[3],
            )

            x = x.to(self.device, self.memory_config)  # to l1
            # re-shape back to original shape (N, H, W, C)
            x = x.reshape(
                original_A_cl_host_shape[0],
                original_A_cl_host_shape[1],
                original_A_cl_host_shape[2],
                original_A_cl_host_shape[3],
            )
        else:
            x_shape = x.get_legacy_shape()
            shard_spec = ttnn.ShardSpec(
                self.shard_grid,
                [
                    x_shape[2] // self.first_conv_num_cores_nhw,
                    x_shape[3],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            )
            mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                shard_spec,
            )
            if write_event is not None:
                ttnn.wait_for_event(0, write_event)
            if x.storage_type() != ttnn.StorageType.DEVICE:
                x = x.to(self.device, mem_config)
            elif x.memory_config().is_sharded():
                if x.memory_config() != mem_config:
                    x = ttnn.reshard(x, mem_config)
                else:
                    x_in = x
            else:
                x = ttnn.interleaved_to_sharded(x, mem_config)
            if op_event is not None:
                ttnn.record_event(0, op_event)

        x = self.conv1(x)
        if x_in is not None:
            x_in.deallocate()
        # Relu is fused with conv1

        if self.batch_size == 20:
            x = ttnn.move(x)

        if not self.sharded:
            x = format_tensor(x, ttnn.ROW_MAJOR_LAYOUT, self.device, self.memory_config)
            x = x.reshape(
                self.conv1_output_shape[0],
                self.conv1_output_shape[1],
                self.conv1_output_shape[2],
                self.conv1_output_shape[3],
            )
        x = self.maxpool(x)

        x = x.reshape(
            1,
            1,
            self.maxpool_output_shape[0] * self.maxpool_output_shape[1] * self.maxpool_output_shape[2],
            self.maxpool_output_shape[3],
        )
        x = ttnn.tilize(
            x,
            memory_config=self.height_sharded_memory_config,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            use_multicore=True,
        )
        if self.batch_size == 20:
            x = ttnn.move(x)

        x = self.layer1_module1(x)
        x = self.layer1_module2(x)
        x = self.layer1_module3(x)

        x = self.layer2_module1(x)
        x = self.layer2_module2(x)
        x = self.layer2_module3(x)
        x = self.layer2_module4(x)
        if self.sharded:
            x_shape = x.get_legacy_shape()
            reshard_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(self.layer_3_grid_size[0] - 1, self.layer_3_grid_size[1] - 1),
                            )
                        }
                    ),
                    [
                        math.ceil((x_shape[-2] // 32) / self.layer_3_grid_size[0]) * 32,
                        x_shape[-1] // self.layer_3_grid_size[1],
                    ],
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
            x = ttnn.reshard(
                x,
                reshard_mem_config,
            )
        x = self.layer3_module1(x)
        x = self.layer3_module2(x)
        x = self.layer3_module3(x)
        x = self.layer3_module4(x)
        x = self.layer3_module5(x)
        x = self.layer3_module6(x)
        if self.sharded:
            x_shape = x.get_legacy_shape()
            reshard_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(self.layer_4_grid_size[0] - 1, self.layer_4_grid_size[1] - 1),
                            )
                        }
                    ),
                    [
                        math.ceil((x_shape[-2] // 32) / self.layer_4_grid_size[0]) * 32,
                        x_shape[-1] // self.layer_4_grid_size[1],
                    ],
                    ttnn.ShardOrientation.COL_MAJOR,
                    False,
                ),
            )
            x = ttnn.reshard(
                x,
                reshard_mem_config,
            )
        x = self.layer4_module1(x)
        x = self.layer4_module2(x)
        x = self.layer4_module3(x)

        if self.sharded:
            x_shape = x.get_legacy_shape()
            reshard_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(self.end_grid_size[0] - 1, self.end_grid_size[1] - 1),
                            )
                        }
                    ),
                    [
                        x.volume() // x_shape[-1],
                        x_shape[-1] // (self.end_grid_size[0] * self.end_grid_size[1]),
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
            x = ttnn.reshard(
                x,
                reshard_mem_config,
            )

        unpadded_shape = x.shape_without_padding()

        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=(
                unpadded_shape[0] - 1,
                unpadded_shape[1] - 1,
                unpadded_shape[2] - 1,
                unpadded_shape[3] - 1,
            ),
            memory_config=self.width_sharded_memory_config,
        )

        x_shape = x.get_legacy_shape()
        x = x.reshape(
            self.batch_size,
            x_shape[1],
            x_shape[2] // self.batch_size,
            x_shape[3],
        )

        unpadded_shape = x.get_legacy_shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        if self.sharded:
            x = ttnn.tilize_with_val_padding(
                x,
                padded_shape,
                0,
                memory_config=self.width_sharded_memory_config,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
        else:
            x = ttnn.pad(
                x,
                padded_shape,
                [0, 0, 0, 0],
                0,
                memory_config=self.memory_config,
                use_multicore=True,
            )
            x = ttnn.tilize(
                x,
                memory_config=self.memory_config,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                use_multicore=True,
            )

        x = self.avgpool(x, self.width_sharded_memory_config)

        x_shape = x.get_legacy_shape()
        unpadded_shape_end = [
            x_shape[0] - 1,
            x_shape[1] - 1,
            1 - 1,
            x_shape[3] - 1,
        ]
        if self.sharded:
            x = ttnn.untilize_with_unpadding(
                x,
                output_tensor_end=unpadded_shape_end,
                memory_config=self.width_sharded_memory_config,
            )
        else:
            x = ttnn.untilize(x, memory_config=self.memory_config, use_multicore=True)
            x = ttnn.slice(x, (0, 0, 0, 0), unpadded_shape_end, memory_config=self.memory_config)

        x_shape = x.get_legacy_shape()
        x = x.reshape(1, x_shape[1], self.batch_size * x_shape[2], x_shape[3])

        unpadded_shape = x.get_legacy_shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        if self.sharded:
            x = ttnn.tilize_with_val_padding(
                x,
                padded_shape,
                0,
                memory_config=self.width_sharded_memory_config,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
        else:
            x = ttnn.pad(
                x,
                padded_shape,
                [0, 0, 0, 0],
                0,
                memory_config=self.memory_config,
                use_multicore=True,
            )
            x = ttnn.tilize(
                x,
                memory_config=self.memory_config,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                use_multicore=True,
            )

        x = self.fc(x)
        x_shape = x.shape_without_padding()
        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=(x_shape[0] - 1, x_shape[1] - 1, x_shape[2] - 1, 1000 - 1),
            memory_config=self.memory_config if final_out_mem_config is None else final_out_mem_config,
        )
        x_shape = x.get_legacy_shape()
        x = x.reshape(
            self.batch_size,
            x_shape[1],
            x_shape[2] // self.batch_size,
            x_shape[3],
        )

        return x
