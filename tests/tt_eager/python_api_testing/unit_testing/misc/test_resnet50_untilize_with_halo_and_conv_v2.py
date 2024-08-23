# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.demos.resnet.tt.metalResnetBlock50 import (
    compute_conv_output_shape,
    resnet50_1x1_conv_as_matmul,
    resnet50_optimized_conv,
    _nearest_32,
    _nearest_y,
    format_tensor,
)
from models.utility_functions import skip_for_grayskull

from ttnn.operations.conv.tt_py_composite_conv import (
    TTPyCompositeConv,
    SlidingWindowOpParamsWithParallelConfig,
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
        (
            25088,
            64,
            64,
        ): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        (
            25088,
            64,
            256,
        ): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        (
            25088,
            256,
            64,
        ): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        (
            25088,
            256,
            128,
        ): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        (
            6272,
            128,
            512,
        ): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        (
            6272,
            512,
            128,
        ): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
            in0_block_w=4,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=5,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 256): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=16,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=5,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=16,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=5,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=16,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=5,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (416, 512, 2048): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=2,
            per_core_N=10,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (416, 2048, 512): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=16,
            out_subblock_h=2,
            out_subblock_w=3,
            per_core_M=2,
            per_core_N=3,
            transpose_mcast=True,
            fused_activation=None,
        ),
    },
}

hardcoded_conv_blocking_and_parallelization_config = {
    8: {
        (100352, 64): [16 * 4, 1024, 1, 64, 128, 64, 1024, (12, 9), 1024, 64, 98],
        (25088, 64): [64 * 3, 256, 1, 64, 128, 64, 256, (12, 9), 256, 64, 98],
        (6272, 128): [128 * 3, 64, 1, 128, 64, 128, 64, (12, 9), 64, 128, 98],
        (1568, 256): [256, 160, 8, 32, 160, 32, 160, (10, 8), 160, 32, 10],
        (416, 512): [512, 64, 8, 64, 64, 64, 64, (7, 8), 64, 64, 7],
    },
    16: {
        (200704, 64): [16 * 4, 1024, 1, 64, 128, 64, 2048, (12, 9), 2048, 64, 98],
        (50176, 64): [64 * 3, 256, 1, 64, 128, 64, 512, (12, 9), 512, 64, 98],
        (12544, 128): [128 * 3, 128, 1, 128, 64, 128, 128, (12, 9), 128, 128, 98],
        (3136, 256): [256, 288, 8, 32, 96, 32, 288, (11, 8), 288, 32, 11],
        (800, 512): [512, 96, 8, 64, 96, 64, 96, (9, 8), 96, 64, 9],
    },
    20: {
        (250880, 64): [16 * 4, 1280, 1, 64, 128, 64, 2560, (12, 9), 2560, 64, 98],  # Won't fit for bfloat16 activations
        (62720, 64): [64 * 3, 320, 1, 64, 64, 64, 640, (12, 9), 640, 64, 98],
        # (62720, 64): [64 * 3, 320, 1, 64, 64, 64, 640, (12, 9), 640, 64, 98], # TODO: This fits for BFLOAT16, but do we need?
        (15680, 128): [128 * 3, 160, 1, 128, 32, 128, 160, (12, 9), 160, 128, 98],
        (3936, 256): [256, 352, 8, 32, 32, 32, 352, (12, 8), 352, 32, 12],
        (992, 512): [512, 96, 8, 64, 96, 64, 96, (11, 8), 96, 64, 11],
    },
}


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("N", (8, 16, 20), ids=["batch_8", "batch_16", "batch_20"])
@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and C padding to tile width
        (64, 16, 115, 115, 4, 4, 1, 1, 0, 0),
        # layer1
        (64, 64, 56, 56, 3, 3, 1, 1, 1, 1),
        # layer2
        # (512, 256, 56, 56, 1, 1, 2, 2, 0, 0), # not supported yet
        (128, 128, 56, 56, 3, 3, 2, 2, 1, 1),
        (128, 128, 28, 28, 3, 3, 1, 1, 1, 1),
        # layer3
        (256, 256, 28, 28, 3, 3, 2, 2, 1, 1),  # not supported yet
        # (1024, 512, 28, 28, 1, 1, 2, 2, 0, 0), # not supported yet
        (256, 256, 14, 14, 3, 3, 1, 1, 1, 1),
        # layer4
        (512, 512, 14, 14, 3, 3, 2, 2, 1, 1),  # not supported yet
        # (2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0), # not supported yet
        (512, 512, 7, 7, 3, 3, 1, 1, 1, 1),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["weights_BFLOAT16", "weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["activations_BFLOAT16", "activations_BFLOAT8_B"],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi], ids=["HiFi4", "LoFi"])
def test_resnet50_conv(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    N,
    K,
    C,
    H,
    W,
    R,
    S,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
):
    if math_fidelity != ttnn.MathFidelity.LoFi:
        pytest.skip(
            "By default, only run tests with LoFi math for pipelines. For local unit testing, enable the other variants by uncommenting the skip here!"
        )

    if (
        activations_dtype == ttnn.bfloat16
        and N == 20
        and (K == 64 or (stride_h == 2 and (K == 256 or (K == 128 and weights_dtype == ttnn.bfloat16))))
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    interleaved_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    for i in range(1):  # increase num of iterations to test op caching
        # assert C % 32 == 0
        assert K % 32 == 0
        torch.manual_seed(0)
        conv_input_shape = [N, C, H, W]
        conv_weight_shape = [K, C, R, S]
        conv_bias_shape = [1, 1, 1, K]
        conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
        conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
        conv_input_shape_nhwc = conv_input_pyt_nhwc.shape
        conv_weight_pyt = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
        conv_bias_pyt = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()

        is_1x1_conv = R == 1 and S == 1 and stride_h == 1 and stride_w == 1 and pad_h == 0 and pad_w == 0

        conv_params = [K, C, R, S, stride_h, stride_w, pad_h, pad_w, 1, 1]
        conv_output_shape = compute_conv_output_shape(conv_params, conv_input_shape_nhwc)
        logger.info(f"Conv output shape - {conv_output_shape}")
        conv_as_mm_padded_act_height = _nearest_32(conv_output_shape[0] * conv_output_shape[1] * conv_output_shape[2])

        assert (conv_as_mm_padded_act_height, K) in hardcoded_conv_blocking_and_parallelization_config[N]
        conv_blocking_and_parallelization_config = hardcoded_conv_blocking_and_parallelization_config[N][
            (conv_as_mm_padded_act_height, K)
        ]
        assert len(conv_blocking_and_parallelization_config) == 11

        [
            act_block_w,
            act_block_h,
            act_c_num_blocks,
            weight_block_w,
            out_subblock_h,
            out_subblock_w,
            out_block_h,
            grid_size,
            per_core_out_matrix_h,
            per_core_weight_matrix_w,
            num_cores_nhw,
        ] = conv_blocking_and_parallelization_config
        if R == 1 and S == 1:
            assert C % act_block_w == 0
        else:
            assert act_block_w == C or act_block_w == C * S

        assert per_core_out_matrix_h % 32 == 0
        per_core_out_matrix_h_ntiles = (int)(per_core_out_matrix_h / 32)
        per_core_weight_matrix_w_ntiles = (int)(per_core_weight_matrix_w / 32)

        compute_grid_size = device.compute_with_storage_grid_size()
        if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
            pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

        out_golden = torch.nn.functional.conv2d(
            conv_input_pyt,
            conv_weight_pyt,
            bias=conv_bias_pyt.reshape(-1),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
        )

        config_override = {
            "act_block_w": act_block_w,
            "act_block_h": act_block_h,
            "out_subblock_h": out_subblock_h,
            "out_subblock_w": out_subblock_w,
            "grid_size": grid_size,
            "per_core_out_matrix_height": per_core_out_matrix_h,
            "per_core_weight_matrix_width": per_core_weight_matrix_w,
            "num_cores_nhw": num_cores_nhw,
        }

        ###############################################
        # Directly run old conv (row major input)
        # NOTE: New conv should have identical output
        ###############################################
        conv = resnet50_optimized_conv(
            conv_weight_pyt.reshape(-1).tolist(),
            conv_params,
            device,
            [act_block_h, act_block_w],
            [act_block_w, weight_block_w],
            [out_subblock_h, out_subblock_w],
            out_block_h,
            grid_size,
            per_core_out_matrix_h_ntiles,
            per_core_weight_matrix_w_ntiles,
            conv_bias_pyt.reshape(-1).tolist(),
            output_mem_config=interleaved_mem_config,
            weights_dtype=weights_dtype,
            output_dtype=activations_dtype,
            math_fidelity=math_fidelity,
        )

        conv_input_on_device = ttnn.Tensor(
            conv_input_pyt_nhwc.reshape(-1).tolist(),
            conv_input_pyt_nhwc.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        ).to(device, interleaved_mem_config)

        output_on_device = conv(conv_input_on_device)

        # convert tiled output to RM
        assert output_on_device.get_layout() == ttnn.TILE_LAYOUT
        output_on_device = format_tensor(output_on_device, ttnn.ROW_MAJOR_LAYOUT, device, interleaved_mem_config)
        output_on_device = output_on_device.reshape(
            conv_output_shape[0],
            conv_output_shape[1],
            conv_output_shape[2],
            conv_output_shape[3],
        )

        # Copy to host
        out = output_on_device.cpu()
        assert out.get_layout() == ttnn.ROW_MAJOR_LAYOUT

        out_result = out.to_torch()
        # NHWC to NCHW
        out_result = torch.transpose(out_result, 2, 3)
        out_result_baseline = torch.transpose(out_result, 1, 2)

        sliding_window_op_params = SlidingWindowOpParamsWithParallelConfig(
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            window_h=R,
            window_w=S,
            batch_size=N,
            input_h=H,
            input_w=W,
            num_cores_h=grid_size[1],
            num_cores_w=grid_size[0],
            num_cores_nhw=num_cores_nhw,
        )
        is_1d_systolic = act_c_num_blocks == 1
        reader_patterns_cache = {}

        tt_tensor_conv_weight = ttnn.Tensor(
            conv_weight_pyt.reshape(-1).tolist(),
            conv_weight_pyt.shape,
            weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        tt_tensor_conv_bias = ttnn.Tensor(
            conv_bias_pyt.reshape(-1).tolist(),
            conv_bias_pyt.shape,
            weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
            ttnn.ROW_MAJOR_LAYOUT,
        )

        conv = TTPyCompositeConv(
            sliding_window_op_params,
            tt_tensor_conv_weight,
            K,
            C,
            device,
            is_1d_systolic,
            reader_patterns_cache,
            bias=tt_tensor_conv_bias,
            conv_blocking_and_parallelization_config_override=config_override,
            weights_dtype=weights_dtype,
            output_dtype=activations_dtype,
            math_fidelity=math_fidelity,
            use_shallow_conv_variant=(C == 16),
            deallocate_activation=True,
            padded_input_channels=16 if C == 16 else None,
        )

        conv_input = ttnn.Tensor(
            conv_input_pyt_nhwc.reshape(-1).tolist(),
            conv_input_pyt_nhwc.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )

        # Convert activation RM to tile layout
        conv_input_on_device = conv_input.reshape(
            1,
            1,
            conv_input_shape_nhwc[0] * conv_input_shape_nhwc[1] * conv_input_shape_nhwc[2],
            conv_input_shape_nhwc[3],
        ).to(device, interleaved_mem_config)
        if C >= 32:
            conv_input_on_device = format_tensor(conv_input_on_device, ttnn.TILE_LAYOUT, device, interleaved_mem_config)

        input_size_to_shard_evenly = _nearest_y(
            conv_input_shape_nhwc[0] * conv_input_shape_nhwc[1] * conv_input_shape_nhwc[2], num_cores_nhw * 32
        )
        untilize_with_halo_input_shard_height = (int)(input_size_to_shard_evenly / num_cores_nhw)
        # Convert interleaved to sharded
        if act_c_num_blocks > 1:  # 2D conv
            conv_input_on_device = ttnn.interleaved_to_sharded(
                conv_input_on_device,
                grid_size,
                [
                    untilize_with_halo_input_shard_height,
                    (int)(C / act_c_num_blocks),
                ],  # act_block_w_datums may include reads of multiple pixels in window
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
            )
        else:
            conv_input_on_device = ttnn.interleaved_to_sharded(
                conv_input_on_device,
                grid_size,
                [
                    untilize_with_halo_input_shard_height,
                    C,
                ],  # act_block_w_datums may include reads of multiple pixels in window
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )

        # Optimized conv v2
        output_on_device = conv(conv_input_on_device)

        # Convert sharded output to tiled interleaved
        output_on_device = ttnn.sharded_to_interleaved(output_on_device, interleaved_mem_config)

        # convert tiled output to RM
        assert output_on_device.get_layout() == ttnn.TILE_LAYOUT
        output_on_device = format_tensor(output_on_device, ttnn.ROW_MAJOR_LAYOUT, device, interleaved_mem_config)
        output_on_device = output_on_device.reshape(
            conv_output_shape[0],
            conv_output_shape[1],
            conv_output_shape[2],
            conv_output_shape[3],
        )

        # Copy to host and compare against pytorch
        out = output_on_device.cpu()
        assert out.get_layout() == ttnn.ROW_MAJOR_LAYOUT

        out_result = out.to_torch()
        # NHWC to NCHW
        out_result = torch.transpose(out_result, 2, 3)
        out_result = torch.transpose(out_result, 1, 2)

        assert len(reader_patterns_cache) == 2
        assert "conv" in reader_patterns_cache and "halo" in reader_patterns_cache
        reader_patterns_cache.clear()

        # Compare baseline against golden
        assert out_result_baseline.shape == out_golden.shape
        passing_pcc_baseline, output_pcc_baseline = comp_pcc(out_golden, out_result_baseline, 0.99)
        logger.debug(f"Passing baseline={passing_pcc_baseline}")
        logger.debug(f"Output pcc baseline={output_pcc_baseline}")

        # Compare out result against golden
        assert out_result.shape == out_golden.shape
        passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
        logger.debug(f"Passing={passing_pcc}")
        logger.debug(f"Output pcc={output_pcc}")
        assert passing_pcc

        # Compare baseline to output (should be identical)
        if activations_dtype == ttnn.bfloat8_b and (K == 256 or K == 512):
            pytest.xfail("PCC of output baseline is slightly lower than with new conv. DEBUG!")
        else:
            ## NOTE: the "old conv" is too old now. We get better PCC with current version.
            # eq = torch.equal(out_result_baseline, out_result)
            # assert eq, "Output should be identical to old conv!"
            assert passing_pcc >= passing_pcc_baseline, "Output pcc should be same or better than old conv pcc!"
            logger.info(f"Output pcc passes and matches old conv pcc")
