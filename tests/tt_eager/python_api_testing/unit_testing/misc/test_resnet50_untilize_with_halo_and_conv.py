# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
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
        (
            25088,
            64,
            64,
        ): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        ): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        ): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        ): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        ): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        ): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        (6272, 512, 256): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=2,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=20,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 256, 1024): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=4,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=5,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 256): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=16,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=5,
            per_core_N=1,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 512): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=16,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=5,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (1568, 1024, 512): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=16,
            out_subblock_h=5,
            out_subblock_w=1,
            per_core_M=5,
            per_core_N=2,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (416, 512, 2048): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=2,
            per_core_N=10,
            transpose_mcast=True,
            fused_activation=None,
        ),
        (416, 2048, 512): tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
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
        (25088, 64): [64 * 3, 256, 1, 64, 128, 64, 256, (12, 9), 256, 64],
        (6272, 128): [128 * 3, 64, 1, 128, 64, 128, 64, (12, 9), 64, 128],
        (1568, 256): [256, 160, 8, 32, 160, 32, 160, (10, 8), 160, 32],
        (416, 512): [512, 64, 8, 64, 64, 64, 64, (7, 8), 64, 64],
    },
    16: {
        (50176, 64): [64 * 3, 256, 1, 64, 128, 64, 512, (12, 9), 512, 64],
        (12544, 128): [128 * 3, 128, 1, 128, 64, 128, 128, (12, 9), 128, 128],
        (3136, 256): [256, 288, 8, 32, 96, 32, 288, (11, 8), 288, 32],
        (800, 512): [512, 96, 8, 64, 96, 64, 96, (9, 8), 96, 64],
    },
}


@pytest.mark.skip(
    "This version of optimized_conv is deprecated. Please see untilize_with_halo_and_conv_v2, which replaces this."
)
@pytest.mark.parametrize("N", (8, 16), ids=["batch_8", "batch_16"])
@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        # unique convs in rn50 (complete list)
        # layer1
        (64, 64, 56, 56, 3, 3, 1, 1, 1, 1),
        # layer2
        # (512, 256, 56, 56, 1, 1, 2, 2, 0, 0), # not supported yet
        # (128, 128, 56, 56, 3, 3, 2, 2, 1, 1), # not supported yet
        (128, 128, 28, 28, 3, 3, 1, 1, 1, 1),
        # layer3
        # (256, 256, 28, 28, 3, 3, 2, 2, 1, 1), # not supported yet
        # (1024, 512, 28, 28, 1, 1, 2, 2, 0, 0), # not supported yet
        (256, 256, 14, 14, 3, 3, 1, 1, 1, 1),
        # layer4
        # (512, 512, 14, 14, 3, 3, 2, 2, 1, 1), # not supported yet
        # (2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0), # not supported yet
        (512, 512, 7, 7, 3, 3, 1, 1, 1, 1),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["weights_BFLOAT16", "weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["activations_BFLOAT16", "activations_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "math_fidelity", [tt_lib.tensor.MathFidelity.HiFi4, tt_lib.tensor.MathFidelity.LoFi], ids=["HiFi4", "LoFi"]
)
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
    interleaved_mem_config = tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
    )

    for i in range(1):  # increase num of iterations to test op caching
        assert C % 32 == 0
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
        logger.info(f"Conv output shape - {conv_output_shape}")
        conv_as_mm_padded_act_height = _nearest_32(conv_output_shape[0] * conv_output_shape[1] * conv_output_shape[2])

        assert (conv_as_mm_padded_act_height, K) in hardcoded_conv_blocking_and_parallelization_config[N]
        conv_blocking_and_parallelization_config = hardcoded_conv_blocking_and_parallelization_config[N][
            (conv_as_mm_padded_act_height, K)
        ]
        assert len(conv_blocking_and_parallelization_config) == 10

        [
            act_block_w_datums,
            act_block_h_datums,
            act_c_num_blocks,
            weight_block_w_datums,
            out_subblock_h_datums,
            out_subblock_w_datums,
            out_block_h_datums,
            grid_size,
            per_core_out_matrix_h,
            per_core_weight_matrix_w,
        ] = conv_blocking_and_parallelization_config
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

        ###############################################
        # Directly run old conv (row major input)
        # NOTE: New conv should have identical output
        ###############################################
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
            output_mem_config=interleaved_mem_config,
            weights_dtype=weights_dtype,
            output_dtype=activations_dtype,
            math_fidelity=math_fidelity,
            act_c_num_blocks=act_c_num_blocks,
        )

        conv_input_on_device = tt_lib.tensor.Tensor(
            conv_input_pyt_nhwc.reshape(-1).tolist(),
            conv_input_pyt_nhwc.shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        ).to(device, interleaved_mem_config)

        output_on_device = conv(conv_input_on_device)

        # convert tiled output to RM
        assert output_on_device.get_layout() == tt_lib.tensor.Layout.TILE
        output_on_device = format_tensor(
            output_on_device, tt_lib.tensor.Layout.ROW_MAJOR, device, interleaved_mem_config
        )
        output_on_device = output_on_device.reshape(
            conv_output_shape[0],
            conv_output_shape[1],
            conv_output_shape[2],
            conv_output_shape[3],
        )

        # Copy to host
        out = output_on_device.cpu()
        assert out.get_layout() == tt_lib.tensor.Layout.ROW_MAJOR

        out_result = out.to_torch()
        # NHWC to NCHW
        out_result = torch.transpose(out_result, 2, 3)
        out_result_baseline = torch.transpose(out_result, 1, 2)

        ########################################################
        # Tilize, untilize_with_halo, conv with reader indices
        # Create interleaved input on device
        ########################################################
        if act_c_num_blocks > 1:  # 2D conv
            in_mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED, tt_lib.tensor.BufferType.L1
            )
            out_memory_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED, tt_lib.tensor.BufferType.L1
            )
        else:
            in_mem_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1
            )
            out_memory_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1
            )

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
            output_mem_config=out_memory_config,
            input_tensor_shape=conv_input_shape_nhwc,
            weights_dtype=weights_dtype,
            output_dtype=activations_dtype,
            math_fidelity=math_fidelity,
            act_c_num_blocks=act_c_num_blocks,
        )

        conv_input_on_device = tt_lib.tensor.Tensor(
            conv_input_pyt_nhwc.reshape(-1).tolist(),
            conv_input_pyt_nhwc.shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        ).to(device, interleaved_mem_config)

        # Convert activation RM to tile layout
        conv_input_on_device = conv_input_on_device.reshape(
            1,
            1,
            conv_input_shape_nhwc[0] * conv_input_shape_nhwc[1] * conv_input_shape_nhwc[2],
            conv_input_shape_nhwc[3],
        )
        conv_input_on_device = format_tensor(
            conv_input_on_device, tt_lib.tensor.Layout.TILE, device, interleaved_mem_config
        )

        # Convert interleaved to sharded
        if act_c_num_blocks > 1:  # 2D conv
            conv_input_on_device = tt_lib.tensor.interleaved_to_sharded(
                conv_input_on_device,
                grid_size,
                [
                    act_block_h_datums,
                    weight_block_w_datums,
                ],  # act_block_w_datums may include reads of multiple pixels in window
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                tt_lib.tensor.ShardOrientation.COL_MAJOR,
            )
        else:
            conv_input_on_device = tt_lib.tensor.interleaved_to_sharded(
                conv_input_on_device,
                grid_size,
                [
                    per_core_out_matrix_h,
                    weight_block_w_datums,
                ],  # act_block_w_datums may include reads of multiple pixels in window
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                tt_lib.tensor.ShardOrientation.ROW_MAJOR,
            )

        # Untilize with halo concat
        conv_input_on_device = tt_lib.tensor.untilize_with_halo(conv_input_on_device, 0x0, N, H, W, 1, in_mem_config)

        # Conv with new reader for sharded untilized with halo inputs
        output_on_device = conv(conv_input_on_device)

        # Convert sharded output to tiled interleaved
        output_on_device = tt_lib.tensor.sharded_to_interleaved(output_on_device, interleaved_mem_config)

        # convert tiled output to RM
        assert output_on_device.get_layout() == tt_lib.tensor.Layout.TILE
        output_on_device = format_tensor(
            output_on_device, tt_lib.tensor.Layout.ROW_MAJOR, device, interleaved_mem_config
        )
        output_on_device = output_on_device.reshape(
            conv_output_shape[0],
            conv_output_shape[1],
            conv_output_shape[2],
            conv_output_shape[3],
        )

        # Copy to host and compare against pytorch
        out = output_on_device.cpu()
        assert out.get_layout() == tt_lib.tensor.Layout.ROW_MAJOR

        out_result = out.to_torch()
        # NHWC to NCHW
        out_result = torch.transpose(out_result, 2, 3)
        out_result = torch.transpose(out_result, 1, 2)

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
        if activations_dtype == tt_lib.tensor.DataType.BFLOAT8_B and (K == 256 or K == 512):
            pytest.xfail("PCC of output baseline is slightly lower than with new conv. DEBUG!")
        else:
            eq = torch.equal(out_result_baseline, out_result)
            assert eq, "Output should be identical to old conv!"
            assert passing_pcc == passing_pcc_baseline, "Output pcc should be identical to old conv pcc!"
            logger.debug(f"Output pcc passes and matches old conv pcc")
