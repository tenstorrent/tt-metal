# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations


class RefinerModelOptimisations(ModelOptimisations):
    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat16,
        conv_w_dtype=ttnn.bfloat16,
        attention_weights_dtype=ttnn.bfloat8_b,
        ff_weights_dtype=ttnn.bfloat8_b,
    ):
        super().__init__(conv_act_dtype, conv_w_dtype, attention_weights_dtype, ff_weights_dtype)

        self.conv_configs["ABH_128_ADB_WDB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )

        self.conv_configs["ABH_128_NO_ADB_WDB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )

        self.conv_configs["ABH_32_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32,
        )

        self.conv_configs["ABH_128_ADB_WDB_NO_MOVE_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )

        self.conv_configs["ABH_256_NO_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.conv_configs["ABH_128_NO_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )

        self.conv_configs["ABH_256_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.conv_configs["ABH_128_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )

        self.conv_configs["ABH_128_ADB_WDB_MOVE_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )

        self.conv_configs["ABH_64_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
        )

        self.matmul_configs["1D_GEGLU_LINEAR_256_1536_SPLIT_GELU"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,
            per_core_M=8,
            per_core_N=3,
            out_subblock_h=1,
            out_subblock_w=1,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=[ttnn.UnaryOpType.GELU, False],
        )

        self.matmul_configs["2D_GEGLU_LINEAR_1024_1536_SPLIT_GELU"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=6,
            per_core_M=4,
            per_core_N=24,
            out_subblock_h=1,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=[ttnn.UnaryOpType.GELU, False],
        )

        self.matmul_configs["2D_GEGLU_LINEAR_4096_768_SPLIT_GELU"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=3,
            per_core_M=16,
            per_core_N=12,
            out_subblock_h=1,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=[ttnn.UnaryOpType.GELU, False],
        )

        self.matmul_configs["1D_RESNET_LINEAR1_32x1536x384"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(6, 2),
            in0_block_w=12,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=None,
        )

        self.matmul_configs["1D_RESNET_LINEAR1_32x1536x768"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(6, 4),
            in0_block_w=24,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=None,
        )

        self.matmul_configs["1D_RESNET_LINEAR1_32x1536x1536"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(6, 8),
            in0_block_w=12,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_LINEAR2_4096x384x768"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=3,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=4,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_LINEAR2_1024x768x1536"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=6,
            per_core_M=4,
            per_core_N=6,
            out_subblock_h=1,
            out_subblock_w=6,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_LINEAR2_256x3072x1536"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=6,
            per_core_M=1,
            per_core_N=6,
            out_subblock_h=1,
            out_subblock_w=6,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_LINEAR2_1024x3072x1536"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=12,
            per_core_M=4,
            per_core_N=6,
            out_subblock_h=1,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_LINEAR2_1024x2304x1536"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=6,
            per_core_M=4,
            per_core_N=6,
            out_subblock_h=1,
            out_subblock_w=6,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_LINEAR2_4096x2304x768"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=6,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=2,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_LINEAR2_4096x1536x768"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=8,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=4,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_LINEAR2_4096x1152x768"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=6,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=2,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_LINEAR2_16384x1152x384"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(6, 8),
            in0_block_w=2,
            per_core_M=64,
            per_core_N=2,
            out_subblock_h=2,
            out_subblock_w=2,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_RESNET_LINEAR2_16384x768x384"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(6, 8),
            in0_block_w=2,
            per_core_M=64,
            per_core_N=2,
            out_subblock_h=2,
            out_subblock_w=2,
            transpose_mcast=False,
            fused_activation=None,
        )

        # 40 cores configs for RESNET_LINEAR2
        self.matmul_configs[
            "2D_RESNET_LINEAR2_4096x384x768_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=3,
            per_core_M=16,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_RESNET_LINEAR2_1024x768x1536_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=4,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_RESNET_LINEAR2_256x3072x1536_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=12,
            per_core_M=1,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=2,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_RESNET_LINEAR2_1024x3072x1536_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=8,
            per_core_M=4,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_RESNET_LINEAR2_1024x2304x1536_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=8,
            per_core_M=4,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_RESNET_LINEAR2_4096x2304x768_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=16,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_RESNET_LINEAR2_4096x1536x768_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=16,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_RESNET_LINEAR2_4096x1152x768_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=16,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_RESNET_LINEAR2_16384x1152x384_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=1,  # TODO: Check if this is correct, should be 2 for 16384x1152x384
            per_core_M=64,
            per_core_N=3,
            out_subblock_h=1,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_RESNET_LINEAR2_16384x768x384_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=1,  # TODO: Check if this is correct, should be 2 for 16384x768x384
            per_core_M=64,
            per_core_N=3,
            out_subblock_h=1,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_GEGLU_LINEAR1_4096x768x3072"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=6,
            per_core_M=16,
            per_core_N=12,
            out_subblock_h=1,
            out_subblock_w=6,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_GEGLU_LINEAR1_1024x1536x6144"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=8,
            per_core_M=4,
            per_core_N=24,
            out_subblock_h=1,
            out_subblock_w=8,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_GEGLU_LINEAR1_256x1536x6144"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            per_core_M=1,
            per_core_N=24,
            out_subblock_h=1,
            out_subblock_w=6,
            transpose_mcast=False,
            fused_activation=None,
        )

        # 40 cores configs for GEGLU_LINEAR2
        self.matmul_configs[
            "2D_GEGLU_LINEAR2_4096x768x3072_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=3,  # TODO: Check if this is correct, should be 6 for 4096x768x3072
            per_core_M=16,
            per_core_N=20,
            out_subblock_h=1,
            out_subblock_w=4,
            transpose_mcast=False,
            fused_activation=[ttnn.UnaryOpType.GELU, False],
        )

        self.matmul_configs[
            "2D_GEGLU_LINEAR2_1024x1536x6144_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=8,
            per_core_M=4,
            per_core_N=39,
            out_subblock_h=1,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=[ttnn.UnaryOpType.GELU, False],
        )

        self.matmul_configs[
            "1D_GEGLU_LINEAR2_256x1536x6144_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=4,
            per_core_M=8,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=[ttnn.UnaryOpType.GELU, False],
        )

        self.matmul_configs["2D_FF_LINEAR_4096x3072x768"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=6,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=2,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_FF_LINEAR_1024x6144x1536"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=16,
            per_core_M=4,
            per_core_N=6,
            out_subblock_h=1,
            out_subblock_w=6,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_FF_LINEAR_256x6144x1536"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=12,
            per_core_M=1,
            per_core_N=6,
            out_subblock_h=1,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        # 40 cores configs
        self.matmul_configs[
            "2D_GEGLU_LINEAR1_4096x768x3072_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=3,  # TODO: Check if this is correct, should be 6 for 4096x768x3072
            per_core_M=16,
            per_core_N=20,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_GEGLU_LINEAR1_1024x1536x6144_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=4,
            per_core_N=39,
            out_subblock_h=1,
            out_subblock_w=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "1D_GEGLU_LINEAR1_256x1536x6144_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=8,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_FF_LINEAR_4096x3072x768_40_CORES"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=16,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_FF_LINEAR_1024x6144x1536_40_CORES"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=4,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_FF_LINEAR_256x6144x1536_40_CORES"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=12,
            per_core_M=1,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        # # # ATTENTION QKV # # #
        self.matmul_configs[
            "2D_ATTENTION_QKV_4096x768x2304_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=1,  # TODO: Check if this is correct, should be 4 for 4096x768x2304
            per_core_M=16,
            per_core_N=15,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_ATTENTION_QKV_1024x1536x4608_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=8,
            per_core_M=4,
            per_core_N=29,
            out_subblock_h=4,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "1D_ATTENTION_QKV_256x1536x4608_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=8,
            per_core_N=4,
            out_subblock_h=1,
            out_subblock_w=4,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=None,
        )

        # # # ATTENTION Q (separate Q projections for cross-attention) # # #
        self.matmul_configs["2D_ATTENTION_Q_4096x768x768_40_CORES"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=2,  # TODO: Check if this is correct, should be 6 for 4096x768x768
            per_core_M=16,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_ATTENTION_Q_1024x1536x1536_40_CORES"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=8,
            per_core_M=4,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_ATTENTION_Q_256x1536x1536_40_CORES"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=1,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=2,
            transpose_mcast=False,
            fused_activation=None,
        )

        # # # ATTENTION K/V (separate K/V projections for cross-attention, same configs) # # #
        self.matmul_configs["1D_ATTENTION_K_96x1280x768_40_CORES"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(3, 8),
            in0_block_w=8,
            per_core_M=3,
            per_core_N=1,
            out_subblock_h=3,
            out_subblock_w=1,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=None,
        )

        self.matmul_configs["1D_ATTENTION_K_96x1280x1536_40_CORES"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(3, 8),
            in0_block_w=8,
            per_core_M=3,
            per_core_N=2,
            out_subblock_h=1,
            out_subblock_w=2,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=None,
        )

        # V uses the same configs as K
        self.matmul_configs["1D_ATTENTION_V_96x1280x768_40_CORES"] = self.matmul_configs[
            "1D_ATTENTION_K_96x1280x768_40_CORES"
        ]

        self.matmul_configs["1D_ATTENTION_V_96x1280x1536_40_CORES"] = self.matmul_configs[
            "1D_ATTENTION_K_96x1280x1536_40_CORES"
        ]

        # # # TRANSFORMER IN (input projections) # # #
        self.matmul_configs[
            "2D_TRANSFORMER_IN_4096x768x768_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=16,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_TRANSFORMER_IN_1024x1536x1536_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=4,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_TRANSFORMER_IN_256x1536x1536_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=1,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        # # # TRANSFORMER OUT (output projections) # # #
        self.matmul_configs[
            "2D_TRANSFORMER_OUT_4096x768x768_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=3,
            per_core_M=16,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_TRANSFORMER_OUT_1024x1536x1536_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=8,
            per_core_M=4,
            per_core_N=10,
            out_subblock_h=4,
            out_subblock_w=2,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs[
            "2D_TRANSFORMER_OUT_256x1536x1536_40_CORES"
        ] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(5, 8),
            in0_block_w=6,
            per_core_M=1,
            per_core_N=10,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_configs["MATH_APPROX_MM_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.compute_configs["RESNET_HIFI4_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def get_matmul_config(self, matmul_path):
        # # # TRANSFORMER IN (input projections) # # #
        if "proj_in" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs["2D_TRANSFORMER_IN_4096x768x768_40_CORES"]
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs["2D_TRANSFORMER_IN_1024x1536x1536_40_CORES"]
            elif "mid_block" in matmul_path:
                return self.matmul_configs["2D_TRANSFORMER_IN_256x1536x1536_40_CORES"]

        # # # TRANSFORMER OUT (output projections) # # #
        if "proj_out" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs["2D_TRANSFORMER_OUT_4096x768x768_40_CORES"]
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs["2D_TRANSFORMER_OUT_1024x1536x1536_40_CORES"]
            elif "mid_block" in matmul_path:
                return self.matmul_configs["2D_TRANSFORMER_OUT_256x1536x1536_40_CORES"]

        # # # ATTENTION Q (cross-attention attn2) and ATTENTION OUT (both attn1 and attn2) # # #
        # Note: Cross-attention Q uses .to_out lookup, and output projections also use .to_out
        if "attn1.to_out" in matmul_path or "attn2.to_out" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs["2D_ATTENTION_Q_4096x768x768_40_CORES"]
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs["2D_ATTENTION_Q_1024x1536x1536_40_CORES"]
            elif "mid_block" in matmul_path:
                return self.matmul_configs["2D_ATTENTION_Q_256x1536x1536_40_CORES"]

        # # # ATTENTION QKV (fused for self-attention attn1) # # #
        if "attn1.to_q" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs["2D_ATTENTION_QKV_4096x768x2304_40_CORES"]
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs["2D_ATTENTION_QKV_1024x1536x4608_40_CORES"]
            elif "mid_block" in matmul_path:
                return self.matmul_configs["1D_ATTENTION_QKV_256x1536x4608_40_CORES"]

        # # # ATTENTION K/V (separate K/V for cross-attention attn2, same configs) # # #
        if (".to_k" in matmul_path or ".to_v" in matmul_path) and "attn2" in matmul_path:
            # K/V projections: 96x1280 shapes, use same configs
            if "mid_block" in matmul_path or "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs["1D_ATTENTION_K_96x1280x1536_40_CORES"]
            else:
                return self.matmul_configs["1D_ATTENTION_K_96x1280x768_40_CORES"]

        # # # GEGLU # # #
        if "net.0.proj" in matmul_path:
            if "mid_block" in matmul_path:
                if "gelu" in matmul_path:
                    return self.matmul_configs["1D_GEGLU_LINEAR2_256x1536x6144_40_CORES"]
                else:
                    return self.matmul_configs["1D_GEGLU_LINEAR1_256x1536x6144_40_CORES"]
            elif "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                if "gelu" in matmul_path:
                    return self.matmul_configs["2D_GEGLU_LINEAR2_4096x768x3072_40_CORES"]
                else:
                    return self.matmul_configs["2D_GEGLU_LINEAR1_4096x768x3072_40_CORES"]
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                if "gelu" in matmul_path:
                    return self.matmul_configs["2D_GEGLU_LINEAR2_1024x1536x6144_40_CORES"]
                else:
                    return self.matmul_configs["2D_GEGLU_LINEAR1_1024x1536x6144_40_CORES"]

        # # # FF LINEAR # # #
        if "ff.net.2" in matmul_path:
            if "mid_block" in matmul_path:
                return self.matmul_configs["2D_FF_LINEAR_256x6144x1536_40_CORES"]
            elif "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs["2D_FF_LINEAR_4096x3072x768_40_CORES"]
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs["2D_FF_LINEAR_1024x6144x1536_40_CORES"]

        # # # RESNET # # #
        if "resnets" in matmul_path:
            if matmul_path in ["down_blocks.1.resnets.0.conv_shortcut"]:
                return self.matmul_configs["2D_RESNET_LINEAR2_4096x384x768_40_CORES"]
            if matmul_path in ["down_blocks.2.resnets.0.conv_shortcut"]:
                return self.matmul_configs["2D_RESNET_LINEAR2_1024x768x1536_40_CORES"]
            if matmul_path in [
                "up_blocks.0.resnets.0.conv_shortcut",
                "up_blocks.0.resnets.1.conv_shortcut",
                "up_blocks.0.resnets.2.conv_shortcut",
            ]:
                return self.matmul_configs["2D_RESNET_LINEAR2_256x3072x1536_40_CORES"]
            if matmul_path in [
                "up_blocks.1.resnets.0.conv_shortcut",
                "up_blocks.1.resnets.1.conv_shortcut",
                "up_blocks.1.resnets.2.conv_shortcut",
            ]:
                return self.matmul_configs["2D_RESNET_LINEAR2_1024x3072x1536_40_CORES"]
            if matmul_path in ["up_blocks.2.resnets.0.conv_shortcut"]:
                return self.matmul_configs["2D_RESNET_LINEAR2_4096x2304x768_40_CORES"]
            if matmul_path in ["up_blocks.2.resnets.1.conv_shortcut"]:
                return self.matmul_configs["2D_RESNET_LINEAR2_4096x1536x768_40_CORES"]
            if matmul_path in ["up_blocks.2.resnets.2.conv_shortcut"]:
                return self.matmul_configs["2D_RESNET_LINEAR2_4096x1152x768_40_CORES"]
            if matmul_path in ["up_blocks.3.resnets.0.conv_shortcut"]:
                return self.matmul_configs["2D_RESNET_LINEAR2_16384x1152x384_40_CORES"]
            if matmul_path in ["up_blocks.3.resnets.1.conv_shortcut"]:
                return self.matmul_configs["2D_RESNET_LINEAR2_16384x768x384_40_CORES"]
            if matmul_path in ["up_blocks.3.resnets.2.conv_shortcut"]:
                return self.matmul_configs["2D_RESNET_LINEAR2_16384x768x384_40_CORES"]
            if matmul_path in [
                "down_blocks.0.resnets.0.linear",
                "down_blocks.0.resnets.1.linear",
                "up_blocks.3.resnets.0.linear",
                "up_blocks.3.resnets.1.linear",
                "up_blocks.3.resnets.2.linear",
            ]:
                return self.matmul_configs["1D_RESNET_LINEAR1_32x1536x384"]
            if matmul_path in [
                "down_blocks.1.resnets.0.linear",
                "down_blocks.1.resnets.1.linear",
                "up_blocks.2.resnets.0.linear",
                "up_blocks.2.resnets.1.linear",
                "up_blocks.2.resnets.2.linear",
            ]:
                return self.matmul_configs["1D_RESNET_LINEAR1_32x1536x768"]
            if matmul_path in [
                "down_blocks.2.resnets.0.linear",
                "down_blocks.2.resnets.1.linear",
                "down_blocks.3.resnets.0.linear",
                "down_blocks.3.resnets.1.linear",
                "mid_block.resnets.0.linear",
                "mid_block.resnets.1.linear",
                "up_blocks.0.resnets.0.linear",
                "up_blocks.0.resnets.1.linear",
                "up_blocks.0.resnets.2.linear",
                "up_blocks.1.resnets.0.linear",
                "up_blocks.1.resnets.1.linear",
                "up_blocks.1.resnets.2.linear",
            ]:
                return self.matmul_configs["1D_RESNET_LINEAR1_32x1536x1536"]

        return None

    def get_mm_compute_config(self, module_path):
        # for now, return default config
        if ".to_q" in module_path:
            return self.compute_configs["MATH_APPROX_MM_COMPUTE_CONFIG"]
        return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]

    def get_mm_output_memory_config(self, module_path):
        if "attn1" in module_path or "attn2" in module_path:
            return ttnn.L1_MEMORY_CONFIG
        if "attentions" in module_path and "proj_in" in module_path:
            return ttnn.L1_MEMORY_CONFIG
        if "resnets" in module_path and "time_emb_proj" in module_path:
            return ttnn.DRAM_MEMORY_CONFIG
        if "resnets" in module_path and "conv_shortcut" in module_path:
            return ttnn.L1_MEMORY_CONFIG
        # TODO: Check if this is correct, should be DRAM for GEGLU and FF operations
        if "ff.net" in module_path or "proj.split" in module_path:
            return ttnn.DRAM_MEMORY_CONFIG
        return None

    def get_conv_config(self, conv_path):
        if "downsamplers" in conv_path:
            if "down_blocks.0" in conv_path:
                return self.conv_configs["ABH_256_ADB_WDB_BS_NO_MOVE"]
            elif "down_blocks.1" in conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_NO_MOVE_BS"]
            elif "down_blocks.2" in conv_path:
                return self.conv_configs["ABH_32_ADB_WDB_BS"]
        if "down_blocks.0" in conv_path:
            return self.conv_configs["ABH_512_ADB_WDB_BS"]
        if "down_blocks.1" in conv_path:
            if "resnets.0" in conv_path and "conv1" in conv_path:
                return self.conv_configs["ABH_256_ADB_WDB_BS_NO_MOVE"]
            else:
                return self.conv_configs["ABH_256_ADB_WDB_BS"]
        if "down_blocks.2" in conv_path:
            if "resnets.0" in conv_path and "conv1" in conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_NO_MOVE_BS"]
            else:
                return self.conv_configs["ABH_128_ADB_WDB_BS"]
        if "down_blocks.3" in conv_path or "mid_block" in conv_path:
            if "conv1" in conv_path:
                return self.conv_configs["ABH_64_ADB_WDB_BS"]
            else:
                return self.conv_configs["ABH_32_ADB_WDB_BS"]
        if "upsamplers" in conv_path:
            if "up_blocks.0" in conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_MOVE_BS"]
            elif "up_blocks.1" in conv_path:
                return self.conv_configs["ABH_256_NO_ADB_WDB_BS"]
            elif "up_blocks.2" in conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_MOVE_BS"]
        if "up_blocks.0" in conv_path:
            if "conv1" in conv_path:
                return self.conv_configs["ABH_32_ADB_WDB_BS"]
            else:
                return self.conv_configs["ABH_32_ADB_WDB_BS"]
        if "up_blocks.1" in conv_path:
            if "conv1" in conv_path:
                if "resnets.2" in conv_path:
                    return self.conv_configs["ABH_128_ADB_WDB_BS"]
                else:
                    return self.conv_configs["ABH_128_NO_ADB_WDB_BS"]
            else:
                return self.conv_configs["ABH_128_ADB_WDB_BS"]
        if "up_blocks.2" in conv_path:
            if "conv1" in conv_path:
                if "resnets.0" in conv_path:
                    return self.conv_configs["ABH_64_ADB_WDB_BS"]
                elif "resnets.1" in conv_path:
                    return self.conv_configs["ABH_128_ADB_WDB_BS"]
                else:
                    return self.conv_configs["ABH_256_ADB_WDB_BS"]
            else:
                return self.conv_configs["ABH_256_ADB_WDB_BS"]
        if "up_blocks.3" in conv_path:
            if "conv1" in conv_path:
                if "resnets.0" in conv_path:
                    return self.conv_configs["ABH_64_ADB_WDB_BS"]
                else:
                    return self.conv_configs["ABH_256_ADB_WDB_BS"]
            else:
                return self.conv_configs["ABH_512_ADB_WDB_BS"]
        if "conv_in" in conv_path:
            return self.conv_configs["ABH_128_ADB_WDB_HS"]
        if "conv_out" in conv_path:
            return self.conv_configs["ABH_128_NO_ADB_WDB_HS"]
        return None

    def get_conv_compute_config(self, module_path):
        return self.compute_configs["CONV_HIFI2_NO_FP32_COMPUTE_CONFIG"]

    def get_conv_output_dtype(self):
        return self.conv_output_dtype

    def _get_groupnorm_config(self, module_path):
        if "up_blocks.3" in module_path and "resnets.0" in module_path and "norm1" in module_path:
            return self.groupnorm_configs["DRAM_GROUPNORM_4X8"]
        if "up_blocks.3" in module_path and "resnets.0" not in module_path and "norm1" in module_path:
            return self.groupnorm_configs["SHARDED_GROUPNORM_INPLACE_NEGATIVE"]
        if "resnets" in module_path:
            return self.groupnorm_configs["SHARDED_GROUPNORM_INPLACE"]
        if "attentions" in module_path:
            return self.groupnorm_configs["SHARDED_GROUPNORM_NON_INPLACE"]
        return self.groupnorm_configs["SHARDED_GROUPNORM_INPLACE"]

    def get_layernorm_config(self, module_path):
        return ttnn.LayerNormDefaultProgramConfig(legacy_reduction=True, legacy_rsqrt=True)

    def get_sdpa_config(self, module_path, is_self_attention):
        if not is_self_attention:
            return self.sdpa_configs["128_K"]
        # TODO: 512 should be possible, latents base optimizations regressed this
        if "down_blocks.1" in module_path or "up_blocks.2" in module_path:
            return self.sdpa_configs["256_K"]
        else:
            return self.sdpa_configs["512_K"]
