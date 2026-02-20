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
        force_full_grid=False,
    ):
        super().__init__(
            conv_act_dtype,
            conv_w_dtype,
            attention_weights_dtype,
            ff_weights_dtype,
            force_full_grid=force_full_grid,
        )

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

        # region MATMUL CONFIGS
        self.matmul_versions = {
            "64_cores": {
                # # # GEGLU # # #
                "2D_GEGLU_LINEAR_768_SPLIT": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=6,
                    per_core_M=16,
                    per_core_N=12,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_GEGLU_LINEAR_768_SPLIT_GELU": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=3,
                    per_core_M=16,
                    per_core_N=12,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    transpose_mcast=False,
                    fused_activation=[ttnn.UnaryOpType.GELU, True],
                ),
                "2D_GEGLU_LINEAR_1536_SPLIT": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=8,
                    per_core_M=4,
                    per_core_N=24,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_GEGLU_LINEAR_1536_SPLIT_GELU": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=6,
                    per_core_M=4,
                    per_core_N=24,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    transpose_mcast=False,
                    fused_activation=[ttnn.UnaryOpType.GELU, True],
                ),
                "1D_GEGLU_LINEAR_1536_SPLIT": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=4,
                    per_core_M=1,
                    per_core_N=24,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "1D_GEGLU_LINEAR_1536_SPLIT_GELU": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=2,
                    per_core_M=8,
                    per_core_N=3,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=[ttnn.UnaryOpType.GELU, True],
                ),
                # # # FF2 # # #
                "2D_FF2_SEQ_LEN_4096": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=6,
                    per_core_M=16,
                    per_core_N=3,
                    out_subblock_h=2,
                    out_subblock_w=3,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_FF2_SEQ_LEN_1024": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=16,
                    per_core_M=4,
                    per_core_N=6,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_FF2_SEQ_LEN_256": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=12,
                    per_core_M=1,
                    per_core_N=6,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                # # # RESNET CONV # # #
                "2D_RESNET_CONV_384_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=3,
                    per_core_M=16,
                    per_core_N=3,
                    out_subblock_h=4,
                    out_subblock_w=1,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_768_1536": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=6,
                    per_core_M=4,
                    per_core_N=6,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_3072_1536": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=6,
                    per_core_M=1,
                    per_core_N=6,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_3072_1536_1024": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=12,
                    per_core_M=4,
                    per_core_N=6,
                    out_subblock_h=1,
                    out_subblock_w=3,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_2304_1536": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=6,
                    per_core_M=4,
                    per_core_N=6,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_2304_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=6,
                    per_core_M=16,
                    per_core_N=3,
                    out_subblock_h=2,
                    out_subblock_w=3,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_1536_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=8,
                    per_core_M=16,
                    per_core_N=3,
                    out_subblock_h=4,
                    out_subblock_w=1,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_1152_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=6,
                    per_core_M=16,
                    per_core_N=3,
                    out_subblock_h=2,
                    out_subblock_w=3,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "1D_RESNET_CONV_1152_384": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(6, 8),
                    in0_block_w=2,
                    per_core_M=64,
                    per_core_N=2,
                    out_subblock_h=2,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "1D_RESNET_CONV_768_384": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(6, 8),
                    in0_block_w=2,
                    per_core_M=64,
                    per_core_N=2,
                    out_subblock_h=2,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
            },
            "40_cores": {
                # # # GEGLU # # #
                "2D_GEGLU_LINEAR_768_SPLIT": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=16,
                    per_core_N=20,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_GEGLU_LINEAR_768_SPLIT_GELU": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=16,
                    per_core_N=20,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    transpose_mcast=False,
                    fused_activation=[ttnn.UnaryOpType.GELU, True],
                ),
                "2D_GEGLU_LINEAR_1536_SPLIT": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=4,
                    per_core_N=39,
                    out_subblock_h=1,
                    out_subblock_w=3,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_GEGLU_LINEAR_1536_SPLIT_GELU": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=4,
                    per_core_N=39,
                    out_subblock_h=1,
                    out_subblock_w=3,
                    transpose_mcast=False,
                    fused_activation=[ttnn.UnaryOpType.GELU, True],
                ),
                "2D_GEGLU_LINEAR_1536_MID_SPLIT": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=1,
                    per_core_N=39,
                    out_subblock_h=1,
                    out_subblock_w=3,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_GEGLU_LINEAR_1536_MID_SPLIT_GELU": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=1,
                    per_core_N=39,
                    out_subblock_h=1,
                    out_subblock_w=3,
                    transpose_mcast=False,
                    fused_activation=[ttnn.UnaryOpType.GELU, True],
                ),
                # # # FF2 # # #
                "2D_FF2_SEQ_LEN_4096": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=16,
                    per_core_N=5,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_FF2_SEQ_LEN_1024": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=3,
                    per_core_M=4,
                    per_core_N=10,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_FF2_SEQ_LEN_256": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=3,
                    per_core_M=1,
                    per_core_N=10,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                # # # TM LINEAR # # #
                "2D_TM_LINEAR_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=6,
                    per_core_M=16,
                    per_core_N=5,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_TM_LINEAR_1536": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=6,
                    per_core_M=4,
                    per_core_N=10,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_TM_LINEAR_1536_MID": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=6,
                    per_core_M=1,
                    per_core_N=10,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_TM_OUT_LINEAR_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=1,
                    per_core_M=16,
                    per_core_N=5,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_TM_OUT_LINEAR_1536": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=4,
                    per_core_N=10,
                    out_subblock_h=4,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_TM_OUT_LINEAR_1536_MID": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=1,
                    per_core_N=10,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                # # # ATTENTION QKV # # #
                "2D_ATTN_QKV_LINEAR_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=1,
                    per_core_M=16,
                    per_core_N=15,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_ATTN_QKV_LINEAR_1536": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=4,
                    per_core_N=29,
                    out_subblock_h=4,
                    out_subblock_w=1,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_ATTN_QKV_LINEAR_1536_MID": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=1,
                    per_core_N=29,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                # # # ATTENTION OUT # # #
                "2D_ATTN_OUT_LINEAR_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=1,
                    per_core_M=16,
                    per_core_N=5,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_ATTN_OUT_LINEAR_1536": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=4,
                    per_core_N=10,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_ATTN_OUT_LINEAR_1536_MID": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=2,
                    per_core_M=1,
                    per_core_N=10,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                # # # ATTENTION K/V # # #
                "2D_ATTEN_K_V_LINEAR_768": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(3, 8),
                    in0_block_w=8,
                    per_core_M=3,
                    per_core_N=1,
                    out_subblock_h=3,
                    out_subblock_w=1,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "1D_ATTEN_K_V_LINEAR_1536": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(3, 8),
                    in0_block_w=8,
                    per_core_M=3,
                    per_core_N=2,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                # # # RESNET LINEAR # # #
                "1D_RESNET_LINEAR_384": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(6, 2),
                    in0_block_w=12,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=1,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "1D_RESNET_LINEAR_768": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(6, 4),
                    in0_block_w=24,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=1,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                # If stress test with 48 cores passes, change config to use 48 cores to avoid perf impact
                "1D_RESNET_LINEAR_1536": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(3, 8),
                    in0_block_w=6,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=1,
                    per_core_N=2,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                # # # RESNET CONV # # #
                "2D_RESNET_CONV_384_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=3,
                    per_core_M=16,
                    per_core_N=5,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_768_1536": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=6,
                    per_core_M=4,
                    per_core_N=10,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_3072_1536": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=12,
                    per_core_M=1,
                    per_core_N=10,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_3072_1536_1024": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=8,
                    per_core_M=4,
                    per_core_N=10,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_2304_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=6,
                    per_core_M=16,
                    per_core_N=5,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_1536_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=6,
                    per_core_M=16,
                    per_core_N=5,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_1152_768": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=6,
                    per_core_M=16,
                    per_core_N=5,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "1D_RESNET_CONV_1152_384": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=3,
                    per_core_M=13,
                    per_core_N=12,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    mcast_in0=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "1D_RESNET_CONV_768_384": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=3,
                    per_core_M=13,
                    per_core_N=12,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    mcast_in0=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "1D_TIME_EMBEDDING_LINEAR_1": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(3, 8),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=2,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=ttnn.UnaryOpType.SILU,
                ),
                "1D_TIME_EMBEDDING_LINEAR_2": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(3, 8),
                    in0_block_w=6,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=1,
                    per_core_N=2,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "1D_ADD_EMBEDDING_LINEAR_1": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(3, 8),
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=2,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=ttnn.UnaryOpType.SILU,
                ),
                "1D_ADD_EMBEDDING_LINEAR_2": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(3, 8),
                    in0_block_w=6,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=1,
                    per_core_N=2,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=None,
                ),
            },
        }

        self.matmul_configs = (
            self.matmul_versions["40_cores"] if not force_full_grid else self.matmul_versions["64_cores"]
        )
        # endregion

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

        self.layernorm_configs = {}
        self.layernorm_configs["1536_mid_config"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            subblock_w=5,
            block_h=1,
            block_w=10,
            inplace=False,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )
        self.layernorm_configs["1536_config"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            subblock_w=5,
            block_h=4,
            block_w=10,
            inplace=False,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )
        self.layernorm_configs["768_config"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            subblock_w=5,
            block_h=16,
            block_w=5,
            inplace=False,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )

    def get_matmul_config(self, matmul_path):
        if matmul_path is None:
            return None

        # # # TM LINEAR (input projections) # # #
        if "proj_in" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs.get("2D_TM_LINEAR_768")
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs.get("2D_TM_LINEAR_1536")
            elif "mid_block" in matmul_path:
                return self.matmul_configs.get("2D_TM_LINEAR_1536_MID")

        # # # TM OUT LINEAR (output projections) # # #
        if "proj_out" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs.get("2D_TM_OUT_LINEAR_768")
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs.get("2D_TM_OUT_LINEAR_1536")
            elif "mid_block" in matmul_path:
                return self.matmul_configs.get("2D_TM_OUT_LINEAR_1536_MID")

        # # # ATTENTION QKV (fused for self-attention attn1) # # #
        if "attn1.to_q" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs.get("2D_ATTN_QKV_LINEAR_768")
            elif "mid_block" in matmul_path:
                return self.matmul_configs.get("2D_ATTN_QKV_LINEAR_1536_MID")
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs.get("2D_ATTN_QKV_LINEAR_1536")

        if "attn2.to_q" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs.get("2D_ATTN_OUT_LINEAR_768")
            elif "mid_block" in matmul_path:
                return self.matmul_configs.get("2D_ATTN_OUT_LINEAR_1536_MID")
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs.get("2D_ATTN_OUT_LINEAR_1536")

        # # # ATTENTION OUT (both attn1 and attn2) # # #
        if "attn1.to_out" in matmul_path or "attn2.to_out" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs.get("2D_ATTN_OUT_LINEAR_768")
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs.get("2D_ATTN_OUT_LINEAR_1536")
            elif "mid_block" in matmul_path:
                return self.matmul_configs.get("2D_ATTN_OUT_LINEAR_1536_MID")

        # # # ATTENTION K/V (separate K/V for cross-attention attn2) # # #
        if (".to_k" in matmul_path or ".to_v" in matmul_path) and "attn2" in matmul_path:
            if "mid_block" in matmul_path or "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs.get("1D_ATTEN_K_V_LINEAR_1536")
            else:
                return self.matmul_configs.get("2D_ATTEN_K_V_LINEAR_768")

        # # # GEGLU # # #
        if "net.0.proj" in matmul_path:
            if "mid_block" in matmul_path:
                if "gelu" in matmul_path:
                    return self.matmul_configs.get("2D_GEGLU_LINEAR_1536_MID_SPLIT_GELU")
                else:
                    return self.matmul_configs.get("2D_GEGLU_LINEAR_1536_MID_SPLIT")
            elif "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                if "gelu" in matmul_path:
                    return self.matmul_configs.get("2D_GEGLU_LINEAR_768_SPLIT_GELU")
                else:
                    return self.matmul_configs.get("2D_GEGLU_LINEAR_768_SPLIT")
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                if "gelu" in matmul_path:
                    return self.matmul_configs.get("2D_GEGLU_LINEAR_1536_SPLIT_GELU")
                else:
                    return self.matmul_configs.get("2D_GEGLU_LINEAR_1536_SPLIT")

        # # # FF2 LINEAR # # #
        if "ff.net.2" in matmul_path:
            if "mid_block" in matmul_path:
                return self.matmul_configs.get("2D_FF2_SEQ_LEN_256")
            elif "down_blocks.1" in matmul_path or "up_blocks.2" in matmul_path:
                return self.matmul_configs.get("2D_FF2_SEQ_LEN_4096")
            elif "down_blocks.2" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs.get("2D_FF2_SEQ_LEN_1024")

        # # # RESNET CONV_SHORTCUT # # #
        if "conv_shortcut" in matmul_path:
            if "down_blocks.1.resnets.0" in matmul_path:
                return self.matmul_configs.get("2D_RESNET_CONV_384_768")
            if "down_blocks.2.resnets.0" in matmul_path:
                return self.matmul_configs.get("2D_RESNET_CONV_768_1536")
            if (
                "up_blocks.0.resnets.0" in matmul_path
                or "up_blocks.0.resnets.1" in matmul_path
                or "up_blocks.0.resnets.2" in matmul_path
            ):
                return self.matmul_configs.get("2D_RESNET_CONV_3072_1536")
            if (
                "up_blocks.1.resnets.0" in matmul_path
                or "up_blocks.1.resnets.1" in matmul_path
                or "up_blocks.1.resnets.2" in matmul_path
            ):
                return self.matmul_configs.get("2D_RESNET_CONV_3072_1536_1024")
            if "up_blocks.2.resnets.0" in matmul_path:
                return self.matmul_configs.get("2D_RESNET_CONV_2304_768")
            if "up_blocks.2.resnets.1" in matmul_path:
                return self.matmul_configs.get("2D_RESNET_CONV_1536_768")
            if "up_blocks.2.resnets.2" in matmul_path:
                return self.matmul_configs.get("2D_RESNET_CONV_1152_768")
            if "up_blocks.3.resnets.0" in matmul_path:
                return self.matmul_configs.get("1D_RESNET_CONV_1152_384")
            if "up_blocks.3.resnets.1" in matmul_path or "up_blocks.3.resnets.2" in matmul_path:
                return self.matmul_configs.get("1D_RESNET_CONV_768_384")

        # # # RESNET TIME_EMB_PROJ LINEAR # # #
        if "resnets" in matmul_path and "linear" in matmul_path:
            if "down_blocks.0.resnets" in matmul_path or "up_blocks.3.resnets" in matmul_path:
                return self.matmul_configs.get("1D_RESNET_LINEAR_384")
            if "down_blocks.1.resnets" in matmul_path or "up_blocks.2.resnets" in matmul_path:
                return self.matmul_configs.get("1D_RESNET_LINEAR_768")
            if (
                "down_blocks.2.resnets" in matmul_path
                or "down_blocks.3.resnets" in matmul_path
                or "mid_block.resnets" in matmul_path
                or "up_blocks.0.resnets" in matmul_path
                or "up_blocks.1.resnets" in matmul_path
            ):
                return self.matmul_configs.get("1D_RESNET_LINEAR_1536")

        if "time_embedding" in matmul_path:
            if "linear_1" in matmul_path:
                return self.matmul_configs.get("1D_TIME_EMBEDDING_LINEAR_1")
            elif "linear_2" in matmul_path:
                return self.matmul_configs.get("1D_TIME_EMBEDDING_LINEAR_2")

        if "add_embedding" in matmul_path:
            if "linear_1" in matmul_path:
                return self.matmul_configs.get("1D_ADD_EMBEDDING_LINEAR_1")
            elif "linear_2" in matmul_path:
                return self.matmul_configs.get("1D_ADD_EMBEDDING_LINEAR_2")

        return None

    def get_mm_compute_config(self, module_path):
        # for now, return default config
        if ".to_q" in module_path:
            return self.compute_configs["MATH_APPROX_MM_COMPUTE_CONFIG"]
        return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]

    def get_mm_output_memory_config(self, module_path):
        if "attn1" in module_path or "attn2" in module_path:
            if "to_out" not in module_path:
                return ttnn.L1_MEMORY_CONFIG
            else:
                if "down_blocks.1" in module_path or "up_blocks.2" in module_path:
                    return ttnn.L1_MEMORY_CONFIG
                else:
                    return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        if "attentions" in module_path and "proj_in" in module_path:
            if "down_blocks.2" in module_path or "up_blocks.1" in module_path or "mid_block" in module_path:
                return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            else:
                return ttnn.L1_MEMORY_CONFIG
        if "resnets" in module_path and "conv_shortcut" in module_path:
            if "up_blocks.3" in module_path:
                return ttnn.L1_MEMORY_CONFIG
            else:
                return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        if "proj.split.gelu" in module_path:
            return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        if "proj.split" in module_path:
            if "down_blocks.2" in module_path or "up_blocks.1" in module_path or "mid_block" in module_path:
                return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            else:
                return ttnn.L1_MEMORY_CONFIG
        if "ff.net.2" in module_path:
            return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG

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
        if "mid_block" in module_path:
            return self.layernorm_configs["1536_mid_config"]
        elif "down_blocks.2" in module_path or "up_blocks.1" in module_path:
            return self.layernorm_configs["1536_config"]
        elif "down_blocks.1" in module_path or "up_blocks.2" in module_path:
            return self.layernorm_configs["768_config"]
        return ttnn.LayerNormDefaultProgramConfig(legacy_reduction=True, legacy_rsqrt=True)

    def get_sdpa_config(self, module_path, is_self_attention):
        if not is_self_attention:
            return self.sdpa_configs["128_K"]
        # TODO: 512 should be possible, latents base optimizations regressed this
        if "down_blocks.1" in module_path or "up_blocks.2" in module_path:
            return self.sdpa_configs["256_K"]
        else:
            return self.sdpa_configs["512_K"]
