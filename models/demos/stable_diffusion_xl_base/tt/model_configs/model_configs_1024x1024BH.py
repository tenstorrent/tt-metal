# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re

import ttnn
from models.demos.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_generated_gn_beta_gamma,
    prepare_gn_beta_gamma,
    prepare_gn_mask,
    prepare_gn_mask_negative_mask,
)


class ModelOptimisations1024x1024BH:
    # Full-grid UNet: the generated GroupNorm (ttnn.operations.groupnorm_sc_N_1_HW_C) instead of ttnn.group_norm,
    # and the whole UNet on the transposed 11x10 grid (COL_MAJOR block shards: HW across the 11 columns, C down the
    # 10 rows) with transpose_mcast matmuls, transpose_shards convs and diagonal in1 / weight senders. Activations
    # stay L1 block-sharded between blocks. Subclasses that reuse these configs for another network set it False.
    FULL_GRID = True

    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat16,
        conv_w_dtype=ttnn.bfloat16,
        attention_weights_dtype=ttnn.bfloat8_b,
        ff_weights_dtype=ttnn.bfloat8_b,
        force_full_grid=False,  # This parameter is not used for BH model configs
    ):
        # full_grid / sdpa_math_fidelity are read by the shared UNet code (the Wormhole configs do not define them)
        self.full_grid = self.FULL_GRID
        self.use_generated_groupnorm = self.full_grid
        # SiLU fused into the generated GN: slower than the standalone unary on the UNet's L1 shards, faster on the
        # VAE's DRAM path (VAE config).
        self.fuse_gn_silu = False
        # LoFi truncation shrinks the SDPA output by 4-6%, which compounds over the denoising loop.
        self.sdpa_math_fidelity = ttnn.MathFidelity.HiFi2 if self.full_grid else ttnn.MathFidelity.LoFi
        self.conv_configs = {}
        self.conv_output_dtype = conv_act_dtype
        self.matmul_configs = {}
        self.compute_configs = {}
        self.prepared_weights = False
        self.conv_w_dtype = conv_w_dtype
        self.conv_ws_dtype = ttnn.bfloat8_b
        self.attention_weights_dtype = attention_weights_dtype
        self.ff_weights_dtype = ff_weights_dtype

        # region CONV2D CONFIGS
        # region HEIGHT SHARDED
        self.conv_configs["ABH_1024_NO_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=1024,
        )
        self.conv_configs["ABH_256_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )
        self.conv_configs["ABH_256_NO_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )
        self.conv_configs["ABH_128_NO_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )
        self.conv_configs["ABH_128_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
        )
        self.conv_configs["ABH_0_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
        )
        self.conv_configs["ABH_1024_NO_ADB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=1024,
        )
        self.conv_configs["ABH_0_NO_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
        )
        self.conv_configs["ABH_512_NO_ADB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=512,
        )
        self.conv_configs["ABH_52T_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=52 * 32,
        )
        self.conv_configs["ABH_52T_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=52 * 32,
        )
        self.conv_configs["ABH_26T_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=26 * 32,
        )
        self.conv_configs["ABH_13T_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=13 * 32,
        )
        self.conv_configs["ABH_26T_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=26 * 32,
        )
        self.conv_configs["ABH_256_ADB_WDB_BS_NO_MOVE"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=False,
            act_block_w_div=1,
            act_block_h_override=256,
        )
        self.conv_configs["ABH_256_NO_ADB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
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
        self.conv_configs["ABH_64_NO_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
        )

        self.conv_configs["ABH_32_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32,
        )
        self.conv_configs["ABH_64_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32 * 2,
        )
        self.conv_configs["ABH_8T_ADB_HS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32 * 8,
        )
        # endregion

        # region BLOCK SHARDED

        self.conv_configs["ABH_0_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=False,
            act_block_w_div=1,
            act_block_h_override=0,
        )

        self.conv_configs["ABH_0_ADB_WDB_NO_DEALLOC_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=False,
            act_block_w_div=1,
            act_block_h_override=0,
        )

        self.conv_configs["ABH_128_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=False,
            act_block_w_div=1,
            act_block_h_override=128,
        )
        self.conv_configs["ABH_256_ADB_WDB_MOVE_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=False,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.conv_configs["ABH_256_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=False,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.conv_configs["ABH_512_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=False,
            act_block_w_div=1,
            act_block_h_override=512,
        )

        self.conv_configs["ABH_512_ADB_WDB_NO_DEALLOC_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=False,
            act_block_w_div=1,
            act_block_h_override=512,
        )

        self.conv_configs["ABH_1024_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=False,
            act_block_w_div=1,
            act_block_h_override=1024,
        )
        # endregion

        # region DEFAULT CONF
        self.conv_configs["DEFAULT"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
        )
        if self.full_grid:
            # every block-sharded conv runs on the transposed (COL_MAJOR) grid; with transpose_shards the default
            # weight senders are one column of DRAM readers, which saturates its NoC links.
            for _cfg in self.conv_configs.values():
                if _cfg.shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
                    _cfg.transpose_shards = True
                    _cfg.diagonal_weight_senders = True

        # DRAM CONF
        self.conv_configs["DEFAULT_DRAM"] = ttnn.Conv2dConfig(
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
            output_layout=ttnn.TILE_LAYOUT,
        )
        # endregion
        # endregion

        # region MATMUL CONFIGS
        self.matmul_versions = {
            "80_cores": {
                "2D_FF2_SEQ_LEN_1024": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=16,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=4,
                    per_core_N=4,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_FF2_SEQ_LEN_4096": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=8,
                    out_subblock_h=4,
                    out_subblock_w=2,
                    per_core_M=16,
                    per_core_N=2,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "1D_RESNET_LINEAR": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(5, 8),
                    in0_block_w=10,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=1,
                    mcast_in0=True,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "2D_GEGLU_LINEAR_640_SPLIT": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=4,
                    per_core_M=16,
                    per_core_N=8,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_GEGLU_LINEAR_640_SPLIT_GELU": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=4,
                    per_core_M=16,
                    per_core_N=8,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    transpose_mcast=False,
                    fused_activation=[ttnn.UnaryOpType.GELU, True],
                ),
                "2D_GEGLU_LINEAR_1280_SPLIT": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=4,
                    per_core_M=4,
                    per_core_N=16,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_GEGLU_LINEAR_1280_SPLIT_GELU": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=4,
                    per_core_M=4,
                    per_core_N=16,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    transpose_mcast=False,
                    fused_activation=[ttnn.UnaryOpType.GELU, True],
                ),
                "2D_TM_LINEAR_640": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=5,
                    per_core_M=16,
                    per_core_N=2,
                    out_subblock_h=2,
                    out_subblock_w=1,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_TM_LINEAR_1280": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=4,
                    per_core_M=4,
                    per_core_N=4,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_ATTN_QKV_LINEAR_640": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=10,
                    per_core_M=16,
                    per_core_N=6,
                    out_subblock_h=4,
                    out_subblock_w=1,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_ATTN_QKV_LINEAR_1280": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=4,
                    per_core_M=4,
                    per_core_N=12,
                    out_subblock_h=1,
                    out_subblock_w=6,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                ),
                "2D_ATTEN_K_V_LINEAR_640": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=2,  # max is 64, 4 seems optimal
                    per_core_M=1,
                    per_core_N=2,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "1D_ATTEN_K_V_LINEAR_1280": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    in0_block_w=16,  # max is 64, 16 seems optimal
                    out_subblock_h=3,
                    out_subblock_w=1,
                    per_core_M=3,
                    per_core_N=1,
                    mcast_in0=True,
                    fuse_batch=True,
                    fused_activation=None,
                ),
                "2D_ATTN_OUT_LINEAR_640": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=4,
                    per_core_M=16,
                    per_core_N=2,
                    out_subblock_h=4,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                ),
                "2D_ATTN_OUT_LINEAR_1280": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=4,
                    per_core_M=4,
                    per_core_N=4,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                ),
                "2D_RESNET_CONV_320_640": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=2,
                    per_core_M=16,
                    per_core_N=2,
                    out_subblock_h=4,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                ),
                "2D_RESNET_CONV_640_1280": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=5,
                    per_core_M=4,
                    per_core_N=4,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_2560_1280": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=8,
                    per_core_M=4,
                    per_core_N=4,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_1920_1280": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=5,
                    per_core_M=4,
                    per_core_N=4,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                ),
                "2D_RESNET_CONV_1920_640": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=4,
                    per_core_M=16,
                    per_core_N=2,
                    out_subblock_h=2,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_1280_640": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=4,
                    per_core_M=16,
                    per_core_N=2,
                    out_subblock_h=2,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "2D_RESNET_CONV_960_640": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(10, 8),
                    in0_block_w=3,
                    per_core_M=16,
                    per_core_N=2,
                    out_subblock_h=2,
                    out_subblock_w=2,
                    transpose_mcast=False,
                    fused_activation=None,
                ),
                "1D_RESNET_CONV_960_320": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(11, 10),
                    in0_block_w=1,
                    out_subblock_h=5,
                    out_subblock_w=1,
                    per_core_M=5,
                    per_core_N=10,
                    mcast_in0=False,
                    gather_in0=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
                "1D_RESNET_CONV_640_320": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(11, 10),
                    in0_block_w=1,
                    out_subblock_h=5,
                    out_subblock_w=1,
                    per_core_M=5,
                    per_core_N=10,
                    mcast_in0=False,
                    gather_in0=False,
                    fuse_batch=False,
                    fused_activation=None,
                ),
            },
        }

        self.matmul_configs = dict(self.matmul_versions["80_cores"])  # copy: the overrides below must not leak
        if self.full_grid:
            # Transformer blocks on the transposed 11x10 grid: M over the 11 columns, N over the 10 rows.
            # 1280 blocks: 1024 rows -> 3 tiles per core (33 >= 32, the last core is ragged).
            T = dict(compute_with_storage_grid_size=(11, 10), transpose_mcast=True)
            self.matmul_configs["2D_TM_LINEAR_1280"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=4,
                per_core_M=3,
                per_core_N=4,
                out_subblock_h=1,
                out_subblock_w=4,
                fused_activation=None,
                **T,
            )
            self.matmul_configs["2D_ATTN_QKV_LINEAR_1280"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=4,
                per_core_M=3,
                per_core_N=12,
                out_subblock_h=1,
                out_subblock_w=6,
                fused_activation=None,
                fuse_batch=True,
                **T,
            )
            self.matmul_configs["2D_ATTN_OUT_LINEAR_1280"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=4,
                per_core_M=3,
                per_core_N=4,
                out_subblock_h=1,
                out_subblock_w=4,
                fused_activation=None,
                fuse_batch=True,
                **T,
            )
            self.matmul_configs["2D_GEGLU_LINEAR_1280_SPLIT"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=4,
                per_core_M=3,
                per_core_N=16,
                out_subblock_h=1,
                out_subblock_w=8,
                fused_activation=None,
                **T,
            )
            self.matmul_configs["2D_GEGLU_LINEAR_1280_SPLIT_GELU"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=4,
                per_core_M=3,
                per_core_N=16,
                out_subblock_h=1,
                out_subblock_w=8,
                fused_activation=[ttnn.UnaryOpType.GELU, True],
                **T,
            )
            # FF2: in0 is the GEGLU output shard [96, 512] = 16 K tiles per core, consumed as one K block
            self.matmul_configs["2D_FF2_SEQ_LEN_1024"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=16,
                per_core_M=3,
                per_core_N=4,
                out_subblock_h=1,
                out_subblock_w=4,
                fused_activation=None,
                **T,
            )
            # 640 blocks: 4096 rows -> 12 tiles per core (132 >= 128). Sharded outputs need subblock_w == per_core_N.
            self.matmul_configs["2D_TM_LINEAR_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=2,
                per_core_M=12,
                per_core_N=2,
                out_subblock_h=4,
                out_subblock_w=2,
                fused_activation=None,
                **T,
            )
            self.matmul_configs["2D_ATTN_QKV_LINEAR_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=2,
                per_core_M=12,
                per_core_N=6,
                out_subblock_h=1,
                out_subblock_w=6,
                fused_activation=None,
                **T,
            )
            self.matmul_configs["2D_ATTN_OUT_LINEAR_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=2,
                per_core_M=12,
                per_core_N=2,
                out_subblock_h=4,
                out_subblock_w=2,
                fused_activation=None,
                fuse_batch=True,
                **T,
            )
            self.matmul_configs["2D_GEGLU_LINEAR_640_SPLIT"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=2,
                per_core_M=12,
                per_core_N=8,
                out_subblock_h=1,
                out_subblock_w=8,
                fused_activation=None,
                **T,
            )
            self.matmul_configs["2D_GEGLU_LINEAR_640_SPLIT_GELU"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=2,
                per_core_M=12,
                per_core_N=8,
                out_subblock_h=1,
                out_subblock_w=8,
                fused_activation=[ttnn.UnaryOpType.GELU, True],
                **T,
            )
            self.matmul_configs["2D_FF2_SEQ_LEN_4096"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=8,
                per_core_M=12,
                per_core_N=2,
                out_subblock_h=4,
                out_subblock_w=2,
                fused_activation=None,
                **T,
            )
            # diagonal in1 senders on every 2D mcast matmul: with transpose_mcast the default senders are one column
            # of DRAM weight readers, which saturates its NoC links
            for _cfg in self.matmul_configs.values():
                if isinstance(_cfg, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig):
                    _cfg.diagonal_in1_senders = True
        # endregion

        # region LAYERNORM CONFIGS
        self.core_grid_x = 10

        self.layernorm_configs = {}
        self.layernorm_configs["640_config"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10) if self.full_grid else ttnn.CoreCoord(10, 8),
            subblock_w=2,
            block_h=12 if self.full_grid else 16,
            block_w=2,
            inplace=False,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )
        self.layernorm_configs["1280_config"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10) if self.full_grid else ttnn.CoreCoord(10, 8),
            subblock_w=4,
            block_h=3 if self.full_grid else 4,
            block_w=4,
            inplace=False,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )
        # endregion

        # region GROUPNORM CONFIGS
        self.groupnorm_configs = {}
        self.groupnorm_configs["SHARDED_GROUPNORM_INPLACE"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=8),
                "num_out_blocks": None,
                "inplace": True,
            },
            "memory_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            "negative_mask": False,
        }
        self.groupnorm_configs["SHARDED_GROUPNORM_INPLACE_NEGATIVE"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=8),
                "num_out_blocks": None,
                "inplace": True,
            },
            "memory_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            "negative_mask": True,
        }
        self.groupnorm_configs["SHARDED_GROUPNORM_NON_INPLACE"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=8),
                "num_out_blocks": None,
                "inplace": False,
            },
            "memory_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            "negative_mask": False,
        }
        self.groupnorm_configs["SHARDED_GROUPNORM_4X8_NON_INPLACE"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=4),
                "num_out_blocks": None,
                "inplace": False,
            },
            "memory_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            "negative_mask": False,
        }
        self.groupnorm_configs["SHARDED_GROUPNORM_NON_INPLACE_NEGATIVE"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=8),
                "num_out_blocks": None,
                "inplace": False,
            },
            "memory_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            "negative_mask": True,
        }
        self.groupnorm_configs["DRAM_GROUPNORM_4X8"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=4),
                "num_out_blocks": 2,
                "inplace": False,
            },
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "negative_mask": False,
        }
        # endregion

        # region SDPA CONFIGS
        self.sdpa_configs = {}
        # 1024-row attention: q256 gives 20 heads x 4 q-chunks = 80 work items, one round on 110 cores (q128 left
        # half the grid idle in a second round).
        self.sdpa_configs["1024_K"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            q_chunk_size=256,
            k_chunk_size=512,
            exp_approx_mode=False,
        )
        self.sdpa_configs["128_K_S1024"] = ttnn.SDPAProgramConfig(  # cross-attn from 1024 rows
            compute_with_storage_grid_size=(11, 10),
            q_chunk_size=256,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self.sdpa_configs["512_K"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            q_chunk_size=128,
            k_chunk_size=512,
            exp_approx_mode=False,
        )
        self.sdpa_configs["128_K"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        # endregion

        # region COMPUTE KERNEL CONFIGS
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

        self.compute_configs["CONV_LOFI_FP32_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.compute_configs["CONV_HIFI2_FP32_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.compute_configs["CONV_HIFI2_NO_FP32_NO_L1_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.compute_configs["CONV_HIFI2_NO_FP32_COMPUTE_CONFIG"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # endregion

    def get_matmul_config(self, matmul_path):
        if matmul_path is None:
            return None

        # # # RESNET CONV MM # # #
        if "conv_shortcut" in matmul_path:
            if "down_blocks.1" in matmul_path:
                return self.matmul_configs["2D_RESNET_CONV_320_640"]
            if "down_blocks.2" in matmul_path:
                return self.matmul_configs["2D_RESNET_CONV_640_1280"]
            if "up_blocks.0.resnets.0" in matmul_path or "up_blocks.0.resnets.1" in matmul_path:
                return self.matmul_configs["2D_RESNET_CONV_2560_1280"]
            if "up_blocks.0.resnets.2" in matmul_path:
                return self.matmul_configs["2D_RESNET_CONV_1920_1280"]
            if "up_blocks.1.resnets.0" in matmul_path:
                return self.matmul_configs["2D_RESNET_CONV_1920_640"]
            if "up_blocks.1.resnets.1" in matmul_path:
                return self.matmul_configs["2D_RESNET_CONV_1280_640"]
            if "up_blocks.1.resnets.2" in matmul_path:
                return self.matmul_configs["2D_RESNET_CONV_960_640"]
            if "up_blocks.2.resnets.0" in matmul_path:
                return self.matmul_configs["1D_RESNET_CONV_960_320"]
            if "up_blocks.2.resnets.1" in matmul_path or "up_blocks.2.resnets.2" in matmul_path:
                return self.matmul_configs["1D_RESNET_CONV_640_320"]
            else:
                return None

        # # # GEGLU # # #
        if "net.0.proj" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                if "gelu" in matmul_path:
                    return self.matmul_configs["2D_GEGLU_LINEAR_640_SPLIT_GELU"]
                else:
                    return self.matmul_configs["2D_GEGLU_LINEAR_640_SPLIT"]

            else:
                if "gelu" in matmul_path:
                    return self.matmul_configs["2D_GEGLU_LINEAR_1280_SPLIT_GELU"]
                else:
                    return self.matmul_configs["2D_GEGLU_LINEAR_1280_SPLIT"]

        # # # TM LINEAR # # #
        if "proj_in" in matmul_path or "proj_out" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs["2D_TM_LINEAR_640"]
            else:
                return self.matmul_configs["2D_TM_LINEAR_1280"]

        # # # ATTN OUT LINEAR # # #
        if "attn1.to_out" in matmul_path or "attn2.to_out" in matmul_path or "attn2.to_q" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs["2D_ATTN_OUT_LINEAR_640"]
            else:
                return self.matmul_configs["2D_ATTN_OUT_LINEAR_1280"]
        if "attn1.to_q" in matmul_path:
            if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs["2D_ATTN_QKV_LINEAR_640"]
            else:
                return self.matmul_configs["2D_ATTN_QKV_LINEAR_1280"]
        if (
            "attn1.to_k" in matmul_path
            or "attn1.to_v" in matmul_path
            or "attn2.to_k" in matmul_path
            or "attn2.to_v" in matmul_path
        ):
            if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                return self.matmul_configs["2D_ATTEN_K_V_LINEAR_640"]
            else:
                return self.matmul_configs["1D_ATTEN_K_V_LINEAR_1280"]

        pattern_down_blocks_1_ff2 = re.compile(
            r"down_blocks\.1\.attentions\.[01]\.transformer_blocks\.[01]\.ff\.net\.2"
        )

        # 4 occurrences
        if pattern_down_blocks_1_ff2.search(matmul_path):
            return self.matmul_configs["2D_FF2_SEQ_LEN_4096"]

        pattern_down_blockcs_2_ff2 = re.compile(
            r"down_blocks\.2\.attentions\.[01]\.transformer_blocks\.[0123456789]\.ff\.net\.2"
        )

        # 20 occurrences
        if pattern_down_blockcs_2_ff2.search(matmul_path):
            return self.matmul_configs["2D_FF2_SEQ_LEN_1024"]

        # # # Mid block  # # #
        pattern_mid_block_ff2 = re.compile(r"mid_block\.attentions\.0\.transformer_blocks\.[0123456789]\.ff\.net\.2")

        # 10 occurrences
        if pattern_mid_block_ff2.search(matmul_path):
            return self.matmul_configs["2D_FF2_SEQ_LEN_1024"]

        pattern_up_blocks_0_ff2 = re.compile(
            r"up_blocks\.0\.attentions\.[012]\.transformer_blocks\.[0123456789]\.ff\.net\.2"
        )

        # 30 occurrences
        if pattern_up_blocks_0_ff2.search(matmul_path):
            return self.matmul_configs["2D_FF2_SEQ_LEN_1024"]

        pattern_up_blocks_1_ff2 = re.compile(r"up_blocks\.1\.attentions\.[012]\.transformer_blocks\.[01]\.ff\.net\.2")

        # 6 occurrences
        if pattern_up_blocks_1_ff2.search(matmul_path):
            return self.matmul_configs["2D_FF2_SEQ_LEN_4096"]

        pattern_resnet_linear = re.compile(
            r"(down_blocks\.[012]\.resnets\.[01]\.linear|up_blocks\.[012]\.resnets\.[012]\.linear|mid_block\.resnets\.[01]\.linear)"
        )

        if pattern_resnet_linear.search(matmul_path):
            return self.matmul_configs["1D_RESNET_LINEAR"]
        return None

    def get_mm_compute_config(self, module_path):
        # for now, return default config
        if ".to_q" in module_path:
            return self.compute_configs["MATH_APPROX_MM_COMPUTE_CONFIG"]
        return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]

    def get_transposed_block_memory_config(self, module_path):
        """The transposed 11x10 activation shard of a transformer block ([96, 128] COL_MAJOR for the 1280 blocks,
        [384, 64] for the 640 blocks), or None off the full grid."""
        if not self.full_grid:
            return None
        is_640_block = "down_blocks.1" in module_path or "up_blocks.1" in module_path
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))})
        shard = [384, 64] if is_640_block else [96, 128]
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.COL_MAJOR),
        )

    def get_mm_output_memory_config(self, module_path):
        # full grid: the 640 blocks stay block-sharded like the 1280 blocks, so LayerNorm / residual adds run sharded
        is_640_block = "down_blocks.1" in module_path or "up_blocks.1" in module_path
        if "attn1" in module_path or "attn2" in module_path:
            if not "to_out" in module_path:
                return ttnn.L1_MEMORY_CONFIG
            else:
                if is_640_block and not self.full_grid:
                    return ttnn.L1_MEMORY_CONFIG
                else:
                    return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        if "ff.net" in module_path:
            return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        if "attentions" in module_path and "proj_in" in module_path:
            if is_640_block and not self.full_grid:
                return ttnn.L1_MEMORY_CONFIG
            else:
                return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        if "resnets" in module_path and "conv_shortcut" in module_path:
            if "up_blocks.2" not in module_path:
                return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            else:
                return ttnn.L1_MEMORY_CONFIG
        return None

    def get_conv_config(self, conv_path):
        if conv_path is None:
            return None

        if "conv_in" == conv_path:
            return self.conv_configs["ABH_256_ADB_HS"]

        # DOWN BLOCK 0
        elif ("down_blocks.0.resnets" in conv_path) and ("conv2" in conv_path):
            # full grid: 16384 rows -> 52 tile rows per core; act blocks must divide it
            return self.conv_configs["ABH_26T_ADB_WDB_BS" if self.full_grid else "ABH_1024_ADB_WDB_BS"]
        elif "down_blocks.0.resnets" in conv_path:
            return self.conv_configs["ABH_26T_ADB_WDB_BS" if self.full_grid else "ABH_1024_ADB_WDB_BS"]
        elif "down_blocks.0.downsamplers.0" == conv_path:
            return self.conv_configs["ABH_512_ADB_WDB_NO_DEALLOC_BS"]

        # DOWN BLOCK 1
        elif "down_blocks.1.resnets.0.conv1" == conv_path:
            return self.conv_configs["ABH_0_ADB_WDB_BS"]
        elif ("down_blocks.1.resnets.0.conv2" == conv_path) or ("down_blocks.1.resnets.1" in conv_path):
            return self.conv_configs["ABH_0_ADB_WDB_BS"]
        elif "down_blocks.1.downsamplers.0" == conv_path:
            return self.conv_configs["ABH_0_ADB_WDB_NO_DEALLOC_BS"]

        # DOWN BLOCK 2
        elif "down_blocks.2.resnets.1.conv1" == conv_path:
            return self.conv_configs["ABH_0_ADB_WDB_BS"]
        elif "down_blocks.2.resnets.0.conv1" == conv_path:
            return self.conv_configs["ABH_0_ADB_WDB_BS"]
        elif ("down_blocks.2.resnets.0.conv2" == conv_path) or ("down_blocks.2.resnets.1.conv2" == conv_path):
            return self.conv_configs["ABH_0_ADB_WDB_BS"]

        # MID BLOCK
        elif "mid_block" in conv_path:
            return self.conv_configs["ABH_0_ADB_WDB_BS"]

        # UP BLOCK 0
        elif ("up_blocks.0.resnets.0.conv1" == conv_path) or ("up_blocks.0.resnets.1.conv1" == conv_path):
            return self.conv_configs["ABH_0_ADB_WDB_BS"]
        elif "up_blocks.0.upsamplers.0" == conv_path:
            # full grid: the upsample lands on the 11x10 shard (12 tile rows per core)
            return self.conv_configs["ABH_0_ADB_WDB_BS" if self.full_grid else "ABH_256_ADB_WDB_BS"]
        elif ("up_blocks.0.resnets" in conv_path) and ("conv2" in conv_path):
            return self.conv_configs["ABH_0_ADB_WDB_BS"]
        elif "up_blocks.0.resnets.2.conv1" == conv_path:
            return self.conv_configs["ABH_0_ADB_WDB_BS"]

        # UP BLOCK 1
        elif "up_blocks.1.resnets.0.conv1" == conv_path:
            return self.conv_configs["ABH_128_ADB_WDB_BS"]
        elif "up_blocks.1.resnets.1.conv1" == conv_path:
            return self.conv_configs["ABH_13T_ADB_WDB_BS" if self.full_grid else "ABH_256_ADB_WDB_BS"]
        elif "up_blocks.1.resnets.2.conv1" == conv_path:
            return self.conv_configs["ABH_13T_ADB_WDB_BS" if self.full_grid else "ABH_256_ADB_WDB_BS"]
        elif ("up_blocks.1.resnets" in conv_path) and ("conv2" in conv_path):
            return self.conv_configs["ABH_0_ADB_WDB_BS"]
        elif "up_blocks.1.upsamplers.0" == conv_path:
            # full grid: the upsample lands on the 10x10 [1664, 64] shard; 26-tile act blocks do not fit L1 next to it
            return self.conv_configs["ABH_13T_ADB_WDB_BS" if self.full_grid else "ABH_256_ADB_WDB_BS"]

        # UP BLOCK 2
        # full grid: these convs consume a 10x10 [1664, C/10] shard (52 tile rows per core), so act blocks divide 52
        elif "up_blocks.2.resnets.0.conv1" == conv_path:
            # Cin=960: a 26-tile act block does not fit L1 next to the shards
            return self.conv_configs["ABH_13T_ADB_WDB_BS" if self.full_grid else "ABH_256_ADB_WDB_MOVE_BS"]
        elif ("up_blocks.2.resnets" in conv_path) and ("conv2" in conv_path):
            return self.conv_configs["ABH_26T_ADB_WDB_BS" if self.full_grid else "ABH_1024_ADB_WDB_BS"]
        elif "up_blocks.2.resnets.1.conv1" == conv_path:
            return self.conv_configs["ABH_26T_ADB_WDB_BS" if self.full_grid else "ABH_512_ADB_WDB_BS"]
        elif "up_blocks.2.resnets.2.conv1" == conv_path:
            return self.conv_configs["ABH_26T_ADB_WDB_BS" if self.full_grid else "ABH_512_ADB_WDB_BS"]

        elif "conv_out" == conv_path:
            return self.conv_configs["ABH_128_ADB_HS"]
        else:
            return self.conv_configs["DEFAULT"]

    def get_conv_compute_config(self, module_path):
        if "conv_in" in module_path or "conv_out" in module_path:
            # packer_l1_acc: with a 16-bit DEST, re-rounding the partial sums per K block inflates the output ~3.7%
            return self.compute_configs["CONV_HIFI2_NO_FP32_COMPUTE_CONFIG"]
        if "resnets" in module_path:
            conv1_no_fp32 = {
                "down_blocks.2.resnets",
                "down_blocks.0",
                "down_blocks.1.resnets.0",
                "up_blocks.0",
                "mid_block",
            }
            conv2_no_fp32 = {"down_blocks.2.resnets", "down_blocks.0", "up_blocks.0", "mid_block"}

            if "conv1" in module_path and any(s in module_path for s in conv1_no_fp32):
                return self.compute_configs["CONV_HIFI2_NO_FP32_COMPUTE_CONFIG"]
            if "conv2" in module_path and any(s in module_path for s in conv2_no_fp32):
                return self.compute_configs["CONV_HIFI2_NO_FP32_COMPUTE_CONFIG"]

            return self.compute_configs["CONV_HIFI2_FP32_COMPUTE_CONFIG"]
        if "upsamplers" in module_path:
            if "up_blocks.0" in module_path:
                return self.compute_configs["CONV_HIFI2_NO_FP32_COMPUTE_CONFIG"]
            else:
                return self.compute_configs["CONV_HIFI2_FP32_COMPUTE_CONFIG"]

        return self.compute_configs["CONV_HIFI2_FP32_COMPUTE_CONFIG"]

    def get_conv_output_dtype(self):
        return self.conv_output_dtype

    def __generate_groupnorm_params(self, config, weights, bias, groups, device):
        if config["memory_config"] != ttnn.DRAM_MEMORY_CONFIG:
            gamma, beta = prepare_gn_beta_gamma(device, weights, bias, config["op_config"]["core_grid"].x)
            mask = prepare_gn_mask(device, weights.shape[0], groups, config["op_config"]["core_grid"].x)
            negative_mask = (
                prepare_gn_mask_negative_mask(device, weights.shape[0], groups, config["op_config"]["core_grid"].x)
                if config["negative_mask"]
                else None
            )
        else:
            [gamma, beta], mask = ttnn.dram_group_norm_params_from_torch(
                [weights, bias],
                weights.shape[0],
                groups,
                device,
                core_grid=config["op_config"]["core_grid"],
                return_mask=True,
            )
            negative_mask = None

        return mask, negative_mask, gamma, beta

    def _get_groupnorm_config(self, module_path):
        if "up_blocks.2" in module_path and "norm1" in module_path:
            return self.groupnorm_configs["SHARDED_GROUPNORM_INPLACE_NEGATIVE"]
        if "resnets" in module_path:
            return self.groupnorm_configs["SHARDED_GROUPNORM_INPLACE"]
        if "attentions" in module_path:
            if "down_blocks.1" in module_path or "up_blocks.1" in module_path:
                return self.groupnorm_configs["SHARDED_GROUPNORM_4X8_NON_INPLACE"]
            else:
                return self.groupnorm_configs["SHARDED_GROUPNORM_NON_INPLACE"]
        return self.groupnorm_configs["SHARDED_GROUPNORM_INPLACE"]

    def get_groupnorm_params(self, module_path, weights, bias, groups, device):
        if self.use_generated_groupnorm:
            gamma, beta = prepare_generated_gn_beta_gamma(device, weights, bias)
            cfg = {"generated": True, "fuse_silu": self.fuse_gn_silu}
            return cfg, ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, None, None, gamma, beta
        config = self._get_groupnorm_config(module_path)

        mask, negative_mask, gamma, beta = self.__generate_groupnorm_params(config, weights, bias, groups, device)
        return config["op_config"], config["memory_config"], mask, negative_mask, gamma, beta

    def get_layernorm_config(self, module_path):
        if "down_blocks.1" in module_path or "up_blocks.1" in module_path:
            return self.layernorm_configs["640_config"]
        else:
            return self.layernorm_configs["1280_config"]

    def get_sdpa_config(self, module_path, is_self_attention):
        rows_4096 = "down_blocks.1" in module_path or "up_blocks.1" in module_path
        if not is_self_attention:
            return self.sdpa_configs["128_K" if rows_4096 else "128_K_S1024"]
        return self.sdpa_configs["512_K" if rows_4096 else "1024_K"]
