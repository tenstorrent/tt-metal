# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import re

import ttnn
from models.demos.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_generated_gn_beta_gamma,
    prepare_gn_beta_gamma,
    prepare_gn_mask,
    prepare_gn_mask_negative_mask,
)


class ModelOptimisations1024x1024BH:
    # "Full transposed" UNet resnet/conv path (see transposed_resnets below). Subclasses that reuse the conv
    # configs for other networks (the VAE) set this False: their convs keep the ROW_MAJOR block shards.
    TRANSPOSED_RESNETS = True

    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat16,
        conv_w_dtype=ttnn.bfloat16,
        attention_weights_dtype=ttnn.bfloat8_b,
        ff_weights_dtype=ttnn.bfloat8_b,
        force_full_grid=False,  # This parameter is not used for BH model configs
        use_generated_groupnorm=None,
    ):
        # Full-grid GroupNorm (ttnn.operations.groupnorm_sc_N_1_HW_C) instead of ttnn.group_norm on 8x8.
        # Env override SDXL_GENERATED_GN=0/1; default on.
        if use_generated_groupnorm is None:
            use_generated_groupnorm = os.environ.get("SDXL_GENERATED_GN", "1") != "0"
        self.use_generated_groupnorm = use_generated_groupnorm
        # Block-shard the 640-channel transformer blocks' activations (see get_mm_output_memory_config).
        self.sharded_640_blocks = use_generated_groupnorm and os.environ.get("SDXL_SHARDED_640", "1") != "0"
        # 1280-channel transformer blocks on the transposed 11x10 grid (see the matmul config block below).
        self.transposed_1280_blocks = use_generated_groupnorm and os.environ.get("SDXL_T1280", "1") != "0"
        # 640-channel transformer blocks (4096 rows = 128 tile rows -> 12 per core over 11 columns) likewise.
        self.transposed_640_blocks = self.sharded_640_blocks and os.environ.get("SDXL_T640", "1") != "0"
        if self.transposed_1280_blocks or self.transposed_640_blocks:
            os.environ.setdefault("TT_MM2D_DIAG_IN1_SENDERS", "1")  # factory knob (this branch)
        # Keep resnet outputs L1 block-sharded for the next block (generated GN only). Env SDXL_GN_KEEP_L1=0 disables.
        self.keep_l1_resnet_output = use_generated_groupnorm and os.environ.get("SDXL_GN_KEEP_L1", "1") != "0"
        # Fuse the resnet SiLU into the generated GN's apply pass. Measured 2026-09-17: NET LOSS (+1.83 ms GN vs
        # -1.50 ms Unary): exact SFPU silu in the op's fp32-DEST chain on 80-100 cores costs more than the
        # standalone bf16 unary on 110 cores. Kept as an opt-in knob (SDXL_GN_FUSE_SILU=1).
        self.fuse_gn_silu = use_generated_groupnorm and os.environ.get("SDXL_GN_FUSE_SILU", "0") == "1"
        # Resnet GN on 10 rows also at HW=4096 (13 tile rows per core; the convs get 13-tile act blocks).
        self.resnet_gn_prime_rows = use_generated_groupnorm and os.environ.get("SDXL_GN_PRIME_ROWS", "1") != "0"
        # "Full transposed" UNet: the resnet / conv path also runs HW across the 11 grid columns and C down the
        # 10 rows (COL_MAJOR block shards, Conv2dConfig.transpose_shards), so the whole model shares one
        # orientation with the transposed transformer blocks and the GN -> proj_in reshards disappear.
        # Env SDXL_T_RESNET=0 disables (needs the transposed transformer blocks).
        self.transposed_resnets = (
            self.TRANSPOSED_RESNETS
            and self.transposed_1280_blocks
            and self.transposed_640_blocks
            and os.environ.get("SDXL_T_RESNET", "1") != "0"
        )
        self.conv_configs = {}
        self.conv_output_dtype = conv_act_dtype
        self.matmul_configs = {}
        self.compute_configs = {}
        self.prepared_weights = False
        self.conv_w_dtype = conv_w_dtype
        self.conv_ws_dtype = ttnn.bfloat8_b
        self.attention_weights_dtype = (
            getattr(ttnn, os.environ["SDXL_ATTN_W_DTYPE"])
            if "SDXL_ATTN_W_DTYPE" in os.environ
            else attention_weights_dtype
        )
        self.ff_weights_dtype = (
            getattr(ttnn, os.environ["SDXL_FF_W_DTYPE"]) if "SDXL_FF_W_DTYPE" in os.environ else ff_weights_dtype
        )

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
        if self.transposed_resnets:
            # every block-sharded conv works on the transposed (COL_MAJOR) grid; the input GN shard already is
            for _cfg in self.conv_configs.values():
                if _cfg.shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
                    _cfg.transpose_shards = True

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
        if self.sharded_640_blocks:
            # in0 of these matmuls is now the [512, 64] block shard (2 K-tiles per core): in0_block_w must divide 2.
            for name in (
                "2D_ATTN_QKV_LINEAR_640",
                "2D_ATTN_OUT_LINEAR_640",
                "2D_GEGLU_LINEAR_640_SPLIT",
                "2D_GEGLU_LINEAR_640_SPLIT_GELU",
            ):
                self.matmul_configs[name].in0_block_w = 2
        if self.use_generated_groupnorm:
            # proj_in in0 = the generated GN's [512, 64] shard (2 K-tiles per core); the reference feeds [512, 160].
            # Its output is block-sharded on this path, so the subblock must span per_core_N (2): 4x2 = 8 DEST tiles.
            self.matmul_configs["2D_TM_LINEAR_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(10, 8),
                in0_block_w=2,
                per_core_M=16,
                per_core_N=2,
                out_subblock_h=4,
                out_subblock_w=2,
                transpose_mcast=False,
                fused_activation=None,
            )
        if self.transposed_1280_blocks:
            # 1280-channel transformer blocks (M = 1024 rows = 32 tile rows) on the full 11x10 grid: M is split
            # over the 11 columns as 3 tiles per core (33 >= 32, ragged last core; the 2D factory handles
            # last_per_core_M), N over the 10 rows. Needs transpose_mcast (COL_MAJOR shards) and the diagonal
            # weight senders (TT_MM2D_DIAG_IN1_SENDERS, this branch's factory change): FF-up 74.1 -> 57.4 us.
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
            # FF2: in0 is the GEGLU output shard [96, 512] = 16 K tiles per core -> one 16-tile K block per
            # core (probe: 72.0 us shipped -> 55.7 us; 8-tile blocks 56.8, 4-tile 61.8)
            self.matmul_configs["2D_FF2_SEQ_LEN_1024"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=16,
                per_core_M=3,
                per_core_N=4,
                out_subblock_h=1,
                out_subblock_w=4,
                fused_activation=None,
                **T,
            )
        if self.transposed_640_blocks:
            # 4096-row blocks: per_core_M 12 (132 tiles for 128), N over the 10 rows: 640 -> 2 tiles, 1920 -> 6,
            # 2560 -> 8. K per core: 640 -> 2 tiles, 2560 (FF2) -> 8. Sharded outputs need subblock_w == per_core_N.
            T = dict(compute_with_storage_grid_size=(11, 10), transpose_mcast=True)
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
        # endregion

        # region LAYERNORM CONFIGS
        self.core_grid_x = 10

        self.layernorm_configs = {}
        self.layernorm_configs["640_config"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10)
            if self.transposed_640_blocks
            else ttnn.CoreCoord(10, 8),
            subblock_w=2,
            block_h=12 if self.transposed_640_blocks else 16,
            block_w=2,
            inplace=False,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )
        self.layernorm_configs["1280_config"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10)
            if self.transposed_1280_blocks
            else ttnn.CoreCoord(10, 8),
            subblock_w=4,
            block_h=3 if self.transposed_1280_blocks else 4,
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
        # SDPA chunking experiments (fullgrid worklog): env overrides, defaults = the shipped values.
        _q1024 = int(os.environ.get("SDXL_SDPA_Q1024", "128"))
        _q512 = int(os.environ.get("SDXL_SDPA_Q512", "128"))
        _k512 = int(os.environ.get("SDXL_SDPA_K512", "512"))
        self.sdpa_configs["1024_K"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            q_chunk_size=_q1024,
            k_chunk_size=1024,
            exp_approx_mode=False,
        )
        self.sdpa_configs["512_K"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            q_chunk_size=_q512,
            k_chunk_size=_k512,
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

        # Precision investigation knobs (env, all default to the shipped configs):
        #   SDXL_MM_FIDELITY=HiFi4 / SDXL_MM_FP32=1 / SDXL_MM_L1ACC=0 : every matmul compute config (DEFAULT + MATH_APPROX)
        #   SDXL_CONV_FIDELITY=HiFi4 / SDXL_CONV_FP32=1 / SDXL_CONV_L1ACC=0|1 : every CONV_* compute config
        #   SDXL_ATTN_W_DTYPE / SDXL_FF_W_DTYPE = bfloat16 : weight dtypes (handled in __init__)
        def _override(names, prefix):
            fid, fp32, l1 = (
                os.environ.get(f"{prefix}_FIDELITY"),
                os.environ.get(f"{prefix}_FP32"),
                os.environ.get(f"{prefix}_L1ACC"),
            )
            if fid is None and fp32 is None and l1 is None:
                return
            for n in names:
                c = self.compute_configs[n]
                self.compute_configs[n] = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=getattr(ttnn.MathFidelity, fid) if fid else c.math_fidelity,
                    math_approx_mode=c.math_approx_mode,
                    fp32_dest_acc_en=(fp32 == "1") if fp32 is not None else c.fp32_dest_acc_en,
                    packer_l1_acc=(l1 == "1") if l1 is not None else c.packer_l1_acc,
                )

        _override([n for n in self.compute_configs if n.endswith("MM_COMPUTE_CONFIG")], "SDXL_MM")
        if os.environ.get("SDXL_MM_FP32") == "1":
            # fp32 DEST halves the register file: cap every matmul program config's out subblock at 4 tiles.
            def _cap(cfg):
                if not hasattr(cfg, "out_subblock_h"):
                    return
                while cfg.out_subblock_h * cfg.out_subblock_w > 4:
                    if cfg.out_subblock_w > 1 and cfg.out_subblock_w % 2 == 0:
                        cfg.out_subblock_w //= 2
                    elif cfg.out_subblock_h > 1 and cfg.out_subblock_h % 2 == 0:
                        cfg.out_subblock_h //= 2
                    elif cfg.out_subblock_w > 1:
                        cfg.out_subblock_w = 1
                    else:
                        cfg.out_subblock_h = 1

            for _d in (self.matmul_configs, *self.matmul_versions.values()):
                for _cfg in _d.values():
                    _cap(_cfg)
        _override([n for n in self.compute_configs if n.startswith("CONV_")], "SDXL_CONV")
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
        [384, 64] for the 640 blocks), or None when that block is not transposed."""
        is_640_block = "down_blocks.1" in module_path or "up_blocks.1" in module_path
        if is_640_block and not self.transposed_640_blocks:
            return None
        if not is_640_block and not self.transposed_1280_blocks:
            return None
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))})
        shard = [384, 64] if is_640_block else [96, 128]
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.COL_MAJOR),
        )

    def get_to_out_in0_memory_config(self, module_path):
        """Shard the concat_heads output onto the block's transposed shard before attn to_out: a transposed
        small-N matmul with interleaved in0 is data-movement bound (31.6 us) while the sharded-in0 one runs 17 us;
        the reshard is L1 -> L1 (3.4 us). None = feed to_out as is."""
        return self.get_transposed_block_memory_config(module_path)

    def get_mm_output_memory_config(self, module_path):
        # sharded_640_blocks: keep the down_blocks.1 / up_blocks.1 (C=640, 4096 rows) transformer activations
        # block-sharded on 10x8 like the 1280 blocks, so LayerNorm / residual adds run sharded instead of on
        # L1-interleaved tensors (LN 4096x640 interleaved measured 42.8 us vs 13 us sharded at 1024x1280).
        is_640_block = "down_blocks.1" in module_path or "up_blocks.1" in module_path
        if "attn1" in module_path or "attn2" in module_path:
            if not "to_out" in module_path:
                return ttnn.L1_MEMORY_CONFIG
            else:
                if is_640_block and not self.sharded_640_blocks:
                    return ttnn.L1_MEMORY_CONFIG
                else:
                    return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        if "ff.net" in module_path:
            return ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        if "attentions" in module_path and "proj_in" in module_path:
            if is_640_block and not self.sharded_640_blocks:
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
            # transposed: 16384 rows -> 10 HW columns x 52 tile rows; 32-tile blocks would make conv2d re-pick 8 cores
            return self.conv_configs["ABH_26T_ADB_WDB_BS" if self.transposed_resnets else "ABH_1024_ADB_WDB_BS"]
        elif "down_blocks.0.resnets" in conv_path:
            return self.conv_configs["ABH_26T_ADB_WDB_BS" if self.transposed_resnets else "ABH_1024_ADB_WDB_BS"]
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
            # 4096x1280->1280 after the upsample: 13-tile act blocks clash with the L1 shards by 56 KB; stays 8 rows.
            # Transposed: the upsample lands on the 11x10 shard (12 tile rows per core); native 12-tile act blocks.
            return self.conv_configs["ABH_0_ADB_WDB_BS" if self.transposed_resnets else "ABH_256_ADB_WDB_BS"]
        elif ("up_blocks.0.resnets" in conv_path) and ("conv2" in conv_path):
            return self.conv_configs["ABH_0_ADB_WDB_BS"]
        elif "up_blocks.0.resnets.2.conv1" == conv_path:
            return self.conv_configs["ABH_0_ADB_WDB_BS"]

        # UP BLOCK 1
        elif "up_blocks.1.resnets.0.conv1" == conv_path:
            # Cin=1920: 13 tiles x the 1920x9 inner dim = 1.63 MB of CBs (> 1.5 MB L1) and act_block_w_div does
            # not shrink a block-sharded conv's CBs; this resnet's GN1 stays on 8 rows (16 tile rows, 4-tile blocks).
            return self.conv_configs["ABH_128_ADB_WDB_BS"]
        elif "up_blocks.1.resnets.1.conv1" == conv_path:
            return self.conv_configs["ABH_13T_ADB_WDB_BS" if self.resnet_gn_prime_rows else "ABH_256_ADB_WDB_BS"]
        elif "up_blocks.1.resnets.2.conv1" == conv_path:
            return self.conv_configs["ABH_13T_ADB_WDB_BS" if self.resnet_gn_prime_rows else "ABH_256_ADB_WDB_BS"]
        elif ("up_blocks.1.resnets" in conv_path) and ("conv2" in conv_path):
            return self.conv_configs["ABH_0_ADB_WDB_BS"]
        elif "up_blocks.1.upsamplers.0" == conv_path:
            # Tried ABH_26T (10 rows): conv2d then reshards the 10x8 upsample output 10x8 -> 10x10 and the reshard
            # kernel overflows its runtime-arg limit (8346 > 4094 args on reshard_reader_diff_width). Stays 8 rows.
            # Transposed: the upsample already lands on the 10x10 [1664, 64] shard (52 tile rows). 26-tile blocks
            # clash with the L1 shards (CB region 1.10 MB vs the shard at 963 KB); 13 tiles fits.
            return self.conv_configs["ABH_13T_ADB_WDB_BS" if self.transposed_resnets else "ABH_256_ADB_WDB_BS"]

        # UP BLOCK 2
        # Full-grid GN: these convs consume a 10x10 [1664, C/10] shard (52 tile rows per core); the 8/16-tile
        # overrides do not divide 52 and conv2d would fall back to 4/13 tiles. 26 tiles is the largest that fits.
        elif "up_blocks.2.resnets.0.conv1" == conv_path:
            # Cin=960: the 26-tile act block clashes with the L1 shards (CB region 1.57 MB); 13 tiles fits.
            return self.conv_configs[
                "ABH_13T_ADB_WDB_BS" if self.use_generated_groupnorm else "ABH_256_ADB_WDB_MOVE_BS"
            ]
        elif ("up_blocks.2.resnets" in conv_path) and ("conv2" in conv_path):
            return self.conv_configs["ABH_26T_ADB_WDB_BS" if self.use_generated_groupnorm else "ABH_1024_ADB_WDB_BS"]
        elif "up_blocks.2.resnets.1.conv1" == conv_path:
            return self.conv_configs["ABH_26T_ADB_WDB_BS" if self.use_generated_groupnorm else "ABH_512_ADB_WDB_BS"]
        elif "up_blocks.2.resnets.2.conv1" == conv_path:
            return self.conv_configs["ABH_26T_ADB_WDB_BS" if self.use_generated_groupnorm else "ABH_512_ADB_WDB_BS"]

        elif "conv_out" == conv_path:
            return self.conv_configs["ABH_128_ADB_HS"]
        else:
            return self.conv_configs["DEFAULT"]

    def get_conv_compute_config(self, module_path):
        if "conv_in" in module_path or "conv_out" in module_path:
            # Precision investigation knob: SDXL_CONVIO_COMPUTE=<compute config name> for conv_in / conv_out.
            return self.compute_configs[
                os.environ.get("SDXL_CONVIO_COMPUTE", "CONV_HIFI2_NO_FP32_NO_L1_COMPUTE_CONFIG")
            ]
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
        if not is_self_attention:
            return self.sdpa_configs["128_K"]
        if "down_blocks.1" in module_path or "up_blocks.1" in module_path:
            return self.sdpa_configs["512_K"]
        else:
            return self.sdpa_configs["1024_K"]
