# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import re


class ModelOptimisations:
    def __init__(self, conv_act_dtype=ttnn.bfloat16, conv_w_dtype=ttnn.bfloat16):
        self.conv_configs = {}
        self.matmul_configs = {}
        self.compute_configs = {}
        self.prepared_weights = False
        self.conv_w_dtype = conv_w_dtype
        self.conv_ws_dtype = ttnn.bfloat8_b

        # HEIGHT SHARDED
        self.conv_configs["ABH_256_ADB"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_128_NO_ADB_HS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=True,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )

        # BLOCK SHARDED
        self.conv_configs["ABH_32_NO_ADB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=32,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_64_NO_ADB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_64_NO_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_64_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )

        self.conv_configs["ABH_64_NO_ADB_BS_BF16"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )

        self.conv_configs["ABH_128_ADB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_128_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_128_NO_ADB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_128_NO_ADB_WDB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_128_ADB_WDB_NO_DEALLOC_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_128_NO_ADB_WDB_NO_DEALLOC_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_256_NO_ADB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )

        # WIDTH SHARDED
        self.conv_configs["ABH_256_NO_ADB_WS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_512_NO_ADB_WS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=self.conv_ws_dtype,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=512,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )

        # DEFAULT CONF
        self.conv_configs["DEFAULT"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )

        # DRAM CONF
        self.conv_configs["ABH_64_NO_ADB_DRAM"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=64,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_128_NO_ADB_DRAM"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_512_NO_ADB_DRAM"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=512,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["DEFAULT_DRAM"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )
        self.conv_configs["ABH_256_NO_ADB_HS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=True,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
            preprocess_weights_on_device=False,
            always_preprocess_weights=False,
        )

        self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_4096"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=1,  # max is 20, 1 seems optimal?
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=8,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_1024"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # max is 40, 4 seems optimal?
            per_core_M=4,
            per_core_N=5,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_FF2_SEQ_LEN_1024"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=16,  # max is 160, 20 seems optimal?
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=4,
            per_core_N=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_FF2_SEQ_LEN_4096"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 8),
            in0_block_w=2,  # max is 80, 2 seems optimal
            out_subblock_h=8,
            out_subblock_w=1,
            per_core_M=16,
            per_core_N=3,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["1D_RESNET_LINEAR"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=10,  # max is 40, 10 seems optimal
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            mcast_in0=True,
            fuse_batch=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_GEGLU_LINEAR_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            per_core_M=16,
            per_core_N=20,
            out_subblock_h=1,
            out_subblock_w=5,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["1D_GEGLU_LINEAR_1280"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=32,
            per_core_N=5,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

        self.matmul_configs["2D_TM_LINEAR_640"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,
            per_core_M=16,
            per_core_N=3,
            out_subblock_h=8,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.matmul_configs["2D_TM_LINEAR_1280"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=5,
            per_core_M=4,
            per_core_N=5,
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

    def clear_weight_preprocess(self):
        if not self.prepared_weights:
            for config_name in self.conv_configs:
                self.conv_configs[config_name].always_preprocess_weights = False
            self.prepared_weights = True

    def get_matmul_config(self, matmul_path):
        if matmul_path is None:
            return None

        if not ("decoder" in matmul_path):
            # # # GEGLU # # #
            if "net.0.proj" in matmul_path:
                if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                    return self.matmul_configs["2D_GEGLU_LINEAR_640"]
                else:
                    return self.matmul_configs["1D_GEGLU_LINEAR_1280"]

            # # # TM LINEAR # # #
            if "proj_in" in matmul_path or "proj_out" in matmul_path:
                if "down_blocks.1" in matmul_path or "up_blocks.1" in matmul_path:
                    return self.matmul_configs["2D_TM_LINEAR_640"]
                else:
                    return self.matmul_configs["2D_TM_LINEAR_1280"]

            # # # Down block 1 # # #
            pattern_downn_block_1_dense_out = re.compile(
                r"down_blocks\.1\.attentions\.[01]\.transformer_blocks\.[01]\.attn[12]\.dense_out"
            )

            # 8 occurences
            if pattern_downn_block_1_dense_out.search(matmul_path):
                return self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_4096"]

            pattern_down_blocks_1_ff2 = re.compile(
                r"down_blocks\.1\.attentions\.[01]\.transformer_blocks\.[01]\.ff\.net\.2"
            )

            # 4 occurences
            if pattern_down_blocks_1_ff2.search(matmul_path):
                return self.matmul_configs["2D_FF2_SEQ_LEN_4096"]

            # # # Down block 2 # # #
            pattern_down_blocks_2_dense_out = re.compile(
                r"down_blocks\.2\.attentions\.[01]\.transformer_blocks\.[0123456789]\.attn[12]\.dense_out"
            )

            # 40 occurences
            if pattern_down_blocks_2_dense_out.search(matmul_path):
                return self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_1024"]

            pattern_down_blockcs_2_ff2 = re.compile(
                r"down_blocks\.2\.attentions\.[01]\.transformer_blocks\.[0123456789]\.ff\.net\.2"
            )

            # 20 occurences
            if pattern_down_blockcs_2_ff2.search(matmul_path):
                return self.matmul_configs["2D_FF2_SEQ_LEN_1024"]

            # # # Mid block  # # #
            pattern_mid_block_ff2 = re.compile(
                r"mid_block\.attentions\.0\.transformer_blocks\.[0123456789]\.ff\.net\.2"
            )

            # 10 occurences
            if pattern_mid_block_ff2.search(matmul_path):
                return self.matmul_configs["2D_FF2_SEQ_LEN_1024"]

            pattern_mid_block_dense_out = re.compile(
                r"mid_block\.attentions\.0\.transformer_blocks\.[0123456789]\.attn[12]\.dense_out"
            )

            # 20 occurences
            if pattern_mid_block_dense_out.search(matmul_path):
                return self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_1024"]

            # # # Up block 0 # # #
            pattern_up_blocks_0_dense_out = re.compile(
                r"up_blocks\.0\.attentions\.[012]\.transformer_blocks\.[0123456789]\.attn[12]\.dense_out"
            )

            # 60 occurences
            if pattern_up_blocks_0_dense_out.search(matmul_path):
                return self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_1024"]

            pattern_up_blocks_0_ff2 = re.compile(
                r"up_blocks\.0\.attentions\.[012]\.transformer_blocks\.[0123456789]\.ff\.net\.2"
            )

            # 30 occurences
            if pattern_up_blocks_0_ff2.search(matmul_path):
                return self.matmul_configs["2D_FF2_SEQ_LEN_1024"]

            # # # Up block 1 # # #
            pattern_up_blocks_1_dense_out = re.compile(
                r"up_blocks\.1\.attentions\.[012]\.transformer_blocks\.[01]\.attn[12]\.dense_out"
            )

            # 12 occurences
            if pattern_up_blocks_1_dense_out.search(matmul_path):
                return self.matmul_configs["2D_LINEAR_ATTENTION_DO_SEQ_LEN_4096"]

            pattern_up_blocks_1_ff2 = re.compile(
                r"up_blocks\.1\.attentions\.[012]\.transformer_blocks\.[01]\.ff\.net\.2"
            )

            # 6 occurences
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
        return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]

    def get_conv_config(self, conv_path):
        if conv_path is None:
            return None

        if not ("decoder" in conv_path):
            if "conv_in" == conv_path:
                return self.conv_configs["ABH_256_ADB"]

            # DOWN BLOCK 0
            elif "down_blocks.0.resnets" in conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_BS"]
            elif "down_blocks.0.downsamplers.0" == conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_NO_DEALLOC_BS"]

            # DOWN BLOCK 1
            elif "down_blocks.1.resnets.0.conv1" == conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_BS"]  # Note: ABH should be 256 with no ABD/WDB (OOM)
            elif ("down_blocks.1.resnets.0.conv2" == conv_path) or ("down_blocks.1.resnets.1" in conv_path):
                return self.conv_configs["ABH_128_NO_ADB_WDB_BS"]  # Note: should have ADB
            elif "down_blocks.1.downsamplers.0" == conv_path:
                return self.conv_configs["ABH_128_ADB_WDB_NO_DEALLOC_BS"]

            # DOWN BLOCK 2
            elif "down_blocks.2.resnets.1.conv1" == conv_path:
                return self.conv_configs["ABH_64_NO_ADB_WDB_BS"]  # Note: should be 128 with ADB/WDB
            elif "down_blocks.2.resnets.0.conv1" == conv_path:
                return self.conv_configs["ABH_128_NO_ADB_WDB_BS"]
            elif ("down_blocks.2.resnets.0.conv2" == conv_path) or ("down_blocks.2.resnets.1.conv2" == conv_path):
                return self.conv_configs["ABH_64_NO_ADB_WDB_BS"]

            # MID BLOCK
            elif "mid_block" in conv_path:
                return self.conv_configs["ABH_64_NO_ADB_WDB_BS"]

            # UP BLOCK 0
            elif ("up_blocks.0.resnets.0.conv1" == conv_path) or ("up_blocks.0.resnets.1.conv1" == conv_path):
                return self.conv_configs["ABH_32_NO_ADB_BS"]
            elif "up_blocks.0.upsamplers.0" == conv_path:
                return self.conv_configs["ABH_64_NO_ADB_BS"]
            elif ("up_blocks.0.resnets" in conv_path) and ("conv2" in conv_path):
                return self.conv_configs["ABH_64_NO_ADB_BS_BF16"]
            elif "up_blocks.0.resnets.2.conv1" == conv_path:
                return self.conv_configs["ABH_512_NO_ADB_WS"]

            # UP BLOCK 1
            elif "up_blocks.1.resnets.0.conv1" == conv_path:
                return self.conv_configs["ABH_32_NO_ADB_BS"]
            elif "up_blocks.1.resnets.1.conv1" == conv_path:
                return self.conv_configs["ABH_64_NO_ADB_BS"]
            elif "up_blocks.1.resnets.2.conv1" == conv_path:
                return self.conv_configs["ABH_32_NO_ADB_BS"]
            elif ("up_blocks.1.resnets" in conv_path) and ("conv2" in conv_path):
                return self.conv_configs["ABH_128_NO_ADB_WDB_BS"]
            elif "up_blocks.1.upsamplers.0" == conv_path:
                return self.conv_configs["ABH_64_NO_ADB_BS"]

            # UP BLOCK 2
            elif "up_blocks.2.resnets.0.conv1" == conv_path:
                return self.conv_configs["ABH_64_ADB_WDB_BS"]
            elif ("up_blocks.2.resnets" in conv_path) and ("conv2" in conv_path):
                return self.conv_configs["ABH_128_ADB_WDB_BS"]
            elif ("up_blocks.2.resnets.1.conv1" == conv_path) or ("up_blocks.2.resnets.2.conv1" == conv_path):
                return self.conv_configs["ABH_128_ADB_WDB_BS"]

            elif "conv_out" == conv_path:
                return self.conv_configs["ABH_128_NO_ADB_HS"]
            else:
                return self.conv_configs["DEFAULT"]
        else:
            # VAE
            if "decoder.conv_in" == conv_path:
                return self.conv_configs["ABH_256_NO_ADB_HS"]
            elif ("decoder.up_blocks.2.resnet.0" in conv_path) and ("conv1" in conv_path):
                return self.conv_configs["ABH_128_NO_ADB_DRAM"]
            elif ("decoder.up_blocks.2.resnet" in conv_path) and ("conv1" in conv_path):
                return self.conv_configs["ABH_64_NO_ADB_DRAM"]  # should be 128, OOM in demo
            # elif ("decoder.up_blocks.2.resnet" in conv_path) and ("conv2" in conv_path):
            #     return self.conv_configs["ABH_32_NO_ADB_DRAM"] # Note: ABH should be 128 (OOM)
            elif "decoder.up_blocks.2.upsamplers.0" == conv_path:
                return self.conv_configs["ABH_64_NO_ADB_DRAM"]  # should be 128, OOM in demo
            elif "decoder.conv_out" == conv_path:
                return self.conv_configs["ABH_512_NO_ADB_DRAM"]
            else:
                return self.conv_configs["DEFAULT_DRAM"]
