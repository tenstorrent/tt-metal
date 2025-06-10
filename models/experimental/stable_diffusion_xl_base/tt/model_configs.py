# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class ModelOptimisations:
    def __init__(self, conv_act_dtype=ttnn.bfloat16, conv_w_dtype=ttnn.bfloat16):
        self.conv_configs = {}
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

    def clear_weight_preprocess(self):
        if not self.prepared_weights:
            for config_name in self.conv_configs:
                self.conv_configs[config_name].always_preprocess_weights = False
            self.prepared_weights = True

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
