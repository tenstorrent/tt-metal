# SPDX-FileCopyrightText: © <YEAR> Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class ModelOptimisations:
    def __init__(self, conv_act_dtype=ttnn.bfloat16, conv_w_dtype=ttnn.bfloat16):
        self.conv_configs = {}
        self.conv_configs["ABH_512_NO_ADB"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=512,
            preprocess_weights_on_device=False,
            always_preprocess_weights=True,
            transpose_shards=True,
        )
        self.conv_configs["ABH_256_ADB"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
            preprocess_weights_on_device=False,
            always_preprocess_weights=True,
            transpose_shards=True,
        )
        self.conv_configs["ABH_256_NO_ADB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,  # should be 256 but hitting OOM
            preprocess_weights_on_device=False,
            always_preprocess_weights=True,
            transpose_shards=True,
        )
        self.conv_configs["ABH_128_ADB"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=True,
            transpose_shards=True,
        )
        self.conv_configs["ABH_128_NO_ADB"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=512,
            preprocess_weights_on_device=False,
            always_preprocess_weights=True,
            transpose_shards=True,
        )
        self.conv_configs["ABH_128_NO_ADB_NO_DEALLOC"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=None,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=True,
            transpose_shards=True,
        )
        self.conv_configs["ABH_128_NO_ADB_NO_DEALLOC_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=128,
            preprocess_weights_on_device=False,
            always_preprocess_weights=True,
            transpose_shards=True,
        )
        self.conv_configs["ABH_128_ADB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
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
            always_preprocess_weights=True,
            transpose_shards=True,
        )
        self.conv_configs["ABH_128_NO_ADB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
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
            always_preprocess_weights=True,
            transpose_shards=True,
        )

        self.conv_configs["ABH_64_NO_ADB_BS"] = ttnn.Conv2dConfig(
            dtype=conv_act_dtype,
            weights_dtype=conv_w_dtype,
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
            always_preprocess_weights=True,
            transpose_shards=True,
        )

    def get_conv_config(self, conv_path):
        if conv_path is None:
            return None
        if "conv_in" == conv_path:
            return self.conv_configs["ABH_256_ADB"]
        elif "down_blocks.0.resnets" in conv_path:
            return self.conv_configs["ABH_128_ADB"]
        elif "down_blocks.2.resnets.1.conv1" == conv_path:
            return self.conv_configs["ABH_64_NO_ADB_BS"]
        elif "down_blocks.0.downsamplers.0" == conv_path:
            return self.conv_configs["ABH_128_NO_ADB_NO_DEALLOC"]
        elif "down_blocks.1.resnets.0.conv1" == conv_path:
            return self.conv_configs["ABH_256_NO_ADB_BS"]
        elif ("down_blocks.1.resnets.0.conv2" == conv_path) or ("down_blocks.1.resnets.1" in conv_path):
            return self.conv_configs["ABH_128_NO_ADB_BS"]
        elif "down_blocks.1.downsamplers.0" == conv_path:
            return self.conv_configs["ABH_128_NO_ADB_NO_DEALLOC_BS"]
        elif "down_blocks.2.resnets.0.conv1" == conv_path:
            return self.conv_configs["ABH_128_NO_ADB_BS"]  # should be ABH_128_ADB_BS, hitting OOM on UNet
        elif ("down_blocks.2.resnets.0.conv2" == conv_path) or ("down_blocks.2.resnets.1.conv2" == conv_path):
            return self.conv_configs["ABH_64_NO_ADB_BS"]
        elif "mid_block" in conv_path:
            return self.conv_configs["ABH_64_NO_ADB_BS"]
        else:
            return None
