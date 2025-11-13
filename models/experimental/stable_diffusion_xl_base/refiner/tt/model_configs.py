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

    def get_matmul_config(self, matmul_path):
        return None

    def get_mm_compute_config(self, module_path):
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
