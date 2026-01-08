# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations


class VAEModelOptimisations(ModelOptimisations):
    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat16,
        conv_w_dtype=ttnn.bfloat16,
        attention_weights_dtype=ttnn.bfloat8_b,
        ff_weights_dtype=ttnn.bfloat8_b,
    ):
        super().__init__(conv_act_dtype, conv_w_dtype, attention_weights_dtype, ff_weights_dtype)

        self.sdpa_configs["64_K"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=64,
            k_chunk_size=64,
            exp_approx_mode=False,
        )

        self.groupnorm_configs["DRAM_GROUPNORM_32"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=8),
                "num_out_blocks": 32,
                "inplace": False,
            },
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "negative_mask": False,
        }

        self.groupnorm_configs["DRAM_GROUPNORM_8"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=8),
                "num_out_blocks": 8,
                "inplace": False,
            },
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "negative_mask": False,
        }

        self.groupnorm_configs["DRAM_GROUPNORM_12"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=8),
                "num_out_blocks": 12,
                "inplace": False,
            },
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "negative_mask": False,
        }

        self.groupnorm_configs["DRAM_GROUPNORM_4"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=8),
                "num_out_blocks": 4,
                "inplace": False,
            },
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "negative_mask": False,
        }

        self.groupnorm_configs["DRAM_GROUPNORM_48"] = {
            "op_config": {
                "core_grid": ttnn.CoreGrid(y=8, x=8),
                "num_out_blocks": 48,
                "inplace": False,
            },
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "negative_mask": False,
        }

    def get_matmul_config(self, matmul_path):
        return None

    def get_mm_compute_config(self, module_path):
        # for now, return default config
        if ".to_q" in module_path:
            return self.compute_configs["MATH_APPROX_MM_COMPUTE_CONFIG"]
        return self.compute_configs["DEFAULT_MM_COMPUTE_CONFIG"]

    def get_mm_output_memory_config(self, module_path):
        return None

    def get_conv_config(self, conv_path):
        if conv_path is None:
            return None

        # VAE
        # DECODER CONV IN
        if "decoder.conv_in" == conv_path:
            return self.conv_configs["ABH_0_ADB_HS"]
        # MID BLOCK (ENCODER + DECODER) and UP BLOCK 0
        elif "mid_block.resnet" in conv_path or "decoder.up_blocks.0.resnet" in conv_path:
            return self.conv_configs["ABH_512_NO_ADB_BS"]
        elif "decoder.up_blocks.0.upsamplers" in conv_path:
            return self.conv_configs["ABH_256_NO_ADB_BS"]
        # UP BLOCK 1
        elif "decoder.up_blocks.1.resnet" in conv_path:
            return self.conv_configs["ABH_256_NO_ADB_BS"]
        elif "decoder.up_blocks.1.upsamplers" in conv_path:
            return self.conv_configs["ABH_256_NO_ADB_BS"]
        # UP BLOCK 2
        elif "decoder.up_blocks.2.resnet" in conv_path:
            return self.conv_configs["ABH_512_NO_ADB_BS"]
        elif "decoder.up_blocks.2.upsamplers" in conv_path:
            return self.conv_configs["ABH_512_NO_ADB_BS"]
        # UP BLOCK 3
        elif "decoder.up_blocks.3.resnet" in conv_path:
            return self.conv_configs["ABH_32_ADB_HS"]
        # DECODER CONV OUT
        elif "decoder.conv_out" == conv_path:
            return self.conv_configs["ABH_256_NO_ADB_HS"]
        # ENCODER DOWNSAMPLERS
        elif "downsamplers" in conv_path:
            if "down_blocks.0" in conv_path:
                return self.conv_configs["ABH_256_NO_ADB_HS"]
            elif "down_blocks.1" in conv_path:
                return self.conv_configs["ABH_1024_NO_ADB_BS"]
            elif "down_blocks.2" in conv_path:
                return self.conv_configs["ABH_512_NO_ADB_BS"]
        # DOWN BLCOK 0
        elif "down_blocks.0" in conv_path:
            return self.conv_configs["ABH_32_ADB_HS"]
        # DOWN BLOCK 1
        elif "down_blocks.1" in conv_path:
            if "resnets.0" in conv_path and "conv1" in conv_path:
                return self.conv_configs["ABH_64_NO_ADB_HS"]
            else:
                return self.conv_configs["ABH_512_NO_ADB_BS"]
        # DOWN BLOCK 2
        elif "down_blocks.2" in conv_path:
            if "resnets.0" in conv_path and "conv1" in conv_path:
                return self.conv_configs["ABH_1024_NO_ADB_BS"]
            else:
                return self.conv_configs["ABH_256_NO_ADB_BS"]
        # DOWN BLOCK 3
        elif "down_blocks.3" in conv_path:
            return self.conv_configs["ABH_512_NO_ADB_BS"]
        # ENCODER CONV IN
        elif "encoder.conv_in" == conv_path:
            return self.conv_configs["ABH_1024_NO_ADB_HS"]
        # ENCODER CONV OUT
        elif "encoder.conv_out" == conv_path:
            return self.conv_configs["ABH_0_NO_ADB_HS"]
        else:
            return self.conv_configs["DEFAULT_DRAM"]

    def get_conv_compute_config(self, module_path):
        return self.compute_configs["CONV_LOFI_FP32_COMPUTE_CONFIG"]

    def get_conv_output_dtype(self):
        return self.conv_output_dtype

    def _get_groupnorm_config(self, module_path):
        if module_path is None:
            return self.groupnorm_configs["DRAM_GROUPNORM_32"]
        elif "attentions" in module_path:
            return self.groupnorm_configs["SHARDED_GROUPNORM_NON_INPLACE"]
        elif "encoder" == module_path:
            return self.groupnorm_configs["SHARDED_GROUPNORM_NON_INPLACE"]
        elif "mid_block" in module_path or "up_blocks.0" in module_path:
            return self.groupnorm_configs["SHARDED_GROUPNORM_NON_INPLACE"]
        elif "down_blocks.0" in module_path:
            return self.groupnorm_configs["DRAM_GROUPNORM_32"]
        elif "down_blocks.1" in module_path:
            if "resnets.0" in module_path and "norm1" in module_path:
                return self.groupnorm_configs["DRAM_GROUPNORM_8"]
            else:
                return self.groupnorm_configs["DRAM_GROUPNORM_12"]
        elif "down_blocks.2" in module_path:
            return self.groupnorm_configs["DRAM_GROUPNORM_4"]
        elif "down_blocks.3" in module_path:
            return self.groupnorm_configs["SHARDED_GROUPNORM_NON_INPLACE"]
        else:
            parts = module_path.split(".")
            block_id = int(parts[parts.index("up_blocks") + 1])
            resnet_id = int(parts[parts.index("resnets") + 1])

            if block_id == 1:
                return self.groupnorm_configs["DRAM_GROUPNORM_4"]
            elif block_id == 2:
                return self.groupnorm_configs["DRAM_GROUPNORM_12"]
            else:
                if "norm1" in module_path and resnet_id == 0:
                    return self.groupnorm_configs["DRAM_GROUPNORM_48"]
                else:
                    return self.groupnorm_configs["DRAM_GROUPNORM_32"]

    def get_layernorm_config(self, module_path):
        return None

    def get_sdpa_config(self, module_path, is_self_attention):
        return self.sdpa_configs["64_K"]
