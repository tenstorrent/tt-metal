# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
import tt_lib

from typing import List
from models.utility_functions import torch2tt_tensor


class TtFalconLayernorm:
    def __init__(self, devices, model_config, config, tt_cache_path, is_sharded=False):
        super().__init__()

        self.model_config = model_config
        self.is_sharded = is_sharded

        layer_name = f"transformer.h.0"

        ln_attn_weights_str = f"{layer_name}.ln_attn.weight"
        ln_attn_bias_str = f"{layer_name}.ln_attn.bias"

        ln_attn_weights_path = (
            tt_cache_path / f"{ln_attn_weights_str}_rm_{self.model_config['LN_ATTN_WEIGHTS_DTYPE'].name}.bin"
        )
        if (ln_attn_weights_path).exists():
            ln_attn_gamma_host = tt_lib.tensor.load_tensor(str(ln_attn_weights_path))
            self.ln_attn_gamma = [
                ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"]) for device in devices
            ]
        else:
            ln_attn_gamma_host = tt_lib.tensor.Tensor(
                self.state_dict[ln_attn_weights_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
            )
            self.ln_attn_gamma = [
                ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"]) for device in devices
            ]
            tt_lib.tensor.dump_tensor(
                str(ln_attn_weights_path),
                ln_attn_gamma_host,
            )

        ln_attn_bias_path = tt_cache_path / f"{ln_attn_bias_str}_rm_{self.model_config['LN_ATTN_BIAS_DTYPE'].name}.bin"
        if (ln_attn_bias_path).exists():
            ln_attn_beta_host = tt_lib.tensor.load_tensor(str(ln_attn_bias_path))
            self.ln_attn_beta = [
                ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"]) for device in devices
            ]
        else:
            ln_attn_beta_host = tt_lib.tensor.Tensor(
                self.state_dict[ln_attn_bias_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_ATTN_BIAS_DTYPE"],
            )
            self.ln_attn_beta = [
                ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"]) for device in devices
            ]
            tt_lib.tensor.dump_tensor(
                str(ln_attn_bias_path),
                ln_attn_beta_host,
            )

        self.layernorm_eps = config.layer_norm_epsilon

    def __call__(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        if self.is_sharded:
            row_height = x.get_legacy_shape()[2]
            shard_width_hidden_dim_across_32_cores = x.get_legacy_shape()[3] // 32
            shard_spec_32_cores_grid = tt_lib.tensor.CoreRangeSet(
                {
                    tt_lib.tensor.CoreRange(
                        tt_lib.tensor.CoreCoord(0, 0),
                        tt_lib.tensor.CoreCoord(7, 3),
                    ),
                }
            )
            # # Option1 : width sharded; produces bad PCC
            # out = tt_lib.operations.primary.layernorm(
            #     x,
            #     self.layernorm_eps,
            #     self.ln_attn_gamma[0],
            #     self.ln_attn_beta[0],
            #     tt_lib.tensor.MemoryConfig(
            #         tt_lib.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            #         tt_lib.tensor.BufferType.L1,
            #         tt_lib.tensor.ShardSpec(
            #             shard_spec_32_cores_grid,
            #             [
            #                 row_height,
            #                 shard_width_hidden_dim_across_32_cores,
            #             ],
            #             tt_lib.tensor.ShardOrientation.ROW_MAJOR,
            #             False,
            #         ),
            #     ),
            #     tt_lib.operations.primary.LayerNormShardedMultiCoreProgramConfig(
            #         compute_with_storage_grid_size=[8, 4],
            #         subblock_w=8,
            #         block_h=row_height // 32,
            #         block_w=8,
            #         math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
            #         im_data_format=tt_lib.tensor.DataType.BFLOAT16,
            #         out_data_format=self.model_config["LN_ATTN_OUTPUT_DTYPE"],
            #         inplace=False,
            #     ),
            # )

            # # option 2: block sharded hardcoded for S=128 and 8x4 grid of cores; produces good PCC!
            # out = tt_lib.operations.primary.layernorm(
            #     x,
            #     self.layernorm_eps,
            #     self.ln_attn_gamma[0],
            #     self.ln_attn_beta[0],
            #     tt_lib.tensor.MemoryConfig(
            #         tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            #         tt_lib.tensor.BufferType.L1,
            #         tt_lib.tensor.ShardSpec(
            #             shard_spec_32_cores_grid,
            #             [
            #                 32,
            #                 1024,
            #             ],
            #             tt_lib.tensor.ShardOrientation.ROW_MAJOR,
            #             False,
            #         ),
            #     ),
            #     tt_lib.operations.primary.LayerNormShardedMultiCoreProgramConfig(
            #         compute_with_storage_grid_size=[8, 4],
            #         subblock_w=8,
            #         block_h=1,
            #         block_w=32, # 8
            #         math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
            #         im_data_format=tt_lib.tensor.DataType.BFLOAT16,
            #         out_data_format=self.model_config["LN_ATTN_OUTPUT_DTYPE"],
            #         inplace=False,
            #     ),
            # )

            # version according to model_config for debug
            out = tt_lib.operations.primary.layernorm(
                x,
                self.layernorm_eps,
                self.ln_attn_gamma[0],
                self.ln_attn_beta[0],
                self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
                self.model_config["LN_ATTN_PROGCFG"],
            )
        else:  # Interleaved does not work for falcon40b dims [32, 8192] since once one core per tile-height is used to process the whole row
            # Option 1: uses only one core; runs out of L1
            # E           Statically allocated circular buffers on core range {} grow to {} B which is beyond max L1 size of {} B
            # E           [(x=0,y=0) - (x=1,y=0)]
            out = tt_lib.operations.primary.layernorm(
                x,
                self.layernorm_eps,
                self.ln_attn_gamma[0],
                self.ln_attn_beta[0],
                # self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
                # self.model_config["LN_ATTN_PROGCFG"],
            )

            # Option 2: uses only one core?!
            # Runs out of L1
            # E                       Statically allocated circular buffers on core range {} grow to {} B which is beyond max L1 size of {} B
            # E                       [(x=0,y=0) - (x=1,y=0)]
            # out = tt_lib.operations.primary.layernorm(
            #     x,
            #     self.layernorm_eps,
            #     self.ln_attn_gamma[0],
            #     self.ln_attn_beta[0],
            #     program_config=tt_lib.operations.primary.LayerNormInterleavedMultiCoreProgramConfig(
            #         math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
            #         im_data_format=tt_lib.tensor.DataType.BFLOAT16,
            #         out_data_format=tt_lib.tensor.DataType.BFLOAT8_B,
            #     ),
            # )

            # # Option 3: uses only one core?!
            # # Runs out of L1
            # # E               Statically allocated circular buffers on core range {} grow to {} B which is beyond max L1 size of {} B
            # # E               [(x=0,y=0) - (x=1,y=0)]
            # out = tt_lib.tensor.layernorm(
            #     x,
            #     self.layernorm_eps,
            #     output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            # )
            # out = tt_lib.tensor.bcast(
            #     out,
            #     self.ln_attn_gamma[0],
            #     tt_lib.tensor.BcastOpMath.MUL,
            #     tt_lib.tensor.BcastOpDim.H,
            #     output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            # )
            # out = tt_lib.tensor.bcast(
            #     out,
            #     self.ln_attn_beta[0],
            #     tt_lib.tensor.BcastOpMath.ADD,
            #     tt_lib.tensor.BcastOpDim.H,
            #     output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            # )

        return out
