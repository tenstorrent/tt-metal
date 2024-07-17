# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
import tt_lib
import ttnn

from typing import List
from models.utility_functions import torch2tt_tensor


class TtFusedFalconLayernorm:
    def __init__(self, device, gamma1, beta1, gamma2, beta2, model_config, config, tt_cache_path):
        super().__init__()

        gamma_beta_rm = False

        self.model_config = model_config

        layer_name = f"transformer.h.0"

        ln_attn_weights_str = f"{layer_name}.ln_attn.weight"
        ln_attn_bias_str = f"{layer_name}.ln_attn.bias"

        ln_mlp_weights_str = f"{layer_name}.ln_mlp.weight"
        ln_mlp_bias_str = f"{layer_name}.ln_mlp.bias"

        if gamma_beta_rm:
            ln_attn_weights_path = (
                tt_cache_path
                / f"{ln_attn_weights_str}_rm_fusedln_{self.model_config['LN_ATTN_WEIGHTS_DTYPE'].name}.bin"
            )
            if (ln_attn_weights_path).exists():
                ln_attn_gamma_host = tt_lib.tensor.load_tensor(str(ln_attn_weights_path))
                self.ln_attn_gamma = ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])
            else:
                ln_attn_gamma_host = tt_lib.tensor.Tensor(
                    gamma1.reshape([1, 1, 1, -1]),
                    self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
                )
                self.ln_attn_gamma = ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])
                tt_lib.tensor.dump_tensor(
                    str(ln_attn_weights_path),
                    ln_attn_gamma_host,
                )

            ln_attn_bias_path = (
                tt_cache_path / f"{ln_attn_bias_str}_rm_fusedln_{self.model_config['LN_ATTN_BIAS_DTYPE'].name}.bin"
            )
            if (ln_attn_bias_path).exists():
                ln_attn_beta_host = tt_lib.tensor.load_tensor(str(ln_attn_bias_path))
                self.ln_attn_beta = ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"])
            else:
                ln_attn_beta_host = tt_lib.tensor.Tensor(
                    beta1.reshape([1, 1, 1, -1]),
                    self.model_config["LN_ATTN_BIAS_DTYPE"],
                )
                self.ln_attn_beta = ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"])
                tt_lib.tensor.dump_tensor(
                    str(ln_attn_bias_path),
                    ln_attn_beta_host,
                )

            ln_mlp_weights_path = (
                tt_cache_path / f"{ln_mlp_weights_str}_rm_fusedln_{self.model_config['LN_MLP_WEIGHTS_DTYPE'].name}.bin"
            )
            if (ln_mlp_weights_path).exists():
                ln_mlp_gamma_host = tt_lib.tensor.load_tensor(str(ln_mlp_weights_path))
                self.ln_mlp_gamma = ln_mlp_gamma_host.to(device, self.model_config["LN_MLP_WEIGHTS_MEMCFG"])
            else:
                ln_mlp_gamma_host = tt_lib.tensor.Tensor(
                    gamma2.reshape([1, 1, 1, -1]),
                    self.model_config["LN_MLP_WEIGHTS_DTYPE"],
                )
                self.ln_mlp_gamma = ln_mlp_gamma_host.to(device, self.model_config["LN_MLP_WEIGHTS_MEMCFG"])
                tt_lib.tensor.dump_tensor(
                    str(ln_mlp_weights_path),
                    ln_mlp_gamma_host,
                )

            ln_mlp_bias_path = (
                tt_cache_path / f"{ln_mlp_bias_str}_rm_fusedln_{self.model_config['LN_MLP_BIAS_DTYPE'].name}.bin"
            )
            if (ln_mlp_bias_path).exists():
                ln_mlp_beta_host = tt_lib.tensor.load_tensor(str(ln_mlp_bias_path))
                self.ln_mlp_beta = ln_mlp_beta_host.to(device, self.model_config["LN_MLP_BIAS_MEMCFG"])
            else:
                ln_mlp_beta_host = tt_lib.tensor.Tensor(
                    beta2.reshape([1, 1, 1, -1]),
                    self.model_config["LN_MLP_BIAS_DTYPE"],
                )
                self.ln_mlp_beta = ln_mlp_beta_host.to(device, self.model_config["LN_MLP_BIAS_MEMCFG"])
                tt_lib.tensor.dump_tensor(
                    str(ln_mlp_bias_path),
                    ln_mlp_beta_host,
                )
        else:
            ln_attn_weights_path = (
                tt_cache_path
                / f"{ln_attn_weights_str}_tilized_fusedln_{self.model_config['LN_ATTN_WEIGHTS_DTYPE'].name}.bin"
            )
            if (ln_attn_weights_path).exists():
                ln_attn_gamma_host = ttnn.experimental.tensor.load_tensor(str(ln_attn_weights_path))
                self.ln_attn_gamma = ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])
            else:
                ln_attn_gamma_torch = gamma1.reshape(1, 1, 1, -1)
                ln_attn_gamma_torch_padded = torch.cat(
                    [ln_attn_gamma_torch, torch.zeros(1, 1, 31, ln_attn_gamma_torch.shape[-1])], dim=2
                )
                ln_attn_gamma_host = torch2tt_tensor(
                    ln_attn_gamma_torch_padded,
                    None,
                    tt_layout=ttnn.experimental.tensor.Layout.TILE,
                    tt_memory_config=self.model_config["LN_ATTN_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
                )
                self.ln_attn_gamma = ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])
                ttnn.experimental.tensor.dump_tensor(
                    str(ln_attn_weights_path),
                    ln_attn_gamma_host,
                )

            ln_attn_bias_path = (
                tt_cache_path / f"{ln_attn_bias_str}_tilized_fusedln_{self.model_config['LN_ATTN_BIAS_DTYPE'].name}.bin"
            )
            if (ln_attn_bias_path).exists():
                ln_attn_beta_host = ttnn.experimental.tensor.load_tensor(str(ln_attn_bias_path))
                self.ln_attn_beta = ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"])
            else:
                ln_attn_beta_torch = beta1.reshape(1, 1, 1, -1)
                ln_attn_beta_torch_padded = torch.cat(
                    [ln_attn_beta_torch, torch.zeros(1, 1, 31, ln_attn_beta_torch.shape[-1])], dim=2
                )
                ln_attn_beta_host = torch2tt_tensor(
                    ln_attn_beta_torch_padded,
                    None,
                    tt_layout=ttnn.experimental.tensor.Layout.TILE,
                    tt_memory_config=self.model_config["LN_ATTN_BIAS_MEMCFG"],
                    tt_dtype=self.model_config["LN_ATTN_BIAS_DTYPE"],
                )
                self.ln_attn_beta = ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"])
                ttnn.experimental.tensor.dump_tensor(
                    str(ln_attn_bias_path),
                    ln_attn_beta_host,
                )

            ln_mlp_weights_path = (
                tt_cache_path
                / f"{ln_mlp_weights_str}_tilized_fusedln_{self.model_config['LN_MLP_WEIGHTS_DTYPE'].name}.bin"
            )
            if (ln_mlp_weights_path).exists():
                ln_mlp_gamma_host = ttnn.experimental.tensor.load_tensor(str(ln_mlp_weights_path))
                self.ln_mlp_gamma = ln_mlp_gamma_host.to(device, self.model_config["LN_MLP_WEIGHTS_MEMCFG"])
            else:
                ln_mlp_gamma_torch = gamma2.reshape(1, 1, 1, -1)
                ln_mlp_gamma_torch_padded = torch.cat(
                    [ln_mlp_gamma_torch, torch.zeros(1, 1, 31, ln_mlp_gamma_torch.shape[-1])], dim=2
                )
                ln_mlp_gamma_host = torch2tt_tensor(
                    ln_mlp_gamma_torch_padded,
                    None,
                    tt_layout=ttnn.experimental.tensor.Layout.TILE,
                    tt_memory_config=self.model_config["LN_MLP_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["LN_MLP_WEIGHTS_DTYPE"],
                )
                self.ln_mlp_gamma = ln_mlp_gamma_host.to(device, self.model_config["LN_MLP_WEIGHTS_MEMCFG"])
                ttnn.experimental.tensor.dump_tensor(
                    str(ln_mlp_weights_path),
                    ln_mlp_gamma_host,
                )

            ln_mlp_bias_path = (
                tt_cache_path / f"{ln_mlp_bias_str}_tilized_fusedln_{self.model_config['LN_MLP_BIAS_DTYPE'].name}.bin"
            )
            if (ln_mlp_bias_path).exists():
                ln_mlp_beta_host = ttnn.experimental.tensor.load_tensor(str(ln_mlp_bias_path))
                self.ln_mlp_beta = ln_mlp_beta_host.to(device, self.model_config["LN_MLP_BIAS_MEMCFG"])
            else:
                ln_mlp_beta_torch = beta2.reshape(1, 1, 1, -1)
                ln_mlp_beta_torch_padded = torch.cat(
                    [ln_mlp_beta_torch, torch.zeros(1, 1, 31, ln_mlp_beta_torch.shape[-1])], dim=2
                )
                ln_mlp_beta_host = torch2tt_tensor(
                    ln_mlp_beta_torch_padded,
                    None,
                    tt_layout=ttnn.experimental.tensor.Layout.TILE,
                    tt_memory_config=self.model_config["LN_MLP_BIAS_MEMCFG"],
                    tt_dtype=self.model_config["LN_MLP_BIAS_DTYPE"],
                )
                self.ln_mlp_beta = ln_mlp_beta_host.to(device, self.model_config["LN_MLP_BIAS_MEMCFG"])
                ttnn.experimental.tensor.dump_tensor(
                    str(ln_mlp_bias_path),
                    ln_mlp_beta_host,
                )

        self.layernorm_eps = config.layer_norm_epsilon

        shard_spec_cores_grid = ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, 0),
                    ttnn.experimental.tensor.CoreCoord(7, 7),
                ),
            }
        )
        H = config.hidden_size  # 8192
        self.sharded_memconfig = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            ttnn.experimental.tensor.ShardSpec(
                shard_spec_cores_grid,
                [self.model_config["SEQ_LEN"] // 8, H // 8],  # 1024
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        self.width_sharded_memconfig = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            ttnn.experimental.tensor.ShardSpec(
                shard_spec_cores_grid,
                [
                    self.model_config["SEQ_LEN"],
                    H // 64,
                ],
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        self.prg_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 8],
            subblock_w=min(8, H // 8),
            block_h=self.model_config["SEQ_LEN"] // 32 // 8,
            block_w=H // 32 // 8,
            inplace=False,
        )

        self.interleaved_memconfig = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            ttnn.experimental.tensor.BufferType.L1,
        )

    def __call__(self, x: ttnn.experimental.tensor.Tensor) -> ttnn.experimental.tensor.Tensor:
        # # OG layernorm
        # out1 = ttnn.layer_norm(
        #     x, epsilon=self.layernorm_eps, weight=self.ln_attn_gamma, bias=self.ln_attn_beta, memory_config=self.sharded_memconfig, program_config=self.prg_config
        # )
        # out2 = ttnn.layer_norm(
        #     x, epsilon=self.layernorm_eps, weight=self.ln_mlp_gamma, bias=self.ln_mlp_beta, memory_config=self.sharded_memconfig, program_config=self.prg_config
        # )

        # all block sharded
        out2 = ttnn.layer_norm(
            x,
            epsilon=self.layernorm_eps,
            weight=None,
            bias=None,
            memory_config=self.sharded_memconfig,
            program_config=self.prg_config,
        )

        out1 = ttnn.experimental.tensor.bcast(
            out2,
            self.ln_attn_gamma,
            math_op=ttnn.experimental.tensor.BcastOpMath.MUL,
            dim=ttnn.experimental.tensor.BcastOpDim.H,
            output_mem_config=self.sharded_memconfig,
        )
        out1 = ttnn.experimental.tensor.bcast(
            out1,
            self.ln_attn_beta,
            math_op=ttnn.experimental.tensor.BcastOpMath.ADD,
            dim=ttnn.experimental.tensor.BcastOpDim.H,
            output_mem_config=self.sharded_memconfig,
        )

        out2 = ttnn.experimental.tensor.bcast(
            out2,
            self.ln_mlp_gamma,
            math_op=ttnn.experimental.tensor.BcastOpMath.MUL,
            dim=ttnn.experimental.tensor.BcastOpDim.H,
            output_mem_config=self.sharded_memconfig,
        )
        out2 = ttnn.experimental.tensor.bcast(
            out2,
            self.ln_mlp_beta,
            math_op=ttnn.experimental.tensor.BcastOpMath.ADD,
            dim=ttnn.experimental.tensor.BcastOpDim.H,
            output_mem_config=self.sharded_memconfig,
        )

        # # Testing bcast only
        # out1 = ttnn.experimental.tensor.bcast(
        #     x,
        #     self.ln_attn_gamma,
        #     math_op=ttnn.experimental.tensor.BcastOpMath.MUL,
        #     dim=ttnn.experimental.tensor.BcastOpDim.H,
        #     output_mem_config=self.sharded_memconfig,
        # )
        # out2 = x

        return out1, out2
