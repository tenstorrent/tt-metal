# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from typing import Optional

import tt_lib
from models.demos.metal_BERT_large_11.tt.mha import TtMultiHeadAttentionModel
from models.demos.metal_BERT_large_11.tt.ffn import TtFeedForwardModel
from tt_lib.utils import pad_weight


class TtBertEncoder(torch.nn.Module):
    def __init__(self, config, encoder_idx, state_dict, device, grid_size, model_config, tt_cache_path):
        super().__init__()
        self.device = device
        self.model_config = model_config

        # MHA sub-graph
        self.mha = TtMultiHeadAttentionModel(
            config, encoder_idx, state_dict, device, grid_size, model_config, tt_cache_path
        )

        attn_layer_name = f"bert.encoder.layer.{encoder_idx}.attention.output"
        layer_name = f"bert.encoder.layer.{encoder_idx}.output"

        if tt_cache_path is not None:
            self.attention_output_weight = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{attn_layer_name}.dense.weight_{self.model_config['OP7_SELFOUT_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP7_SELFOUT_WEIGHTS_MEMCFG"])
            self.attention_output_bias = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{attn_layer_name}.dense.bias_{self.model_config['OP7_SELFOUT_BIAS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP7_SELFOUT_BIAS_MEMCFG"])
            self.mha_gamma = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{attn_layer_name}.LayerNorm.weight_{self.model_config['OP8_LAYERNORM_GAMMA_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP8_LAYERNORM_GAMMA_MEMCFG"])
            self.mha_beta = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{attn_layer_name}.LayerNorm.bias_{self.model_config['OP8_LAYERNORM_BETA_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP8_LAYERNORM_BETA_MEMCFG"])
            self.ffn_gamma = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layer_name}.LayerNorm.weight_{self.model_config['OP11_LAYERNORM_GAMMA_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP11_LAYERNORM_GAMMA_MEMCFG"])
            self.ffn_beta = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layer_name}.LayerNorm.bias_{self.model_config['OP11_LAYERNORM_BETA_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP11_LAYERNORM_BETA_MEMCFG"])
        else:
            self.attention_output_weight = pad_weight(
                torch.transpose(
                    state_dict[f"{attn_layer_name}.dense.weight"],
                    -2,
                    -1,
                )
            )
            self.attention_output_weight = (
                tt_lib.tensor.Tensor(
                    self.attention_output_weight.reshape(-1).tolist(),
                    self.attention_output_weight.shape,
                    model_config["OP7_SELFOUT_WEIGHTS_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(device, model_config["OP7_SELFOUT_WEIGHTS_MEMCFG"])
            )
            self.attention_output_bias = pad_weight(state_dict[f"{attn_layer_name}.dense.bias"])
            self.attention_output_bias = (
                tt_lib.tensor.Tensor(
                    self.attention_output_bias.reshape(-1).tolist(),
                    self.attention_output_bias.shape,
                    model_config["OP7_SELFOUT_BIAS_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(device, model_config["OP7_SELFOUT_BIAS_MEMCFG"])
            )

            # MHA layernorm
            gamma0 = state_dict[f"{attn_layer_name}.LayerNorm.weight"]
            beta0 = state_dict[f"{attn_layer_name}.LayerNorm.bias"]
            mha_gamma = gamma0.reshape(1, 1, -1, 32)
            self.mha_gamma = tt_lib.tensor.Tensor(
                mha_gamma.reshape(-1).tolist(),
                mha_gamma.shape,
                model_config["OP8_LAYERNORM_GAMMA_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            ).to(device, model_config["OP8_LAYERNORM_GAMMA_MEMCFG"])
            mha_beta = beta0.reshape(1, 1, -1, 32)
            self.mha_beta = tt_lib.tensor.Tensor(
                mha_beta.reshape(-1).tolist(),
                mha_beta.shape,
                model_config["OP8_LAYERNORM_BETA_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            ).to(device, model_config["OP8_LAYERNORM_BETA_MEMCFG"])

            # FFN layernorm
            gamma1 = state_dict[f"{layer_name}.LayerNorm.weight"]
            beta1 = state_dict[f"{layer_name}.LayerNorm.bias"]
            ffn_gamma = gamma1.reshape(1, 1, -1, 32)
            self.ffn_gamma = tt_lib.tensor.Tensor(
                ffn_gamma.reshape(-1).tolist(),
                ffn_gamma.shape,
                model_config["OP11_LAYERNORM_GAMMA_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            ).to(device, model_config["OP11_LAYERNORM_GAMMA_MEMCFG"])
            ffn_beta = beta1.reshape(1, 1, -1, 32)
            self.ffn_beta = tt_lib.tensor.Tensor(
                ffn_beta.reshape(-1).tolist(),
                ffn_beta.shape,
                model_config["OP11_LAYERNORM_BETA_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            ).to(device, model_config["OP11_LAYERNORM_BETA_MEMCFG"])

            # configs
            self.selfout_program_config = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=4,
                out_subblock_h=2,
                out_subblock_w=4,
                per_core_M=12,
                per_core_N=4,
                transpose_mcast=True,
                fused_activation=None,
            )
            self.mha_ln_program_config = tt_lib.operations.primary.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid_size,
                subblock_w=4,
                block_h=12,
                block_w=4,
                math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
                im_data_format=model_config["OP8_LAYERNORM_GAMMA_DTYPE"],
                out_data_format=model_config["OP8_LAYERNORM_OUTPUT_DTYPE"],
                inplace=True,
            )
            self.ffn_ln_program_config = tt_lib.operations.primary.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid_size,
                subblock_w=4,
                block_h=12,
                block_w=4,
                math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
                im_data_format=model_config["OP11_LAYERNORM_GAMMA_DTYPE"],
                out_data_format=model_config["OP11_LAYERNORM_OUTPUT_DTYPE"],
                inplace=True,
            )

        # FFN sub-graph
        self.ffn = TtFeedForwardModel(encoder_idx, state_dict, device, grid_size, model_config, tt_cache_path)

        self.layer_norm_eps = config.layer_norm_eps

    def op7_mm_plus_bias(self, mha_res, attention_output_weight, attention_output_bias):
        mha_out = tt_lib.operations.primary.matmul(
            mha_res,
            attention_output_weight,
            bias=attention_output_bias,
            program_config=self.selfout_program_config,
            output_mem_config=self.model_config["OP7_SELFOUT_OUTPUT_MEMCFG"],
            math_fidelity=tt_lib.tensor.MathFidelity.LoFi,
        )
        return mha_out

    def op8_add_layernorm(self, activation, mha_out):
        mha_out_add_and_norm = tt_lib.operations.primary.add_layernorm(
            activation,
            mha_out,
            self.layer_norm_eps,
            self.mha_gamma,
            self.mha_beta,
            output_mem_config=self.model_config["OP8_LAYERNORM_OUTPUT_MEMCFG"],
            program_config=self.mha_ln_program_config,
        )
        return mha_out_add_and_norm

    def op11_add_layernorm(self, mha_out_add_and_norm, ffn_out):
        ffn_out_add_and_norm = tt_lib.operations.primary.add_layernorm(
            mha_out_add_and_norm,
            ffn_out,
            self.layer_norm_eps,
            self.ffn_gamma,
            self.ffn_beta,
            output_mem_config=self.model_config["OP11_LAYERNORM_OUTPUT_MEMCFG"],
            program_config=self.ffn_ln_program_config,
        )
        return ffn_out_add_and_norm

    def forward(
        self, activation: tt_lib.tensor.Tensor, attention_mask: Optional[tt_lib.tensor.Tensor] = None
    ) -> tt_lib.tensor.Tensor:
        activation_shape = activation.shape()

        # MHA - OP1 - OP10 ------------------------------->
        mha_res = self.mha(activation, attention_mask)

        # do it only for  testing
        mha_out = self.op7_mm_plus_bias(activation, self.attention_output_weight, self.attention_output_bias)
        mha_res.deallocate()
        mha_out_add_and_norm = self.op8_add_layernorm(activation, mha_out)
        activation.deallocate()
        mha_out.deallocate()

        # FFN - OP9 - OP10 ----------------------------->
        ffn_out = self.ffn(mha_out_add_and_norm)

        ffn_out_add_and_norm = self.op11_add_layernorm(mha_out_add_and_norm, ffn_out)
        mha_out_add_and_norm.deallocate()
        ffn_out.deallocate()
        return ffn_out_add_and_norm
