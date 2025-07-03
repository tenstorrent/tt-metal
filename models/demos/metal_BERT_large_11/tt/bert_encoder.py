# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import torch
from tt_lib.utils import pad_weight

import ttnn
from models.demos.metal_BERT_large_11.tt import custom_matmuls
from models.demos.metal_BERT_large_11.tt.ffn import TtFeedForwardModel
from models.demos.metal_BERT_large_11.tt.mha import TtMultiHeadAttentionModel
from models.demos.metal_BERT_large_11.tt.tensor_utils import load_or_compute_and_cache


class TtBertEncoder:
    def __init__(self, config, encoder_idx, state_dict, device, model_config, tt_cache_path):
        self.device = device
        self.model_config = model_config

        # MHA sub-graph
        self.mha = TtMultiHeadAttentionModel(config, encoder_idx, state_dict, device, model_config, tt_cache_path)

        attn_layer_name = f"bert.encoder.layer.{encoder_idx}.attention.output"
        layer_name = f"bert.encoder.layer.{encoder_idx}.output"

        attention_output_weight_path = None
        attention_output_bias_path = None
        mha_gamma_path = None
        mha_beta_path = None
        ffn_gamma_path = None
        ffn_beta_path = None

        if tt_cache_path is not None:
            attention_output_weight_path = str(
                f"{tt_cache_path}/"
                f"{attn_layer_name}.dense.weight_{self.model_config['OP7_SELFOUT_WEIGHTS_DTYPE'].name}.bin"
            )
            attention_output_bias_path = str(
                f"{tt_cache_path}/"
                f"{attn_layer_name}.dense.bias_{self.model_config['OP7_SELFOUT_BIAS_DTYPE'].name}.bin"
            )
            mha_gamma_path = str(
                f"{tt_cache_path}/"
                f"{attn_layer_name}.LayerNorm.weight_{self.model_config['OP8_LAYERNORM_GAMMA_DTYPE'].name}.bin"
            )
            mha_beta_path = str(
                f"{tt_cache_path}/"
                f"{attn_layer_name}.LayerNorm.bias_{self.model_config['OP8_LAYERNORM_BETA_DTYPE'].name}.bin"
            )
            ffn_gamma_path = str(
                f"{tt_cache_path}/"
                f"{layer_name}.LayerNorm.weight_{self.model_config['OP11_LAYERNORM_GAMMA_DTYPE'].name}.bin"
            )
            ffn_beta_path = str(
                f"{tt_cache_path}/"
                f"{layer_name}.LayerNorm.bias_{self.model_config['OP11_LAYERNORM_BETA_DTYPE'].name}.bin"
            )

        def compute_attention_output_weight():
            weight_torch = pad_weight(
                torch.transpose(
                    state_dict[f"{attn_layer_name}.dense.weight"],
                    -2,
                    -1,
                )
            )
            return ttnn.from_torch(
                weight_torch,
                model_config["OP7_SELFOUT_WEIGHTS_DTYPE"],
                layout=ttnn.TILE_LAYOUT,
            )

        def compute_attention_output_bias():
            bias_torch = pad_weight(state_dict[f"{attn_layer_name}.dense.bias"])
            return ttnn.from_torch(
                bias_torch,
                model_config["OP7_SELFOUT_BIAS_DTYPE"],
                layout=ttnn.TILE_LAYOUT,
            )

        def compute_mha_gamma():
            gamma_torch = state_dict[f"{attn_layer_name}.LayerNorm.weight"].reshape(1, 1, -1, 32)
            return ttnn.from_torch(
                gamma_torch,
                model_config["OP8_LAYERNORM_GAMMA_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        def compute_mha_beta():
            beta_torch = state_dict[f"{attn_layer_name}.LayerNorm.bias"].reshape(1, 1, -1, 32)
            return ttnn.from_torch(
                beta_torch,
                model_config["OP8_LAYERNORM_BETA_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        def compute_ffn_gamma():
            gamma_torch = state_dict[f"{layer_name}.LayerNorm.weight"].reshape(1, 1, -1, 32)
            return ttnn.from_torch(
                gamma_torch,
                model_config["OP11_LAYERNORM_GAMMA_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        def compute_ffn_beta():
            beta_torch = state_dict[f"{layer_name}.LayerNorm.bias"].reshape(1, 1, -1, 32)
            return ttnn.from_torch(
                beta_torch,
                model_config["OP11_LAYERNORM_BETA_DTYPE"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        self.attention_output_weight = load_or_compute_and_cache(
            attention_output_weight_path,
            compute_attention_output_weight,
            device=device,
            mem_config=model_config["OP7_SELFOUT_WEIGHTS_MEMCFG"],
        )
        self.attention_output_bias = load_or_compute_and_cache(
            attention_output_bias_path,
            compute_attention_output_bias,
            device=device,
            mem_config=model_config["OP7_SELFOUT_BIAS_MEMCFG"],
        )
        self.mha_gamma = load_or_compute_and_cache(
            mha_gamma_path,
            compute_mha_gamma,
            device=device,
            mem_config=model_config["OP8_LAYERNORM_GAMMA_MEMCFG"],
        )
        self.mha_beta = load_or_compute_and_cache(
            mha_beta_path,
            compute_mha_beta,
            device=device,
            mem_config=model_config["OP8_LAYERNORM_BETA_MEMCFG"],
        )
        self.ffn_gamma = load_or_compute_and_cache(
            ffn_gamma_path,
            compute_ffn_gamma,
            device=device,
            mem_config=model_config["OP11_LAYERNORM_GAMMA_MEMCFG"],
        )
        self.ffn_beta = load_or_compute_and_cache(
            ffn_beta_path,
            compute_ffn_beta,
            device=device,
            mem_config=model_config["OP11_LAYERNORM_BETA_MEMCFG"],
        )

        # FFN sub-graph
        self.ffn = TtFeedForwardModel(encoder_idx, state_dict, device, model_config, tt_cache_path)

        self.layer_norm_eps = config.layer_norm_eps

        if "OP7_SELFOUT_CONFIG" in model_config:

            def op7_mm_plus_bias(mha_res, attention_output_weight, attention_output_bias):
                mha_out = ttnn.linear(
                    mha_res,
                    attention_output_weight,
                    bias=attention_output_bias,
                    program_config=model_config["OP7_SELFOUT_CONFIG"],
                    memory_config=model_config["OP7_SELFOUT_OUTPUT_MEMCFG"],
                    dtype=model_config["OP7_SELFOUT_OUTPUT_DTYPE"],
                )
                return mha_out

        else:

            def op7_mm_plus_bias(mha_res, attention_output_weight, attention_output_bias):
                mha_out = custom_matmuls.bert_large_selfout_matmul(
                    mha_res,
                    attention_output_weight,
                    bias=attention_output_bias,
                    output_mem_config=model_config["OP7_SELFOUT_OUTPUT_MEMCFG"],
                    output_dtype=model_config["OP7_SELFOUT_OUTPUT_DTYPE"],
                )
                return mha_out

        self.op7_mm_plus_bias = op7_mm_plus_bias
        self.mha_ln_program_config = model_config.get("OP8_LAYERNORM_CONFIG", ttnn.LayerNormDefaultProgramConfig())
        self.ffn_ln_program_config = model_config.get("OP11_LAYERNORM_CONFIG", ttnn.LayerNormDefaultProgramConfig())

    def op8_add_layernorm(self, activation, mha_out):
        mha_out_add_and_norm = ttnn.layer_norm(
            activation,
            residual_input_tensor=mha_out,
            epsilon=self.layer_norm_eps,
            weight=self.mha_gamma,
            bias=self.mha_beta,
            program_config=self.mha_ln_program_config,
            memory_config=self.model_config["OP8_LAYERNORM_OUTPUT_MEMCFG"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
        )
        return mha_out_add_and_norm

    def op11_add_layernorm(self, mha_out_add_and_norm, ffn_out):
        ffn_out_add_and_norm = ttnn.layer_norm(
            mha_out_add_and_norm,
            residual_input_tensor=ffn_out,
            epsilon=self.layer_norm_eps,
            weight=self.ffn_gamma,
            bias=self.ffn_beta,
            program_config=self.ffn_ln_program_config,
            memory_config=self.model_config["OP11_LAYERNORM_OUTPUT_MEMCFG"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
        )
        return ffn_out_add_and_norm

    def __call__(self, activation: ttnn.Tensor, attention_mask: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        # MHA - OP1 - OP6 ------------------------------->
        mha_res = self.mha(activation, attention_mask)
        # Don't deallocate activations here since it is used by more ops

        mha_out = self.op7_mm_plus_bias(mha_res, self.attention_output_weight, self.attention_output_bias)
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
