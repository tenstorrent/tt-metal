# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

import tt_lib as ttl
from tt_lib.utils import pad_weight


def feed_forward(
    ffn_dim,
    hidden_dim,
    ff1_weighta,
    ff1_biasa,
    ff2_weighta,
    ff2_biasa,
    device,
    model_config,
    ff1_fused_program_config,
    ff2_fused_program_config,
):
    def op9_MM_bias_gelu(activation, ff1_weighta, ff1_biasa):
        output_plus_bias_act = ttl.operations.primary.matmul(
            activation,
            ff1_weighta,
            bias=ff1_biasa,
            program_config=ff1_fused_program_config,
            output_mem_config=model_config["OP9_FF1_MM_OUTPUT_MEMCFG"],
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
        )
        return output_plus_bias_act

    def op10_MM_bias(activation, ff2_weighta, ff2_biasa):
        output_plus_bias = ttl.operations.primary.matmul(
            activation,
            ff2_weighta,
            bias=ff2_biasa,
            program_config=ff2_fused_program_config,
            output_mem_config=model_config["OP10_FF2_MM_OUTPUT_MEMCFG"],
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
        )
        return output_plus_bias

    def feed_forward_(activation: ttl.tensor.Tensor) -> ttl.tensor.Tensor:
        ff1_output_plus_bias_act = op9_MM_bias_gelu(activation, ff1_weighta, ff1_biasa)
        ff2_output_plus_bias = op10_MM_bias(ff1_output_plus_bias_act, ff2_weighta, ff2_biasa)
        ff1_output_plus_bias_act.deallocate()

        return ff2_output_plus_bias

    return feed_forward_


class TtFeedForwardModel(torch.nn.Module):
    def __init__(self, encoder_idx, state_dict, device, grid_size, model_config, tt_cache_path):
        super().__init__()

        # FF1 params
        layer_name = f"bert.encoder.layer.{encoder_idx}"
        encoder_ff1_str = f"{layer_name}.intermediate.dense"
        encoder_ff2_str = f"{layer_name}.output.dense"
        if tt_cache_path is not None:
            encoder0_ff1_weight = ttl.tensor.load_tensor(
                str(tt_cache_path / f"{encoder_ff1_str}.weight_{model_config['OP9_FF1_MM_WEIGHTS_DTYPE'].name}.bin")
            ).to(device, model_config["OP9_FF1_MM_WEIGHTS_MEMCFG"])
            encoder0_ff1_bias = ttl.tensor.load_tensor(
                str(tt_cache_path / f"{encoder_ff1_str}.bias_{model_config['OP9_FF1_MM_BIAS_DTYPE'].name}.bin")
            ).to(device, model_config["OP9_FF1_MM_BIAS_MEMCFG"])
            encoder0_ff1_weight_shape = encoder0_ff1_weight.shape()

            encoder0_ff2_weight = ttl.tensor.load_tensor(
                str(tt_cache_path / f"{encoder_ff2_str}.weight_{model_config['OP10_FF2_MM_WEIGHTS_DTYPE'].name}.bin")
            ).to(device, model_config["OP10_FF2_MM_WEIGHTS_MEMCFG"])
            encoder0_ff2_bias = ttl.tensor.load_tensor(
                str(tt_cache_path / f"{encoder_ff2_str}.bias_{model_config['OP10_FF2_MM_BIAS_DTYPE'].name}.bin")
            ).to(device, model_config["OP10_FF2_MM_BIAS_MEMCFG"])
        else:
            encoder0_ff1_weight = pad_weight(
                torch.transpose(
                    state_dict[f"{encoder_ff1_str}.weight"],
                    -2,
                    -1,
                )
            )
            encoder0_ff1_bias = pad_weight(state_dict[f"{encoder_ff1_str}.bias"])

            encoder0_ff1_weight_shape = encoder0_ff1_weight.shape
            encoder0_ff1_bias_shape = encoder0_ff1_bias.shape

            encoder0_ff1_weight = (
                ttl.tensor.Tensor(
                    encoder0_ff1_weight.reshape(-1).tolist(),
                    encoder0_ff1_weight.shape,
                    model_config["OP9_FF1_MM_WEIGHTS_DTYPE"],
                    ttl.tensor.Layout.ROW_MAJOR,
                )
                .to(ttl.tensor.Layout.TILE)
                .to(device, model_config["OP9_FF1_MM_WEIGHTS_MEMCFG"])
            )
            encoder0_ff1_bias = (
                ttl.tensor.Tensor(
                    encoder0_ff1_bias.reshape(-1).tolist(),
                    encoder0_ff1_bias.shape,
                    model_config["OP9_FF1_MM_BIAS_DTYPE"],
                    ttl.tensor.Layout.ROW_MAJOR,
                )
                .to(ttl.tensor.Layout.TILE)
                .to(device, model_config["OP9_FF1_MM_BIAS_MEMCFG"])
            )

            # FF2 params
            encoder0_ff2_weight = pad_weight(
                torch.transpose(
                    state_dict[f"{encoder_ff2_str}.weight"],
                    -2,
                    -1,
                )
            )
            encoder0_ff2_bias = pad_weight(state_dict[f"{encoder_ff2_str}.bias"])

            encoder0_ff2_weight_shape = encoder0_ff2_weight.shape
            encoder0_ff2_bias_shape = encoder0_ff2_bias.shape

            encoder0_ff2_weight = (
                ttl.tensor.Tensor(
                    encoder0_ff2_weight.reshape(-1).tolist(),
                    encoder0_ff2_weight.shape,
                    model_config["OP10_FF2_MM_WEIGHTS_DTYPE"],
                    ttl.tensor.Layout.ROW_MAJOR,
                )
                .to(ttl.tensor.Layout.TILE)
                .to(device, model_config["OP10_FF2_MM_WEIGHTS_MEMCFG"])
            )
            encoder0_ff2_bias = (
                ttl.tensor.Tensor(
                    encoder0_ff2_bias.reshape(-1).tolist(),
                    encoder0_ff2_bias.shape,
                    model_config["OP10_FF2_MM_BIAS_DTYPE"],
                    ttl.tensor.Layout.ROW_MAJOR,
                )
                .to(ttl.tensor.Layout.TILE)
                .to(device, model_config["OP10_FF2_MM_BIAS_MEMCFG"])
            )

        # configs
        ff1_fused_program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=12,
            per_core_N=16,
            transpose_mcast=True,
            fused_activation=(ttl.tensor.FusibleActivation.GELU, True),
        )
        ff2_fused_program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=16,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=12,
            per_core_N=4,
            transpose_mcast=True,
            fused_activation=(ttl.tensor.FusibleActivation.GELU, True),
        )

        self.ffn = feed_forward(
            *encoder0_ff1_weight_shape[-2:],
            encoder0_ff1_weight,
            encoder0_ff1_bias,
            encoder0_ff2_weight,
            encoder0_ff2_bias,
            device,
            model_config,
            ff1_fused_program_config,
            ff2_fused_program_config,
        )

    def forward(self, activation: ttl.tensor.Tensor) -> ttl.tensor.Tensor:
        return self.ffn(activation)
