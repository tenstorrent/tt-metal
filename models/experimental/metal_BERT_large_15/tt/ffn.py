# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from loguru import logger

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
):
    # Weights pre-transposed on host​. No on-the fly transpose of W.
    # activation = [1, 9, 384, 1024]
    # ff1_weighta = [1, 1, 1024, 4096]
    # output = [1, 9, 384, 4096]
    def op13_MM_bias_gelu(activation, ff1_weighta, ff1_biasa):
        output_plus_bias_act = ttl.tensor.bert_large_ff1_matmul(
            activation,
            ff1_weighta,
            ff1_biasa,
            (ttl.tensor.FusibleActivation.GELU, True),
            output_mem_config=model_config["OP13_FF1_MM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP13_FF1_MM_OUTPUT_DTYPE"],
        )
        return output_plus_bias_act

    # activation = [1, 9, 384, 4096]
    # ff2_weighta = [1, 1, 4096, 1024]
    # output = [1, 9, 384, 1024]
    def op14_MM_bias(activation, ff2_weighta, ff2_biasa):

        output_plus_bias = ttl.tensor.bert_large_ff2_matmul(
            activation,
            ff2_weighta,
            ff2_biasa,
            output_mem_config=model_config["OP14_FF2_MM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP14_FF2_MM_OUTPUT_DTYPE"],
        )
        return output_plus_bias

    def feed_forward_(activation):
        ff1_output_plus_bias_act = op13_MM_bias_gelu(activation, ff1_weighta, ff1_biasa)

        # Don't deallocate activations here since it is used by more ops in encoder

        ff2_output_plus_bias = op14_MM_bias(
            ff1_output_plus_bias_act, ff2_weighta, ff2_biasa
        )
        ff1_output_plus_bias_act.deallocate()

        return ff2_output_plus_bias

    return feed_forward_


class TtFeedForwardModel(torch.nn.Module):
    def __init__(self, encoder_idx, state_dict, device, model_config):
        super().__init__()

        # FF1 params
        encoder0_ff1_weight = pad_weight(
            torch.transpose(
                state_dict[
                    f"bert.encoder.layer.{encoder_idx}.intermediate.dense.weight"
                ],
                -2,
                -1,
            )
        )
        encoder0_ff1_bias = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.intermediate.dense.bias"]
        )

        encoder0_ff1_weight_shape = encoder0_ff1_weight.shape
        encoder0_ff1_bias_shape = encoder0_ff1_bias.shape

        encoder0_ff1_weight = (
            ttl.tensor.Tensor(
                encoder0_ff1_weight.reshape(-1).tolist(),
                encoder0_ff1_weight.shape,
                model_config["OP13_FF1_MM_WEIGHTS_DTYPE"],
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device, model_config["OP13_FF1_MM_WEIGHTS_MEMCFG"])
        )
        encoder0_ff1_bias = (
            ttl.tensor.Tensor(
                encoder0_ff1_bias.reshape(-1).tolist(),
                encoder0_ff1_bias.shape,
                model_config["OP13_FF1_MM_BIAS_DTYPE"],
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device, model_config["OP13_FF1_MM_BIAS_MEMCFG"])
        )

        # FF2 params
        encoder0_ff2_weight = pad_weight(
            torch.transpose(
                state_dict[f"bert.encoder.layer.{encoder_idx}.output.dense.weight"],
                -2,
                -1,
            )
        )
        encoder0_ff2_bias = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.output.dense.bias"]
        )

        encoder0_ff2_weight_shape = encoder0_ff2_weight.shape
        encoder0_ff2_bias_shape = encoder0_ff2_bias.shape

        encoder0_ff2_weight = (
            ttl.tensor.Tensor(
                encoder0_ff2_weight.reshape(-1).tolist(),
                encoder0_ff2_weight.shape,
                model_config["OP14_FF2_MM_WEIGHTS_DTYPE"],
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device, model_config["OP14_FF2_MM_WEIGHTS_MEMCFG"])
        )
        encoder0_ff2_bias = (
            ttl.tensor.Tensor(
                encoder0_ff2_bias.reshape(-1).tolist(),
                encoder0_ff2_bias.shape,
                model_config["OP14_FF2_MM_BIAS_DTYPE"],
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device, model_config["OP14_FF2_MM_BIAS_MEMCFG"])
        )

        self.ffn = feed_forward(
            *encoder0_ff1_weight_shape[-2:],
            encoder0_ff1_weight,
            encoder0_ff1_bias,
            encoder0_ff2_weight,
            encoder0_ff2_bias,
            device,
            model_config,
        )

    def forward(self, activation):
        return self.ffn(activation)
