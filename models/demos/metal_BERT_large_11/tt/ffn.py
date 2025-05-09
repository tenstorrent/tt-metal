# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from tt_lib.utils import pad_weight

import ttnn
from models.demos.metal_BERT_large_11.tt import custom_matmuls
from models.demos.metal_BERT_large_11.tt.tensor_utils import load_or_compute_and_cache


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
    if "OP9_FF1_MM_CONFIG" in model_config:

        def op9_MM_bias_gelu(activation, ff1_weighta, ff1_biasa):
            output_plus_bias_act = ttnn.linear(
                activation,
                ff1_weighta,
                bias=ff1_biasa,
                program_config=model_config["OP9_FF1_MM_CONFIG"],
                memory_config=model_config["OP9_FF1_MM_OUTPUT_MEMCFG"],
                dtype=model_config["OP9_FF1_MM_OUTPUT_DTYPE"],
            )
            return output_plus_bias_act

    else:

        def op9_MM_bias_gelu(activation, ff1_weighta, ff1_biasa):
            output_plus_bias_act = custom_matmuls.bert_large_ff1_matmul(
                activation,
                ff1_weighta,
                bias=ff1_biasa,
                fused_activation=(ttnn.UnaryOpType.GELU, True),
                output_mem_config=model_config["OP9_FF1_MM_OUTPUT_MEMCFG"],
                output_dtype=model_config["OP9_FF1_MM_OUTPUT_DTYPE"],
            )
            return output_plus_bias_act

    if "OP10_FF2_MM_CONFIG" in model_config:

        def op10_MM_bias(activation, ff2_weighta, ff2_biasa):
            output_plus_bias = ttnn.linear(
                activation,
                ff2_weighta,
                bias=ff2_biasa,
                program_config=model_config["OP10_FF2_MM_CONFIG"],
                memory_config=model_config["OP10_FF2_MM_OUTPUT_MEMCFG"],
                dtype=model_config["OP10_FF2_MM_OUTPUT_DTYPE"],
            )
            return output_plus_bias

    else:

        def op10_MM_bias(activation, ff2_weighta, ff2_biasa):
            output_plus_bias = custom_matmuls.bert_large_ff2_matmul(
                activation,
                ff2_weighta,
                bias=ff2_biasa,
                output_mem_config=model_config["OP10_FF2_MM_OUTPUT_MEMCFG"],
                output_dtype=model_config["OP10_FF2_MM_OUTPUT_DTYPE"],
            )
            return output_plus_bias

    def feed_forward_(activation: ttnn.Tensor) -> ttnn.Tensor:
        ff1_output_plus_bias_act = op9_MM_bias_gelu(activation, ff1_weighta, ff1_biasa)

        # Don't deallocate activations here since it is used by more ops in encoder

        ff2_output_plus_bias = op10_MM_bias(ff1_output_plus_bias_act, ff2_weighta, ff2_biasa)
        ff1_output_plus_bias_act.deallocate()

        return ff2_output_plus_bias

    return feed_forward_


class TtFeedForwardModel:
    def __init__(self, encoder_idx, state_dict, device, model_config, tt_cache_path):
        layer_name = f"bert.encoder.layer.{encoder_idx}"
        encoder_ff1_str = f"{layer_name}.intermediate.dense"
        encoder_ff2_str = f"{layer_name}.output.dense"

        ff1_weight_path = None
        ff1_bias_path = None
        ff2_weight_path = None
        ff2_bias_path = None

        if tt_cache_path is not None:
            ff1_weight_path = str(
                f"{tt_cache_path}/" f"{encoder_ff1_str}.weight_{model_config['OP9_FF1_MM_WEIGHTS_DTYPE'].name}.bin"
            )
            ff1_bias_path = str(
                f"{tt_cache_path}/" f"{encoder_ff1_str}.bias_{model_config['OP9_FF1_MM_BIAS_DTYPE'].name}.bin"
            )
            ff2_weight_path = str(
                f"{tt_cache_path}/" f"{encoder_ff2_str}.weight_{model_config['OP10_FF2_MM_WEIGHTS_DTYPE'].name}.bin"
            )
            ff2_bias_path = str(
                f"{tt_cache_path}/" f"{encoder_ff2_str}.bias_{model_config['OP10_FF2_MM_BIAS_DTYPE'].name}.bin"
            )

        def compute_ff1_weight():
            ff1_weight_torch = pad_weight(
                torch.transpose(
                    state_dict[f"{encoder_ff1_str}.weight"],
                    -2,
                    -1,
                )
            )
            return ttnn.from_torch(
                ff1_weight_torch,
                dtype=model_config["OP9_FF1_MM_WEIGHTS_DTYPE"],
                layout=ttnn.TILE_LAYOUT,
            )

        def compute_ff1_bias():
            ff1_bias_torch = pad_weight(state_dict[f"{encoder_ff1_str}.bias"])
            return ttnn.from_torch(
                ff1_bias_torch,
                dtype=model_config["OP9_FF1_MM_BIAS_DTYPE"],
                layout=ttnn.TILE_LAYOUT,
            )

        def compute_ff2_weight():
            ff2_weight_torch = pad_weight(
                torch.transpose(
                    state_dict[f"{encoder_ff2_str}.weight"],
                    -2,
                    -1,
                )
            )
            return ttnn.from_torch(
                ff2_weight_torch,
                dtype=model_config["OP10_FF2_MM_WEIGHTS_DTYPE"],
                layout=ttnn.TILE_LAYOUT,
            )

        def compute_ff2_bias():
            ff2_bias_torch = pad_weight(state_dict[f"{encoder_ff2_str}.bias"])
            return ttnn.from_torch(
                ff2_bias_torch,
                dtype=model_config["OP10_FF2_MM_BIAS_DTYPE"],
                layout=ttnn.TILE_LAYOUT,
            )

        encoder0_ff1_weight = load_or_compute_and_cache(
            ff1_weight_path,
            compute_ff1_weight,
            device=device,
            mem_config=model_config["OP9_FF1_MM_WEIGHTS_MEMCFG"],
        )
        encoder0_ff1_bias = load_or_compute_and_cache(
            ff1_bias_path,
            compute_ff1_bias,
            device=device,
            mem_config=model_config["OP9_FF1_MM_BIAS_MEMCFG"],
        )
        encoder0_ff2_weight = load_or_compute_and_cache(
            ff2_weight_path,
            compute_ff2_weight,
            device=device,
            mem_config=model_config["OP10_FF2_MM_WEIGHTS_MEMCFG"],
        )
        encoder0_ff2_bias = load_or_compute_and_cache(
            ff2_bias_path,
            compute_ff2_bias,
            device=device,
            mem_config=model_config["OP10_FF2_MM_BIAS_MEMCFG"],
        )

        encoder0_ff1_weight_shape = encoder0_ff1_weight.padded_shape

        self.ffn = feed_forward(
            *tuple(encoder0_ff1_weight_shape)[-2:],
            encoder0_ff1_weight,
            encoder0_ff1_bias,
            encoder0_ff2_weight,
            encoder0_ff2_bias,
            device,
            model_config,
        )

    def __call__(self, activation: ttnn.Tensor) -> ttnn.Tensor:
        return self.ffn(activation)
