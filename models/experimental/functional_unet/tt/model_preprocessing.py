# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ttnn.model_preprocessing import infer_ttnn_module_args

from models.experimental.functional_unet.tt import unet_shallow_torch

# Import from common location for backward compatibility


def create_unet_model_parameters(
    model: unet_shallow_torch.UNet, input_tensor: torch.Tensor, groups: int, device: ttnn.Device
):
    parameters = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.keys():
        parameters[key].module = getattr(model, key)

    parameters.c1["conv_blocking_and_parallelization_config_override"] = None
    parameters.c1["use_activation_double_buffer"] = True
    parameters.c1["enable_activation_reuse"] = True
    parameters.c1_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 12 * 32}
    parameters.c1_2["use_activation_double_buffer"] = True

    parameters.c2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c2["use_activation_double_buffer"] = True
    parameters.c2["enable_activation_reuse"] = True
    parameters.c2_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c2_2["use_activation_double_buffer"] = True
    parameters.c2_2["enable_activation_reuse"] = True

    parameters.c3["conv_blocking_and_parallelization_config_override"] = None
    parameters.c3["use_activation_double_buffer"] = True
    parameters.c3_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c3_2["use_activation_double_buffer"] = True

    parameters.c4["conv_blocking_and_parallelization_config_override"] = None
    parameters.c4["use_activation_double_buffer"] = True
    parameters.c4_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c4_2["use_activation_double_buffer"] = True

    parameters.bnc["conv_blocking_and_parallelization_config_override"] = None
    parameters.bnc["use_activation_double_buffer"] = False
    parameters.bnc_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.bnc_2["use_activation_double_buffer"] = False

    parameters.c5["conv_blocking_and_parallelization_config_override"] = None
    parameters.c5["use_activation_double_buffer"] = False
    parameters.c5_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c5_2["use_activation_double_buffer"] = False
    parameters.c5_3["conv_blocking_and_parallelization_config_override"] = None
    parameters.c5_3["use_activation_double_buffer"] = False

    parameters.c6["conv_blocking_and_parallelization_config_override"] = None
    parameters.c6["use_activation_double_buffer"] = True
    parameters.c6_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c6_2["use_activation_double_buffer"] = True
    parameters.c6_3["conv_blocking_and_parallelization_config_override"] = None
    parameters.c6_3["use_activation_double_buffer"] = True

    parameters.c7["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 3 * 32}
    parameters.c7["use_activation_double_buffer"] = True
    parameters.c7_2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c7_2["use_activation_double_buffer"] = True
    parameters.c7_2["enable_activation_reuse"] = True
    parameters.c7_3["conv_blocking_and_parallelization_config_override"] = None
    parameters.c7_3["use_activation_double_buffer"] = True
    parameters.c7_3["enable_activation_reuse"] = True

    parameters.c8["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 4 * 32}
    parameters.c8["use_activation_double_buffer"] = True
    parameters.c8_2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 12 * 32}
    parameters.c8_2["use_activation_double_buffer"] = True
    parameters.c8_3["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 12 * 32}

    parameters.c8_3["use_activation_double_buffer"] = True

    parameters.output_layer["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 42 * 32}
    parameters.output_layer["use_activation_double_buffer"] = True

    return parameters
