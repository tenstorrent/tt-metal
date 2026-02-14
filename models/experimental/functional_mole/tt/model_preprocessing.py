# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

"""
def pad_channels_up_to_target(input_tensor, target=16):
    assert len(input_tensor.shape) == 4, "Expected input tensor to rank 4"
    N, C, H, W = input_tensor.shape
    if C < target:
        return torch.nn.functional.pad(input_tensor, (0, 0, 0, 0, 0, target - C), mode="constant", value=0)
    else:
        return input_tensor

"""


def create_mole_input_tensors(
    batch_size: int,
    seq_len: int,
    enc_in: int,
    time_features: int,
    torch_dtype=torch.float32,
    ttnn_dtype=ttnn.float32,
    device=None,
):
    torch_input_tensor = torch.randn(batch_size, seq_len, enc_in).to(torch_dtype)
    torch_input_x_tensor = torch.randn(batch_size, seq_len, time_features).to(torch_dtype)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, device=device)
    ttnn_input_x_tensor = ttnn.from_torch(torch_input_x_tensor, dtype=ttnn_dtype, device=device)

    return torch_input_tensor, torch_input_x_tensor, ttnn_input_tensor, ttnn_input_x_tensor


"""

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
    """
